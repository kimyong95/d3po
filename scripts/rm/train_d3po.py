from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import convert_state_dict_to_diffusers
import numpy as np
import d3po_pytorch.prompts
import d3po_pytorch.rewards
from d3po_pytorch.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import uuid
import copy
from dotenv import load_dotenv

load_dotenv()

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + "_" + str(uuid.uuid4())[:8]
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # load scheduler, tokenizer and models.
    if config.sd_model == "sd1":
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipeline.enable_vae_slicing()

        num_steps = 50
        guidance_scale = 5.0

    elif config.sd_model == "sdxl-lightning":
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
        # only tested and works with DDIM for now
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipeline.enable_vae_slicing()
        pipeline.vae.encoder = None

        num_steps = 8
        guidance_scale = 0.0
    elif config.sd_model == "sdxl":
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        # only tested and works with DDIM for now
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipeline.enable_vae_slicing()
        pipeline.vae.encoder = None

        num_steps = 30
        guidance_scale = 7.0

    assert config.train.batch_size % 2 == 0, "Requires even batch size for D3PO for win/loss pairs"

    if config.fixed_prompt and config.prompt_fn == "fixed":
        config.prompt_fn_kwargs = { "prompt": config.fixed_prompt }

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    wandb_name = f"d3po-{config.name_subfix}" if config.name_subfix else "d3po"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="finetune-stable-diffusion",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": wandb_name, "mode": config.wandb_mode}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # freeze parameters of models to save more memory
    for k, c in pipeline.components.items():
        if isinstance(c, torch.nn.Module):
            c.requires_grad_(False)

    # pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline = pipeline.to(accelerator.device, dtype=inference_dtype)

    ref_unet = copy.deepcopy(pipeline.unet)
    ref_unet.requires_grad_(False)

    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:

        unet_lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        pipeline.unet.add_adapter(unet_lora_config)

        # To avoid accelerate unscaling problems in FP16, see: https://github.com/huggingface/peft/issues/341
        for param in pipeline.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        unet = pipeline.unet
        trainable_parameters = filter(lambda p: p.requires_grad, unet.parameters())
    else:
        pipeline.unet.requires_grad_(True)
        unet = pipeline.unet
        trainable_parameters = unet.parameters()

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not config.use_lora:
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Support multi-dimensional comparison. Default demension is 1. You can add many rewards instead of only one to judge the preference of images.
    # For example: A: clipscore-30 blipscore-10 LAION aesthetic score-6.0 ; B: 20, 8, 5.0  then A is prefered than B
    # if C: 40, 4, 4.0 since C[0] = 40 > A[0] and C[1] < A[1], we do not think C is prefered than A or A is prefered than C 
    def compare(a, b):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        if len(a.shape)==1:
            a = a[...,None]
            b = b[...,None]

        a_dominates = torch.logical_and(torch.all(a <= b, dim=1), torch.any(a < b, dim=1))
        b_dominates = torch.logical_and(torch.all(b <= a, dim=1), torch.any(b < a, dim=1))


        c = torch.zeros([a.shape[0],2],dtype=torch.float,device=a.device)

        c[a_dominates] = torch.tensor([-1., 1.],device=a.device)
        c[b_dominates] = torch.tensor([1., -1.],device=a.device)

        return c

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(d3po_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(d3po_pytorch.rewards, config.reward_fn)()
    reward_fn_choice = getattr(d3po_pytorch.rewards, config.reward_fn_choice)() if config.reward_fn_choice else None

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)


    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts_list = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            prompts = list(prompts)

            # sample
            with autocast():
                images, latents, log_probs, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_dict=False,
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 128, 128)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            rewards = reward_fn(images, prompts, prompt_metadata)

            samples.append(
                {
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs, # "log_probs" not used in d3po
                    "rewards": rewards,
                    "images": images,
                }
            )
            prompts_list.extend(prompts)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"]
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        
        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "epoch": epoch,
                "train/reward_mean": rewards.mean(),
            },
            step=global_step,
        )

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch)
        assert num_timesteps == num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            prompts = [prompts_list[i] for i in perm]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            prompts_batched = [ prompts[i:i+config.train.batch_size] for i in range(0, len(prompts), config.train.batch_size) ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, (sample, prompts) in tqdm(
                list(enumerate(zip(samples_batched, prompts_batched))),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):  
                sample["latents"].requires_grad = False
                unet = pipeline.unet
                pipeline.unet = ref_unet # warning: not multithread-safe
                _, _, ref_log_probs, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    latents=sample["latents"][:, 0], # x0
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_dict=False,
                    callback_on_step_end=lambda self, index, timestep, kwargs: { "latents": sample["next_latents"][:, index] },
                    log_prob_given_prev_sample=sample["next_latents"], # x1, ..., xT
                    enable_grad=False,
                )
                
                pipeline.unet = unet
                sample["latents"].requires_grad = True

                def callback_func(self, index, timestep, kwargs):
                    
                    nonlocal accelerator, optimizer, unet, info, global_step, ref_log_probs, sample
                    with accelerator.accumulate(unet), autocast():
                        log_prob = kwargs["log_prob"]
                        ref_log_prob = ref_log_probs[index]

                        # half batch size
                        hb = len(sample["rewards"]) // 2

                        if reward_fn_choice is not None: # use "which images is better" as reward function
                            choices, rewards_meta = reward_fn_choice(sample["images"][hb:], sample["images"][:hb], prompts, [{}] * hb)
                            choices_map = dict([
                                (0, [0., 0.]),
                                (1, [1., -1.]),
                                (2, [-1., 1.]),
                            ])
                            human_prefer = [ choices_map[c] for c in choices]
                            human_prefer = torch.tensor(human_prefer, device=accelerator.device)

                        else:
                            human_prefer = compare(sample["rewards"][hb:],sample["rewards"][:hb])

                        # clip the Q value
                        ratio_0 = torch.clamp(torch.exp(log_prob[hb:]-ref_log_prob[hb:]),1 - config.train.eps, 1 + config.train.eps)
                        ratio_1 = torch.clamp(torch.exp(log_prob[:hb]-ref_log_prob[:hb]),1 - config.train.eps, 1 + config.train.eps)
                        loss = -torch.log(torch.sigmoid(config.train.beta*(torch.log(ratio_0))*human_prefer[:,0] + config.train.beta*(torch.log(ratio_1))*human_prefer[:, 1])).mean()

                        # debugging values
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            assert (index == num_train_timesteps - 1) and (
                                i + 1
                            ) % config.train.gradient_accumulation_steps == 0
                            # log training-related stuff
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                            accelerator.log(info, step=global_step)
                            global_step += 1
                            info = defaultdict(list)

                    return {
                        "latents": sample["next_latents"][:, index]
                    }

                images, latents, log_probs, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    latents=sample["latents"][:, 0], # x0
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_dict=False,
                    callback_on_step_end=callback_func,
                    callback_on_step_end_tensor_inputs=["log_prob"],
                    log_probs_given_trajectory=sample["next_latents"], # x1, ..., xT
                    enable_grad=True,
                )
            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

        #################### EVALUATION ####################
        # (epoch+1) because model already trained for (epoch+1) epochs
        if (epoch+1)%config.sample.eval_epoch==0:
            if config.get("eval_prompt_fn") == "fixed":
                eval_prompts = config.eval_fixed_prompt
                assert len(eval_prompts) == config.sample.eval_batch_size
                eval_prompt_metadata = [{}] * config.sample.eval_batch_size
            else:
                eval_prompts, eval_prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.eval_batch_size)])
                eval_prompts = list(eval_prompts)
            
            eval_generator = torch.Generator(device=accelerator.device)
            eval_generator.manual_seed(config.seed+1)

            # sample
            with autocast():
                eval_images, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=eval_prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_dict=False,
                    generator=eval_generator,
                )

            eval_rewards, eval_rewards_meta = reward_fn(eval_images, eval_prompts, eval_prompt_metadata)
            eval_rewards = eval_rewards.cpu().numpy()

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                wandb_images = []
                for i, (image, prompt, reward, reward_meta) in enumerate(zip(eval_images, eval_prompts, eval_rewards, eval_rewards_meta)):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((512, 512))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                    caption = f"{prompt} | {reward:.2f}"
                    if reward_meta is not None and "output" in reward_meta:
                        caption += f" | {reward_meta['output']}"
                    wandb_images.append(
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=caption)
                    )
                accelerator.log({
                    "validation/images": wandb_images,
                    "validation/reward_mean": eval_rewards.mean(),
                    "epoch": epoch },
                    step=global_step-1 # log at the end of the training epoch
                )

if __name__ == "__main__":
    app.run(main)
