from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import random
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
import copy
from seiko.surrogate_model import SurrogateModel

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
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

    elif config.sd_model == "sdxl":
        from d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
        # only tested and works with DDIM for now
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipeline.enable_vae_slicing()

        num_steps = 8
        guidance_scale = 0.0

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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    wandb_name = f"seiko-{config.name_subfix}" if config.name_subfix else "seiko"
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
        training_unet = pipeline.unet
        trainable_parameters = filter(lambda p: p.requires_grad, training_unet.parameters())
    else:
        pipeline.unet.requires_grad_(True)
        training_unet = pipeline.unet
        trainable_parameters = unet.parameters()
    
    current_unet = copy.deepcopy(training_unet)
    current_unet.requires_grad_(False)

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        # not saving the models[0] (surrogate model)
        assert isinstance(models[-1], type(pipeline.unet))
        if config.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[-1]))
            pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not config.use_lora:
            models[-1].save_pretrained(os.path.join(output_dir, "unet"))
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

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # prepare prompt and reward fn
    prompt_fn = getattr(d3po_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(d3po_pytorch.rewards, config.reward_fn)()


    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    surrogate_model = SurrogateModel()
    surrogate_model = surrogate_model.to(accelerator.device)
    surrogate_model.model = accelerator.prepare(surrogate_model.model)

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

    num_samples_per_outerloop = config.num_samples_per_outerloop
    global_step = 0
    for outer_loop, num_samples in enumerate(num_samples_per_outerloop):

        #################### SAMPLING ####################
        pipeline.unet.eval()

        assert num_samples % config.sample.batch_size == 0
        num_samples_batches = num_samples // config.sample.batch_size

        all_images_tensor = []
        all_prompts = []

        # config.sample.num_batches_per_epoch is not used
        for i in tqdm(
            range(num_samples_batches),
            desc="Obtain fresh samples and feedbacks",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            prompts = list(prompts)
            all_prompts.extend(prompts)

            # sample
            with autocast():
                images_tensor, latents, log_probs, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_dict=False,
                )

            all_images_tensor.append(images_tensor)
        
        all_images_tensor = torch.cat(all_images_tensor, dim=0)

        assert(len(all_images_tensor) == num_samples), "Number of fresh online samples does not match the target number" 

        ##### Generate a feedback y(i) = r(x(i)) + ε, (we set ε = 0)
        scores, outputs = reward_fn(all_images_tensor, all_prompts, [{}]*num_samples)

        ##### Construct a new dataset: D(i) = D(i−1) + (x(i), y(i))
        surrogate_model.update(all_images_tensor.type(torch.float32), scores)

        del all_images_tensor

        ##### Update a diffusion model as {p(i)} by finetuning.
        optimizer = torch.optim.AdamW(
            training_unet.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        # Prepare everything with our `accelerator`.
        training_unet, optimizer = accelerator.prepare(training_unet, optimizer)
    
        #################### TRAINING ####################
        for epoch in range(first_epoch, config.num_epochs):
            
            num_batches_per_epoch = config.train.total_samples_per_epoch // config.train.batch_size

            for inner_iters in tqdm(
                    list(range(num_batches_per_epoch)),
                    position=0,
                    disable=not accelerator.is_local_main_process
                ):

                with accelerator.accumulate(training_unet), autocast(), torch.enable_grad():
                    prompts, prompt_metadata = zip(
                        *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.train.batch_size)]
                    )
                    prompts = list(prompts)

                    # train
                    pipeline.unet.train()
                    info = defaultdict(list)

                    truncated_backprop_at = []
                    for i in range(num_steps):
                        if i < random.randint(0, num_steps):
                            truncated_backprop_at.append(i)
                    if len(truncated_backprop_at) == num_steps:
                        print("Warning: all timesteps are truncated!")

                    images_tensor, latents_trajectory, log_probs, training_prev_sample_means = pipeline_with_logprob(
                        pipeline,
                        prompt=prompts,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        return_dict=False,
                        enable_grad=True,
                        enable_grad_checkpointing=True,
                        truncated_backprop_at=truncated_backprop_at,
                    )
                    assert len(latents) == num_steps + 1

                    pipeline.unet = current_unet
                    _, _, _, current_prev_sample_means = pipeline_with_logprob(
                        pipeline,
                        prompt=prompts,
                        latents=latents_trajectory[0],
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        return_dict=False,
                        enable_grad=True,
                        enable_grad_checkpointing=True,
                        callback_on_step_end=lambda self, index, timestep, kwargs: { "latents": latents_trajectory[index+1] }, 
                    )
                    pipeline.unet = training_unet

                    kl_loss = 0.0
                    for timestep, training_prev_sample_mean, current_prev_sample_mean in zip(pipeline.scheduler.timesteps, training_prev_sample_means, current_prev_sample_means):
                        scheduler = pipeline.scheduler

                        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
                        variance = scheduler._get_variance(timestep, prev_timestep)
                        std_dev_t = config.sample.eta * variance ** (0.5)

                        kl_term = (training_prev_sample_mean - current_prev_sample_mean)**2 / (2 * (std_dev_t**2))
                        kl_term = kl_term.mean(dim=tuple(range(1, kl_term.ndim))) # (batch_size, C, H, W) -> (batch_size,)
                        kl_loss += kl_term.mean()

                    loss, rewards = surrogate_model(
                        images = images_tensor.type(torch.float32),
                        osm_alpha = config.train.osm_alpha,
                        osm_lambda = config.train.osm_lambda,
                        osm_clipping = config.train.osm_clipping,
                        aesthetic_target = config.aesthetic_target,
                        grad_scale = config.grad_scale
                    )
                    loss = loss.mean() * config.train.loss_coeff
                    
                    total_loss = loss + config.train.kl_weight*kl_loss
                    
                    rewards_mean = rewards.mean()
                    rewards_std = rewards.std()
                    
                    info["loss"].append(loss)
                    info["kl_loss"].append(kl_loss)
                    
                    # backward pass
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()  

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0

                    #################### LOGGING ####################
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({
                            "outer_loop": outer_loop,
                            "train/reward_mean": rewards_mean,
                            "train/dataset_size": len(surrogate_model),
                            "train/dataset_y_mean": torch.mean(surrogate_model.y),
                        }
                    )
                    
                    accelerator.log(info, step=global_step)
                    
                    global_step += 1
                    info = defaultdict(list)

        #################### EVALUATION PER OUTERLOOP ####################
        
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
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                caption = f"{prompt} | {reward:.2f}"
                if reward_meta is not None and "output" in reward_meta:
                    caption += f" | {reward_meta['output']}"
                wandb_images.append(
                    wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=caption)
                )
            accelerator.log({
                    "outer_loop": outer_loop,
                    "validation/images": wandb_images,
                    "validation/reward_mean": eval_rewards.mean(),
                },
                step=global_step-1 # log at the end of the training epoch
            )

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

            if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state()

            
        current_unet.load_state_dict(training_unet.state_dict())
        current_unet.requires_grad_(False)

if __name__ == "__main__":
    app.run(main)
