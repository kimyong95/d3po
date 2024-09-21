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
import einops
from PIL import Image
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import related_works.targetdiff.utils.transforms as trans
from torch_geometric.transforms import Compose
from related_works.targetdiff.datasets import get_dataset
from related_works.targetdiff.models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from related_works.targetdiff.scripts.sample_diffusion import sample_diffusion_ligand
from d3po_pytorch.targetdiff_patch.targetdiff_with_logprob import sample_diffusion_ligand_with_logprob
from related_works.targetdiff.utils.evaluation.docking_vina import VinaDockingTask, PrepLig
from dotenv import load_dotenv

load_dotenv()

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)



def resolve_targetdiff_relative_dir(path):
    targetdiff_root = "../../related_works/targetdiff"
    return os.path.normpath(os.path.join(targetdiff_root, path))

def get_targetdiff(data_id=0):

    ckpt_path = "./pretrained_models/pretrained_diffusion.pt"

    ckpt = torch.load(resolve_targetdiff_relative_dir(ckpt_path))

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    ckpt['config'].data["path"] = resolve_targetdiff_relative_dir(ckpt['config'].data["path"])
    ckpt['config'].data["split"] = resolve_targetdiff_relative_dir(ckpt['config'].data["split"])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']

    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        scheduler="ddim",
    )
    model.load_state_dict(ckpt['model'])

    data = test_set[data_id]

    return model, data

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

    data_id = int(config.data_id)

    model, data = get_targetdiff(data_id=data_id)
    receptor_info = {
        "ligand_filename": data.ligand_filename,
        "protein_root": resolve_targetdiff_relative_dir("data/test_set"),
        "vina_web_url": config.vina_web_url if config.vina_web_url else None,
    }
    num_steps = 200

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

    wandb_name = f"ddpo-data={data_id}-{config.name_subfix}" if config.name_subfix else f"ddpo-data={data_id}"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="finetune-targetdiff",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": wandb_name, "mode": config.wandb_mode}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    model = model.to(accelerator.device, inference_dtype)
    if config.use_lora:
        
        target_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=[
                "hk_func.net.0", "hk_func.net.3",
                "hq_func.net.0", "hq_func.net.3",
                "hv_func.net.0", "hv_func.net.3",
                "refine_net.edge_pred_layer.net.0",
                "refine_net.edge_pred_layer.net.3",
            ],
        )
        model.requires_grad_(False)
        model = get_peft_model(model, lora_config)

        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model.requires_grad_(True)
        trainable_parameters = model.parameters()

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1

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
    reward_fn = getattr(d3po_pytorch.rewards, config.reward_fn)()

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(model, optimizer)

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
        model.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):

            # sample
            with autocast():

                pred_pos, pred_v, log_probs, pred_pos_traj, _, _, _, _ = sample_diffusion_ligand_with_logprob(
                    model, data, config.sample.batch_size,
                    batch_size=config.sample.batch_size, device=accelerator.device,
                    num_steps=num_steps,
                    pos_only=True,
                    center_pos_mode="protein",
                    sample_num_atoms="ref",
                )

            pred_pos_traj = torch.FloatTensor(np.array(pred_pos_traj)).to(accelerator.device)  # (batch_size, num_steps + 1, N, D)
            log_probs = torch.FloatTensor(np.array(log_probs)).to(accelerator.device)  # (batch_size, num_steps)
            timesteps = model.sampling_scheduler.timesteps.repeat(config.sample.batch_size, 1).to(accelerator.device)  # (batch_size, num_steps)

            rewards = reward_fn(zip(pred_pos, pred_v), receptor_info, [{}]*len(pred_pos))

            samples.append(
                {
                    "timesteps": timesteps,
                    "pred_pos": pred_pos_traj[:, :-1],  # each entry is the latent before timestep t
                    "next_pred_pos": pred_pos_traj[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

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
                "train/raw_score_mean": (-rewards).mean(),
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            advantages = stat_tracker.update([0.0]*len(rewards), rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch)
        assert num_timesteps == num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # train
            model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):

                def callback_func(self, index, timestep, kwargs):
                    
                    nonlocal accelerator, optimizer, model, info, global_step
                    with accelerator.accumulate(model), autocast():
                        log_prob = kwargs["log_prob"]
                        log_prob = einops.rearrange(log_prob, "(B N) -> B N", B=config.train.batch_size)
                        log_prob = log_prob.mean(dim=1)

                        offset = kwargs["offset"]

                        advantages = torch.clamp(
                            sample["advantages"], -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, index])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, index]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
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

                    next_ligand_pos = einops.rearrange(sample["next_pred_pos"][:, index], "B N D -> (B N) D") - offset

                    return {
                        "ligand_pos": next_ligand_pos
                    }
                
                _ = sample_diffusion_ligand_with_logprob(
                    model, data, config.train.batch_size,
                    batch_size=config.train.batch_size, device=accelerator.device,
                    num_steps=num_steps,
                    pos_only=True,
                    center_pos_mode="protein",
                    sample_num_atoms="ref",
                    log_probs_given_trajectory=sample["next_pred_pos"], # x1, ..., xT
                    callback_on_step_end=callback_func,
                    enable_grad=True,
                    init_pos=sample["pred_pos"][:, 0],
                )
            
            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

        #################### EVALUATION ####################
        # (epoch+1) because model already trained for (epoch+1) epochs
        if (epoch+1)%config.sample.eval_epoch==0:
            eval_generator = torch.Generator(device=accelerator.device)
            eval_generator.manual_seed(config.seed+1)

            # sample
            with autocast():

                eval_pred_pos, eval_pred_v, _, _, _, _, _, _ = sample_diffusion_ligand_with_logprob(
                    model, data, config.sample.eval_batch_size,
                    batch_size=config.sample.eval_batch_size, device=accelerator.device,
                    num_steps=num_steps,
                    pos_only=True,
                    center_pos_mode="protein",
                    sample_num_atoms="ref",
                    generator=eval_generator,
                )

            eval_rewards, _ = reward_fn(zip(eval_pred_pos, eval_pred_v), receptor_info, [{}]*len(eval_pred_pos))
            eval_rewards = eval_rewards.cpu().numpy()

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        

            molecules_list = []
            ligand_list = []
            receptor_list = []

            for i, (pos, v) in enumerate(zip(eval_pred_pos, eval_pred_v)):
                mol = d3po_pytorch.rewards.reconstruct_molecule(pos, v)
                if mol is not None:
                    vina_task = VinaDockingTask.from_generated_mol(mol, data.ligand_filename, protein_root=resolve_targetdiff_relative_dir("data/test_set"))
                    ligand_str = PrepLig(vina_task.ligand_str, 'sdf').get_pdbqt()
                    receptor_file = vina_task.receptor_path[:-4] + '.pdbqt'

                    molecules_list.append(mol)
                    ligand_list.append(ligand_str)
                    with open(receptor_file, 'r') as f:
                        receptor_str = f.read()
                    receptor_list.append(receptor_str)
                else:
                    molecules_list.append(None)
                    ligand_list.append(None)
                    receptor_list.append(None)

            ligand_table = wandb.Table(data=[[ligand_list]], columns=["ligand"])
            receptor_table = wandb.Table(data=[[receptor_list]], columns=["receptor"])

            eval_info = {
                "validation/molecules_ligand": ligand_table,
                "validation/reward_mean": eval_rewards.mean(),
                "validation/raw_score_mean": (-eval_rewards).mean(),
                "epoch": epoch
            }

            if (epoch+1) == config.sample.eval_epoch:
                eval_info["validation/molecules_receptor"] = receptor_table
            accelerator.log(
                eval_info,
                step=global_step-1 # log at the end of the training epoch
            )

if __name__ == "__main__":
    app.run(main)
