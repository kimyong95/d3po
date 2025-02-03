import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ############ General ############
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    config.name_subfix = ""
    # random seed for reproducibility.
    config.seed = 0
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.num_epochs = 500
    # number of epochs between saving model checkpoints.
    config.save_freq = 10
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    config.wandb_mode = "online"

    ############ Pretrained Model ############
    config.sd_model = "sdxl"

    ############ Sampling ############
    config.sample = sample = ml_collections.ConfigDict()
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # save interval
    sample.save_interval = 100
    # eval batch_size
    sample.eval_batch_size = 16
    # eval epoch
    sample.eval_epoch = 5
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 32
    # number of batches to sample per epoch.
    sample.num_batches_per_epoch = 1
    # the total number of samples per epoch is `num_batches_per_epoch * batch_size * num_gpus`.

    ############ Training ############
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 2
    # number of gradient accumulation steps.
    train.gradient_accumulation_steps = 2
    # the effective batch size is `batch_size * num_gpus * gradient_accumulation_steps`.
    # effective batch size: 2*2 = 4
    # gradient updates per epoch: 32 / 4 = 8

    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8

    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    train.activation_checkpoint = True
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # coefficient of the KL divergence
    train.beta = 0.1
    # The coefficient constraining the probability ratio. Equivalent to restricting the Q-values within a certain range.
    train.eps = 0.1
    # save_interval
    train.save_interval = 50
    # sample path
    train.sample_path = ""
    # json path
    train.json_path = ""

    ############ Other method Config ############
    # DDPO: clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # DDPO: the PPO clip range.
    train.clip_range = 1e-4
    # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
    # advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
    # advantages will be calculated using the mean and std of the entire batch.
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    config.per_prompt_stat_tracking.buffer_size = 16
    # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
    # contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    config.per_prompt_stat_tracking.min_count = 16

    # DPOK: the KL coefficient
    config.kl_ratio = 0.01

    ############ Prompt Function ############
    # prompt function to use. see `prompts.py` for available prompt functisons.
    config.prompt_fn = "fixed"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = { }
    config.fixed_prompt = "A bear riding a red bicycle with his puppy friend sitting on the back, he holding a colourful umbrella and wearing a hat."

    ############ Reward Function ############
    # reward function to use. see `rewards.py` for available reward functions.
    # if the reward_fn is "jpeg_compressibility" or "jpeg_incompressibility", using the default config can reproduce our results.
    # if the reward_fn is "aesthetic_score" and you want to reproduce our results, 
    # set config.num_epochs = 1000, sample.num_batches_per_epoch=1, sample.batch_size=8 and sample.eval_batch_size=8
    # config.reward_fn = "gemini"
    config.reward_fn = "gemini_binary"
    config.reward_prompts = [
        "A bear riding a bicycle with his puppy friend sitting on the back.",
        "A bear riding a bicycle with his puppy friend sitting on the back, he holding an umbrella.",
        "A bear riding a red bicycle with his puppy friend sitting on the back, he holding a colourful umbrella.",
        "A bear riding a red bicycle with his puppy friend sitting on the back, he holding a colourful umbrella and wearing a hat.",
    ]

    ############ D3PO Specific ############
    config.reward_fn_choice = "gemini_choice"

    return config
