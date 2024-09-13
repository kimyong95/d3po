import ml_collections

import config.base as base_config

def get_config():
    config = base_config.get_config()

    config.num_samples_per_outerloop = [160,320,480,640] # total budget = 32*50
    config.train.total_samples_per_epoch = 64
    config.train.gradient_accumulation_steps = 8
    config.num_epochs = 5
    config.train.batch_size = 4
    config.train.osm_alpha = 1e-2   # this is the most important hyperparameter, a big alpha corresponds to stronger bonus
    config.train.osm_lambda = 1e-3  # this is just to ensure cov mat is invertible, not very important
    config.train.osm_clipping = 10  # this is not important, LCB bonus should not be big (probably < 1)
    config.aesthetic_target = 1.0
    config.grad_scale = 1
    config.train.loss_coeff = 0.01
    config.train.kl_weight = 0.01
    config.train.learning_rate = 3e-4

    # debug:
    # config.num_samples_per_outerloop = [32,64,96,128]
    # config.train.total_samples_per_epoch = 4
    # config.wandb_mode = "disabled"
    # config.sd_model="sd1"
    
    return config