import ml_collections
from config.base import get_config as base_get_config

def get_config():
    config = base_get_config()
    config.num_epochs = 1000
    config.sample.eval_epoch = 1

    ############ Prompt Function ############
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "fixed"
    config.eval_fixed_prompt = ["elephant", "eagle", "pigeon", "hippo", "hamster", "otter", "panda", "reindeer", "owl", "penguin", "flamingo", "seal", "koala", "giraffe", "parrot", "cheetah"]

    ############ Reward Function ############
    config.reward_fn = "jpeg_compressibility"

    return config
