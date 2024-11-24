import ml_collections
from config.base import get_config as base_get_config

def get_config():
    config = base_get_config()

    ############ General ############
    config.num_epochs = 1000
    config.sample.eval_epoch = 1

    ############ Training ############
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2

    ############ Prompt Function ############
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "fixed"
    config.eval_fixed_prompt = ["elephant", "eagle", "pigeon", "hippo", "hamster", "otter", "panda", "reindeer", "owl", "penguin", "flamingo", "seal", "koala", "giraffe", "parrot", "cheetah"]

    ############ Reward Function ############
    config.reward_fn = "jpeg_compressibility"

    ############ D3PO Specific ############
    config.reward_fn_choice = ""

    return config
