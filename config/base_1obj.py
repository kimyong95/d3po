import ml_collections
from config.base import get_config as base_get_config

def get_config():
    config = base_get_config()

    config.reward_prompts = [
        "A cat and a dog on the moon.",
    ]

    return config
