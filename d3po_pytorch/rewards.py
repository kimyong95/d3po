from PIL import Image
import io
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
import inspect
import re
def light_reward():
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach()),{}
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from d3po_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def gemini():
    from utils.rewards import GeminiQuestion
    gemini = GeminiQuestion().to(torch.device("cuda"))

    def _fn(images, prompts, metadata):
        assert all(x == prompts[0] for x in prompts)
        prompt = prompts[0]

        images = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(images))
        scores, outputs = gemini(images, prompt, "")
        for i, output in enumerate(outputs):
            metadata[i]["output"] = output
        return scores, metadata

    return _fn

def gemini_choice():
    from utils.rewards import GeminiQuestion
    gemini = GeminiQuestion().to(torch.device("cuda"))

    # images_0: [batch_size // 2, 2]
    # images_1: [batch_size // 2, 2]
    # prompts: [batch_size]
    def _fn(images_0, images_1, prompts, metadata):
        assert all(x == prompts[0] for x in prompts)
        assert len(images_0) * 2 == len(prompts)
        assert len(images_1) * 2 == len(prompts)

        prompt = prompts[0]

        images_0 = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(images_0))
        images_1 = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(images_1))

        images_merged = [list(img) for img in zip(images_0, images_1)]

        query_prompt = inspect.cleandoc(f"""
            Given this two images, which images are better aligned with the prompt '{prompt}'?
            Answer in the format (without bracket): Choice=(1/2), Reason=(reason).
        """)

        # ignore score and manually extract outputs
        _, outputs = gemini(images_merged, prompt, query_prompt)
        
        choices = []
        for i, output in enumerate(outputs):
            match = re.search(r"Choice=(\d+)", output)
            choice = int(match.group(1)) if match else 0
            choices.append(choice)
            metadata[i]["output"] = output
        
        return choices, metadata

    return _fn