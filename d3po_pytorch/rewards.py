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

def gemini_binary():
    from utils.rewards import GeminiQuestion
    gemini = GeminiQuestion().to(torch.device("cuda"))
    query = "Does the image accurately, precisely and comprehensively described by the prompt '{target_prompt}'? Answer score=0 (no) or score=1 (yes).\nAnswer in the format: Score=(score), Reason=(reason)."
    
    def _fn(images, prompts, metadata):
        assert all(x == prompts[0] for x in prompts)
        prompt = prompts[0]

        images = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(images))
        scores, outputs = gemini(images, prompt, query, max_reward=1.0)
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
            Given this two images, which image is better aligned with the prompt '{prompt}'?
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

from related_works.targetdiff.utils import misc, reconstruct, transforms
import related_works.targetdiff.utils.transforms as trans
from rdkit import Chem
def reconstruct_molecule(pos, v):
    # reconstruction
    pred_atom_type = transforms.get_atomic_number_from_index(v, mode="add_aromatic")
    try:
        pred_aromatic = transforms.is_aromatic_from_index(v, mode="add_aromatic")
        mol = reconstruct.reconstruct_from_generated(pos, pred_atom_type, pred_aromatic)
    except reconstruct.MolReconsError:
        return None

    return mol

async def get_score(pos, v, receptor_info):

    # Tang S, Chen R, Lin M, Lin Q, Zhu Y, Ding J, Hu H, Ling M, Wu J. Accelerating AutoDock Vina with GPUs. Molecules. 2022 May 9;27(9):3041. doi: 10.3390/molecules27093041. PMID: 35566391; PMCID: PMC9103882.
    # cite: The AutoDock Vina score for drug-like compounds can reach as low as -11.6 kcal/mol.
    # used for normalizing the score to [0, 1]
    MAX_VINA_SCORE = 11.6

    ligand_filename = receptor_info["ligand_filename"]
    protein_root = receptor_info["protein_root"]
    vina_web_url = receptor_info["vina_web_url"]

    score = 0.0
    mol = reconstruct_molecule(pos, v)
    if mol is not None:
        vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=protein_root, web_dock_url=vina_web_url)
        score = (await vina_task.run(mode='score_only', exhaustiveness=16))[0]["affinity"]
        score = score if score < 0.0 else 0.0
    failed = bool(mol is None)

    return score, failed

import random
async def get_scores_async(pos_list, v_list, receptor_info):

    tasks = [get_score(pos,v, receptor_info) for pos, v in zip(pos_list, v_list)]
    results = await asyncio.gather(*tasks)
    scores, failed = map(list, zip(*results))
    failed_count = sum(failed)
    return torch.FloatTensor(scores), failed_count


import asyncio
from related_works.targetdiff.utils.evaluation.docking_vina import VinaDockingTask
def vina():

    def _fn(pos_v_zip, receptor_info, metadata):

        pos_list, v_list = map(list, zip(*pos_v_zip))
        
        scores, failed_count = asyncio.run(get_scores_async(pos_list, v_list, receptor_info))
        
        # minimize scores (lower is better)
        # maximize rewards (higher is better)
        rewards = - scores

        return rewards, metadata

    return _fn