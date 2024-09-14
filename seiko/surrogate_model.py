


import os, pdb
from tqdm import tqdm

import sys
import os
from accelerate.utils import broadcast
import copy
cwd = os.getcwd()
sys.path.append(cwd)

import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
import argparse
import datetime
import ml_collections
import torchvision
from transformers import CLIPModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class ClipEmbedder:
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.eval()
        self.clip.train = lambda mode: self.clip
        self.clip.requires_grad_(False)
        
        self.target_size = 224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # not sure why official SEIKO not using [CLIPImageProcessor] from packahe
    # below follow the SEIKO's preprocess
    def preprocess(self, images: torch.Tensor):
        images = torchvision.transforms.Resize(self.target_size, antialias=False)(images)
        images = self.normalize(images).to(images.dtype)
        return images

    # images: tensor range [0,1]
    def __call__(self, images: torch.Tensor):
        images = self.preprocess(images)
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed

class SurrogateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.x = torch.empty(0)
        self.y = torch.empty(0)

        self.clip_embedder = ClipEmbedder()

        self.noise = 0.1        
        self.model = MLPDiff()

        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        self.clip_embedder.clip.to(self.device)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        return self

    # new_x: images tensor range [0,1]
    def update(self, new_x_images, new_y):

        new_x_embedding = self.clip_embedder(new_x_images)

        if self.x.numel() == 0:
            self.x = new_x_embedding
        else:
            self.x = torch.cat((self.x, new_x_embedding), dim=0)
                
        if self.y.numel() == 0:
            self.y = new_y
        else:
            self.y = torch.cat((self.y, new_y), dim=0)

    @torch.no_grad()
    def cov(self): # used if we have non-optimism or UCB optimism
        features = self.model.forward_up_to_second_last(self.x)
        return torch.cov(features.t())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def train_MLP(self, accelerator, config):
        
        assert self.x.requires_grad == False
        assert self.y.requires_grad == False
        
        args = ml_collections.ConfigDict()

        # Arguments
        args.num_epochs = 300
        args.train_bs = 512
        args.val_bs = 512
        args.lr = 0.001
        
        args.SGLD_base_noise = 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        self.model.requires_grad_(True)
        self.model.train()
        
        val_percentage = 0.05 # 5% of the trainingdata will be used for validation
        train_border = int(self.x.shape[0] * (1 - val_percentage) )
        
        train_dataset = TensorDataset(self.x[:train_border],self.y[:train_border])
        train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True) # create your dataloader

        val_dataset = TensorDataset(self.x[train_border:],self.y[train_border:])
        val_loader = DataLoader(val_dataset, batch_size=args.val_bs) # create your dataloader
        
        best_loss = 999
        best_model = {k: torch.empty_like(v) for k, v in self.model.state_dict().items()}
            
        def adjust_noise(learning_rate, batch_size):
            return args.SGLD_base_noise * (learning_rate ** 0.5) / (batch_size ** 0.5)   
    
        with torch.enable_grad():
            for epoch in range(args.num_epochs):
                
                noise_level = adjust_noise(args.lr, args.train_bs)
                
                losses = []
                for batch_num, (x,y) in enumerate(train_loader):
                    optimizer.zero_grad()

                    output = self.model(x)
                    
                    loss = criterion(output, y.detach())
                    accelerator.backward(loss)
                    losses.append(loss.item())
                    
                    # add Gaussian noise to gradients
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad += noise_level * torch.randn_like(param.grad)
                

                    optimizer.step()
                
                if accelerator.is_main_process:
                    losses_val = []
                    
                    for _, (x,y) in enumerate(val_loader):
                        self.model.eval()
                        optimizer.zero_grad()
                        output = self.model(x)
                        loss = criterion2(output, y.detach())

                        losses_val.append(loss.item())

                    print('Epoch %d | Loss %6.4f | val-loss %6.4f' % (epoch, (sum(losses)/len(losses)), sum(losses_val)/len(losses_val)))

                    if sum(losses_val)/len(losses_val) < best_loss:
                        best_loss = sum(losses_val)/len(losses_val)
                        print("Best MAE val loss so far: %6.4f" % (best_loss))
                        best_model = self.model.state_dict()
        
        best_model = broadcast(best_model)
        self.model.load_state_dict(best_model)
        self.model.requires_grad_(False)
        self.model.eval()
            
        del optimizer, criterion, criterion2, train_dataset, train_loader, val_dataset, val_loader


    # return loss
    # images: tensor range [0,1]
    def __call__(self, images, osm_alpha, osm_lambda, osm_clipping, aesthetic_target=1.0, grad_scale=0):
        
        embed = self.clip_embedder(images)

        feats = self.model.forward_up_to_second_last(embed)  #(B,16)
            
        raw_rewards = self.model(embed)
        
        bonuses = torch.zeros_like(raw_rewards).to(feats.device)
        
        cov_mat = self.cov()

        for idx in range(raw_rewards.shape[0]):
            feat = feats[[idx,],].t()
            invertible_cov_mat = cov_mat.to(feat.device) + osm_lambda * torch.eye(feat.shape[0]).to(feat.device)
            bonus = osm_alpha*torch.sqrt(torch.mm(feat.t(), torch.mm(torch.linalg.inv(invertible_cov_mat), feat)))
            bonuses[idx,] = bonus.squeeze(1)
        
        rewards = raw_rewards + torch.clamp(bonuses, max=osm_clipping)
        rewards = rewards.squeeze(1)

        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        
        return loss * grad_scale, rewards