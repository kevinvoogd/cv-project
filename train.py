
from __future__ import print_function
from __future__ import division
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from timm.scheduler.cosine_lr import CosineLRScheduler
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
from datetime import datetime
import os, shutil
import copy
import pickle

from config import Config
from swin_dataset import SwinDataset
from swin_transformer import SwinTransformer


config = Config()
### ----- LOAD DATA ----- ###
train_xform = transforms.Compose([
    transforms.Resize((config.input_size, config.input_size)),
    transforms.ToTensor()
])

train_ds = SwinDataset("train", train_xform)
train_dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=config.batch_size, num_workers=4, shuffle=True)

model = SwinTransformer(hidden_dim = 96, layers= (2, 2, 6, 2), heads= (3, 6, 12, 24), channels=3, num_classes=10, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define optimizer and parameter

optimizer = torch.optim.AdamW(model,
                              lr=config.training_opt_lr,
                              betas=config.training_opt_betas,
                              eps=config.training_opt_eps,
                              weight_decay=config.training_opt_weight_decay,
                              amsgrad=False)
                              
'''
# Define Scheduler and parameters
lr_scheduler = CosineLRScheduler(optimizer,
                                 t_initial=config.num_steps,
                                 t_mul=1.,
                                 lr_min=config.training_sch_minimum_lr,
                                 warmup_lr_init=config.training_sch_warmup_lr,
                                 warmup_t=config.training_sch_warmup_steps,
                                 cycle_limit=1,
                                 t_in_epochs=False
                                 )

# Define Loss Function
loss = torch.nn.CrossEntropyLoss()'''