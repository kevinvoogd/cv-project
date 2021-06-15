# Implementation of Swin Tansformer
# in Pytorch.
# Created by: Guru Deep Singh, Kevin Luis Voogd

# Script to train Swin Transformer model on Imagenet Subset (10 classes)
from __future__ import print_function
from __future__ import division
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from timm.scheduler.cosine_lr import CosineLRScheduler
import torchvision
from torchvision import datasets, models, transforms
from torchinfo import summary

import matplotlib.pyplot as plt
import time
from datetime import datetime
import os, shutil
import copy
import pickle

from config import Config
from swin_dataset import SwinDataset
from swin_transformer import SwinTransformer, swin_t
from tqdm import tqdm

def trainmodel(train_loader, model, optimizer, criterion, scheduler, device, epoch, config):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. AdamW).
        scheduler: LR Cosine Scheduler
        criterion: Loss function (e.g. cross-entropy loss).
        device: cpu/gpu.
    """

    avg_loss = 0
    correct = 0
    total = 0
    loss_print = []
    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels, _, _ = data

        # Move data to target device
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.argmax(labels, dim=1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        print('Predictions size', outputs.shape)
        print('Targets', labels.shape)
        loss = criterion(outputs, labels)
        loss_print.append(loss) #for printing the loss
        loss.backward()
        optimizer.step()
        scheduler.step_update(epoch * config.num_steps + i)

        # Keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted label: ', predicted)
        print('Ground truth label:', labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print('Number of correct predictions: ', correct)

    return avg_loss / len(train_loader), 100 * correct / total, loss_print


def validate(validation_loader, model, optimizer, scheduler, criterion, device):
    valid_loss = 0
    correct = 0
    total = 0
    min_valid_loss = np.inf


    for inputs, labels in validation_loader:
        # Transfer Data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward Pass
        outputs = model(inputs)
        # Find the Loss
        loss = criterion(outputs,labels)
        # Calculate Loss
        valid_loss = loss.item() * inputs.size(0)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f\
        }--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

    return avg_loss / len(train_loader), 100 * correct / total



def main():
    config = Config()

    ### ----- LOAD DATA ----- ###

    # Transform input images (train and validation)
    train_xform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor()
        ])


    # Train and validation datasets
    train_ds = SwinDataset("train", train_xform)

    # Train and Validation dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, num_workers=4, shuffle=True)


    # Create classifier model
    model = SwinTransformer(hidden_dim = 96,
                            layers= (2, 2, 6, 2),
                            heads= (3, 6, 12, 24),
                            channels=3,
                            num_classes=10,
                            head_dim=32,
                            window_size=7,
                            downscaling_factors=(4, 2, 2, 2),
                            relative_pos_embedding=True
                            )


    # Define optimizer

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config.training_opt_lr,
                                betas=config.training_opt_betas,
                                eps=config.training_opt_eps,
                                weight_decay=config.training_opt_weight_decay,
                                amsgrad=False)

    # Define Scheduler
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
    loss = torch.nn.CrossEntropyLoss()

    # Print number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has: ', "{:.2f}".format(n_parameters/1e6), 'million parameters')

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.train()


    for epoch in tqdm(range(config.num_epochs)):
        # Train on data
        print('\n Epoch number:', epoch)
        train_loss, train_acc, loss_print = trainmodel(train_dataloader,
                                      model,
                                      optimizer,
                                      loss,
                                      lr_scheduler,
                                      device,
                                      epoch,
                                      config)

        # Save the model
        torch.save(model.state_dict(), 'saved_model.pth')
        print('Training Loss', train_loss )


if __name__ == "__main__":
    main()
