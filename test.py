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

config = Config()

def testmodel(test_loader, model, criterion, device):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
        device: cpu/gpu
    """

    avg_loss = 0
    correct = 0
    total = 0
    test_loss = []
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, data in enumerate(test_loader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels, _, _ = data
            labels = labels.type(torch.LongTensor)
            labels = torch.argmax(labels, dim=1)

            # Move data to target device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss.append(loss)
            # Keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    return avg_loss / len(test_loader), 100 * correct / total, test_loss

def main():

    val_xform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor()
        ])

    val_ds = SwinDataset("val", val_xform)

    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=4, shuffle=True)

    print('Test!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    model.load_state_dict(torch.load('./models/100_epoch.pth'))
    model = model.to(device)
    # Validate on data
    model.eval()
    loss = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_loss_print = testmodel(val_dataloader,
                                   model,
                                   loss,
                                   device)


    print('Test Loss', test_loss)
    print('\n\n')
    print('Test Accuracy', test_acc)

    with open('Test_loss.pkl','ab') as f:
      pickle.dump(test_loss_print, f)

if __name__ == "__main__":
    main()
