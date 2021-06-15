# Implementation of Swin Tansformer
# in Pytorch.
# Created by: Guru Deep Singh, Kevin Luis Voogd

# Script to try the dataloader and plot the accuracies generated on training and test data
# Note - This script can be ignored as it was created by the author to check the dataloader and plot the curve for accuracies

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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


def main():
    config = Config()
    train_xform = transforms.Compose([
        transforms.Resize((config.input_size,config.input_size)),
        transforms.ToTensor()
    ])

    train_ds = SwinDataset("train",train_xform)
    ep = np.array([1,5,10,30,43,65,85,95,100])
    test_acc = np.array([33.73, 51.84, 55.10, 63.00, 65.12, 63.66, 65.55, 64.81, 64.58 ])
    train_acc = np.array([35.94, 53.25, 58.24, 90.36, 99.94, 99.61, 97.89, 99.98, 100])
    fig = plt.figure(figsize=(8, 8))
    plt.plot(ep, test_acc, label= "Test")
    plt.plot(ep, train_acc, label= "Train")
    plt.xlabel("Number of Trained Epochs")
    plt.ylabel("Accuracies")
    plt.title("Training Epochs vs Accuracy")
    plt.legend()
    plt.show()

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=4, shuffle=True)

    for i, [input,class_encoded, class_names,output] in enumerate(train_dataloader,0):
        print(input.shape)
        print(output.shape)
        #print(input[0])
        print(class_encoded)
        print(class_names)
        img = input.permute(0, 2, 3, 1)
        fig = plt.figure(figsize=(8, 8))
        for i in range(config.batch_size):
            j =i+1
            fig.add_subplot(4,4,j)
            plt.imshow(img[i])
        plt.show()

if __name__ == "__main__":
    main()
