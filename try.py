
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
            fig.add_subplot(2,3,j)
            plt.imshow(img[i])
        plt.show()

if __name__ == "__main__":
    main()
