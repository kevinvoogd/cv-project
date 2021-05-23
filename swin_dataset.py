from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import glob
import time

from config import Config

config = Config()

file_ext = ".JPEG"

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

# Assumes given data directory (train, val, etc) has a directory called "images"
# Loads image as both inputs and outputs
# Applies different transforms to both input and output
class SwinDataset(Dataset):
    def __init__(self, mode, input_transforms):
        self.mode = mode
        self.data_path  = os.path.join(config.data_dir, mode)
        #self.images_dir = os.path.join(self.data_path, 'images')
        self.image_list, self.class_encoded, self.class_name = self.get_image_list()
        self.transforms = input_transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # Get the ith item of the dataset
        filepath = self.image_list[i]
        class_encoded = self.class_encoded[i,:]
        class_name = self.class_name[i]
        input = self.load_pil_image(filepath)
        input = self.transforms(input)

        input = toPIL(input)
        output = input.copy()
        if self.mode == "train" and config.variationalTranslation > 0:
            output = randomCrop(input)
        #input = toTensor(centerCrop(input))
        input = toTensor(input)
        output = toTensor(output)
        #class_encoded = toTensor(class_encoded)
        return input, class_encoded, class_name, output

    def get_image_list(self):
        image_list = []
        class_name = []
        self.unique_class = []
        for classes in os.listdir(self.data_path):
            self.unique_class.append(classes)
            self.image_dir = os.path.join(self.data_path, classes)
            #print(self.image_dir)
            for file in os.listdir(self.image_dir):
                if file.endswith(file_ext):
                    path = os.path.join(self.image_dir, file)
                    image_list.append(path)
                    class_name.append(classes)

        encoded_class = self.one_hot_encoding(class_name, self.unique_class)
        return image_list, encoded_class, class_name

    def load_pil_image(self, path):
    # open path as file to avoid ResourceWarning
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def one_hot_encoding(self, classes, unique_class):
        no_class = len(unique_class)
        class_encoding = np.zeros((len(classes),no_class))
        for i in range(no_class):
            for j in range(len(classes)):
                if unique_class[i] == classes[j]:
                    class_encoding[j, i] =1
        return class_encoding
