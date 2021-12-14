
import pandas as pd

import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


import albumentations as A
from albumentations.pytorch import ToTensorV2






train_transforms = A.Compose(
    [
        A.RandomCrop(height=280, width=280, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(range , p=0.5),
        A.Resize(width=300, height=300),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Resize(height=300, width=300),
        ToTensorV2(),
    ]
)


class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, transform, train=True):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpg", "")

        try:
            image = np.array(Image.open(os.path.join(self.images_folder, image_file + ".jpeg")))
        except FileNotFoundError:
            image = np.array(Image.open(os.path.join(self.images_folder, image_file + ".jpg")))

        # print(image)
        image = self.transform(image=image)["image"]

        return image, label, image_file