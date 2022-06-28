from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

car_dataset = datasets.ImageFolder(r"E:\Text2CarsGeneration\Text2CarsGeneration\CarDataset",
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                   ]))
print(car_dataset)
length = len(car_dataset)
print(length)

training_data, validation_data = random_split(car_dataset, [122165, 30541])

for image, label in training_data:
    print(image, label)
