"""
File: load.py
Author: Chuncheng Zhang
Date: 2024-02-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load CIFAR 10/100 image toolbox in torchvision module

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-02-29 ------------------------
# Requirements and constants
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from pathlib import Path

from . import root

data_path = root.joinpath('data')

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

kwargs = dict(
    train=True,
    transform=transform,
    download=True
)

dataset10 = CIFAR10(data_path, **kwargs)
dataset100 = CIFAR100(data_path, **kwargs)

# %% ---- 2024-02-29 ------------------------
# Function and class
def base_check():
    return dict(
        name='CIFAR',
        data_path=data_path,
        cuda_available=torch.cuda.is_available(),
    )



# %% ---- 2024-02-29 ------------------------
# Play ground


# %% ---- 2024-02-29 ------------------------
# Pending



# %% ---- 2024-02-29 ------------------------
# Pending
