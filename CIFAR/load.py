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

root = Path(__file__).parent
data_path = root.joinpath('data')

# %%

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[1.0, 1.0, 1.0]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

data_kwargs = dict(
    root=data_path,
    train=True,
    transform=transform,
    download=True
)

loader_kwargs = dict(
    batch_size=10,
    shuffle=True,
    num_workers=10
)

dataset10 = CIFAR10(**data_kwargs)
dataset10_loader = DataLoader(
    dataset=dataset10,
    **loader_kwargs
)

dataset100 = CIFAR100(**data_kwargs)
dataset100_loader = DataLoader(
    dataset=dataset100,
    **loader_kwargs
)

# %% ---- 2024-02-29 ------------------------
# Function and class


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except:
            raise AttributeError()

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)


def base_check() -> AttrDict:
    return AttrDict(
        name='CIFAR',
        data_path=data_path,
        cuda_available=torch.cuda.is_available(),
    )


def get_image(dataset: CIFAR10, i: int = 0) -> AttrDict:
    rgb = dataset.data[i]
    class_idx = dataset.targets[i]
    class_name = dataset.classes[class_idx]

    return AttrDict(
        rgb=rgb,
        class_idx=class_idx,
        class_name=class_name,
    )


# %% ---- 2024-02-29 ------------------------
# Play ground
if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    from rich import print

    got = dataset100_loader.__iter__().__next__()
    print(got)

    print(f'Using backend: {plt.get_backend()}')

    im = get_image(dataset100, i=random.randint(0, 1000))
    plt.imshow(im.rgb)
    plt.suptitle(f'Image: {im.class_name} ({im.class_idx})')
    plt.tight_layout()
    plt.show()




# %% ---- 2024-02-29 ------------------------
# Pending


# %% ---- 2024-02-29 ------------------------
# Pending
