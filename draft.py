"""
File: draft.py
Author: Chuncheng Zhang
Date: 2024-02-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Draft area.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-02-29 ------------------------
# Requirements and constants
import matplotlib.pyplot as plt

from rich import print, inspect

# from CIFAR.load import base_check, dataset10, dataset100
from CIFAR.load import *


# %% ---- 2024-02-29 ------------------------
# Function and class


# %%
im = get_image(dataset10)
plt.imshow(im.rgb)
plt.suptitle(f'Image: {im.class_name} ({im.class_idx})')
plt.tight_layout()
plt.show()

# %%
for e in dataset10_loader:
    break

e[0].shape

# %%
dir(transform.transforms[1])


# %% ---- 2024-02-29 ------------------------
# Play ground
if __name__ == '__main__':
    print(base_check())
    print(base_check().name)


# %% ---- 2024-02-29 ------------------------
# Pending


# %% ---- 2024-02-29 ------------------------
# Pending
