"""
File: image_window.py
Author: Chuncheng Zhang
Date: 2024-03-08
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Show what will happen to image's psd before and after they are windowed.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-08 ------------------------
# Requirements and constants
import time
import cv2
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from scipy import signal
from rich import inspect
from IPython.display import display

sys.path.append('..')  # noqa
from ImageNet.load import ImageNetDataSet

# --------------------
window_names = [
    e.strip()
    for e in """
boxcar

triang

blackman

hamming

hann

bartlett

flattop

parzen

bohman

blackmanharris

nuttall

barthann

cosine

exponential

tukey

taylor

lanczos
""".split(
        "\n"
    )
    if e.strip()
]

dataset = ImageNetDataSet()
# %% ---- 2024-03-08 ------------------------
# Function and class


def get_window_shape(name: str, Nx: int = 101) -> np.ndarray:
    return signal.windows.get_window(name, Nx)


def get_2d_window(name1: str, n1: int, name2: str = None, n2: int = None):
    if name2 is None:
        name2 = name1
    if n2 is None:
        n2 = n1

    s1 = get_window_shape(name1, n1)
    s2 = get_window_shape(name2, n2)

    return np.outer(s2, s1), name1, name2


def compute_image_fft(image: Image, dot_mask: np.ndarray = None):
    if isinstance(image, Image.Image):
        if image.mode == 'RGB':
            mat = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        elif image.mode == 'L':
            mat = np.asarray(image)
        else:
            raise ValueError('Invalid image mode: {image.mode}')
    elif isinstance(image, np.ndarray):
        # Assume the input is (width x height) matrix
        mat = image
        if np.ndim(mat) != 2:
            raise ValueError(f'Invalid input matrix shape: {mat.shape}')
    else:
        raise ValueError('Invalid input: {image}')

    if dot_mask is not None:
        mat = mat.astype(dot_mask.dtype)
        mat *= dot_mask

    f = np.fft.fft2(mat)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum, mat


# %% ---- 2024-03-08 ------------------------
# Play ground

# --------------------
width, height = 400, 300
k = 3  # for low frequency
k = 30  # for high frequency
texture = np.cos(np.linspace(0, k*np.pi, width*height)).reshape(height, width)
image = texture

# --------------------
# got = dataset.getitem(random.choice(range(50000000)))
# image = got['image']
# width, height = image.size
# display(got)
# display(image)

# --------------------

random.seed(time.time())
name = random.choice(window_names)
window_2d, name1, name2 = get_2d_window(name1=name,  # 'hamming',
                                        n1=width,
                                        n2=height)

fft_image, _ = compute_image_fft(image)
fft_window, _ = compute_image_fft(window_2d)
fft_windowed_image, windowed_image = compute_image_fft(image, window_2d)

# --------------------
df = sns.load_dataset("penguins")
display(df)

df = pd.DataFrame()
df['imageFFT'] = fft_image.ravel()
df['windowedFFT'] = fft_windowed_image.ravel()
display(df)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(df, x='imageFFT', y='windowedFFT', stat='density',
             cbar=True, cbar_kws={'shrink': 0.5}, ax=ax)
ax.grid()
ax.set_aspect(1.0)
ax.set_title('Cross Density')
plt.show()

# --------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 8/width*height))

ax = axs[0][0]
ax.imshow(image)
ax.set_title('Raw Image')

ax = axs[1][0]
im = ax.imshow(fft_image)
plt.colorbar(im, shrink=0.5)
ax.set_title('Raw Image FFT')

ax = axs[0][1]
im = ax.imshow(window_2d, cmap='grey')
plt.colorbar(im, shrink=0.5)
ax.set_title(f'Window {name}')

ax = axs[1][1]
im = ax.imshow(fft_window)
plt.colorbar(im, shrink=0.5)
ax.set_title('Window FFT')

ax = axs[0][2]
ax.imshow(windowed_image, cmap='grey')
ax.set_title('Windowed Image')

ax = axs[1][2]
im = ax.imshow(fft_windowed_image)
plt.colorbar(im, shrink=0.5)
ax.set_title('Windowed Image FFT')

plt.tight_layout()
plt.show()

# %% ---- 2024-03-08 ------------------------
# Pending

# %% ---- 2024-03-08 ------------------------
# Pending

# %%
