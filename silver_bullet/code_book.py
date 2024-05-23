"""
File: code_book.py
Author: Chuncheng Zhang
Date: 2024-03-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    It is the code book for the functional

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-07 ------------------------
# Requirements and constants
from . import AttrDict


# %% ---- 2024-03-07 ------------------------
# Function and class

monocular_depth_websocket_server = AttrDict(
    host='localhost',
    port=23401,
    processing='image-to-image',
    title='Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging',
    url='https://yaksoy.github.io/highresdepth/'
)

informative_drawings_websocket_server = AttrDict(
    host='localhost',
    port=23402,
    processing='image-to-image',
    title='Informative Drawings: Learning to generate line drawings that convey geometry and semantics',
    url='https://github.com/carolineec/informative-drawings'
)

clip_analysis_websocket_server = AttrDict(
    host='localhost',
    port=23403,
    processing='image-to-array',
    title='Learning Transferable Visual Models From Natural Language Supervision',
    url='https://huggingface.co/openai/clip-vit-large-patch14'
)

depth_anything_websocket_server = AttrDict(
    host='localhost',
    port=23404,
    processing='image-to-image',
    title='Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data',
    url='https://github.com/LiheYoung/Depth-Anything'
)

# %% ---- 2024-03-07 ------------------------
# Play ground


# %% ---- 2024-03-07 ------------------------
# Pending


# %% ---- 2024-03-07 ------------------------
# Pending
