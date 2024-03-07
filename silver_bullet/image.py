"""
File: image.py
Author: Chuncheng Zhang
Date: 2024-03-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Image processing with opencv, PIL, AI or everything

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-06 ------------------------
# Requirements and constants
import io
import cv2
import base64
import requests
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from . import logger

# %% ---- 2024-03-06 ------------------------
# Function and class


def image2bytes(image: Image, format: str = 'JPEG') -> bytes:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue())
    logger.debug(
        f'Encoded image {image} to buffer {encoded[:8]}...x {len(encoded)} bytes')
    return encoded


def bytes2image(encoded: bytes, format: str = 'JPEG') -> Image:
    img_str = io.BytesIO(base64.b64decode(encoded))
    image = Image.open(img_str)
    logger.debug(f'Decoded image: {image}')
    return image

# %% ---- 2024-03-06 ------------------------
# Play ground


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
