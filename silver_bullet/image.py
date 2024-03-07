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


def check_is_image(image: Image):
    """
    Checks if the given object is an instance of the Image class.

    Args:
        image: The object to be checked.

    Returns:
        bool: True if the object is an instance of Image, False otherwise.
    """

    return isinstance(image, Image.Image)


def image_to_bytes(image: Image, format: str = 'JPEG') -> bytes:
    """
    Encodes an image to bytes using the specified format.

    Args:
        image: The image to be encoded.
        format: The format to use for encoding the image. Defaults to 'JPEG'.

    Returns:
        bytes: The encoded image as bytes.

    Examples:
        >>> image = Image.open('image.jpg')
        >>> encoded_image = image2bytes(image, format='PNG')
    """

    if not check_is_image(image):
        msg = f'Image is required, but get {type(image)}'
        logger.error(msg)
        raise ValueError(msg)

    buffered = io.BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue())
    logger.debug(
        f'Encoded image {image} to buffer {encoded[:8]}...x {len(encoded)} bytes')
    return encoded


def bytes_to_image(encoded: bytes) -> Image:
    """
    Decodes bytes to an image.

    Args:
        encoded: The encoded image as bytes.

    Returns:
        Image: The decoded image.

    Raises:
        N/A
    """

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
