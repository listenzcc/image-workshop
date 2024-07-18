"""
File: app.py
Author: Chuncheng Zhang
Date: 2024-04-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Gradio web server

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-12 ------------------------
# Requirements and constants
import sys
import asyncio
import numpy as np
import gradio as gr
from gradio_imageslider import ImageSlider

from PIL import Image, ImageDraw
from pathlib import Path

from loguru import logger

sys.path.append(Path(__file__).parent.parent.as_posix())  # noqa

from silver_bullet.web_io import WebsocketClient
from silver_bullet import AttrDict

logger.add('log/gradio.log', rotation='5MB')

# %% ---- 2024-04-12 ------------------------
# Function and class
wc_1 = WebsocketClient(host='localhost', port=23401)
wc_2 = WebsocketClient(host='localhost', port=23402)
wc_4 = WebsocketClient(host='localhost', port=23404)


def get_async_loop():
    # --------------------
    # Get or make a loop
    try:
        # Use the existing loop
        loop = asyncio.get_event_loop()
        logger.debug(f'Using existing loop: {loop}')
    except Exception as err:
        # or, Create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug(f'Create new loop: {loop}')
    return loop


def __process_image(image: Image):
    image = Image.fromarray(image)
    print(image)

    mass_1 = AttrDict(msg=image, recv=None, require_image=True)
    mass_2 = AttrDict(msg=image, recv=None, require_image=True)

    loop = get_async_loop()

    jobs = [
        wc_1.get(mass_1),
        wc_2.get(mass_2),
    ]

    tasks = asyncio.gather(*jobs, return_exceptions=True)

    res = loop.run_until_complete(tasks)
    print(res)

    return mass_1.recv, mass_2.recv


def process_image_with_wc(image: Image, wc: WebsocketClient):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    mass = AttrDict(msg=image, recv=None, require_image=True)

    loop = get_async_loop()

    jobs = [wc.get(mass)]
    tasks = asyncio.gather(*jobs, return_exceptions=True)
    res = loop.run_until_complete(tasks)
    print(res)

    return mass.recv


def process_image_wc1(mat: np.ndarray):
    print(mat.shape)
    image = Image.fromarray(mat)
    return image, process_image_with_wc(image, wc_1)


def process_image_wc2(mat: np.ndarray):
    print(mat.shape)
    image = Image.fromarray(mat)
    return image, process_image_with_wc(image, wc_2)


def process_image_wc4(mat: np.ndarray):
    print(mat.shape)
    image = Image.fromarray(mat)
    return image, process_image_with_wc(image, wc_4)


# %% ---- 2024-04-12 ------------------------
# Play ground
if __name__ == '__main__':
    # demo = gr.Interface(process_image, 'image', ['image'] * 2)

    with gr.Blocks() as demo:
        with gr.Row():
            input_image = gr.Image(label='Input')

        with gr.Row():
            image_slider_1 = ImageSlider(label='Boosting Monocular Depth')
            image_slider_2 = ImageSlider(label='Line Drawing')
            image_slider_3 = ImageSlider(label='Depth Anything')

        input_image.change(fn=process_image_wc1,
                           inputs=input_image, outputs=image_slider_1)
        input_image.change(fn=process_image_wc2,
                           inputs=input_image, outputs=image_slider_2)
        input_image.change(fn=process_image_wc4,
                           inputs=input_image, outputs=image_slider_3)

    demo.launch(server_port=7911)
    print('Done.')

# %% ---- 2024-04-12 ------------------------
# Pending


# %% ---- 2024-04-12 ------------------------
# Pending
