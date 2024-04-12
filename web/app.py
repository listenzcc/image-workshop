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
import gradio as gr

from PIL import Image
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())  # noqa

from silver_bullet.web_io import WebsocketClient
from silver_bullet import AttrDict


# %% ---- 2024-04-12 ------------------------
# Function and class
wc_1 = WebsocketClient(host='localhost', port=23401)
wc_2 = WebsocketClient(host='localhost', port=23402)


def get_async_loop():
    # --------------------
    # Get or make a loop
    try:
        # Use the existing loop
        loop = asyncio.get_event_loop()
    except Exception as err:
        # or, Create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def process_image(image: Image):
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


# %% ---- 2024-04-12 ------------------------
# Play ground
if __name__ == '__main__':
    demo = gr.Interface(process_image, 'image', ['image'] * 2)
    demo.launch()

# %% ---- 2024-04-12 ------------------------
# Pending


# %% ---- 2024-04-12 ------------------------
# Pending
