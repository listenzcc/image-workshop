"""
File: silver_bullet-example.py
Author: Chuncheng Zhang
Date: 2024-03-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-06 ------------------------
# Requirements and constants
import asyncio
from PIL import Image
from pathlib import Path
from threading import Thread

from silver_bullet import AttrDict
from silver_bullet.shoot import WhoAmI
from silver_bullet.image import image2bytes, bytes2image
from silver_bullet.web_io import serve_forever, WebsocketClient


# %% ---- 2024-03-06 ------------------------
# Function and class


# %% ---- 2024-03-06 ------------------------
# Play ground
if __name__ == '__main__':
    # --------------------
    wai = WhoAmI()
    image = Image.open('./NSD/shared1000/shared0001_nsd02951.png')
    print(image)
    encoded = image2bytes(image)
    print(type(encoded))
    img = bytes2image(encoded)
    print(img)

    # --------------------
    Thread(target=asyncio.run, args=(serve_forever(),), daemon=True).start()
    # asyncio.run(serve_forever())

    wc = WebsocketClient()

    # mass = AttrDict(msg=encoded, recv=None)
    # print(asyncio.run(wc.get(mass)))
    # print(mass)

    inp = ''
    while inp != 'q':
        inp = input('>> ')
        mass = AttrDict(msg=inp.encode(), recv=None)
        print(asyncio.run(wc.get(mass)))
        print(mass)


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
