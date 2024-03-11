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
from silver_bullet.image import image_to_bytes, bytes_to_image
from silver_bullet.web_io import serve_forever, WebsocketClient


# %% ---- 2024-03-06 ------------------------
# Function and class


# %% ---- 2024-03-06 ------------------------
# Play ground
if __name__ == '__main__':
    # --------------------
    wai = WhoAmI()
    image = Image.open('./NSD/shared1000/shared0001_nsd02951.png')
    image = Image.open(
        './ImageNet/data/val/n01440764/ILSVRC2012_val_00000293.JPEG')
    print(image)
    encoded = image_to_bytes(image)
    print(type(encoded))
    img = bytes_to_image(encoded)
    print(img)

    # --------------------
    # Goes along with the process
    # Thread(target=asyncio.run, args=(serve_forever(),), daemon=True).start()

    # or Block the process
    # asyncio.run(serve_forever())

    wc = WebsocketClient(host='localhost', port=23401)

    # --------------------
    mass = AttrDict(msg=image, recv=None, require_image=True)
    mass1 = AttrDict(msg=image, recv=None, require_image=True)

    # --------------------
    # ! Block the process
    print(asyncio.run(wc.get(mass)))
    print(mass)
    mass.recv.save('a.jpg')

    # --------------------
    # ! Run in background, the thread will wait the wc.get
    Thread(target=asyncio.run, args=(wc.get(mass), ), daemon=True).start()
    Thread(target=asyncio.run, args=(wc.get(mass1), ), daemon=True).start()
    print(mass1, mass)

    while True:
        inp = input('>> ')
        if inp == 'q':
            break

        print(mass1, mass)

        if inp == '':
            continue

        mass = AttrDict(msg=inp.encode(), recv=None)
        print(asyncio.run(wc.get(mass)))
        print(mass)


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
