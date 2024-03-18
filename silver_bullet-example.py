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

from copy import deepcopy

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

    # --------------------
    # Make the WebsocketClient
    wc = WebsocketClient(host='localhost', port=23401)

    # --------------------
    # Make the mass, the input to the remote websocket server
    mass = AttrDict(msg=image, recv=None, require_image=True)
    mass0 = AttrDict(msg=image, recv=None, require_image=True)
    mass1 = AttrDict(msg=image, recv=None, require_image=True)

    # --------------------
    # ! Block the process
    # print(asyncio.run(wc.get(mass)))
    # print(mass)
    # mass.recv.save('recv.jpg')

    # --------------------
    # ! Run in background, the thread will wait the wc.get
    # Thread(target=asyncio.run, args=(wc.get(mass), ), daemon=True).start()
    # Thread(target=asyncio.run, args=(wc.get(mass1), ), daemon=True).start()
    print(mass1, mass)

    # --------------------
    help_msg = '''
Empty message is not allowed.
Known inputs are:
- q: Quit the app;
- e: Use the example;
...
    '''

    # async def go(mass):
    #     wc.get(mass)

    while True:
        inp = input('>> ')
        if inp == 'q':
            break

        if inp == '':
            print(help_msg)
            continue

        if inp == 'e':
            _mass = deepcopy(mass0)
            print(_mass, type(_mass))
        else:
            _mass = AttrDict(msg=inp.encode(), recv=None)
        # print(f'Start getting data {asyncio.run(wc.get(mass))}')

        # Use the existing loop
        loop = asyncio.get_event_loop()

        # or, Create a new loop
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)

        tasks = asyncio.gather(*[wc.get(_mass)], return_exceptions=True)
        print(loop.run_until_complete(tasks))

        print(f'Got: {_mass}')


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
