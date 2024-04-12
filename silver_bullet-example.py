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
    image.save('example.jpg')
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
    wc_1 = WebsocketClient(host='localhost', port=23401)
    wc_2 = WebsocketClient(host='localhost', port=23402)

    # --------------------
    # Make the mass, the input to the remote websocket server
    mass = AttrDict(msg=image, recv=None, require_image=True)

    # --------------------
    # ! Run blocking the process
    # print(asyncio.run(wc.get(mass)))
    # print(mass)
    # mass.recv.save('recv.jpg')

    # --------------------
    # ! Run in background, the thread will wait the wc.get
    # Thread(target=asyncio.run, args=(wc.get(mass), ), daemon=True).start()
    # Thread(target=asyncio.run, args=(wc.get(mass1), ), daemon=True).start()

    # --------------------
    help_msg = '''
Empty message is not allowed.
Known inputs are:
- q: Quit the app;
- e: Use the example;
...
    '''

    while True:
        inp = input('>> ')
        if inp == 'q':
            break

        if inp == '':
            print(help_msg)
            continue

        if inp == 'e':
            # >> e: Using the example mass
            mass_1 = deepcopy(mass)
            mass_2 = deepcopy(mass)
            print(mass_1, type(mass_1))
            print(mass_2, type(mass_2))
        else:
            # >> [others]: Using the input directly
            mass_1 = AttrDict(msg=inp.encode(), recv=None)
            mass_2 = AttrDict(msg=inp.encode(), recv=None)

        if False:
            # Single run
            print(f'Start getting data {asyncio.run(wc_1.get(mass_1))}')
            print(f'Start getting data {asyncio.run(wc_2.get(mass_2))}')

        # --------------------
        # Get or make a loop
        try:
            # Use the existing loop
            loop = asyncio.get_event_loop()
        except Exception as err:
            # or, Create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # --------------------
        # Run the tasks
        tasks = asyncio.gather(
            *[wc_1.get(mass_1), wc_2.get(mass_2)],
            return_exceptions=True)
        # Summary the results
        print(loop.run_until_complete(tasks))
        print(f'Got mass_1: {mass_1}')
        print(f'Got mass_2: {mass_2}')
        mass_1.recv.save('example-processed-1.jpg')
        mass_2.recv.save('example-processed-2.jpg')


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
