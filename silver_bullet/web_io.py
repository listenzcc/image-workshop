"""
File: web_io.py
Author: Chuncheng Zhang
Date: 2024-03-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Web interface for signal exchange.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-07 ------------------------
# Requirements and constants
import asyncio
import websockets

from . import logger, AttrDict
from .image import image2bytes, bytes2image

# %%
example_options = AttrDict(
    host='localhost',
    port=28765
)

# %% ---- 2024-03-07 ------------------------
# Function and class

# %%
# Example server


async def _handler(websocket: websockets.ServerProtocol):
    print(websocket, type(websocket))
    recv = await websocket.recv()
    logger.debug(f'Received: {type(recv)}, {len(recv)} : {recv[:20]}')
    resp = recv
    await websocket.send(resp)


async def serve_forever():
    async with websockets.serve(_handler, example_options.host, example_options.port):
        await asyncio.Future()  # run forever

# %%
# Websocket client


class WebsocketClient(object):
    uri = f'ws://{example_options.host}:{example_options.port}'

    def __init__(self, host: str = None, port: int = None):
        if host is not None and port is not None:
            self.uri = f'ws://{host}:{port}'
        logger.info(f'Initialized {__class__} with {self.uri}')

    async def get(self, mass: AttrDict):
        msg = mass.msg
        async with websockets.connect(self.uri) as ws:
            await ws.send(msg)
            recv = await ws.recv()
            mass.recv = recv
            return recv


# %% ---- 2024-03-07 ------------------------
# Play ground


# %% ---- 2024-03-07 ------------------------
# Pending


# %% ---- 2024-03-07 ------------------------
# Pending
