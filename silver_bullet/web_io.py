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
import time
import asyncio
import websockets

from . import logger, AttrDict
from .image import image_to_bytes, bytes_to_image, check_is_image

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
    """
    Handles the WebSocket connection.

    Args:
        websocket: The WebSocket connection object.

    Returns:
        None

    Raises:
        N/A
    """

    print(websocket, type(websocket))
    recv = await websocket.recv()
    logger.debug(f'Received: {type(recv)}, {len(recv)} : {recv[:20]}')
    resp = recv
    await websocket.send(resp)


async def serve_forever():
    """
    Starts serving the WebSocket indefinitely.

    Args:
        N/A

    Returns:
        None

    Raises:
        N/A
    """

    async with websockets.serve(_handler, example_options.host, example_options.port):
        await asyncio.Future()  # run forever

# %%
# Websocket client


class WebsocketClient(object):
    uri = f'ws://{example_options.host}:{example_options.port}'

    def __init__(self, host: str = None, port: int = None):
        """
        Initializes the WebSocket client.

        Args:
            host: The host address to connect to. Defaults to None.
            port: The port number to connect to. Defaults to None.

        Returns:
            None

        Raises:
            N/A
        """
        if host is not None and port is not None:
            self.uri = f'ws://{host}:{port}'
        logger.info(f'Initialized {__class__} with {self.uri}')

    async def get(self, mass: AttrDict):
        """
        Sends a message to the WebSocket server and waits for a response.

        Args:
            mass: An AttrDict object containing the message to send.

        Returns:
            str: The received response.

        Raises:
            N/A
        """

        msg = mass.msg

        # Convert to bytes if it is an Image.Image
        if check_is_image(msg):
            msg = image_to_bytes(msg)

        async with websockets.connect(self.uri) as ws:
            tic = time.perf_counter()
            await ws.send(msg)
            recv = await ws.recv()

            # Convert back to image if require_image option is set
            if mass.get('require_image', False):
                recv = bytes_to_image(recv)

            mass.recv = recv
            mass.cost_secs = time.perf_counter() - tic
            return recv


# %% ---- 2024-03-07 ------------------------
# Play ground


# %% ---- 2024-03-07 ------------------------
# Pending


# %% ---- 2024-03-07 ------------------------
# Pending
