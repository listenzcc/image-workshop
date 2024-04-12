"""
File: save-slaves.py
Author: Chuncheng Zhang
Date: 2024-04-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Save the slaves' run-server.py file

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-12 ------------------------
# Requirements and constants
import shutil

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm

# --------------------
pwd = Path(__file__).parent
slaves_folder = pwd.joinpath('slaves')
saved_folder = pwd.joinpath('saved-slaves')

# %% ---- 2024-04-12 ------------------------
# Function and class


def copy_file(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    assert src.is_file(), f'Invalid file: {src}'
    dst.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(src, dst)
    logger.debug('\n'.join([
        'Copied file:',
        f'- src: {src}',
        f'- dst: {dst}',
        '',
    ]))
    return dst


def save_slave(folder: Path):
    src = folder.joinpath('run-server.py')

    if not src.is_file():
        logger.warning(f'Invalid slave directory: {src}')
        return

    logger.info(f'---- Found slave: {folder.name} ----')

    dst = saved_folder.joinpath(folder.name, 'run-server.py')
    return copy_file(src, dst)


def save_util():
    folder = slaves_folder.joinpath('util')
    for src in [e for e in folder.iterdir() if e.is_file()]:
        logger.info(f'---- Found slave/util: {src.name} ----')
        dst = saved_folder.joinpath('util', src.name)
        dst.parent.mkdir(exist_ok=True, parents=True)
        copy_file(src, dst)


# %% ---- 2024-04-12 ------------------------
# Play ground
if __name__ == '__main__':
    for folder in tqdm([e for e in slaves_folder.iterdir() if e.is_dir()], 'Save slaves'):
        save_slave(folder)

    save_util()


# %% ---- 2024-04-12 ------------------------
# Pending


# %% ---- 2024-04-12 ------------------------
# Pending
