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
from tqdm.auto import tqdm

# --------------------
pwd = Path(__file__).parent
slaves_folder = pwd.joinpath('slaves')
saved_folder = pwd.joinpath('saved-slaves')

# %% ---- 2024-04-12 ------------------------
# Function and class


def save_slave(folder: Path):
    src_path = folder.joinpath('run-server.py')

    if not src_path:
        print(f'Invalid slave directory: {src_path}')
        return

    dst_path = saved_folder.joinpath(folder.name, 'run-server.py')
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src=src_path, dst=dst_path)
    print('\n'.join([
        'Copy file:',
        f'- src: {src_path}',
        f'- dst: {dst_path}',
    ]))
    return


# %% ---- 2024-04-12 ------------------------
# Play ground
if __name__ == '__main__':
    for folder in tqdm([e for e in slaves_folder.iterdir() if e.is_dir()], 'Save slaves'):
        print(f'---- Found slave: {folder.name} ----')
        save_slave(folder)


# %% ---- 2024-04-12 ------------------------
# Pending


# %% ---- 2024-04-12 ------------------------
# Pending
