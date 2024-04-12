"""
File: __init__.py
Author: Chuncheng Zhang
Date: 2024-03-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The magic bullet which solves the problem for me.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-03-06 ------------------------
# Requirements and constants
from loguru import logger


# %% ---- 2024-03-06 ------------------------
# Function and class

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except:
            raise AttributeError(f'Invalid attribute: .{key}')

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)


# %% ---- 2024-03-06 ------------------------
# Play ground
logger.add('log/silver-bullet.log', rotation='5MB')


# %% ---- 2024-03-06 ------------------------
# Pending


# %% ---- 2024-03-06 ------------------------
# Pending
