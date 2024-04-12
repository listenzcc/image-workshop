"""
File: task_manager.py
Author: Chuncheng Zhang
Date: 2024-04-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Task manager

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-12 ------------------------
# Requirements and constants
import time
import random
import threading
import contextlib

from typing import Any
from loguru import logger


# %% ---- 2024-04-12 ------------------------
# Function and class

logger.add('log/taskmanager.log', rotation='5MB')


class TaskManager(object):
    '''
    Automatical task manager
    '''
    uid = 0
    tasks = {}
    rlock = threading.RLock()
    logger = logger

    def __init__(self, logger=None):
        """
        Initializes the task manager.

        Args:
            logger: The logger object to be used for logging.

        Returns:
            None
        """

        if logger is not None:
            self.logger = logger

        self.logger.info('Initialized task manager')

    @contextlib.contextmanager
    def lock(self):
        """
        A context manager for acquiring and releasing a lock.

        Yields:
            The result of acquiring the lock.
        """

        try:
            yield self.rlock.acquire()
        finally:
            self.rlock.release()

    def rotate_uid(self):
        """
        Generates a new unique identifier by rotating the current identifier.

        Returns:
            The generated unique identifier.
        """

        uid = self.uid
        self.uid += 1
        return uid

    def new_task(self, uid: Any = None, description: str = ''):
        """
        Creates a new task with an optional unique identifier and description.

        Args:
            uid: An optional unique identifier for the task.
            description: A description of the task.

        Returns:
            The unique identifier of the created task.
        """

        with self.lock():
            if uid is None:
                # The uid is not provided, using self accending interger
                uid = self.rotate_uid()
            else:
                # The uid is provided, try to make it unique
                stem = str(uid)
                uid = stem
                while self.tasks.get(uid) is not None:
                    uid = f'{stem}-{time.time():.8f}-{random.random():.8f}'

            dct = dict(
                tic=time.time(),
                uid=uid,
                state='running',
                description=description
            )

            self.tasks[uid] = dct
            self.logger.debug(f'Created new task: {dct}')
            return uid

    def task_finished(self, uid, state: str = 'finished'):
        """
        Updates the state of a task to 'finished' and calculates the task duration.

        Args:
            uid: The unique identifier of the task to mark as finished.
            state: The state to set for the task (default is 'finished').

        Returns:
            A dictionary containing the updated task details.
        """

        with self.lock():
            if uid not in self.tasks:
                self.logger.warning(f'Invalid task ID: {uid}')
                return None

            dct = self.tasks[uid]
            dct['state'] = state
            dct['toc'] = time.time()
            dct['costs'] = dct['toc'] - dct['tic']
            self.logger.debug(f'Task finished: {dct}')

            return dct

# %% ---- 2024-04-12 ------------------------
# Play ground


# %% ---- 2024-04-12 ------------------------
# Pending


# %% ---- 2024-04-12 ------------------------
# Pending
