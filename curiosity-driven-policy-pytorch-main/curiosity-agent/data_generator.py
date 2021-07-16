import time
import threading
import queue
# import vizdoom as vzd
import numpy as np
import torch

import random
from collections import namedtuple, deque
SEQ_LEN = 5


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, seq):
        """Save a transition"""
        self.memory.append(seq)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)