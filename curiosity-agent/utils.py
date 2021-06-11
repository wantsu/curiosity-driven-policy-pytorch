import torch
import threading
import six.moves.queue as queue
import torchvision.transforms as tfs
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import random
import scipy.signal


transform_train = tfs.Compose([tfs.ToPILImage(),
                           tfs.Resize([128, 128]),
                           tfs.RandomHorizontalFlip(),
                           tfs.ToTensor(),
                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                           ])

transform_val = tfs.Compose([tfs.ToPILImage(),
                           tfs.Resize([128, 128]),
                           tfs.ToTensor(),
                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                           ])


# grab from official implementation
class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        self.bonuses = []
        self.end_state = None

    def add(self, state, action, reward, value, terminal, features, bonus=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.bonuses += [bonus]
        self.end_state = end_state

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        self.bonuses.extend(other.bonuses)
        self.end_state = other.end_state


def get_rollout(env, policy, episode_lens, predictor):
    """
    TODO
    collect a trajectory by allowing agent interacting with environment.
    :param env:
    :param policy:
    :param episode_lens:
    :param predictor: action-state predictor
    :return: rollout of a trajectory,
    """
    rollout = []
    return rollout


class RunnerThread(threading.Thread):
    """
    TODO
    multi-thread data collection.
    """
    def __init__(self, env, policy, roll_out_lens, predictor):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)     # ideally, should be 1. Mostly doesn't matter in our case.
        self.roll_out_lens = roll_out_lens
        self.env = env
        self.policy = policy
        self.predictor = predictor

    def start_runner(self):
        self.start()

    def run(self):
        self._run()

    def _run(self):
        rollout = get_rollout(self.env, self.policy, self.roll_out_lens, self.predictor)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(rollout), timeout=600.0)


def discount(x, gamma):
    """
    compute discounted reward
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

