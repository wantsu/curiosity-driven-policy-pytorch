import time

import vizdoom as vzd
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as trans

DEFAULT_CONFIG = '/home/yf302/Desktop/Visual goal/curiosity-agent/default.cfg'
VIZDOOM_TO_TF = [0, 1, 2]
EPISODE_TIMEOUT = 128
game = vzd.DoomGame()
game.load_config(DEFAULT_CONFIG)
game.set_episode_timeout(EPISODE_TIMEOUT)

# 创造可能的action
actions_num = game.get_available_buttons_size()
ACTIONS = np.zeros(actions_num, dtype=np.int64).tolist()


def data_generator(episodes=10, d_max=5, batch_size=128, shuffle=True, transforms=None):
    X, Y = [], []
    game.init()
    for e in range(episodes):
        game.new_episode()
        x,  y = [], []
        while not game.is_episode_finished():
            # Gets the state
            obs = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
            action = ACTIONS.copy()
            #action_index, states = agent(transforms(obs), states)
            action_index = int(np.random.randint(0, actions_num, 1))
            action[action_index] = 1
            game.make_action(action)
            x.append(obs)
            y.append(action_index)
            time.sleep(0.1)
        X.append(np.stack(x))
        Y.append(np.stack(y))

    game.close()
    #X, y = generate_batch(trajectory, batch_size, d_max=d_max, shuffle=shuffle, transforms=transforms, episodes=episodes)
    return X, y


def generate_batch(trajectory, batch_size, d_max, shuffle=True, transforms=None, episodes=10):
    batch = []
    for i in range(batch_size):
        idx_trajectory = random.randint(0, episodes - 1)
        idx_img = random.randint(0, EPISODE_TIMEOUT - 11) + idx_trajectory * EPISODE_TIMEOUT
        goal_obs = transforms(trajectory[idx_img + d_max - 1][0])
        obs_seq, action_seq = [], []
        for j in range(d_max):
            curr_obs, action = trajectory[idx_img + j]
            obs = torch.cat((transforms(curr_obs), goal_obs), dim=0)
            obs_seq.append(obs)
            action_seq.append(action)

        batch.append((torch.stack(obs_seq), torch.tensor(action_seq)))

    if shuffle:
        random.shuffle(batch)

    X, Y = [], []
    for i in range(batch_size):
        x, y = batch[i]
        X.append(x)
        Y.append(y)

    return torch.stack(X), torch.stack(Y)