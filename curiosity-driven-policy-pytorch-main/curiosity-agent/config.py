import vizdoom as vzd
import numpy as np
import torch

viz_config = './default.cfg'
game = vzd.DoomGame()
game.load_config(viz_config)

VIZDOOM_TO_TF = [0, 1, 2]
EPISODE_TIMEOUT = 100000
game.set_episode_timeout(EPISODE_TIMEOUT)

# action space
n_actions = game.get_available_buttons_size()
ACTIONS = np.zeros(n_actions, dtype=np.int64).tolist()

cfg = {
    'max_episode_len':10000,  # episode len should more than batch_size*repeated_step
    'repeated_step': 4,
    'epochs': 20,
    'batch_size': 16,
    'lr': 1e-4,
    'state_feature_dim': 250,
    'n_actions': n_actions,
    'alpha': 0.5,
    'beta': 0.5,
    'gamma': 0.2,
    'seq_len': 5
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')