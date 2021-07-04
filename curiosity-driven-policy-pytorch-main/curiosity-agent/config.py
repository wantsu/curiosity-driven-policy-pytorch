import vizdoom as vzd
import numpy as np

viz_config = './default.cfg'
game = vzd.DoomGame()
game.load_config(viz_config)

VIZDOOM_TO_TF = [0, 1, 2]
EPISODE_TIMEOUT = 10000
game.set_episode_timeout(EPISODE_TIMEOUT)

# action space
n_actions = game.get_available_buttons_size()
ACTIONS = np.zeros(n_actions, dtype=np.int64).tolist()