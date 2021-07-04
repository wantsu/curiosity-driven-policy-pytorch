import time
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import transform_train, transform_val
from data_generator import ReplayMemory, Transition
from model import ICMAgent
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = {
    'max_episode_len':10000,  # episode len should more than batch_size*repeated_step
    'repeated_step': 4,
    'epochs': 20,
    'batch_size': 64,
    'lr': 1e-4,
    'state_feature_dim': 250,
    'n_actions': n_actions,
    'alpha': 0.5,
    'beta': 0.5
}

def setup(cfg):
    memory = ReplayMemory(10000)
    agent = ICMAgent(cfg['state_feature_dim'], cfg['n_actions'])
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg['lr'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    return memory, agent, optimizer, scheduler


def optimize_model(memory, model, optimizer, scheduler):
    # get batch of samples
    transitions = memory.sample(cfg['batch_size'])
    batch = Transition(*zip(*transitions))
    # construct a training batch
    state_batch = torch.cat(batch.state)
    one_hot_action_batch = torch.cat(batch.action)
    action_batch = torch.argmax(one_hot_action_batch, dim=-1)
    next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.stack(batch.reward)

    # compute logps
    logits, values = model.policy_net(state_batch)
    logps = Categorical(logits=logits).log_prob(action_batch)

    # compute forward and inverse loss
    real_next_state_feature, pred_next_state_feature, pred_action = model(state_batch, next_state_batch, one_hot_action_batch)
    inverse_loss = F.cross_entropy(pred_action, action_batch)
    forward_loss = F.mse_loss(pred_next_state_feature, real_next_state_feature)
    # backward and update weights
    loss = -torch.mean(logps*reward_batch) + inverse_loss*cfg['alpha'] + forward_loss*cfg['beta']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()


def train(epoch, memory, model, optimizer, scheduler, episode_len=32):
    model.train()
    game.init()
    # initialize state
    state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
    state = transform_train(state).unsqueeze(0).to(device)
    for t in tqdm(range(episode_len)):
        # sampling an action from policy
        action = model.get_action(state)
        # execute an action
        one_hot_action = ACTIONS.copy()
        one_hot_action[action] = 1
        for _ in range(cfg['repeated_step']):
            game.make_action(one_hot_action)

        # compute reward
        next_state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
        next_state = transform_train(next_state).unsqueeze(0).to(device)
        one_hot_action = torch.tensor(one_hot_action, dtype=torch.float32, device=device).unsqueeze(0)
        reward = model.compute_intrinsic(state, next_state, one_hot_action)

        # Store the transition in memory
        memory.push(state, one_hot_action, next_state, reward)
        # Move to the next state
        state = next_state
        if len(memory) > cfg['batch_size']:
              optimize_model(memory, model, optimizer, scheduler)


def val(epoch, model, episode_len=16):
    model.eval()
    game.init()
    with torch.no_grad():
        for t in range(episode_len):
            state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
            state = transform_val(state).unsqueeze(0).to(device)
            action = model.get_action(state)
            one_hot_action = ACTIONS.copy()
            one_hot_action[action] = 1
            for _ in range(cfg['repeated_step']):
                game.make_action(one_hot_action)

def main():
    memory, model, optimizer, scheduler = setup(cfg)
    best_val = 0
    start_time = time.time()
    for epoch in range(cfg['epochs']):
        train(epoch, memory, model, optimizer, scheduler, cfg['max_episode_len'])
        if (epoch + 1) % 20 == 0:
            acc = val(epoch, model)
            if acc >= best_val:
                torch.save(model, f'model_{epoch}.pth')

            # logging here
if __name__ == '__main__':
    main()



