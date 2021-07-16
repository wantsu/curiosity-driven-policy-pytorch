import os
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
import argparse
import time
from tensorboardX import SummaryWriter
from collections import deque



parser = argparse.ArgumentParser()

parser.add_argument("--mode", default='train')
parser.add_argument("--model", default='model_19.pth')
args = parser.parse_args()


write = SummaryWriter()



def setup(cfg):
    memory = ReplayMemory(10000)
    if args.mode == 'train':
        agent = ICMAgent(cfg['state_feature_dim'], cfg['n_actions'])
    else:
        agent = torch.load(args.model)
    agent = agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg['lr'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    return memory, agent, optimizer, scheduler


def optimize_model(memory, model, optimizer, scheduler):
    # get batch of samples
    transitions = memory.sample(cfg['batch_size'])
    # memory = deque([], maxlen=cfg['batch_size'])
    # for i in range(cfg['batch_size']):
    #     memory.append(transitions[i][cfg['seq_len']-1])
    # batch = Transition(*zip(*memory))
    # # construct a training batch
    # state_batch = torch.cat(batch.state)
    # one_hot_action_batch = torch.cat(batch.action)
    # action_batch = torch.argmax(one_hot_action_batch, dim=-1)
    # next_state_batch = torch.cat(batch.next_state)
    # reward_batch = torch.stack(batch.reward)
    #
    # # compute logps
    # logits, values = model.policy_net(state_batch.to(device))
    # logps = Categorical(logits=logits).log_prob(action_batch.to(device))
    #
    # # compute forward and inverse loss
    # forward_loss, pred_action = model(transitions)
    # inverse_loss = F.cross_entropy(pred_action, action_batch.to(device))
    # # forward_loss = F.mse_loss(pred_next_state_feature, real_next_state_feature)
    # # backward and update weights
    # loss = -torch.mean(logps*reward_batch) + inverse_loss*cfg['alpha'] + forward_loss*cfg['beta']
    loss = model(transitions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


def train(epoch, memory, model, optimizer, scheduler, episode_len=32):
    model.train()
    game.init()
    # initialize state
    state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
    state = transform_train(state).unsqueeze(0)
    loss = None
    for t in tqdm(range(episode_len)):
        seq = []
        for _ in range(cfg['seq_len']):
            # sampling an action from policy
            action = model.get_action(state.to(device))
            # execute an action
            one_hot_action = ACTIONS.copy()
            one_hot_action[action] = 1
            for _ in range(cfg['repeated_step']):
                game.make_action(one_hot_action)
            # compute reward
            next_state = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
            next_state = transform_train(next_state).unsqueeze(0)
            one_hot_action = torch.tensor(one_hot_action, dtype=torch.float32).unsqueeze(0)
            reward = model.compute_intrinsic(state.to(device), next_state.to(device), one_hot_action.to(device))

            # Store the transition in memory

            seq.append(Transition(*(state, one_hot_action, next_state, reward)))
            #memory.push(state, one_hot_action, next_state, reward)
            # Move to the next state
            state = next_state
        memory.push(seq)
        if len(memory) > cfg['batch_size']:
            loss = optimize_model(memory, model, optimizer, scheduler)

    game.close()
    return loss



def val(model, episode_len=100):
    model.eval()
    #game.episode_timeout = 1000
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
                time.sleep(0.1)
            #time.sleep(0.1)

def main():
    memory, model, optimizer, scheduler = setup(cfg)
    if args.mode == 'train':
        for epoch in range(cfg['epochs']):
            loss = train(epoch, memory, model, optimizer, scheduler, cfg['max_episode_len'])
            write.add_scalar('train_loss', loss, epoch)
            if (epoch + 1) % 20 == 0:
                torch.save(model, f'model_{epoch+1}_{n_actions}.pth')
        write.close()
    elif args.mode == 'test':
        val(model)
            # logging here
if __name__ == '__main__':
    main()



