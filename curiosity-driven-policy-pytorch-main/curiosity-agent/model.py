import math
import random
from config import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from data_generator import Transition
from utils import discount
from collections import deque
from torch.autograd import Variable

from torchvision import models


GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0

class ActorCritic(nn.Module):
    """
    """
    def __init__(self, n_actions, state_feature_dim):
        super(ActorCritic, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.lstm_cell = nn.LSTMCell(self.backbone.fc.out_features, state_feature_dim)
        self.actor = nn.Linear(state_feature_dim, n_actions)
        self.critic = nn.Linear(state_feature_dim, 1)

    def forward(self, state, h_t, c_t):
        """
        :param state:
        :return:

        """
        x = self.backbone(state)
        (h_t, c_t) = self.lstm_cell(x, (h_t, c_t))
        logits = self.actor(h_t)
        value = self.critic(h_t)
        return logits, value, (h_t, c_t)

class ICMModel(nn.Module):
    def __init__(self, state_feature_dim, n_actions):
        super(ICMModel, self).__init__()
        self.policy_net = ActorCritic(n_actions=n_actions,
                                      state_feature_dim=state_feature_dim)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, state_feature_dim)
        self.forward_head = nn.Linear(state_feature_dim+n_actions, state_feature_dim)
        self.inverse_head = nn.Linear(state_feature_dim*2, n_actions)

    def forward(self, transitions):
        """
        predict next state feature and current action
        :param
        state: shape of tensor (bs, h, w, c), current observation
        next_state: shape of tensor (bs, h, w, c), next observation
        actoin: shape of tensor (bs, n_actions), onehot encoding of actions
        :return:
        real_next_state_feature: shape of tensor (bs, state_feature_dim), extracted feature of next observation
        pred_next_state_feature: shape of tensor (bs, state_feature_dim), predicted feature of next observation
        pred_action: shape of tensor (bs, n_actions), predicted current action
        """
        bz = cfg['batch_size']
        memory = deque([], maxlen=bz)
        encode_state_seq = []
        encode_next_state_seq = []
        one_hot_action_batch = []
        action_batch = []
        reward_batch = []
        state = []
        for ts in range(cfg['seq_len']):
            for i in range(cfg['batch_size']):
                memory.append(transitions[i][ts])
            batch = Transition(*zip(*memory))
            # construct a training batch
            state.append(torch.cat(batch.state).to(device))
            one_hot_action_batch.append((torch.cat(batch.action)).to(device))
            action_batch.append(torch.argmax(torch.cat(batch.action), dim=-1).to(device))
            next_state = torch.cat(batch.next_state).to(device)
            reward_batch.append(torch.stack(batch.reward).to(device))

            # embedding current and next state
            encode_state = self.backbone(state[ts])
            encode_next_state = self.backbone(next_state)

            # caching encoded feature of a trajectory
            encode_state_seq += [encode_state]
            encode_next_state_seq += [encode_next_state]

        loss = 0
        # init state for lstm cell
        h_t = c_t = Variable(torch.randn(bz, cfg['state_feature_dim'], requires_grad=True, device=device))
        for t in range(cfg['seq_len']):
            # compute logps
            logits, values, (h_t, c_t) = self.policy_net(state[t], h_t, c_t)
            logps = Categorical(logits=logits).log_prob(action_batch[t])
            # forward
            pred_next_state_feature = self.forward_head(torch.cat((encode_state_seq[t], one_hot_action_batch[t]), -1))
            forward_loss = F.mse_loss(pred_next_state_feature, encode_next_state_seq[t])
            # inverse
            pred_action = self.inverse_head(torch.cat((encode_next_state_seq[t], encode_state_seq[t]), -1))
            inverse_loss = F.cross_entropy(pred_action, action_batch[t])
            # compute loss
            loss += -torch.mean(logps * reward_batch[t]) + inverse_loss * cfg['alpha'] + forward_loss * cfg['beta']

        return loss / cfg['seq_len']

class ICMAgent(nn.Module):
    def __init__(self, state_feature_dim, n_actions):
        super().__init__()
        self.policy_net = ActorCritic(n_actions=n_actions,
                                      state_feature_dim=state_feature_dim)
        self.icm = ICMModel(n_actions=n_actions,
                            state_feature_dim=state_feature_dim,)

    @torch.no_grad()
    def get_action(self, state, h_t, c_t):
        "the exploration and exploitation trade-off trick right here borrowed from PyTorch tutorial."
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            logits, value, (h_t, c_t) = self.policy_net(state, h_t, c_t)
            _, action = logits.max(-1)
            return action, (h_t, c_t)
        else:
            return np.random.choice(n_actions), (h_t, c_t)

    @torch.no_grad()
    def compute_intrinsic(self, state, next_state, action, MEMORY:list = []):
        """
        compute episodic curiosity:
        G_t = sum_{k=0}^{k=t} gamma^k \bar_{s}(o_k, o_t)
        :param state:
        :param next_state:
        :param action: shape of tensor (bs, n_actions)
        :param MEMORY: list of history state feature, optional for computing episodic curiosity
        :return:
        """
        encode_state = self.icm.backbone(state)
        encode_next_state = self.icm.backbone(next_state)
        # forward
        pred_next_state_feature = self.icm.forward_head(torch.cat((encode_state, action), -1))
        reward = F.mse_loss(pred_next_state_feature, encode_next_state)
        # compute memory curiosity
        # if MEMORY != []:
        #     T = []
        #     for i in MEMORY:
        #         T += [F.mse_loss(i, encode_state)]
        #     G_t = discount(T.reverse(), cfg['gamma'])
        #     reward += G_t
        # else:
        #   MEMORY.append(encode_state)
        return reward

    def forward(self, transitions):
        return self.icm(transitions)
