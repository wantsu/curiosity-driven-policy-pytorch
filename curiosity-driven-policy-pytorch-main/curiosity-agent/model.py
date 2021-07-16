import math
import random
from config import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from data_generator import Transition
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
    def __init__(self, n_actions):
        super(ActorCritic, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.actor = nn.Linear(self.backbone.fc.out_features, n_actions)
        self.critic = nn.Linear(self.backbone.fc.out_features, 1)

    def forward(self, state):
        """
        :param state:
        :return:

        """
        x = self.backbone(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class ICMModel(nn.Module):
    def __init__(self, state_feature_dim, n_actions, hidden_dim=512):
        super(ICMModel, self).__init__()
        self.policy_net = ActorCritic(n_actions=n_actions)
        self.backbone = models.resnet18(pretrained=True)
        self.lstm = nn.LSTM(self.backbone.fc.out_features, hidden_dim, num_layers=1, batch_first=True)
        #self.backbone.fc = nn.Linear(self.backbone.fc.in_features, state_feature_dim)
        self.linear = nn.Linear(hidden_dim, state_feature_dim)
        self.forward_head = nn.Linear(state_feature_dim+n_actions, state_feature_dim)
        self.inverse_head = nn.Linear(state_feature_dim*2, n_actions)

    #def forward(self, state, next_state, action):
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
        memory = deque([], maxlen=cfg['batch_size'])
        encode_state_seq = None
        encode_next_state_seq = None
        one_hot_action_batch = []
        action_batch = []
        reward_batch = []
        state = []
        for seq in range(cfg['seq_len']):
            for bs in range(cfg['batch_size']):
                memory.append(transitions[bs][seq])
            batch = Transition(*zip(*memory))
            # construct a training batch
            state.append(torch.cat(batch.state).to(device))
            one_hot_action_batch.append((torch.cat(batch.action)).to(device))
            action_batch.append(torch.argmax(torch.cat(batch.action), dim=-1).to(device))
            next_state = torch.cat(batch.next_state)
            next_state = next_state.to(device)
            reward_batch.append(torch.stack(batch.reward).to(device))

            encode_state = self.backbone(state[seq])
            encode_state = torch.unsqueeze(encode_state, 1)

            encode_next_state = self.backbone(next_state)
            encode_next_state = torch.unsqueeze(encode_next_state, 1)

            if seq == 0:
                encode_state_seq = encode_state
                encode_next_state_seq = encode_next_state
            else:
                encode_state_seq = torch.cat((encode_state_seq, encode_state), dim=1)
                encode_next_state_seq = torch.cat((encode_next_state_seq, encode_next_state), dim=1)

        encode_state, (h_n, c_n) = self.lstm(encode_state_seq, None)
        encode_next_state, (h_n, c_n) = self.lstm(encode_next_state_seq, None)
        loss = None
        forward_loss = 0
        for i in range(cfg['seq_len']):
            # compute logps
            logits, values = self.policy_net(state[i])
            logps = Categorical(logits=logits).log_prob(action_batch[i])
            encode_state_every_time = self.linear(encode_state[:, i, :])
            encode_next_state_every_time = self.linear(encode_next_state[:, i, :])
            # forward
            pred_next_state_feature = self.forward_head(torch.cat((encode_state_every_time, one_hot_action_batch[i]), 1))
            forward_loss_time = F.mse_loss(pred_next_state_feature, encode_state_every_time)
            forward_loss = forward_loss_time + cfg['gamma'] * forward_loss
            # inverse
            pred_action = self.inverse_head(torch.cat((encode_state_every_time, encode_next_state_every_time), 1))
            inverse_loss = F.cross_entropy(pred_action, action_batch[i])
            if i == 0:
                loss = -torch.mean(logps * reward_batch[i]) + inverse_loss * cfg['alpha'] + forward_loss * cfg['beta']
            else:
                loss += -torch.mean(logps * reward_batch[i]) + inverse_loss * cfg['alpha'] + forward_loss * cfg['beta']
        return loss

class ICMAgent(nn.Module):
    def __init__(self, state_feature_dim, n_actions):
        super().__init__()
        self.policy_net = ActorCritic(n_actions=n_actions)
        self.icm = ICMModel(state_feature_dim=state_feature_dim,
                            n_actions=n_actions)
    @torch.no_grad()
    def get_action(self, state):
        "we can apply the exploration and exploitation trade-off trick right here."
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            logits, value = self.policy_net(state)
            # policy = Categorical(logits=logits)
            # action = policy.sample()
            _, action = logits.max(-1)
            return action
        else:
            return np.random.choice(n_actions)

    @torch.no_grad()
    def compute_intrinsic(self, state, next_state, action):
        """

        :param state:
        :param next_state:
        :param action: shape of tensor (bs, n_actions)
        :return:
        """

        # encode_state = self.icm.backbone(state)
        # encode_next_state = self.icm.backbone(next_state)

        encode_state = self.icm.backbone(state)
        encode_state = torch.unsqueeze(encode_state, 0)
        encode_state, (h_n, c_n) = self.icm.lstm(encode_state, None)
        encode_state = self.icm.linear(encode_state[:, -1, :])
        encode_next_state = self.icm.backbone(next_state)
        encode_next_state = torch.unsqueeze(encode_next_state, 0)
        encode_next_state, (h_n, c_n) = self.icm.lstm(encode_next_state, None)
        encode_next_state = self.icm.linear(encode_next_state[:, -1, :])


        # forward
        pred_next_state_feature = self.icm.forward_head(torch.cat((encode_state, action), -1))
        reward = F.mse_loss(pred_next_state_feature, encode_next_state)
        return reward

    def forward(self, transitions):
        return self.icm(transitions)
