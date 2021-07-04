import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torchvision import models


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
    def __init__(self, state_feature_dim, n_actions):
        super(ICMModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, state_feature_dim)
        self.forward_head = nn.Linear(state_feature_dim+n_actions, state_feature_dim)
        self.inverse_head = nn.Linear(state_feature_dim*2, n_actions)

    def forward(self, state, next_state, action):
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
        encode_state = self.backbone(state)
        encode_next_state = self.backbone(next_state)

        # forward
        pred_next_state_feature = self.forward_head(torch.cat((encode_state, action), 1))
        # inverse
        pred_action = self.inverse_head(torch.cat((encode_state, encode_next_state), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

class ICMAgent(nn.Module):
    def __init__(self, state_feature_dim, n_actions):
        super().__init__()
        self.policy_net = ActorCritic(n_actions=n_actions)
        self.icm = ICMModel(state_feature_dim=state_feature_dim,
                            n_actions=n_actions)
    @torch.no_grad()
    def get_action(self, state):
        "we can apply the exploration and exploitation trade-off trick right here."
        logits, value = self.policy_net(state)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action

    @torch.no_grad()
    def compute_intrinsic(self, state, next_state, action):
        """

        :param state:
        :param next_state:
        :param action: shape of tensor (bs, n_actions)
        :return:
        """

        encode_state = self.icm.backbone(state)
        encode_next_state = self.icm.backbone(next_state)
        # forward
        pred_next_state_feature = self.icm.forward_head(torch.cat((encode_state, action), -1))
        reward = F.mse_loss(pred_next_state_feature, encode_next_state)
        return reward

    def forward(self, states, next_states, actions):
        return self.icm(states, next_states, actions)
