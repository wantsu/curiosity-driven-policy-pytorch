import torch.nn as nn
import torch.nn.functional
from torch.nn import functional as F
import torchvision.models as models
from utils import *


class Policy(nn.Module):
    """
    policy network which act as an agent keeping interaction with environment
    num_class: action space
    state_feature_size: state feature dim
    rnn_size: hidden dim of lstm
    """

    def __init__(self, num_class=7, state_feature_size=256, rnn_size=256):
        super(Policy, self).__init__()
        self.rnn_size = rnn_size
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, state_feature_size)
        self.lstm = nn.LSTMCell(input_size=state_feature_size, hidden_size=rnn_size)
        self.action_head = nn.Linear(rnn_size, num_class)

    def forward(self, x, h_t, c_t):
        """

        :param x: current state, shape of tensor (h, w, c)
        :param h_t: last hidden state, shape of tensor (1, rnn_size)
        :param c_t: last context state, shape of tensor (1, rnn_size)
        :return: action logits, shape of tensor (num_class
        ) state dict (h_t, c_t)
        """
        obs_feature = self.resnet18(x)
        h_t, c_t = self.lstm(obs_feature, (h_t, c_t))
        o_t = self.action_head(h_t)
        return o_t, (h_t, c_t)

    def init_state(self):
        return (
            torch.zeros(self.rnn_size),
            torch.zeros(self.rnn_size)
        )


class StateActionPredictor(nn.Module):
    """
    ICM module proposed in paper, which try to predict current action given current and next state, and to predict the feature
    of next state given current state and action. The predicted error of next state is treated as bonus to provide intrinsic
    reward.
    feature_head: cnn head for extracting state feature from observation
    num_class: action space
    state_feature_size: state feature dim
    """
    def __init__(self, feature_head=None, num_class=7, state_feature_size=256):
        super(StateActionPredictor, self).__init__()
        self.feature_head = feature_head
        self.num_class = num_class
        self.state_feature_size = state_feature_size

    def forward(self, x_1, x_2, a):
        """
        predict action and bonus computation
        :param x_1: current state, shape of tensor(3, w, h), will converted to state feature by the feature_head
        :param x_2: next state, shape of tensor(3, w, h), will converted to state feature by the feature_head
        :param a: current action, int, action will converted to one-hot embedding for bonus computation. see paper for detail
        :return:
        s_hat: predicted next state feature, shape of tensor()
        a_hat: predicted current action
        """
        pass

    def pred_act(self, s1, s2):
        """
        predict current action
        :param s1:
        :param s2:
        :return:
        """
        pass

    def pred_bonus(self, s1, s2, onehot_a):
        """
        bonus computation
        :param s1: next state feature, shape of tensor(state_feature_size)
        :param s2: predicted next state feature, shape of tensor(state_feature_size)
        :param onehot_a:
        :return:
        """
        pass


class A3C(nn.Module):
    """
    TODO
    Actor Critic framework for training policy and predictor
    see curiosity-driven official code for details on how to compute A3C loss
    """
    def __init__(self):
        super(A3C, self).__init__()
        pass

    def forward(self):
        """
        1. retrieve a trajectory rollout
        2. return and bonus computation
        3. backpropagation
        :return:
        """
        beta = 0.1   # scaling entropy loss
        alpha = 0.5  # scaling predictor loss
        rollout = []
        # compute return
        # backprop
        # compute loss for policy
        actions = []
        adv = []
        rtn = []
        value = []
        logits = []
        logps = torch.log_softmax(logits)
        prob = []
        pi_loss = - torch.mean(torch.sum(logps * actions, dim=1)*adv)
        value_loss = 0.5*torch.mean(torch.sqrt(value - rtn))  # TD
        entropy = - torch.mean(torch.sum(prob * logps, dim=1))
        policy_loss = pi_loss + 0.5*value_loss - entropy*beta
        # compute loss for predictor
        forwardloss = []
        inverseloss = []
        predictor_loss = forwardloss + inverseloss

        loss = policy_loss + alpha*predictor_loss
        pass