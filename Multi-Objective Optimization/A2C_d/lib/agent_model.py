import ptan
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import logging

from ptan import actions

CRUMOR_NUMBER = 5

SOURCE_TARGET_PATH = '../data/facebook/facebook_combined.csv'

data_set = pd.read_csv(SOURCE_TARGET_PATH)
edges = np.asarray(data_set, np.int64)
edges = edges.T

class ModelActor(nn.Module):
    def __init__(self, n_features, n_actions, hidden=32):
        super(ModelActor, self).__init__()
        self.device = torch.device('cuda')
        self.edges = torch.LongTensor(edges)
        self.edges_gpu = self.edges.to(self.device)
        self.gcn1 = GCNConv(n_features[1], hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        self.logstd = nn.Parameter(torch.zeros(n_actions))

        conv_out_size = self._get_conv_out(n_features)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            # nn.Tanh()
        )

    def _get_conv_out(self, shape):
        zeros = torch.zeros(*shape)
        features = self.gcn1(zeros, self.edges)
        # features = F.relu(features)
        # features = torch.tanh(features)
        features = torch.relu(features)
        features = self.gcn2(features, self.edges)
        # o = F.relu(features)
        # o = torch.tanh(features)
        o = torch.relu(features)
        # o = self.conv(zeros, self.edges)
        return int(np.prod(o.size()))

    def forward(self, features):
        features = self.gcn1(features, self.edges_gpu)
        # features = F.tanh(features)
        # features = torch.tanh(features)
        features = torch.relu(features)
        features = self.gcn2(features, self.edges_gpu)
        # features = F.tanh(features)
        # features = torch.tanh(features)
        features = torch.relu(features)
        conv_out = features.view(features.size()[0], -1)
        policy = self.policy(conv_out)
        return policy

class ModelCritic(nn.Module):
    def __init__(self, n_features, hidden=32):
        super(ModelCritic, self).__init__()
        self.device = torch.device('cuda')
        self.edges = torch.LongTensor(edges)
        self.edges_gpu = self.edges.to(self.device)
        self.gcn1 = GCNConv(n_features[1], hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        conv_out_size = self._get_conv_out(n_features)

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(512, 1),
            # nn.Tanh()
        )

    def _get_conv_out(self, shape):
        zeros = torch.zeros(*shape)
        features = self.gcn1(zeros, self.edges)
        # features = F.relu(features)
        # features = torch.tanh(features)
        features = torch.relu(features)

        features = self.gcn2(features, self.edges)
        # o = F.relu(features)
        # o = torch.tanh(features)
        o = torch.relu(features)
        # o = self.conv(zeros, self.edges)
        return int(np.prod(o.size()))

    def forward(self, features):
        features = self.gcn1(features, self.edges_gpu)
        # features = F.tanh(features)
        # features = torch.tanh(features)
        features = torch.relu(features)
        features = self.gcn2(features, self.edges_gpu)
        # features = F.tanh(features)
        # features = torch.tanh(features)
        features = torch.relu(features)
        conv_out = features.view(features.size()[0], -1)
        policy = self.value(conv_out)
        return policy


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


class SelectorPPO(actions.ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            '''不重复的选择离散动作'''
            actions.append(np.random.choice(len(prob), size=(CRUMOR_NUMBER, ), p=prob, replace=False))
        return np.array(actions)


class AgentPPO(ptan.agent.BaseAgent):
    def __init__(self, model, action_selector=SelectorPPO(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):

        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs_v = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs_v)
        return np.array(actions), agent_states





# class AgentA2C(ptan.agent.BaseAgent):
#     def __init__(self, net, device="cpu"):
#         self.net = net
#         self.device = device
#
#     def __call__(self, states, agent_states):
#         states_v = ptan.agent.float32_preprocessor(states)
#         states_v = states_v.to(self.device)
#
#         mu_v = self.net(states_v)
#         mu = mu_v.data.cpu().numpy()
#         if True in np.isnan(mu):
#             print("*************2")
#         logstd = self.net.logstd.data.cpu().numpy()
#         rnd = np.random.normal(size=logstd.shape)
#
#         actions = mu + np.exp(logstd) * rnd #add noise  mu + (e^(logstd))*rnd
#         # actions = mu    #not add noise
#
#         actions = np.clip(actions, -1, 1)
#
#
#         return actions, agent_states