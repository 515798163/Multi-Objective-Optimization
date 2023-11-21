import ptan
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

SOURCE_TARGET_PATH = '../data/facebook/facebook_combined.csv'

data_set = pd.read_csv(SOURCE_TARGET_PATH)
edges = np.asarray(data_set, np.int64)
edges = edges.T

class ModelActor(nn.Module):
    def __init__(self, n_features, n_actions, hidden=64):
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
            nn.Tanh(),
            nn.Linear(512, n_actions),
            nn.Tanh()
        )

    def _get_conv_out(self, shape):
        zeros = torch.zeros(*shape)
        features = self.gcn1(zeros, self.edges)
        features = F.relu(features)
        features = self.gcn2(features, self.edges)
        o = F.relu(features)
        # o = self.conv(zeros, self.edges)
        return int(np.prod(o.size()))

    def forward(self, features):
        features = self.gcn1(features, self.edges_gpu)
        features = F.tanh(features)
        features = self.gcn2(features, self.edges_gpu)
        features = F.tanh(features)
        conv_out = features.view(features.size()[0], -1)
        policy = self.policy(conv_out)
        return policy

class ModelCritic(nn.Module):
    def __init__(self, n_features, hidden=64):
        super(ModelCritic, self).__init__()
        self.device = torch.device('cuda')
        self.edges = torch.LongTensor(edges)
        self.edges_gpu = self.edges.to(self.device)
        self.gcn1 = GCNConv(n_features[1], hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        conv_out_size = self._get_conv_out(n_features)

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def _get_conv_out(self, shape):
        zeros = torch.zeros(*shape)
        features = self.gcn1(zeros, self.edges)
        features = F.relu(features)
        features = self.gcn2(features, self.edges)
        o = F.relu(features)
        # o = self.conv(zeros, self.edges)
        return int(np.prod(o.size()))

    def forward(self, features):
        features = self.gcn1(features, self.edges_gpu)
        features = F.tanh(features)
        features = self.gcn2(features, self.edges_gpu)
        features = F.tanh(features)
        conv_out = features.view(features.size()[0], -1)
        policy = self.value(conv_out)
        return policy


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


