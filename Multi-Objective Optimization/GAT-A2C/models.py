import torch.nn as nn
from torch_geometric.nn import GAT

class DQN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            GAT(nfeat, nhid),
            nn.ReLU(),
            GAT(nhid, nhid),
            nn.Linear(nhid, dropout)
        )