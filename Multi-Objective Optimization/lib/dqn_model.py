import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_action, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.fc(x)