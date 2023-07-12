import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np
from typing import Tuple


# Actor network
class ActorNetwork(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_layers: Tuple[int, int] = (64, 64)):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], out_channels)

        self.reset_params()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.bn1(self.fc1(state)))
        x = f.relu(self.fc2(x))
        return f.tanh(self.fc3(x))

    def reset_params(self) -> None:
        self.fc1.weight.data.uniform_(*self.param_init_fcn(self.fc1))
        self.fc2.weight.data.uniform_(*self.param_init_fcn(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @staticmethod
    def param_init_fcn(layer: nn.Linear) -> Tuple[float, float]:
        f_in = layer.weight.data.size()[0]
        lm = 1. / np.sqrt(f_in)
        return -lm, lm


# Critic network
class CriticNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers: Tuple[int, int] = (64, 64)):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0] + out_channels, hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], 1)

        self.reset_params()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = f.relu(self.bn1(self.fc1(state)))
        x = torch.cat((xs, action), dim=1)
        x = f.relu(self.fc2(x))
        return self.fc3(x)

    def reset_params(self) -> None:
        self.fc1.weight.data.uniform_(*self.param_init_fcn(self.fc1))
        self.fc2.weight.data.uniform_(*self.param_init_fcn(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @staticmethod
    def param_init_fcn(layer: nn.Linear) -> Tuple[float, float]:
        f_in = layer.weight.data.size()[0]
        lm = 1. / np.sqrt(f_in)
        return -lm, lm
