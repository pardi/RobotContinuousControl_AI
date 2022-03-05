import torch.nn as nn
import torch.nn.functional as f


# Actor network
class ActorNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_layers=(64, 64)):
        super(ActorNetwork, self).__init__()

        self.__lin = []

        self.__lin.append(nn.Linear(in_channels, hidden_layers[0]))

        for hidden_layer_in, hidden_layer_out in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.__line.append(nn.Linear(hidden_layer_in, hidden_layer_out))

        self.__lin.append(nn.Linear(hidden_layers[-1], out_channels))

    def forward(self, x):
        # Execute all layers
        for layer in self.__lin[:-1]:
            x = f.relu(layer(x))

        # Use softmax or tanh to return a probability!
        return f.tanh(self.__lin[-1](x))


# Critic network
class CriticNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers=(64, 64)):
        super(CriticNetwork, self).__init__()

        self.__lin = []

        self.__lin.append(nn.Linear(in_channels, hidden_layers[0]))

        for hidden_layer_in, hidden_layer_out in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.__line.append(nn.Linear(hidden_layer_in, hidden_layer_out))

        self.__lin.append(nn.Linear(hidden_layers[-1], out_channels))

    def forward(self, x):
        # Execute all layers
        for layer in self.__lin[:-1]:
            x = f.relu(layer(x))

        return self.__lin[-1](x)
