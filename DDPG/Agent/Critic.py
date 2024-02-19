import torch
import torch.nn as nn
from DDPG.StaticAlgorithms import hidden_layer_init


class Critic(nn.Module):
    """
        This class is the critic network as suggested in the DDPG paper (https://arxiv.org/abs/1509.02971)
    """

    def __init__(self, observation_dim, action_dim, fcs1_units=400, fc2_units=300, device="cpu"):
        super(Critic, self).__init__()
        self.__device = device

        self.__state_net = nn.Sequential(
            nn.Linear(observation_dim, fcs1_units),
            nn.ReLU(),
        ).to(self.__device)

        self.__net = nn.Sequential(
            nn.Linear(fcs1_units + action_dim, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        ).to(self.__device)

        self.reset_params()

    def forward(self, state, action) -> torch.Tensor:
        state_net_output = self.__state_net(state)
        return self.__net(torch.cat((state_net_output, action), 1))

    def reset_params(self):
        for i, layer in enumerate(self.__state_net):
            if isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(hidden_layer_init(layer))

        for i, layer in enumerate(self.__net):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    torch.nn.init.uniform_(hidden_layer_init(layer))
                else:
                    torch.nn.init.uniform_(layer.weight, a=-3e-3, b=3e-3)

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))
