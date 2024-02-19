import torch
from torch import nn

from DDPG.StaticAlgorithms import hidden_layer_init


class Actor(nn.Module):
    """
        This class is the actor network as suggested in the DDPG paper (https://arxiv.org/abs/1509.02971)
    """

    def __init__(self, observations_dim, action_dim, fc1_units=400, fc2_units=300, device="cpu"):
        super(Actor, self).__init__()
        self.__device = device

        self.__net = nn.Sequential(
            nn.Linear(observations_dim, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_dim),
            nn.Tanh()
        ).to(self.__device)

        self.reset_params()

    def forward(self, state) -> torch.Tensor:
        return self.__net(state)

    def reset_params(self):
        for i, layer in enumerate(self.__net):
            if isinstance(layer, nn.Linear):
                if i == 0 or i == 1:
                    torch.nn.init.uniform_(hidden_layer_init(layer))
                else:
                    torch.nn.init.uniform_(layer.weight, a=-3e-3, b=3e-3)

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))
