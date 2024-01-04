import torch
import torch.nn as nn
from GymnasiumDDPGTrainer.StaticAlgorithms import hidden_layer_init


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, device="cpu", dtype=torch.float32, fcs1_units=400, fc2_units=300):
        super(CriticNet, self).__init__()
        self.device = device
        self.dtype = dtype

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, fcs1_units),
            nn.ReLU(),
        ).to(device).to(dtype)

        self.net = nn.Sequential(
            nn.Linear(fcs1_units + action_dim, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        ).to(device).to(dtype)
        self.reset_params()

    def reset_params(self):
        for i, layer in enumerate(self.state_net):
            if isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(hidden_layer_init(layer))

        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    torch.nn.init.uniform_(hidden_layer_init(layer))
                else:
                    torch.nn.init.uniform_(layer.weight, a=-3e-3, b=3e-3)

    def forward(self, state, action) -> torch.Tensor:
        state_net_output = self.state_net(state)
        return self.net(torch.cat((state_net_output, action), 1))

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))
