import torch
import torch.nn as nn

from GymnasiumDDPGTrainer.OUNoise import OUNoise
from GymnasiumDDPGTrainer.StaticAlgorithms import hidden_layer_init


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim,
                 device="cpu", dtype=torch.float32, fc1_units=400, fc2_units=300, noise: str | OUNoise = "default"
                 ):
        super(ActorNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.noise = noise

        self.net = nn.Sequential(
            nn.Linear(state_dim, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_dim),
            nn.Tanh()
        ).to(self.device).to(self.dtype)
        self.reset_params()

    def reset_params(self):
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i == 0 or i == 1:
                    torch.nn.init.uniform_(hidden_layer_init(layer))
                else:
                    torch.nn.init.uniform_(layer.weight, a=-3e-3, b=3e-3)

    def forward(self, state) -> torch.Tensor:
        return self.net(state)

    def select_action(self, state, env) -> torch.Tensor:
        with (torch.no_grad()):
            if self.noise == "default":
                action = self.forward(state) + torch.randn(
                    size=env.action_space.shape, device=self.device, dtype=self.dtype
                )
            else:
                action = self.forward(state).cpu() + self.noise.sample()
                action.to(self.device)
        return action

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))
