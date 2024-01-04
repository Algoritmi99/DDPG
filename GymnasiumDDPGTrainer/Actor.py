import numpy as np
import torch
import torch.nn as nn
from .LayersInit import hidden_layer_init


class ActorNet(nn.Module):
    def __init__(self, n_states, action_dim, device="cpu", dtype=torch.float32, fc1_units=400, fc2_units=300):
        super(ActorNet, self).__init__()
        self.device = device
        self.dtype = dtype

        self.net = nn.Sequential(
            nn.Linear(n_states, fc1_units),
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
            action = self.forward(state) + torch.randn(size=env.action_space.shape, device=self.device, dtype=self.dtype)

        return action
