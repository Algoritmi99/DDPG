import numpy as np
import torch
import torch.nn as nn

from params import device, dtype


class ActorNet(nn.Module):
    def __init__(self, n_states, actor_layer):
        super(ActorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, actor_layer),
            nn.ReLU(),
            nn.Linear(actor_layer, actor_layer),
            nn.ReLU(),
            nn.Linear(actor_layer, 1),
            nn.Tanh()
        ).to(device).to(dtype)

    def forward(self, state) -> torch.Tensor:
        return self.net(state)

    def select_action(self, state, env) -> torch.Tensor:
        with (torch.no_grad()):
            noise = np.random.normal(0, 1, 1)
            action = self.forward(state) + \
                noise[0] * torch.randn(size=env.action_space.shape, device=device, dtype=dtype)

            return action

