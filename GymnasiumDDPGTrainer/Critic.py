import torch
import torch.nn as nn


class CriticNet(nn.Module):
    def __init__(self, n_states, action_dim, critic_layer, device="cpu", dtype=torch.float32):
        super(CriticNet, self).__init__()
        self.device = device
        self.dtype = dtype

        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, critic_layer),
            nn.ReLU(),
            nn.Linear(critic_layer, critic_layer),
            nn.ReLU(),
            nn.Linear(critic_layer, critic_layer),
            nn.Linear(critic_layer, 1)
        ).to(device).to(dtype)

    def forward(self, state, action) -> torch.Tensor:
        return self.net(torch.cat((state, action), 1))
