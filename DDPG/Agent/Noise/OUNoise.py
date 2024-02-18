import copy
import random

import numpy as np
import torch

from DDPG.Agent.Noise.Noise import Noise


class OUNoise(Noise):
    """
        This class implements Ornstein Uhlenbeck Process to be applied to the chosen state by the agent for
        the purposes of exploration.
    """

    def __init__(self, outSample: torch.Tensor, mu=0.0, theta=0.15, sigma=0.2, dtype="float32"):
        super(OUNoise, self).__init__(outSample)
        size = self.outSample.shape[0]
        self.state = None
        self.dtype = dtype
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self) -> torch.Tensor:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state.astype(self.dtype)
