import copy
import random

import numpy as np


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dtype="float64"):
        self.state = None
        self.dtype = dtype
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state.astype(self.dtype)
