import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from DDPG.Agent.Agent import Agent
from DDPG.Plotter import Plotter


class Evaluator(object):
    """
        This class evaluates a trained agent on an environment in visual rendering mode.
    """

    def __init__(self, environment: gym.Env, agent: Agent, plotter: Plotter = None):
        self.__environment = environment
        self.__agent = agent
        self.__plotter = plotter

    def evaluate(self, num_episodes: int):
        state, info = self.__environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)
        returns = []

        for _ in tqdm(range(num_episodes)):
            rewards = []
            while not (terminated or truncated):
                numpy_action = self.__agent.take_greedyAction(state).squeeze().numpy()
                state, reward, terminated, truncated, info = self.__environment.step([numpy_action])
                rewards.append(reward)
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            episodeReward = sum(rewards)
            returns.append(episodeReward)

            if self.__plotter is not None:
                self.__plotter.add_evaluationReward(episodeReward)

            state, info = self.__environment.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = (False, False)

        return np.mean(returns)
