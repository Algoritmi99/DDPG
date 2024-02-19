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

    def __init__(self,
                 environment: gym.Env,
                 agent: Agent,
                 plotter: Plotter = None,
                 train_mode=False,
                 device: str = 'cpu',
                 mujoco_mode: bool = False):
        self.__environment = environment
        self.__agent = agent
        self.__plotter = plotter
        self.__trainMode = train_mode
        self.__device = device
        self.__mujoco_mode = mujoco_mode

    def evaluate(self, num_episodes: int):
        state, info = self.__environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)
        returns = []

        iterable = range(num_episodes) if self.__trainMode else tqdm(range(num_episodes))
        for _ in iterable:
            rewards = []
            while not (terminated or truncated):
                state = state.to(self.__device)
                numpy_action = self.__agent.take_greedyAction(state).squeeze().to("cpu").numpy()
                numpy_action = [numpy_action] if not self.__mujoco_mode else numpy_action
                state, reward, terminated, truncated, info = self.__environment.step(numpy_action)
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

    def set_trainMode(self, trainMode: bool):
        self.__trainMode = trainMode
