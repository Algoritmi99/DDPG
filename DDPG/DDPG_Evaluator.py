import time

import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from DDPG.Agent.Agent import Agent
from DDPG.Plotter import Plotter

# TODO:
#  now - save the reward value for each step of each episode,
#  taking mean of the reward value and add it to a global table

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

    def evaluate(self, num_episodes: int, current_iteration=0):
        state, info = self.__environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)

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
                if not self.__trainMode:
                    time.sleep(0.05)

            if self.__plotter is not None:
                self.__plotter.add_evaluationRewardPerStep(sum(rewards)/len(rewards), current_iteration)

            state, info = self.__environment.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = (False, False)

    def set_trainMode(self, trainMode: bool):
        self.__trainMode = trainMode
