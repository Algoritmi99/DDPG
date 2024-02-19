import gymnasium as gym
import torch
from tqdm import tqdm

from DDPG.Agent.Agent import Agent
from DDPG.DDPG_Evaluator import Evaluator
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.Plotter import Plotter


class Trainer(object):
    """
        This class os the trainer of the DDPG algorithm,
        that can be used to train an agent on an environment.
    """

    def __init__(self, environment: gym.Env,
                 agent: Agent,
                 replay_buffer: ReplayBuffer,
                 plotter: Plotter = None,
                 evaluator: Evaluator = None,
                 device: str = 'cpu',
                 mujoco_mode: bool = False
                 ) -> None:
        self.__environment = environment
        self.__agent = agent
        self.__replay_buffer = replay_buffer
        self.__plotter = plotter
        self.__evaluator = evaluator
        self.__evaluator.set_trainMode(True)
        self.__device = device
        self.__mujoco_mode = mujoco_mode

    def train(self, num_episodes: int, max_steps: int):
        state, info = self.__environment.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = (False, False)
        for i in tqdm(range(num_episodes)):
            timeStep = 0
            while not terminated and not truncated and timeStep < max_steps:
                state = state.to(self.__device)
                action = self.__agent.take_Action(state)

                numpy_action = action.to('cpu').squeeze().numpy()
                numpy_action = [numpy_action] if not self.__mujoco_mode else numpy_action
                next_state, reward, terminated, truncated, info = self.__environment.step(numpy_action)

                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.__device)
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.__device)
                terminated = torch.tensor(terminated, dtype=torch.bool).unsqueeze(0).to(self.__device)
                action = action.to(self.__device)

                self.__replay_buffer.push(state, action, reward, next_state, terminated)
                state = next_state

                if self.__replay_buffer.canSample():
                    (
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        terminated_batch
                    ) = self.__replay_buffer.sample_batch()

                    self.__agent.update(
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        terminated_batch
                    )

                if self.__plotter is not None:
                    self.__plotter.add_trainingReward(float(reward), i)

                timeStep += 1

            if self.__evaluator is not None and i % 10 == 0:
                self.__evaluator.evaluate(10)

            state, info = self.__environment.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = (False, False)

    def get_agent(self) -> Agent:
        return self.__agent
