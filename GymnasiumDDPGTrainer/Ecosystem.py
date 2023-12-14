import gymnasium as gym
import torch
from gymnasium.core import ObsType

from GymnasiumDDPGTrainer.Actor import ActorNet


class Ecosystem:
    def __init__(self, environment: gym.Env, actor: ActorNet, device="cpu", dtype=torch.double):
        self.__environment = environment
        self.__actor = actor
        self.__device = device
        self.__dtype = dtype

    def take_action(self, state) -> tuple[gym.Env, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.__actor.select_action(state, self.__environment)
        action = action.cpu()
        next_state, reward, terminated, truncated, _ = self.__environment.step(action[0])
        action = action.to(self.__device)

        next_state = torch.tensor(next_state, device=self.__device, dtype=self.__dtype).unsqueeze(0)
        reward = torch.tensor([reward], device=self.__device, dtype=self.__dtype).unsqueeze(0)
        done = torch.tensor([terminated or truncated], device=self.__device, dtype=self.__dtype).unsqueeze(0)

        return self.__environment, action, next_state, reward, done

    def get_environment(self) -> gym.Env:
        return self.__environment

    def reset_environment(self) -> ObsType:
        return self.__environment.reset()[0]

    def get_actor(self) -> ActorNet:
        return self.__actor

    def set_actor(self, actor: ActorNet):
        self.__actor = actor
