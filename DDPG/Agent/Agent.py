import copy
import torch
from torch import nn
from DDPG.Agent.Noise import Noise
from DDPG.StaticAlgorithms import network_update, update_target_net


class Agent(object):
    """
        The Agent class is a composite class consisting of an Actor and a Critic.
        It resembles a DDPG agent, which is capable of decision-making through the Actor and can
        also learn using both the Actor and the Critic.
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 noiseObj: Noise,
                 discount: float,
                 tau: float) -> None:
        self.__actor = actor
        self.__actor_target_net = copy.deepcopy(self.__actor)
        self.__critic = critic
        self.__critic_target_net = copy.deepcopy(self.__critic)
        self.__critic_optimizer = critic_optimizer
        self.__actor_optimizer = actor_optimizer
        self.__noiseObj = noiseObj
        self.__discount = discount
        self.__tau = tau
        self.__critic_criterion = nn.MSELoss()

    def take_greedyAction(self, state: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            return self.__actor(state)

    def take_Action(self, state: torch.tensor) -> torch.tensor:
        raw_action = self.take_greedyAction(state)
        noise = self.__noiseObj.sample()
        return raw_action + noise

    def update(
            self,
            state_batch: torch.tensor,
            action_batch: torch.tensor,
            reward_batch: torch.tensor,
            next_state_batch: torch.tensor,
            terminated_batch: torch.tensor) -> None:
        not_terminated = ~terminated_batch

        # Update critic
        critic_target = reward_batch
        with torch.no_grad():
            next_action_batch = self.__actor_target_net(next_state_batch)
            critic_target[not_terminated] += self.__discount * self.__critic_target_net(
                next_state_batch[not_terminated],
                next_action_batch[not_terminated]
            )
        prediction = self.__critic(state_batch, action_batch)
        critic_loss = self.__critic_criterion(prediction, critic_target)
        network_update(critic_loss, self.__critic_optimizer)

        # Update actor
        new_actions = self.__actor(state_batch)
        actor_loss = -self.__critic(state_batch, new_actions).mean()
        network_update(actor_loss, self.__actor_optimizer)

        # Update target nets
        update_target_net(self.__critic, self.__critic_target_net, self.__tau)
        update_target_net(self.__actor, self.__actor_target_net, self.__tau)
