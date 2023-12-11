from collections import deque
import random

import gymnasium as gym
import torch
import torch.nn as nn

from actor import ActorNet
from critic import CriticNet
from params import *
import torch.nn as nn
import numpy as np


class ReplayBuffer:
    replay_buffer = None

    def __init__(self):
        self.replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= MAX_BUFFER_SIZE:
            self.replay_buffer.popleft()

        self.replay_buffer.append(
            transition(state, action, reward, next_state, done)
        )

    def sample_batch(self, batch_size=BATCH_SIZE):
        return transition(
            *[torch.cat(i) for i in
              [*zip(*random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size)))]]
        )


def network_update(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def target_network_update(net_params, target_net_params):
    for net_param, target_net_param in zip(net_params, target_net_params):
        target_net_param.data = POLYAK * net_param.data + (1 - POLYAK) * target_net_param.data


class DDPG:
    def __init__(self, env: gym.Env,
                 actor: ActorNet,
                 critic: CriticNet,
                 target_actor: ActorNet,
                 target_critic: CriticNet
                 ):
        self.__env = env
        self.__actor = actor
        self.__critic = critic
        self.__target_actor = target_actor
        self.__target_critic = target_critic

        self.__replay_buffer = ReplayBuffer()
        self.__train_rewards_list = None
        self.__actor_error_history = []
        self.__critic_error_history = []

    def train(self, actor_optimizer, critic_optimizer, max_episodes=MAX_EPISODES):
        self.__train_rewards_list = []

        print("Starting Training:\n running on " + device)
        for episode in range(max_episodes):
            state = self.__env.reset()
            state = torch.tensor(state[0], device=device, dtype=dtype).unsqueeze(0)
            episode_reward = 0

            for time in range(MAX_TIME_STEPS):
                action = self.__actor.select_action(state, self.__env)
                action = action.clone().detach().cpu()
                next_state, reward, terminated, truncated, _ = self.__env.step(action[0])
                action = action.clone().detach().to(device)
                done = terminated or truncated
                episode_reward += reward

                next_state = torch.tensor(next_state, device=device, dtype=dtype).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=dtype).unsqueeze(0)
                self.__replay_buffer.store_transition(state, action, reward, next_state, done)

                state = next_state
                sample_batch = self.__replay_buffer.sample_batch(BATCH_SIZE)

                with torch.no_grad():
                    target = sample_batch.reward + \
                             (1 - sample_batch.done) * GAMMA * \
                             self.__target_critic.forward(sample_batch.next_state,
                                                          self.__target_actor(sample_batch.next_state))

                critic_loss = nn.MSELoss()(
                    target, self.__critic.forward(sample_batch.state, sample_batch.action))
                self.__critic_error_history.append(critic_loss.item())

                network_update(critic_loss, critic_optimizer)

                actor_loss = -1 * torch.mean(
                    self.__critic.forward(sample_batch.state, self.__actor(sample_batch.state))
                )
                self.__actor_error_history.append(actor_loss.item())
                network_update(actor_loss, actor_optimizer)

                target_network_update(self.__actor.parameters(), self.__target_actor.parameters())
                target_network_update(self.__critic.parameters(), self.__target_critic.parameters())

                if done:
                    print("Completed episode {}/{}".format(
                            episode + 1, MAX_EPISODES))
                    break

            self.__train_rewards_list.append(episode_reward)

        self.__env.close()

    def get_reward_list(self):
        reward_list = self.__train_rewards_list
        return reward_list

    def get_losses(self):
        return self.__actor_error_history, self.__critic_error_history

