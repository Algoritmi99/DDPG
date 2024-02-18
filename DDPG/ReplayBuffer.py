from collections import deque, namedtuple
import random

import torch

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'terminated')
)


class ReplayBuffer(object):
    """
        This class implements a simple replay buffer that stores transitions after taken actions
    """

    def __init__(self, capacity, batch_size):
        self.__replay_buffer = deque([], maxlen=capacity)
        self.__batch_size = batch_size
        self.__capacity = capacity

    def push(self, *args):
        if len(self.__replay_buffer) >= self.__capacity:
            self.__replay_buffer.popleft()

        self.__replay_buffer.append(Transition(*args))

    def sample_batch(self):
        if not self.canSample():
            return None

        transitions = random.sample(self.__replay_buffer, self.__batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminated_batch = torch.cat(batch.terminated)

        return state_batch, action_batch, reward_batch, next_state_batch, terminated_batch

    def canSample(self):
        return len(self.__replay_buffer) >= self.__batch_size

    def __len__(self):
        return len(self.__replay_buffer)
