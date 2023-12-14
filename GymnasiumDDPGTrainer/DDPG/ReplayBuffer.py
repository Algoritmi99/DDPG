import random
import torch
from collections import deque
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, max_buffer_size, batch_size):
        self.max_bufferSize = max_buffer_size
        self.batch_size = batch_size
        self.__replay_buffer = deque(maxlen=self.max_bufferSize)

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.__replay_buffer) >= self.max_bufferSize:
            self.__replay_buffer.popleft()

        self.__replay_buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample_batch(self):
        return Transition(
            *[torch.cat(i) for i in
              [*zip(*random.sample(self.__replay_buffer, min(len(self.__replay_buffer), self.batch_size)))]]
        )
