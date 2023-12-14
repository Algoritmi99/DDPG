import random
from collections import deque

from params import *


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
