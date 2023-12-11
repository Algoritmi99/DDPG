import torch
from collections import namedtuple

MAX_BUFFER_SIZE = 1000000
GAMMA = 0.99
LR = 0.001
POLYAK = 0.001
MAX_TIME_STEPS = 3000
MAX_EPISODES = 100
UPDATE_AFTER = 1
BATCH_SIZE = 128

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))