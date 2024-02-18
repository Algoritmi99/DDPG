import torch
import torch.nn as nn
import math

"""
    This file contains all the static algorithms used in the entire project to avoid rewriting and 
    make for reusing to keep the code cleaner and easier to read.
"""


def hidden_layer_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / math.sqrt(fan_in)
    return torch.tensor((-lim, lim), requires_grad=True)


def network_update(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def update_target_net(source_net: nn.Module, target_net: nn.Module, tau) -> None:
    target_net_state_dict = target_net.state_dict()
    source = source_net.state_dict()
    for key in source:
        target_net_state_dict[key] = source[key] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)
