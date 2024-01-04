import math
import torch


def hidden_layer_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / math.sqrt(fan_in)
    return torch.tensor((-lim, lim), requires_grad=True)
