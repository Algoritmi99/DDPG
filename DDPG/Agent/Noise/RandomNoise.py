import torch

from DDPG.Agent.Noise.Noise import Noise


class RandomNoise(Noise):
    """
        Random noise is used to do standard random noise creation to add to a selected action.
    """

    def __init__(self, outSample: torch.Tensor, noiseVariance):
        super(RandomNoise, self).__init__(outSample)
        self.noiseVariance = noiseVariance

    def sample(self) -> torch.Tensor:
        return (torch.randn_like(self.outSample) * self.noiseVariance).to(self.device)
