import torch


class Noise:
    """
        The base class for all noise classes to implement.
    """
    def __init__(self, outSample: torch.Tensor):
        self.outSample = outSample

    def sample(self) -> torch.Tensor:
        pass
