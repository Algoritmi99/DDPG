import torch


class Noise:
    """
        The base class for all noise classes to implement.
    """
    def __init__(self, outSample: torch.Tensor, device: str = "cpu"):
        self.outSample = outSample
        self.device = device

    def sample(self) -> torch.Tensor:
        pass

    def to(self, device: str):
        self.device = device
