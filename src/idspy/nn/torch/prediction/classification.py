import torch

from . import PredFactory


@PredFactory.register()
class ArgMax:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)
