import torch

from .base import BasePrediction


class ArgMax(BasePrediction):

    def predict(self, x: torch.Tensor, *args) -> torch.Tensor:
        return torch.argmax(x, dim=1)

    def confidence_scores(self, x: torch.Tensor, *args) -> torch.Tensor:
        probs = torch.softmax(x, dim=1)
        return probs.max(dim=1).values
