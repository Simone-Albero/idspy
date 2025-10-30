from abc import ABC, abstractmethod

import torch


class BasePrediction(ABC):
    """Base class for computing predictions."""

    @abstractmethod
    def predict(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Compute predictions for the given inputs."""
        raise NotImplementedError

    @abstractmethod
    def confidence_scores(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Compute confidence scores for the given inputs."""
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self.predict(x, *args)
