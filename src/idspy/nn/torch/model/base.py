from typing import Optional, Tuple

from torch import nn, Tensor
import torch


class ModelOutput(dict):
    """Model output container.

    Expected to contain at least 'logits' or 'latents' keys and other optional tensors.
    """

    @property
    def logits(self) -> Tensor:
        if "logits" not in self:
            raise KeyError("ModelOutput must contain 'logits'")
        return self["logits"]

    @property
    def latents(self) -> Optional[Tensor]:
        if "latents" not in self:
            raise KeyError("ModelOutput must contain 'latents'")
        return self.get("latents")

    def detach(self) -> "ModelOutput":
        detached = {}
        for k, v in self.items():
            detached[k] = v.detach() if isinstance(v, Tensor) else v
        return ModelOutput(detached)

    def to(self, device: torch.device, non_blocking: bool = True) -> "ModelOutput":
        moved = {}
        for k, v in self.items():
            moved[k] = (
                v.to(device, non_blocking=non_blocking) if isinstance(v, Tensor) else v
            )
        return ModelOutput(moved)


class BaseModel(nn.Module):
    """
    Base class for models. Defines the interface for forward and loss_inputs methods.
    """

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass. Must be implemented by subclasses.
        Args:
            x: Input tensor of shape [batch_size, num_features].
        Returns:
            ModelOutput: NamedTuple, expected to contain at least 'logits'.
        """
        raise NotImplementedError

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepares arguments for the loss function. Default: pred=output['logits'].
        Override if your model/loss requires different fields.
        """
        return output.logits, target
