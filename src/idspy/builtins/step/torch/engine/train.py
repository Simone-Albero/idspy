from typing import Any, Dict, Optional

import torch

from .....core.step.base import Step
from .....nn.torch.model.base import BaseModel
from .....nn.torch.loss.base import BaseLoss
from .....nn.torch.engine.loops import train_epoch


@Step.needs("dataloader", "model", "loss_fn", "optimizer", "device")
class TrainOneEpoch(Step):
    """Train model for one epoch."""

    def __init__(
        self,
        clip_grad_max_norm: Optional[float] = 1.0,
        dataloader_key: str = "train.dataloader",
        model_key: str = "model",
        loss_key: str = "loss_fn",
        optimizer_key: str = "optimizer",
        device_key: str = "device",
        metrics_key: str = "train.metrics",
        name: Optional[str] = None,
    ) -> None:
        self.clip_grad_max_norm = clip_grad_max_norm
        super().__init__(name=name or "train_one_epoch")

        self.key_map = {
            "dataloader": dataloader_key,
            "model": model_key,
            "loss_fn": loss_key,
            "optimizer": optimizer_key,
            "device": device_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: BaseModel,
        loss_fn: BaseLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        context: Optional[any] = None,
    ) -> Optional[Dict[str, Any]]:

        average_loss, grad_norm, lr = train_epoch(
            dataloader=dataloader,
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            clip_grad_max_norm=self.clip_grad_max_norm,
            profiler=context,
        )

        return {
            "model": model,
            "metrics": {
                "avg_loss": average_loss,
                "grad_norm": grad_norm,
                "lr": lr,
            },
        }
