from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory
from ......nn.torch.loss import LossFactory


@StepFactory.register()
@Step.needs("device")
class BuildLoss(Step):
    """build a loss function instance from config and loss name."""

    def __init__(
        self,
        loss_args: Dict[str, Any],
        loss_key: str = "loss_fn",
        reduction: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_loss")
        self.loss_args = loss_args
        self.reduction = reduction
        self.key_map = {
            "loss": loss_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, device: torch.device) -> Optional[Dict[str, Any]]:
        loss = LossFactory.create(self.loss_args).to(device)
        if self.reduction is not None:
            loss.reduction = self.reduction

        return {"loss": loss}
