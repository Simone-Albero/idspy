from typing import Any, Callable, Dict, Optional

import torch

from ......core.step.base import Step
from ......nn.torch.engine.forward import make_predictions
from .... import StepFactory


@StepFactory.register()
@Step.needs("logits", "labels")
class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        pred_fn: Callable,
        logits_key: str = "logits",
        outputs_key: str = "outputs",
        labels_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "make_predictions")
        self.pred_fn = pred_fn
        self.key_map = {
            "logits": logits_key,
            "outputs": outputs_key,
        }

        if labels_key is not None:
            self.key_map["labels"] = labels_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        predictions = (
            make_predictions(self.pred_fn, logits, labels).detach().cpu().numpy()
        )
        return {"outputs": predictions}
