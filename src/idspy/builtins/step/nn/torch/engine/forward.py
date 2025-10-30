from typing import Any, Callable, Dict, Optional

import torch

from ......core.step.base import Step
from .... import StepFactory


@StepFactory.register()
@Step.needs("logits", "labels")
class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        pred_fn: Callable,
        logits_key: str = "logits",
        prediction_key: str = "predictions",
        confidences_key: str = "confidences",
        labels_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "make_predictions")
        self.pred_fn = pred_fn
        self.key_map = {
            "logits": logits_key,
            "predictions": prediction_key,
            "confidences": confidences_key,
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
        predictions = self.pred_fn(logits, labels).detach().cpu().numpy()
        confidence_scores = (
            self.pred_fn.confidence_scores(logits, labels).detach().cpu().numpy()
        )
        return {"predictions": predictions, "confidences": confidence_scores}
