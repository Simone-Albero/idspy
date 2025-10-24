from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory
from ......nn.torch.prediction import PredFactory


@StepFactory.register()
class BuildPredFn(Step):
    """build a prediction function instance from config and function name."""

    def __init__(
        self,
        pred_config: Dict[str, Any],
        pred_key: str = "pred_fn",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_pred_fn")
        self.pred_config = pred_config
        self.key_map = {
            "pred": pred_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self) -> Optional[Dict[str, Any]]:
        pred_fn = PredFactory.create(self.pred_config)
        return {"pred_fn": pred_fn}
