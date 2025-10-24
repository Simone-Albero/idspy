from typing import Any, Dict, Optional
import logging

import torch

from ......core.step.base import Step
from .... import StepFactory

logger = logging.getLogger(__name__)


@StepFactory.register()
@Step.needs("metrics", "scheduler")
class EpochWiseScheduler(Step):

    def __init__(
        self,
        scheduler_key: str = "scheduler",
        metrics_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "early_stopping")

        self.key_map = {
            "scheduler": scheduler_key,
        }

        if metrics_key is not None:
            self.key_map["metrics"] = metrics_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:

        if metrics is not None:
            avg_loss = metrics.get("avg_loss")
            scheduler.step(avg_loss)
        else:
            scheduler.step()
