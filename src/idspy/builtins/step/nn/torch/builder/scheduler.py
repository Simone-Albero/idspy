from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory


@StepFactory.register()
@Step.needs("optimizer")
class BuildScheduler(Step):
    """build a learning rate scheduler instance from config and optimizer."""

    def __init__(
        self,
        scheduler_config: Dict[str, Any],
        optimizer_key: str = "optimizer",
        scheduler_key: str = "scheduler",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_scheduler")
        self.scheduler_config = scheduler_config
        self.key_map = {
            "optimizer": optimizer_key,
            "scheduler": scheduler_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
        SchedulerClass = getattr(
            torch.optim.lr_scheduler, self.scheduler_config._target_
        )
        scheduler = SchedulerClass(optimizer, **self.scheduler_config._params_)
        return {"scheduler": scheduler}
