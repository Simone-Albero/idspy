from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory


@StepFactory.register()
@Step.needs("optimizer", "dataloader")
class BuildScheduler(Step):
    """build a learning rate scheduler instance from config and optimizer."""

    def __init__(
        self,
        scheduler_name: str,
        scheduler_args: Dict[str, Any],
        optimizer_key: str = "optimizer",
        scheduler_key: str = "scheduler",
        dataloader_key: str = "dataloader",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_scheduler")
        self.scheduler_name = scheduler_name
        self.scheduler_args = scheduler_args
        self.key_map = {
            "optimizer": optimizer_key,
            "scheduler": scheduler_key,
            "dataloader": dataloader_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader
    ) -> Optional[Dict[str, Any]]:
        SchedulerClass = getattr(torch.optim.lr_scheduler, self.scheduler_name)

        if self.scheduler_args.steps_per_epoch == "auto":
            steps_per_epoch = len(dataloader)
            self.scheduler_args.steps_per_epoch = steps_per_epoch

        scheduler = SchedulerClass(optimizer, **self.scheduler_args)
        return {"scheduler": scheduler}
