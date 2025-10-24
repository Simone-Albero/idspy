from typing import Optional, Callable, Any, Dict

from torch.utils.data import Dataset, DataLoader

from ......core.step.base import Step
from .... import StepFactory
from ......data.torch.batch import default_collate


@StepFactory.register()
@Step.needs("dataset")
class BuildDataLoader(Step):
    """Build dataloader from dataset in state."""

    def __init__(
        self,
        dataloader_args: Dict[str, Any],
        collate_fn: Optional[Callable] = default_collate,
        dataset_key: str = "dataset",
        dataloader_key: str = "dataloader",
        name: Optional[str] = None,
    ) -> None:
        self.collate_fn = collate_fn

        super().__init__(name=name or "build_dataloader")
        self.dataloader_args = dataloader_args
        self.key_map = {
            "dataset": dataset_key,
            "dataloader": dataloader_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, dataset: Dataset) -> Optional[Dict[str, Any]]:
        dataloader = DataLoader(
            dataset, collate_fn=self.collate_fn, **self.dataloader_args
        )
        return {"dataloader": dataloader}
