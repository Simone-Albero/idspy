from typing import Optional, Callable, Any, Dict

from torch.utils.data import Dataset, DataLoader

from ......core.step.base import Step
from ....factory import StepFactory
from ......data.torch.batch import default_collate


@StepFactory.register()
@Step.needs("dataset")
class BuildDataLoader(Step):
    """Build dataloader from dataset in state."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = default_collate,
        dataset_key: str = "dataset",
        dataloader_key: str = "dataloader",
        name: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        super().__init__(name=name or "build_dataloader")
        self.key_map = {
            "dataset": dataset_key,
            "dataloader": dataloader_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, dataset: Dataset) -> Optional[Dict[str, Any]]:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )
        return {"dataloader": dataloader}
