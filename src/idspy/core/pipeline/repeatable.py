from typing import Any, Sequence, Dict, Optional

from .base import Pipeline
from ..storage.base import Storage, StoragePredicate
from ..step.base import Step


class RepeatablePipeline(Pipeline):
    """A pipeline that can be executed multiple times."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        name: Optional[str] = None,
        count: int = 1,
        clear_storage: bool = True,
        predicate: Optional[StoragePredicate] = None,
    ) -> None:
        super().__init__(steps, storage, name=name)
        self.count = count
        self.clear_storage = clear_storage
        self.predicate = predicate

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the pipeline multiple times."""
        # Clear storage to ensure isolated state for each run
        if self.clear_storage:
            self._storage.clear()

        for _ in range(self.count):
            if self.predicate is not None and self.predicate(self._storage):
                break
            super().run(**kwargs)
