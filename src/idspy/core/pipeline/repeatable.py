import logging
from typing import Any, Sequence, Dict, Optional

from .base import Pipeline
from ..storage.base import Storage, StoragePredicate
from ..step.base import Step

logger = logging.getLogger(__name__)


class RepeatablePipeline(Pipeline):
    """A pipeline that can be executed multiple times."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        count: int = 1,
        clear_storage: bool = True,
        predicate: Optional[StoragePredicate] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(steps, storage, name=name)
        self.count = count
        self.clear_storage = clear_storage
        self.predicate = predicate

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the pipeline multiple times."""
        for i in range(self.count):
            if self.predicate is not None and self.predicate(self._storage):
                break

            logger.info(f"Starting run {i + 1}/{self.count} of '{self.name}':")

            # Clear storage to ensure isolated state for each run
            if self.clear_storage:
                self._storage.clear()

            super().run(**kwargs)
