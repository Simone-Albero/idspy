from typing import Optional, Sequence


from .base import Pipeline, PipelineEvent
from ..step.base import Step
from ..step.fittable import FittableStep
from ..storage.base import Storage
from ..storage.proxy import BindedStorage


class FittablePipeline(Pipeline):
    """A Pipeline that can be fitted and refitted."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        refit: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(steps, storage, name=name or "fittable_pipeline")
        self.refit = refit
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has executed its fit hook at least once."""
        return self._is_fitted

    @Pipeline.hook(PipelineEvent.PIPELINE_START)
    def _fit_on_start(self) -> None:
        for step in self._steps:
            if isinstance(step, FittableStep):
                if self.refit or not step.is_fitted:
                    storage = BindedStorage(self._storage, step.bindings() or {})
                    inputs = storage.get(step.requires)
                    step.fit(**inputs)
        self._is_fitted = True

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted}, refit={self.refit})"
