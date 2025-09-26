from typing import Any, Sequence


from .base import Pipeline, PipelineEvent
from ..storage.base import Storage
from ..step.base import Step
from ..step.fittable import FittableStep


class FittablePipeline(Pipeline):
    """A Pipeline that can be fitted and refitted."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        refit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(steps, storage, **kwargs)
        self.refit = refit
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has executed its fit hook at least once."""
        return self._is_fitted

    @Pipeline.hook(PipelineEvent.PIPELINE_START, priority=1)
    def _fit_on_start(self) -> None:
        for step in self.steps:
            if isinstance(step, FittableStep):
                if self.refit or not step.is_fitted:
                    inputs = self._storage.get(step.required_ports)
                    step.fit(**inputs)
        self._is_fitted = True

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted}, refit={self.refit})"
