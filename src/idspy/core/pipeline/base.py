from enum import Enum, auto
from collections import defaultdict
import logging
from typing import Any, Sequence, Dict, List, Optional

from ..storage.base import Storage
from ..storage.proxy import BindedStorage
from ..step.base import Step

logger = logging.getLogger(__name__)


class PipelineEvent(Enum):
    PIPELINE_START = auto()
    PIPELINE_END = auto()
    STEP_START = auto()
    STEP_END = auto()
    STEP_ERROR = auto()


class Pipeline(Step):
    """
    A pipeline is itself a Step. It runs child steps in order, using Storage.
    Optionally binds step keys to storage keys.

    Emits hooks at key points:

      PIPELINE_START: {}
      PIPELINE_END:   {result}
      STEP_START:     {step, index, inputs}
      STEP_END:       {step, index, outputs}
      STEP_ERROR:     {step, index, error}
    """

    def __init__(
        self, steps: Sequence[Step], storage: Storage, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        self._steps: List[Step] = list(steps)
        self._storage = storage
        self._hooks: Dict[PipelineEvent, List[Any]] = defaultdict(list)
        self._collect_decorated_hooks()

    @classmethod
    def hook(cls, event: PipelineEvent, priority: int = 0):
        """Decorator: mark a method as a hook for `event` on subclasses."""

        def decorator(func):
            func._hook_event = event
            func._hook_priority = priority
            return func

        return decorator

    def on(self, event: PipelineEvent, callback, priority: int = 0) -> None:
        """Register a callback at runtime."""
        self._hooks[event].append((callback, priority))

    def _collect_decorated_hooks(self) -> None:
        """
        Find methods on this instance marked by @Pipeline.hook and register them.
        Works per-subclass without polluting base class/global state.
        """
        for name in dir(self):
            try:
                method = getattr(self, name)
                if callable(method) and hasattr(method, "_hook_event"):
                    event = method._hook_event
                    priority = getattr(method, "_hook_priority", 0)
                    self._hooks[event].append((method, priority))
            except (AttributeError, TypeError):
                # Skip attributes that can't be accessed or aren't callable
                continue

        # Sort all hooks by priority
        for event in self._hooks:
            self._hooks[event].sort(key=lambda x: x[1])

    def _emit(self, event: PipelineEvent, **ctx: Any) -> None:
        """Invoke callbacks; hook failures are isolated and logged to stderr."""
        for cb, _ in self._hooks.get(event, []):
            try:
                cb(**ctx)
            except Exception as e:
                logger.error(f"Hook {getattr(cb, '__name__', str(cb))} failed: {e}")
                raise e

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute all steps sequentially using storage for I/O.
        Returns the final snapshot of *all* ports currently persisted for the pipeline's provided ports.
        """
        if kwargs:
            self.storage.set(kwargs)

        self._emit(PipelineEvent.PIPELINE_START)
        try:
            for idx, step in enumerate(self._steps):
                binded_storage = BindedStorage(self._storage, step.bindings() or {})
                inputs = binded_storage.get(step.requires)
                self._emit(
                    PipelineEvent.STEP_START, step=step, index=idx, inputs=inputs
                )
                try:
                    outputs = step.run(**inputs) or {}
                except Exception as e:
                    # hook first, then re-raise
                    self._emit(PipelineEvent.STEP_ERROR, step=step, index=idx, error=e)
                    raise

                binded_storage.set(outputs)

                self._emit(
                    PipelineEvent.STEP_END,
                    step=step,
                    index=idx,
                    outputs=outputs,
                )

            result = self._storage.as_dict()
            self._emit(PipelineEvent.PIPELINE_END, result=result)
            return result

        finally:
            pass

    def bindings(self) -> Dict[str, str]:
        return {}

    def __call__(self, *args, **kwds) -> Dict[str, Any]:
        return self.run(*args, **kwds)

    def __repr__(self) -> str:
        step_reprs = ", ".join(repr(s) for s in self._steps)
        return f"{self.__class__.__name__}(steps=[{step_reprs}])"
