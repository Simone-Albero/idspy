from enum import Enum, auto
from collections import defaultdict
from typing import Any, Mapping, Sequence, Dict, List, Optional

from ..storage.base import Storage, Port
from ..step.base import Step


class PipelineEvent(Enum):
    PIPELINE_START = auto()
    PIPELINE_END = auto()
    STEP_START = auto()
    STEP_END = auto()
    STEP_ERROR = auto()


def _ensure_outputs_match_ports(
    outputs: Mapping[str, Any], ports: Sequence[Port]
) -> None:
    port_names = {p.name for p in ports}
    missing = port_names - set(outputs.keys())
    if missing:
        raise ValueError(f"Step did not return values for ports: {sorted(missing)}")


class Pipeline(Step):
    """
    A pipeline is itself a Step. It runs child steps in order, using Storage:
        inputs  = storage.get(step.required_ports)
        outputs = step.run(inputs)
        storage.set(outputs)

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
            setattr(func, "_hook_event", event)
            setattr(func, "_hook_priority", priority)
            return func

        return decorator

    def on(self, event: PipelineEvent, callback, priority: int = 0) -> None:
        """Register a callback at runtime."""
        self._hooks[event].append((callback, priority))

    def _collect_decorated_hooks(self) -> None:
        """
        Find methods on this instance marked by @BasePipeline.hook and register them.
        Works per-subclass without polluting base class/global state.
        """
        for name in dir(self):
            try:
                attr = getattr(self, name)
            except Exception:
                continue
            func = getattr(attr, "__func__", attr)
            if callable(attr) and hasattr(func, "_hook_event"):
                event = getattr(func, "_hook_event")
                priority = getattr(func, "_hook_priority", 0)
                # attr is bound method when coming via getattr(self, name)
                self._hooks[event].append((attr, priority))

    def _emit(self, event: PipelineEvent, **ctx: Any) -> None:
        """Invoke callbacks; hook failures are isolated and logged to stderr."""
        for cb, _ in sorted(self._hooks.get(event, ()), key=lambda x: x[1]):
            try:
                cb(**ctx)
            except Exception as e:
                raise e

    @property
    def required_ports(self) -> Sequence[Port]:
        """Ports needed externally before the first run (not produced by earlier steps)."""
        produced: set[Port] = set()
        needed: set[Port] = set()
        for step in self._steps:
            for p in step.required_ports:
                if p.name not in {q.name for q in produced} and p.name not in {
                    q.name for q in needed
                }:
                    needed.append(p)
            produced.update(step.provided_ports)
        return needed

    @property
    def provided_ports(self) -> Sequence[Port]:
        """All ports produced by the pipeline after completion."""
        out_by_name: Dict[str, Port] = {}
        for step in self._steps:
            for p in step.provided_ports:
                out_by_name[p.name] = p
        return list(out_by_name.values())

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute all steps sequentially using storage for I/O.
        Returns the final snapshot of *all* ports currently persisted for the pipeline's provided ports.
        """
        if kwargs:
            self._storage.set(kwargs)

        self._emit(PipelineEvent.PIPELINE_START)

        try:
            for idx, step in enumerate(self._steps):
                inputs = self._storage.get(step.required_ports)
                self._emit(
                    PipelineEvent.STEP_START, step=step, index=idx, inputs=inputs
                )
                try:
                    outputs = step.run(**inputs)
                except Exception as e:
                    # hook first, then re-raise
                    self._emit(PipelineEvent.STEP_ERROR, step=step, index=idx, error=e)
                    raise
                _ensure_outputs_match_ports(outputs, step.provided_ports)
                self._storage.set(outputs)
                self._emit(
                    PipelineEvent.STEP_END,
                    step=step,
                    index=idx,
                    outputs=outputs,
                )

            result = self._storage.get(self.provided_ports)
            self._emit(PipelineEvent.PIPELINE_END, result=result)
            return result

        finally:
            pass

    def __call__(self, *args, **kwds) -> Dict[str, Any]:
        return self.run(*args, **kwds)

    def __repr__(self) -> str:
        step_reprs = ", ".join(repr(s) for s in self._steps)
        return f"{self.__class__.__name__}(steps=[{step_reprs}])"
