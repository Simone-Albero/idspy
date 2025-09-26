from typing import Any, Sequence, Dict, Mapping
from types import MappingProxyType

from .base import Pipeline, PipelineEvent
from ..step.base import Step
from ..events.bus import EventBus


class ObservablePipeline(Pipeline):
    """Pipeline that publishes lifecycle events to an EventBus."""

    def __init__(self, steps: Sequence[Step], bus: EventBus, **kwargs: Any) -> None:
        super().__init__(steps, **kwargs)
        self.bus = bus

    @Pipeline.hook(PipelineEvent.PIPELINE_START)
    def start(self) -> None:
        self.bus.publish(
            event_type=PipelineEvent.PIPELINE_START.value,
            source=self.name,
            payload={},
        )

    @Pipeline.hook(PipelineEvent.PIPELINE_END)
    def end(self, result: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=PipelineEvent.PIPELINE_END.value,
            source=self.name,
            payload=result,
        )

    @Pipeline.hook(PipelineEvent.BEFORE_STEP)
    def before_step(self, step: Step, index: int, inputs: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=PipelineEvent.BEFORE_STEP.value,
            source=self.name + "." + step.name,
            payload=inputs,
        )

    @Pipeline.hook(PipelineEvent.AFTER_STEP)
    def after_step(self, step: Step, index: int, outputs: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=PipelineEvent.AFTER_STEP.value,
            source=self.name + "." + step.name,
            payload=outputs,
        )

    @Pipeline.hook(PipelineEvent.STEP_ERROR)
    def step_error(self, step: Step, index: int, error: Exception) -> None:
        self.bus.publish(
            event_type=PipelineEvent.STEP_ERROR.value,
            source=self.name + "." + step.name,
            payload={"error": str(error)},
        )
