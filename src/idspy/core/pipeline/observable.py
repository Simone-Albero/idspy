from typing import Any, Optional, Sequence, Dict

from .base import Pipeline, PipelineEvent
from .fittable import FittablePipeline
from .repeatable import RepeatablePipeline, StoragePredicate
from ..step.base import Step
from ..events.bus import EventBus
from ..storage.base import Storage


class ObservablePipeline(Pipeline):
    """Pipeline that publishes lifecycle events to an EventBus."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        bus: EventBus,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(steps, storage, name=name or "observable_pipeline")
        self.bus = bus

    @Pipeline.hook(PipelineEvent.PIPELINE_START)
    def start(self) -> None:
        self.bus.publish(
            event_type=str(PipelineEvent.PIPELINE_START),
            source=self.name,
            payload={},
        )

    @Pipeline.hook(PipelineEvent.PIPELINE_END)
    def end(self, result: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=str(PipelineEvent.PIPELINE_END),
            source=self.name,
            payload=result,
        )

    @Pipeline.hook(PipelineEvent.STEP_START)
    def before_step(self, step: Step, index: int, inputs: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=str(PipelineEvent.STEP_START),
            source=self.name + "." + step.name,
            payload=inputs,
        )

    @Pipeline.hook(PipelineEvent.STEP_END)
    def after_step(self, step: Step, index: int, outputs: Dict[str, Any]) -> None:
        self.bus.publish(
            event_type=str(PipelineEvent.STEP_END),
            source=self.name + "." + step.name,
            payload=outputs,
        )

    @Pipeline.hook(PipelineEvent.STEP_ERROR)
    def step_error(self, step: Step, index: int, error: Exception) -> None:
        self.bus.publish(
            event_type=str(PipelineEvent.STEP_ERROR),
            source=self.name + "." + step.name,
            payload={"error": str(error)},
        )


class ObservableFittablePipeline(ObservablePipeline, FittablePipeline):
    """A FittablePipeline that also publishes lifecycle events to an EventBus."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        bus: EventBus,
        refit: bool = False,
        name: Optional[str] = None,
    ) -> None:
        ObservablePipeline.__init__(self, steps, storage, bus=bus, name=name)
        FittablePipeline.__init__(self, steps, storage, refit=refit, name=name)


class ObservableRepeatablePipeline(ObservablePipeline, RepeatablePipeline):
    """A RepeatablePipeline that also publishes lifecycle events to an EventBus."""

    def __init__(
        self,
        steps: Sequence[Step],
        storage: Storage,
        bus: EventBus,
        count: int = 1,
        clear_storage: bool = True,
        predicate: Optional[StoragePredicate] = None,
        name: Optional[str] = None,
    ) -> None:
        ObservablePipeline.__init__(self, steps, storage, bus=bus, name=name)
        RepeatablePipeline.__init__(
            self,
            steps,
            storage,
            count=count,
            clear_storage=clear_storage,
            predicate=predicate,
            name=name,
        )
