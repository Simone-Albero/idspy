from typing import Dict, Optional

import pandas as pd
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from ....core.step.base import Step
from ....core.step.contextual import ContextualStep
from .. import StepFactory
from ....plot.dict import dict_to_table


@StepFactory.register()
class TorchProfiler(ContextualStep):
    """Wrap a step inside a torch.profiler profile that writes TensorBoard traces."""

    def __init__(
        self,
        step: Step,
        log_dir: str,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(step=step, name=name)

        # store config
        self.log_dir = log_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

    def context(self) -> Optional[any]:
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        elif torch.backends.mps.is_available():
            acts.append(ProfilerActivity.MPS)

        # Standard TB schedule: wait -> warmup -> active (repeat)
        sched = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )

        profiler = profile(
            activities=acts,
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )

        return profiler


@StepFactory.register()
@Step.needs("df")
class DataFrameProfiler(Step):
    """Compute simple ml driven statistics from a DataFrame."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        output_key: str = "data.profile_report",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "data_frame_profiler")
        self.key_map = {
            "df": df_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, any]]:
        n_rows, _ = df.shape
        report = {
            "num_rows": n_rows,
            "num_numerical_columns": df.tab.numerical.shape[1],
            "num_categorical_columns": df.tab.categorical.shape[1],
        }

        labels_count = df.tab.label.value_counts().to_dict()
        report.update({f"label_count_{k}": v for k, v in labels_count.items()})
        return {"output": {"report": dict_to_table(report, title="Dataset Statistics")}}
