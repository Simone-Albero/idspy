from typing import Any, Dict, Optional
import logging

from ...core.step import Step
from ...core.state import State
from ...nn.models.base import BaseModel

logger = logging.getLogger(__name__)


class EarlyStopping(Step):
    """Early stopping step to monitor a metric and stop training if no improvement is seen."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        in_scope: str = "val",
        out_scope: str = "",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, in_scope=in_scope, out_scope=out_scope)
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.best_model: Optional[BaseModel] = None
        self.num_bad_epochs = 0
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float("inf")
        else:  # mode == "max"
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = -float("inf")

    @Step.requires(history=list, model=BaseModel)
    @Step.provides(stop_pipeline=bool, model=BaseModel)
    def run(
        self, state: State, history: list, model: BaseModel
    ) -> Optional[Dict[str, Any]]:
        current_score = history[-1]

        if self.best_score is None:
            self.best_score = current_score
            self.num_bad_epochs = 0
            self.best_model = model
        elif self.monitor_op(current_score, self.best_score):
            logger.info(f"EARLY_STOPPING {self.best_score:.6f} -> {current_score:.6f}.")
            self.best_score = current_score
            self.num_bad_epochs = 0
            self.best_model = model
        else:
            logger.info(f"EARLY_STOPPING {self.num_bad_epochs+1}/{self.patience}")
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            logger.info("EARLY_STOPPING Triggered")
            return {"stop_pipeline": True, "model": self.best_model}
        else:
            return {"stop_pipeline": False, "model": model}
