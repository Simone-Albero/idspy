from typing import Any, Dict, Optional
import logging

from .....core.step.base import Step
from .....nn.torch.model.base import BaseModel

logger = logging.getLogger(__name__)


@Step.needs(avg_loss=float, model=BaseModel)
class EarlyStopping(Step):
    """Early stopping step to monitor a metric and stop training if no improvement is seen."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        model_key: str = "model",
        avg_loss_key: str = "val.avg_loss",
        stop_key: str = "stop_pipeline",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "early_stopping")
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

        self.key_map = {
            "history": avg_loss_key,
            "model": model_key,
            "stop_training": stop_key,
        }

    def compute(self, avg_loss: float, model: BaseModel) -> Optional[Dict[str, Any]]:
        if self.best_score is None:
            self.best_score = avg_loss
            self.num_bad_epochs = 0
            self.best_model = model
        elif self.monitor_op(avg_loss, self.best_score):
            logger.info(f"EARLY_STOPPING {self.best_score:.6f} -> {avg_loss:.6f}.")
            self.best_score = avg_loss
            self.num_bad_epochs = 0
            self.best_model = model
        else:
            logger.info(f"EARLY_STOPPING {self.num_bad_epochs+1}/{self.patience}")
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            logger.info("EARLY_STOPPING Triggered")
            return {"stop_training": True, "model": self.best_model}
        else:
            return {"stop_training": False}
