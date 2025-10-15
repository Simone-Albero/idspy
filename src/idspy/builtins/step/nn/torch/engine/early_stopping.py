from typing import Any, Dict, Optional
import logging

from ......core.step.base import Step
from ......nn.torch.model.base import BaseModel
from ....factory import StepFactory

logger = logging.getLogger(__name__)


@StepFactory.register()
@Step.needs("metrics", "model")
class EarlyStopping(Step):
    """Early stopping step to monitor a metric and stop training if no improvement is seen."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        model_key: str = "model",
        metrics_key: str = "val.metrics",
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
            "metrics": metrics_key,
            "model": model_key,
            "stop_training": stop_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, metrics: Dict[str, Any], model: BaseModel
    ) -> Optional[Dict[str, Any]]:
        avg_loss = metrics.get("avg_loss")
        if avg_loss is None:
            raise ValueError("metrics must contain 'avg_loss' key")

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
            logger.info(f"EARLY_STOPPING {self.best_score:.6f} // {avg_loss:.6f}.")
            logger.info(f"EARLY_STOPPING {self.num_bad_epochs+1}/{self.patience}")
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            logger.info("EARLY_STOPPING Triggered")
            return {"stop_training": True, "model": self.best_model}
        else:
            return {"stop_training": False}
