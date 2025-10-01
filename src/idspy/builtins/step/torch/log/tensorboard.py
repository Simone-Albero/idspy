from typing import Optional, Dict, Any

from torch.utils.tensorboard import SummaryWriter

from .....core.step.base import Step


@Step.needs("metrics")
class TensorBoardLogger(Step):
    """Log metrics to TensorBoard."""

    def __init__(
        self, log_dir: str, metrics_key: str, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name or "tensorboard_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

        self.key_map = {
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.step)
        self.writer.flush()
        self.step += 1
