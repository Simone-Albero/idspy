from typing import Optional, Dict, Any

from torch.utils.tensorboard import SummaryWriter

from ......core.step.base import Step


@Step.needs("metrics")
class TensorBoardLogger(Step):
    """Log metrics to TensorBoard."""

    def __init__(
        self,
        log_dir: str,
        metrics_key: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "tensorboard_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

        # Auto-detect prefix from metrics_key if not provided
        if prefix is None:
            if "train" in metrics_key.lower():
                self.prefix = "train/"
            elif "val" in metrics_key.lower() or "validation" in metrics_key.lower():
                self.prefix = "val/"
            elif "test" in metrics_key.lower():
                self.prefix = "test/"
            else:
                self.prefix = ""
        else:
            self.prefix = (
                f"{prefix}/" if prefix and not prefix.endswith("/") else prefix or ""
            )

        self.key_map = {
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        for key, value in metrics.items():
            # Add prefix to distinguish train/validation/test metrics
            metric_name = f"{self.prefix}{key}"
            self.writer.add_scalar(metric_name, value, self.step)
        self.writer.flush()
        self.step += 1
