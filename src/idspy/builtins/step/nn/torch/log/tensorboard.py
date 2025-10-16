from typing import Optional, Dict, Any, Union

from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch

from ......core.step.base import Step
from .... import StepFactory


def _auto_detect_prefix(metrics_key: str) -> str:
    """Auto-detect prefix from metrics_key."""
    if "train" in metrics_key.lower():
        return "train"
    elif "val" in metrics_key.lower() or "validation" in metrics_key.lower():
        return "val"
    elif "test" in metrics_key.lower():
        return "test"
    else:
        return ""


def _make_tag(separator: str = "/", *args: str) -> str:
    """Generate writer tag from prefix and key."""
    return separator.join(filter(None, args))


def _to_cpu_flat(x: torch.Tensor) -> torch.Tensor:
    """Flatten to 1D CPU tensor."""
    return x.detach().reshape(-1).to("cpu")


@StepFactory.register()
@Step.needs("metrics")
class MetricsLogger(Step):
    """Log metrics to TensorBoard."""

    SEPARATOR = "/"

    def __init__(
        self,
        log_dir: str,
        metrics_key: str,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "tensorboard_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.prefix = _auto_detect_prefix(metrics_key) if prefix is None else prefix
        self.key_map = {"metrics": metrics_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def _log_scalar(self, name: str, value: Union[int, float], step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def _log_figure(self, name: str, figure: plt.Figure, step: int) -> None:
        self.writer.add_figure(name, figure, step)

    def _log_metric(self, key: str, value: Any, step: int) -> None:
        metric_name = _make_tag(self.SEPARATOR, self.prefix, key)

        if isinstance(value, (int, float)):
            self._log_scalar(metric_name, value, step)

        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._log_metric(key, v, step * len(value) + i)

        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._log_metric(
                    _make_tag(key, sub_key, self.SEPARATOR), sub_value, step
                )

        elif isinstance(value, plt.Figure):
            self._log_figure(metric_name, value, step)
            plt.close(value)

        else:
            try:
                self.writer.add_text(metric_name, str(value), step)
            except Exception:
                raise ValueError(
                    f"Unsupported metric type for {metric_name}: {type(value)}"
                )

    def compute(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        for key, value in metrics.items():
            self._log_metric(key, value, self.step)
        self.writer.flush()
        self.step += 1

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def __del__(self) -> None:
        self.close()


@StepFactory.register()
@Step.needs("model")
class WeightsLogger(Step):
    """Log model weights/gradients to TensorBoard."""

    def __init__(
        self,
        log_dir: str,
        model_key: str,
        prefix: Optional[str] = None,
        dead_params_ratio_threshold: float = 1e-6,
        dead_grad_ratio_threshold: float = 1e-8,
        exploding_grad_ratio_threshold: float = 10.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "tensorboard_weights_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.dead_params_ratio_threshold = dead_params_ratio_threshold
        self.dead_grad_ratio_threshold = dead_grad_ratio_threshold
        self.exploding_grad_ratio_threshold = exploding_grad_ratio_threshold
        self.prefix = "model" if prefix is None else prefix
        self.key_map = {"model": model_key}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    @staticmethod
    def _get_layer_name(param_name: str) -> str:
        for suffix in (
            ".weight",
            ".bias",
            ".running_mean",
            ".running_var",
            ".num_batches_tracked",
        ):
            if param_name.endswith(suffix):
                return param_name[: -len(suffix)]
        return param_name

    def _log_layer_stats(self, model: torch.nn.Module, step: int) -> None:
        layer_stats: Dict[str, Dict[str, Any]] = {}

        # Collect per-layer weights/grads
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            layer_name = self._get_layer_name(name)
            entry = layer_stats.setdefault(layer_name, {"weights": [], "grads": []})

            entry["weights"].append(_to_cpu_flat(param.data))
            if param.grad is not None:
                entry["grads"].append(_to_cpu_flat(param.grad))

        # Log summaries
        for layer_name, stats in layer_stats.items():
            if not stats["weights"]:
                continue

            clean = layer_name.replace(".", "_")
            all_w = torch.cat(stats["weights"], dim=0).float()

            weight_stats = {
                "mean": all_w.mean().item(),
                "std": all_w.std().item(),
                "dead_ratio": (all_w.abs() < self.dead_params_ratio_threshold)
                .float()
                .mean()
                .item(),
            }

            for stat_name, stat_value in weight_stats.items():
                metric_name = _make_tag(
                    self.separator, self.prefix, "weights", clean, stat_name
                )
                self.writer.add_scalar(
                    metric_name,
                    stat_value,
                    step,
                )

            self.writer.add_histogram(
                _make_tag(
                    self.separator, self.prefix, "weights", clean, "distribution"
                ),
                all_w,
                step,
            )

            if stats["grads"]:
                all_g = torch.cat(stats["grads"], dim=0).float()
                grad_stats = {
                    "mean": all_g.mean().item(),
                    "std": all_g.std().item(),
                    "max": all_g.abs().max().item(),
                    "dead_ratio": (all_g.abs() < self.dead_grad_ratio_threshold)
                    .float()
                    .mean()
                    .item(),
                    "exploding_ratio": (
                        all_g.abs() > self.exploding_grad_ratio_threshold
                    )
                    .float()
                    .mean()
                    .item(),
                }

                for stat_name, stat_value in grad_stats.items():
                    metric_name = _make_tag(
                        self.separator, self.prefix, "grads", clean, stat_name
                    )

                    self.writer.add_scalar(
                        metric_name,
                        stat_value,
                        step,
                    )

                self.writer.add_histogram(
                    _make_tag(
                        self.separator, self.prefix, "grads", clean, "distribution"
                    ),
                    all_g,
                    step,
                )

    def compute(self, model: torch.nn.Module) -> None:
        self._log_layer_stats(model, self.step)
        self.writer.flush()
        self.step += 1

    def close(self) -> None:
        try:
            self.writer.flush()
        finally:
            try:
                self.writer.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
