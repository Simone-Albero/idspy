from typing import Optional, Dict, Any, Union

from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch

from ......core.step.base import Step
from ....factory import StepFactory


def _auto_detect_prefix(metrics_key: str) -> str:
    """Auto-detect prefix from metrics_key."""
    if "train" in metrics_key.lower():
        return "train/"
    elif "val" in metrics_key.lower() or "validation" in metrics_key.lower():
        return "val/"
    elif "test" in metrics_key.lower():
        return "test/"
    else:
        return ""


@StepFactory.register()
@Step.needs("metrics")
class MetricsLogger(Step):
    """Log metrics to TensorBoard."""

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

        self.key_map = {
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def _log_scalar(self, name: str, value: Union[int, float], step: int) -> None:
        """Log scalar value."""
        self.writer.add_scalar(name, value, step)

    def _log_figure(self, name: str, figure: plt.Figure, step: int) -> None:
        """Log matplotlib figure."""
        self.writer.add_figure(name, figure, step)

    def _log_metric(self, key: str, value: Any, step: int) -> None:
        """Log a single metric based on its type."""
        # Struttura gerarchica: phase/category/metric_name
        metric_name = f"{self.prefix}/{key}"

        if isinstance(value, (int, float)):
            self._log_scalar(metric_name, value, step)

        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._log_scalar(f"{metric_name}/item_{i:02d}", v, step)

        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._log_metric(f"{key}/{sub_key}", sub_value, step)

        elif isinstance(value, plt.Figure):
            # Le figure vanno in una sezione separata
            figure_name = f"figures/{self.prefix}/{key}"
            self._log_figure(figure_name, value, step)
            plt.close(value)

        else:
            try:
                # I testi vanno in una sezione separata
                text_name = f"text/{self.prefix}/{key}"
                self.writer.add_text(text_name, str(value), step)
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
    """Log model weights and gradients to TensorBoard with focus on detecting dead/exploding gradients and parameters."""

    def __init__(
        self,
        log_dir: str,
        model_key: str,
        prefix: Optional[str] = None,
        log_individual_layers: bool = False,
        dead_params_ratio_threshold: float = 1e-6,
        dead_grad_ratio_threshold: float = 1e-8,
        exploding_grad_ratio_threshold: float = 10.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "tensorboard_weights_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.log_individual_layers = log_individual_layers
        self.dead_params_ratio_threshold = dead_params_ratio_threshold
        self.dead_grad_ratio_threshold = dead_grad_ratio_threshold
        self.exploding_grad_ratio_threshold = exploding_grad_ratio_threshold

        self.prefix = "model/" if prefix is None else f"{prefix}/"

        self.key_map = {
            "model": model_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def _get_layer_name(self, param_name: str) -> str:
        """Extract layer name from parameter name (remove .weight, .bias suffixes)."""
        # Remove common parameter suffixes
        for suffix in [
            ".weight",
            ".bias",
            ".running_mean",
            ".running_var",
            ".num_batches_tracked",
        ]:
            if param_name.endswith(suffix):
                return param_name[: -len(suffix)]
        return param_name

    def _log_layer_stats(self, model: torch.nn.Module, step: int) -> None:
        """Log essential statistics for detecting dead/exploding gradients and parameters."""

        # Collect statistics per layer name
        layer_stats = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            layer_name = self._get_layer_name(name)

            # Initialize layer if not exists
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {"weights": [], "grads": [], "names": []}

            # Store parameter data
            layer_stats[layer_name]["weights"].append(param.data)
            layer_stats[layer_name]["names"].append(name)

            if param.grad is not None:
                layer_stats[layer_name]["grads"].append(param.grad)

        # Log aggregated statistics by layer name
        for layer_name, stats in layer_stats.items():
            if not stats["weights"]:
                continue

            # Clean layer name for logging (replace dots with underscores)
            clean_layer_name = layer_name.replace(".", "_")

            # Combine all weights of this layer
            all_weights = torch.cat([w.flatten() for w in stats["weights"]])

            # Weight statistics for detecting dead parameters
            weight_mean = all_weights.mean().item()
            weight_std = all_weights.std().item()
            weight_abs_mean = all_weights.abs().mean().item()
            dead_params_ratio = (
                (all_weights.abs() < self.dead_params_ratio_threshold)
                .float()
                .mean()
                .item()
            )

            self.writer.add_scalar(
                f"{self.prefix}weights/{clean_layer_name}/mean", weight_mean, step
            )
            self.writer.add_scalar(
                f"{self.prefix}weights/{clean_layer_name}/std", weight_std, step
            )
            self.writer.add_scalar(
                f"{self.prefix}weights/{clean_layer_name}/abs_mean",
                weight_abs_mean,
                step,
            )
            self.writer.add_scalar(
                f"{self.prefix}weights/{clean_layer_name}/dead_ratio",
                dead_params_ratio,
                step,
            )

            # Single histogram for weight distribution (key for detecting dead parameters)
            self.writer.add_histogram(
                f"{self.prefix}weights/{clean_layer_name}/distribution",
                all_weights,
                step,
            )

            # Gradient statistics for detecting dead/exploding gradients
            if stats["grads"]:
                all_grads = torch.cat([g.flatten() for g in stats["grads"]])

                grad_mean = all_grads.mean().item()
                grad_std = all_grads.std().item()
                grad_abs_mean = all_grads.abs().mean().item()
                grad_max = all_grads.abs().max().item()
                dead_grads_ratio = (
                    (all_grads.abs() < self.dead_grad_ratio_threshold)
                    .float()
                    .mean()
                    .item()
                )
                exploding_grads_ratio = (
                    (all_grads.abs() > self.exploding_grad_ratio_threshold)
                    .float()
                    .mean()
                    .item()
                )

                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/mean", grad_mean, step
                )
                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/std", grad_std, step
                )
                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/abs_mean",
                    grad_abs_mean,
                    step,
                )
                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/max", grad_max, step
                )
                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/dead_ratio",
                    dead_grads_ratio,
                    step,
                )
                self.writer.add_scalar(
                    f"{self.prefix}grads/{clean_layer_name}/exploding_ratio",
                    exploding_grads_ratio,
                    step,
                )

                # Single histogram for gradient distribution (key for detecting issues)
                self.writer.add_histogram(
                    f"{self.prefix}grads/{clean_layer_name}/distribution",
                    all_grads,
                    step,
                )

            # Individual parameter logging if requested
            if self.log_individual_layers:
                for i, (name, weight) in enumerate(
                    zip(stats["names"], stats["weights"])
                ):
                    clean_name = name.replace(".", "_")
                    self.writer.add_scalar(
                        f"{self.prefix}individual/{clean_name}/abs_mean",
                        weight.abs().mean().item(),
                        step,
                    )
                    if i < len(stats["grads"]):
                        grad = stats["grads"][i]
                        self.writer.add_scalar(
                            f"{self.prefix}individual/{clean_name}/grad_abs_mean",
                            grad.abs().mean().item(),
                            step,
                        )

    def compute(self, model: torch.nn.Module) -> None:
        self._log_layer_stats(model, self.step)
        self.writer.flush()
        self.step += 1

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def __del__(self) -> None:
        self.close()
