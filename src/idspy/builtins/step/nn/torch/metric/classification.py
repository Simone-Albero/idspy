from typing import Optional, Dict, Any

import numpy as np

from ......core.step.base import Step


@Step.needs("predictions", "targets")
class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        name: Optional[str] = None,
        predictions_key: str = "predictions",
        targets_key: str = "targets",
        metrics_key: str = "classification_metrics",
    ) -> None:

        super().__init__(name=name or "classification_metrics")
        self.key_map = {
            "predictions": predictions_key,
            "targets": targets_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
        }

        return metrics

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        metrics = self.compute_metrics(predictions, targets)
        return {"metrics": metrics}
