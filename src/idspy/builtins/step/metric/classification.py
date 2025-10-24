from typing import Optional, Dict, Any

import numpy as np
from sklearn.metrics import roc_curve, auc

from ....core.step.base import Step
from ....plot.score import confusion_matrix_to_plot, dict_to_bar_plot, roc_auc_plot
from .. import StepFactory


@StepFactory.register()
@Step.needs("predictions", "targets")
class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        name: Optional[str] = None,
        predictions_key: str = "predictions",
        targets_key: str = "targets",
        metrics_key: str = "classification_metrics",
        class_names: Optional[list] = None,
    ) -> None:

        super().__init__(name=name or "classification_metrics")
        self.class_names = class_names
        self.key_map = {
            "predictions": predictions_key,
            "targets": targets_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def _compute_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        base_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }

        class_metrics = {}

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        class_names = (
            [str(i) for i in range(len(f1_per_class))]
            if self.class_names is None
            else self.class_names
        )

        for class_name, f1 in zip(class_names, f1_per_class):
            class_metrics[f"f1_{class_name}"] = f1

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "base_metrics": dict_to_bar_plot(base_metrics),
            "class_metrics": dict_to_bar_plot(class_metrics),
            "confusion_matrix": confusion_matrix_to_plot(cm),
        }

        return metrics

    def compute(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        metrics = self._compute_metrics(predictions, targets)
        return {"metrics": metrics}


@StepFactory.register()
@Step.needs("predictions", "targets")
class UnsupervisedClassificationMetrics(Step):

    def __init__(
        self,
        name: Optional[str] = None,
        predictions_key: str = "predictions",
        targets_key: str = "targets",
        metrics_key: str = "roc_auc_metrics",
    ) -> None:

        super().__init__(name=name or "unsupervised_classification_metrics")
        self.key_map = {
            "predictions": predictions_key,
            "targets": targets_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Compute ROC AUC for unsupervised classification."""
        targets = 1 - targets  # invert targets for anomaly detection
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)

        return {"metrics": {"roc_auc": roc_auc_plot(fpr, tpr, roc_auc)}}
