from typing import Optional, Dict, Any

import numpy as np
from sklearn.metrics import roc_curve, auc

from ....core.step.base import Step
from ....plot.array import confusion_matrix_to_plot, roc_auc_plot, distribution_plot
from ....plot.dict import dict_to_bar_plot
from .. import StepFactory


@StepFactory.register()
@Step.needs("predictions", "confidences", "labels")
class SupervisedClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        name: Optional[str] = None,
        predictions_key: str = "predictions",
        confidences_key: str = "confidences",
        labels_key: str = "labels",
        metrics_key: str = "classification_metrics",
    ) -> None:

        super().__init__(name=name or "classification_metrics")
        self.key_map = {
            "predictions": predictions_key,
            "confidences": confidences_key,
            "labels": labels_key,
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

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        f1_per_class = f1_score(
            y_true, y_pred, average=None, labels=unique_classes, zero_division=0
        ).tolist()

        class_names = [str(i) for i in unique_classes]

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
        self, predictions: np.ndarray, confidences: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        metrics = self._compute_metrics(predictions, labels)
        metrics["confidence_distribution"] = distribution_plot(
            confidences,
            title="Confidence Scores Distribution",
            x_range=(0, 1),
        )
        return {"metrics": metrics}


@StepFactory.register()
@Step.needs("predictions", "labels")
class UnsupervisedClassificationMetrics(Step):

    def __init__(
        self,
        name: Optional[str] = None,
        predictions_key: str = "predictions",
        labels_key: str = "labels",
        metrics_key: str = "roc_auc_metrics",
    ) -> None:

        super().__init__(name=name or "unsupervised_classification_metrics")
        self.key_map = {
            "predictions": predictions_key,
            "labels": labels_key,
            "metrics": metrics_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def _find_optimal_threshold(
        self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
    ) -> float:
        """Find optimal threshold using Youden's J statistic."""
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    def _compute_binary_metrics(
        self, y_true: np.ndarray, y_pred_binary: np.ndarray
    ) -> Dict[str, float]:
        """Compute binary classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        return {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
        }

    def compute(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Compute ROC AUC for unsupervised classification."""
        labels = 1 - labels  # invert labels for anomaly detection
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        optimal_threshold = self._find_optimal_threshold(fpr, tpr, thresholds)

        # Apply optimal threshold to get binary predictions
        predictions_binary = (predictions >= optimal_threshold).astype(int)

        # Compute binary classification metrics
        binary_metrics = self._compute_binary_metrics(labels, predictions_binary)

        # Compute confidence as normalized distance from optimal threshold
        # Use actual min/max of predictions for normalization
        distance_from_threshold = np.abs(predictions - optimal_threshold)
        max_distance = max(
            optimal_threshold - predictions.min(),  # max distance below threshold
            predictions.max() - optimal_threshold,  # max distance above threshold
        )
        confidences = (
            distance_from_threshold / max_distance
            if max_distance > 0
            else np.zeros_like(distance_from_threshold)
        )

        metrics = {
            "roc_auc": roc_auc_plot(fpr, tpr, roc_auc),
            "binary_metrics": dict_to_bar_plot(binary_metrics),
            "confidence_distribution": distribution_plot(
                confidences,
                title="Confidence Scores Distribution",
                x_range=(0, 1),
            ),
        }

        return {"metrics": metrics}
