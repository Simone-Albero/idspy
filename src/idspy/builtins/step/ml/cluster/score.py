from typing import Optional, Any, Dict
import numpy as np

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from pyclustertend import hopkins as hopkins_score

from .....core.step.base import Step
from ...factory import StepFactory


@StepFactory.register()
@Step.needs("vectors", "targets")
class ClusteringScores(Step):
    """Compute clustering metrics: Hopkins, Silhouette, and Adjusted Rand Index."""

    def __init__(
        self,
        vectors_key: str = "vectors",
        targets_key: str = "targets",
        outputs_key: str = "clustering_scores",
        scale_inputs: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_clustering_scores")
        self.scale_inputs = scale_inputs
        self.key_map = {
            "vectors": vectors_key,
            "targets": targets_key,
            "outputs": outputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, vectors: np.ndarray, targets: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        if self.scale_inputs:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(vectors.reshape(-1, 1))
        else:
            X_scaled = vectors.reshape(-1, 1)

        hopkins = hopkins_score(X_scaled, sampling_size=X_scaled.shape[0])
        silhouette = silhouette_score(X_scaled, targets)
        ari = adjusted_rand_score(targets, vectors)

        outputs = {
            "hopkins": hopkins,
            "silhouette": silhouette,
            "ari": ari,
        }

        return {"outputs": outputs}
