from typing import Optional, Any, Dict
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .....core.step.base import Step
from ... import StepFactory


@StepFactory.register()
@Step.needs("inputs")
class KMeans(Step):
    """Compute K-Means clustering on input data."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        inputs_key: str = "inputs",
        outputs_key: str = "kmeans_model",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_kmeans")
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )

        self.key_map = {
            "inputs": inputs_key,
            "outputs": outputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, inputs: np.ndarray) -> Optional[Dict[str, Any]]:
        self.kmeans.fit(inputs)
        outputs = {
            "kmeans_model": self.kmeans,
            "cluster_centers": self.kmeans.cluster_centers_,
            "labels": self.kmeans.labels_,
            "inertia": self.kmeans.inertia_,
        }
        return {"outputs": outputs}


@StepFactory.register()
@Step.needs("inputs")
class GaussianMixture(Step):
    """Compute Gaussian Mixture clustering on input data."""

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        inputs_key: str = "inputs",
        outputs_key: str = "gm_model",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_gm")
        self.gm = GaussianMixture(
            n_components=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )

        self.key_map = {
            "inputs": inputs_key,
            "outputs": outputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, inputs: np.ndarray) -> Optional[Dict[str, Any]]:
        self.gm.fit(inputs)
        outputs = {
            "gm_model": self.gm,
            "means": self.gm.means_,
            "covariances": self.gm.covariances_,
            "weights": self.gm.weights_,
            "labels": self.gm.predict(inputs),
            "aic": self.gm.aic(inputs),
            "bic": self.gm.bic(inputs),
        }
        return {"outputs": outputs}
