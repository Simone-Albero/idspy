from typing import Optional, Any, Dict
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .....core.step.base import Step
from ... import StepFactory


@StepFactory.register()
@Step.needs("data")
class KMeans(Step):
    """Compute K-Means clustering on input data."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        data_key: str = "data",
        output_key: str = "kmeans_model",
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
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        self.kmeans.fit(data)
        output = {
            "kmeans_model": self.kmeans,
            "cluster_centers": self.kmeans.cluster_centers_,
            "labels": self.kmeans.labels_,
            "inertia": self.kmeans.inertia_,
        }
        return {"output": output}


@StepFactory.register()
@Step.needs("data")
class GaussianMixture(Step):
    """Compute Gaussian Mixture clustering on input data."""

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        data_key: str = "data",
        output_key: str = "gm_model",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_gm")
        self.gm = GaussianMixture(
            n_components=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        self.gm.fit(data)
        output = {
            "gm_model": self.gm,
            "means": self.gm.means_,
            "covariances": self.gm.covariances_,
            "weights": self.gm.weights_,
            "labels": self.gm.predict(data),
            "aic": self.gm.aic(data),
            "bic": self.gm.bic(data),
        }
        return {"output": output}
