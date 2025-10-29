from typing import Optional, Any, Dict
import numpy as np

from sklearn.cluster import KMeans as KmeansAlgorithm
from sklearn.mixture import GaussianMixture as GaussianMixtureAlgorithm
import hdbscan

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
        output_key: str = "kmeans_labels",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_kmeans")
        self.kmeans = KmeansAlgorithm(
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
        return {"output": self.kmeans.labels_}


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
        output_key: str = "gm_labels",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_gm")
        self.gm = GaussianMixtureAlgorithm(
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
        return {"output": self.gm.predict(data)}


@StepFactory.register()
@Step.needs("data")
class HDBSCAN(Step):
    """Compute HDBSCAN clustering on input data."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        data_key: str = "data",
        output_key: str = "hdbscan_labels",
        core_dist_n_jobs: int = 8,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_hdbscan")
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            core_dist_n_jobs=core_dist_n_jobs,
        )

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        self.hdbscan.fit(data)
        print("HDBSCAN found clusters:", np.unique(self.hdbscan.labels_))
        return {"output": self.hdbscan.labels_}
