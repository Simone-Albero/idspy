from typing import Optional, Any, Dict
import numpy as np

from sklearn.cluster import (
    KMeans as KmeansAlgorithm,
    OPTICS as OPTICSAlgorithm,
    SpectralClustering as SpectralClusteringAlgorithm,
    AgglomerativeClustering as AgglomerativeClusteringAlgorithm,
)
from sklearn.mixture import GaussianMixture as GaussianMixtureAlgorithm
from sklearn.preprocessing import StandardScaler
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
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_kmeans")
        self.kmeans = KmeansAlgorithm(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
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
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_gm")
        self.gm = GaussianMixtureAlgorithm(
            n_components=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
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
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        alpha: float = 1.0,
        data_key: str = "data",
        output_key: str = "hdbscan_labels",
        core_dist_n_jobs: int = 8,
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_hdbscan")
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            alpha=alpha,
            core_dist_n_jobs=core_dist_n_jobs,
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
        self.hdbscan.fit(data)
        return {"output": self.hdbscan.labels_}


@StepFactory.register()
@Step.needs("data")
class OPTICS(Step):
    """Compute OPTICS clustering on input data."""

    def __init__(
        self,
        min_samples: int = 5,
        max_eps: float = np.inf,
        metric: str = "euclidean",
        cluster_method: str = "xi",
        xi: float = 0.05,
        data_key: str = "data",
        output_key: str = "optics_labels",
        n_jobs: Optional[int] = None,
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_optics")
        self.optics = OPTICSAlgorithm(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            cluster_method=cluster_method,
            xi=xi,
            n_jobs=n_jobs,
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
        self.optics.fit(data)
        return {"output": self.optics.labels_}


@StepFactory.register()
@Step.needs("data")
class SpectralClustering(Step):
    """Compute Spectral Clustering on input data."""

    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        data_key: str = "data",
        output_key: str = "spectral_labels",
        n_jobs: Optional[int] = None,
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_spectral")
        self.spectral = SpectralClusteringAlgorithm(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
        labels = self.spectral.fit_predict(data)
        return {"output": labels}


@StepFactory.register()
@Step.needs("data")
class AgglomerativeClustering(Step):
    """Compute Agglomerative Clustering on input data."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        linkage: str = "ward",
        metric: str = "euclidean",
        data_key: str = "data",
        output_key: str = "agglomerative_labels",
        standardize: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_agglomerative")
        # if distance_threshold is set, n_clusters must be None
        if distance_threshold is not None:
            n_clusters = None
        elif n_clusters is None:
            n_clusters = 2  # default to 2 clusters if neither is set

        self.agglomerative = AgglomerativeClusteringAlgorithm(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage,
            metric=metric if linkage != "ward" else "euclidean",
        )
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.key_map = {
            "data": data_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.standardize:
            data = self.scaler.fit_transform(data)
        labels = self.agglomerative.fit_predict(data)
        return {"output": labels}
