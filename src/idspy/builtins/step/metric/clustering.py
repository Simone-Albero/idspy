import logging
from typing import Optional, Any, Dict


import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from ....core.step.base import Step
from ....plot.dict import dict_to_table
from .. import StepFactory

logger = logging.getLogger(__name__)


def hopkins_statistic(X: np.ndarray) -> float:
    """Compute the Hopkins statistic for the dataset X. Values close to 1 indicate
    a highly clustered dataset, while values close to 0.5 suggest randomness.
    """

    X = MinMaxScaler().fit_transform(X)
    n, d = X.shape

    # Generate random points uniformly in the data space
    X_random = np.random.uniform(size=(n, d))

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X)

    u_dist, _ = neigh.kneighbors(X_random, n_neighbors=1)
    w_dist, _ = neigh.kneighbors(X, n_neighbors=2)

    w_dist = w_dist[:, 1]  # exclude the distance to itself

    H = np.sum(u_dist) / (np.sum(u_dist) + np.sum(w_dist))
    return H


def class_separation_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute ratio of between-class to within-class variance.
    Higher values indicate better separation between true classes.
    """
    unique_labels = np.unique(labels)
    overall_mean = np.mean(X, axis=0)

    # Within-class scatter
    within_class_var = 0.0
    for label in unique_labels:
        class_samples = X[labels == label]
        class_mean = np.mean(class_samples, axis=0)
        within_class_var += np.sum((class_samples - class_mean) ** 2)

    # Between-class scatter
    between_class_var = 0.0
    for label in unique_labels:
        class_samples = X[labels == label]
        class_mean = np.mean(class_samples, axis=0)
        n_samples = len(class_samples)
        between_class_var += n_samples * np.sum((class_mean - overall_mean) ** 2)

    # Avoid division by zero
    if within_class_var < 1e-10:
        return float("inf")

    return between_class_var / within_class_var


def intra_class_cohesion(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute average within-class cosine similarity.
    Values close to 1 indicate high cohesion within classes.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    unique_labels = np.unique(labels)
    cohesion_scores = []

    for label in unique_labels:
        class_samples = X[labels == label]
        if len(class_samples) > 1:
            sim_matrix = cosine_similarity(class_samples)
            # Get upper triangle without diagonal
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            if len(upper_tri) > 0:
                cohesion_scores.append(np.mean(upper_tri))

    return np.mean(cohesion_scores) if cohesion_scores else 0.0


def inter_class_separation(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute average between-class distance (centroid-based).
    Higher values indicate better separation between classes.
    """
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        class_samples = X[labels == label]
        centroids.append(np.mean(class_samples, axis=0))

    centroids = np.array(centroids)

    # Compute pairwise distances between centroids
    from scipy.spatial.distance import pdist

    distances = pdist(centroids, metric="euclidean")

    return np.mean(distances) if len(distances) > 0 else 0.0


@StepFactory.register()
@Step.needs("vectors", "labels", "ground_truth")
class ClusteringMetrics(Step):
    """Compute clustering metrics for latent space clusterability analysis.

    Metrics computed:
    - Hopkins: measures clustering tendency (closer to 1 = more clusterable)
    - Silhouette: measures separation quality using true labels (-1 to 1, higher is better)
    - Class Separation: ratio of between-class to within-class variance (higher is better)
    - Intra-class Cohesion: average cosine similarity within classes (0 to 1, higher is better)
    - Inter-class Separation: average distance between class centroids (higher is better)

    If ground truth labels provided, also computes comparison metrics:
    - ARI: Adjusted Rand Index (-1 to 1, higher is better)
    - V-measure: harmonic mean of homogeneity and completeness (0 to 1, higher is better)
    - Homogeneity: each cluster contains only members of a single class (0 to 1, higher is better)
    - Completeness: all members of a class are in the same cluster (0 to 1, higher is better)
    """

    def __init__(
        self,
        vectors_key: str = "vectors",
        labels_key: str = "labels",
        ground_truth_key: Optional[str] = None,
        metrics_key: str = "clustering_scores",
        scale_inputs: bool = True,
        pca_components: int = 16,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_clustering_scores")
        self.scale_inputs = scale_inputs
        self.pca_components = pca_components
        self.ground_truth_key = ground_truth_key
        self.key_map = {
            "vectors": vectors_key,
            "labels": labels_key,
            "outputs": metrics_key,
        }
        if ground_truth_key:
            self.key_map["ground_truth"] = ground_truth_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:

        if self.scale_inputs:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(vectors)
        else:
            X_scaled = vectors

        if self.pca_components < X_scaled.shape[1]:
            X_compressed = PCA(n_components=self.pca_components).fit_transform(X_scaled)
        else:
            X_compressed = X_scaled

        scores = {}

        # Hopkins statistic (clustering tendency)
        logger.info("Computing Hopkins statistic...")
        scores["Hopkins"] = hopkins_statistic(X_compressed)

        # Metrics based on true labels
        logger.info("Computing Silhouette score...")
        scores["Silhouette"] = silhouette_score(X_compressed, labels)

        # Custom metrics
        logger.info("Computing class separation...")
        scores["Class Separation"] = class_separation_score(X_compressed, labels)

        logger.info("Computing intra-class cohesion...")
        scores["Intra-class Cohesion"] = intra_class_cohesion(X_compressed, labels)

        logger.info("Computing inter-class separation...")
        scores["Inter-class Separation"] = inter_class_separation(X_compressed, labels)

        # Ground truth comparison metrics
        if ground_truth is not None:
            logger.info("Computing ground truth comparison metrics...")

            scores["ARI"] = adjusted_rand_score(ground_truth, labels)

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                ground_truth, labels
            )
            scores["V-measure (considering ground truth)"] = v_measure
            scores["Homogeneity (considering ground truth)"] = homogeneity
            scores["Completeness (considering ground truth)"] = completeness

        return {
            "outputs": {
                "clustering_scores": dict_to_table(scores, title="Clustering Scores")
            }
        }
