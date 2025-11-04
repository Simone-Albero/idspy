from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split


def sample_vectors_and_labels(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample data with optional stratification by labels.

    Returns:
        Tuple of (sampled_X, sampled_labels). sampled_labels is None if labels is None.
    """
    if sample_size is None or len(X) <= sample_size:
        return X, labels

    if labels is not None and stratify:
        _, X_sampled, _, labels_sampled = train_test_split(
            X, labels, test_size=sample_size, stratify=labels, random_state=random_state
        )
        return X_sampled, labels_sampled
    else:
        # Simple random sampling
        np.random.seed(random_state)
        indices = np.random.choice(len(X), sample_size, replace=False)
        sampled_labels = labels[indices] if labels is not None else None
        return X[indices], sampled_labels


def sample_labels(
    labels: np.ndarray,
    sample_size: int,
    stratify: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Sample indices from array with optional stratification.

    Args:
        labels: 1D array of values to sample from
        sample_size: Number of samples to select
        stratify: If True, maintain the distribution of values in labels
        random_state: Random seed for reproducibility

    Returns:
        Array of selected indices
    """
    n_samples = len(labels)

    if sample_size >= n_samples:
        return np.arange(n_samples)

    if stratify:
        # Create dummy indices array for train_test_split
        indices = np.arange(n_samples)
        _, selected_indices = train_test_split(
            indices, test_size=sample_size, stratify=labels, random_state=random_state
        )
        return selected_indices
    else:
        # Simple random sampling
        np.random.seed(random_state)
        return np.random.choice(n_samples, sample_size, replace=False)
