import matplotlib.pyplot as plt
import numpy as np


def vectors_plot(
    vectors: np.ndarray, colors: list | np.ndarray | None = None
) -> plt.Figure:
    """Plot 2D or 3D vectors with optional color coding.

    Args:
        vectors (np.ndarray): Array of shape (n_samples, n_features) where n_features is 2 or 3.
    """
    if vectors.ndim != 2 or vectors.shape[1] not in (2, 3):
        raise ValueError(
            "`vectors` must be a 2D array with shape (n_samples, 2) or (n_samples, 3)."
        )

    n_features = vectors.shape[1]
    fig = plt.figure()
    if n_features == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            vectors[:, 0], vectors[:, 1], vectors[:, 2], c=colors or "b"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:  # n_features == 2
        ax = fig.add_subplot(111)
        scatter = ax.scatter(vectors[:, 0], vectors[:, 1], c=colors or "b")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if colors is not None:
        plt.colorbar(scatter, ax=ax, label="Color Scale")

    plt.title(f"{n_features}D Vector Plot")
    plt.tight_layout()
    return fig
