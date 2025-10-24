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
    is_3d = n_features == 3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)

    # Prepare color argument
    color_arg = colors if colors is not None and len(colors) > 0 else "b"

    # Create scatter plot
    if is_3d:
        scatter = ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            c=color_arg,
            cmap="tab10",
            s=50,
            alpha=0.7,
        )
        ax.set_zlabel("Z")
    else:
        scatter = ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            c=color_arg,
            cmap="tab10",
            s=50,
            alpha=0.7,
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{n_features}D Vector Plot")

    # Add legend if colors are provided
    if colors is not None and len(colors) > 0:
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        legend.set_draggable(True)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig
