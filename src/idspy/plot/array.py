import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_matrix_to_plot(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: list[str] | None = None,
    normalize: str | None = "true",  # one of: None, 'true', 'pred', 'all'
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    values_decimals: int | None = None,  # override decimals used in annotations
) -> plt.Figure:
    """Plot a confusion matrix with optional normalization."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")

    cm = np.asarray(cm, dtype=float)

    # Compute normalized matrix if requested (with safe division)
    if normalize is None:
        cm_to_plot = cm.copy()
    elif normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_to_plot = np.divide(cm, np.where(col_sums == 0, 1, col_sums))
    elif normalize == "all":
        total = cm.sum()
        cm_to_plot = cm / (total if total != 0 else 1.0)
    else:
        raise ValueError("`normalize` must be one of {None, 'true', 'pred', 'all'}.")

    # Decide annotation format
    if values_decimals is None:
        values_decimals = 0 if normalize is None else 2
    values_format = f".{values_decimals}f"

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_to_plot, display_labels=class_names
    )
    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=values_format,
        colorbar=show_colorbar,
        im_kw={"interpolation": "nearest"},
        xticks_rotation=45,
        include_values=True,
    )

    ax.set_title(title, fontsize=14, pad=16)
    if normalize is not None:
        norm_map = {
            "true": "rows (by true label)",
            "pred": "columns (by predicted)",
            "all": "entire matrix",
        }
        ax.set_xlabel(
            f"Predicted label\nNormalization: {norm_map[normalize]}", labelpad=10
        )
    else:
        ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig


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


def roc_auc_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    """Plot ROC curve with AUC annotation."""
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color="blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    return fig


def distribution_plot(
    values: np.ndarray,
    bins: int = 50,
    title: str = "Distribution",
    color: str = "steelblue",
    show_stats: bool = True,
    x_label: str = "Value",
    y_label: str = "Frequency",
    x_range: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot distribution of values with optional statistics.

    Args:
        values (np.ndarray): Array of values to plot.
        bins (int): Number of bins for the histogram.
        title (str): Plot title.
        figsize (tuple): Figure size.
        color (str): Color for the histogram bars.
        show_stats (bool): Whether to show statistics (mean, median, std) on the plot.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        x_range (tuple): Optional range for x-axis (min, max). If None, auto-determined.
    """
    if values.ndim != 1:
        values = values.flatten()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create histogram
    n, bins_edges, patches = ax.hist(
        values,
        bins=bins,
        range=x_range,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add vertical lines for statistics
    if show_stats:
        mean_val = values.mean()
        median_val = np.median(values)
        std_val = values.std()

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.3f}",
        )

        # Add text box with statistics
        stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}\nMin: {values.min():.3f}\nMax: {values.max():.3f}\nSamples: {len(values)}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.legend(loc="upper right")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=16)

    if x_range is not None:
        ax.set_xlim(x_range)

    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig
