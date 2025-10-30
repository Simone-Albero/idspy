import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dict_to_bar_plot(d: dict, title: str = "Metrics") -> plt.Figure:
    """Plot dictionary as a bar plot."""
    fig, ax = plt.subplots()
    sns.barplot(x=list(d.keys()), y=list(d.values()), ax=ax)
    ax.set_title(title)
    return fig


def dict_to_table(d: dict, title: str = "Metrics") -> plt.Figure:
    """Plot dictionary as a table."""
    fig, ax = plt.subplots()
    ax.axis("off")
    table_data = [
        [k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in d.items()
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title)
    fig.tight_layout()
    return fig
