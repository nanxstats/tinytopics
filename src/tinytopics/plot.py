from typing import Tuple
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .colors import scale_color_tinytopics


def plot_loss(
    losses: Sequence[float],
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    title: str = "Loss curve",
    color_palette: Sequence[str] | str | None = None,
    output_file: str | None = None,
) -> None:
    """
    Plot the loss curve over training epochs.

    Args:
        losses: List of loss values for each epoch.
        figsize: Plot size.
        dpi: Plot resolution.
        title: Plot title.
        color_palette: Custom color palette.
        output_file: File path to save the plot. If None, displays the plot.
    """
    if isinstance(color_palette, list):
        color = color_palette[0]
    elif isinstance(color_palette, str):
        color = color_palette
    else:
        color = scale_color_tinytopics(1)(0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(losses, color=color)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_structure(
    L_matrix: np.ndarray,
    normalize_rows: bool = False,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    title: str = "Structure Plot",
    color_palette: Sequence[str] | str | None = None,
    output_file: str | None = None,
) -> None:
    """
    Structure plot for visualizing document-topic distributions.

    Args:
        L_matrix: Document-topic distribution matrix.
        normalize_rows: If True, normalizes each row of L_matrix to sum to 1.
        figsize: Plot size.
        dpi: Plot resolution.
        title: Plot title.
        color_palette: Custom color palette.
        output_file: File path to save the plot. If None, displays the plot.
    """
    matrix = (
        L_matrix / L_matrix.sum(axis=1, keepdims=True) if normalize_rows else L_matrix
    )
    n_documents, n_topics = matrix.shape
    ind = np.arange(n_documents)
    cumulative = np.zeros(n_documents)
    if isinstance(color_palette, list):
        colors = color_palette
    else:
        colors = [scale_color_tinytopics(n_topics)(k) for k in range(n_topics)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    for k in range(n_topics):
        plt.bar(
            ind,
            matrix[:, k],
            bottom=cumulative,
            color=colors[k],
            width=1.0,
        )
        cumulative += matrix[:, k]

    plt.title(title)
    plt.xlabel("Documents (sorted)")
    plt.ylabel("Topic Proportions")
    plt.xlim([0, n_documents])
    plt.ylim(0, 1)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_top_terms(
    F_matrix: np.ndarray,
    n_top_terms: int = 10,
    term_names: Sequence[str] | None = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    title: str = "Top Terms",
    color_palette: Sequence[str] | str | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    output_file: str | None = None,
) -> None:
    """
    Plot top terms for each topic in horizontal bar charts.

    Args:
        F_matrix: Topic-term distribution matrix.
        n_top_terms: Number of top terms to display per topic.
        term_names: List of term names corresponding to indices.
        figsize: Plot size.
        dpi: Plot resolution.
        title: Plot title.
        color_palette: Custom color palette.
        nrows: Number of rows in the subplot grid.
        ncols: Number of columns in the subplot grid.
        output_file: File path to save the plot. If None, displays the plot.
    """
    n_topics = F_matrix.shape[0]
    top_terms_indices = np.argsort(-F_matrix, axis=1)[:, :n_top_terms]
    top_terms_probs = np.take_along_axis(F_matrix, top_terms_indices, axis=1)
    top_terms_labels = (
        np.array(term_names)[top_terms_indices]
        if term_names is not None
        else top_terms_indices.astype(str)
    )
    if isinstance(color_palette, list):
        colors = color_palette
    else:
        colors = [scale_color_tinytopics(n_topics)(k) for k in range(n_topics)]

    # Calculate grid dimensions
    if nrows is None and ncols is None:
        ncols = 5
        nrows = int(np.ceil(n_topics / ncols))
    elif nrows is None:
        nrows = int(np.ceil(n_topics / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_topics / nrows))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    axes_flat = axes.flatten()

    def plot_topic(ax: Axes, topic_idx: int) -> None:
        probs = top_terms_probs[topic_idx]
        labels = top_terms_labels[topic_idx]
        y_pos = np.arange(n_top_terms)[::-1]

        ax.barh(y_pos, probs, color=colors[topic_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Probability")
        ax.set_title(f"Topic {topic_idx}")
        ax.set_xlim(0, top_terms_probs.max() * 1.1)

    for i in range(n_topics):
        plot_topic(axes_flat[i], i)

    # Hide unused subplots
    for j in range(n_topics, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(title)

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
