import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses, figsize=(10, 8), dpi=300, title="Loss curve", output_file=None):
    """
    Plot the loss curve over training epochs.

    Args:
        losses (list): List of loss values for each epoch.
        figsize (tuple, optional): Plot size. Default is (10, 8).
        dpi (int, optional): Plot resolution. Default is 300.
        title (str, optional): Plot title. Default is "Loss curve".
        output_file (str, optional): File path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_structure(
    L_matrix,
    figsize=(12, 6),
    dpi=300,
    title="Structure Plot",
    color_palette=None,
    output_file=None,
):
    """
    STRUCTURE plot for visualizing document-topic distributions.

    Args:
        L_matrix (np.ndarray): Document-topic distribution matrix.
        figsize (tuple, optional): Plot size. Default is (12, 6).
        dpi (int, optional): Plot resolution. Default is 300.
        title (str): Plot title.
        color_palette (list or matplotlib colormap, optional): Custom color palette.
        output_file (str, optional): File path to save the plot. If None, displays the plot.
    """
    n_documents, n_topics = L_matrix.shape
    ind = np.arange(n_documents)  # Document indices
    cumulative = np.zeros(n_documents)

    # Color palette
    if color_palette is None:
        colors = plt.cm.tab20(np.linspace(0, 1, n_topics))
    else:
        if isinstance(color_palette, list):
            colors = color_palette
        else:
            colors = color_palette(np.linspace(0, 1, n_topics))

    plt.figure(figsize=figsize, dpi=dpi)
    for k in range(n_topics):
        plt.bar(
            ind,
            L_matrix[:, k],
            bottom=cumulative,
            color=colors[k % len(colors)],
            width=1.0,
        )
        cumulative += L_matrix[:, k]
    plt.title(title)
    plt.xlabel("Documents (sorted)")
    plt.ylabel("Topic Proportions")
    plt.xlim([0, n_documents])
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_top_terms(
    F_matrix,
    n_top_terms=10,
    term_names=None,
    figsize=(10, 8),
    dpi=300,
    title="Top Terms",
    color_palette=None,
    nrows=None,
    ncols=None,
    output_file=None,
):
    """
    Plot top terms for each topic in horizontal bar charts.

    Args:
        F_matrix (np.ndarray): Topic-term distribution matrix.
        n_top_terms (int, optional): Number of top terms to display per topic. Default is 10.
        term_names (list, optional): List of term names corresponding to indices.
        figsize (tuple, optional): Plot size. Default is (10, 8).
        dpi (int, optional): Plot resolution. Default is 300.
        title (str): Plot title.
        color_palette (list or matplotlib colormap, optional): Custom color palette.
        nrows (int, optional): Number of rows in the subplot grid.
        ncols (int, optional): Number of columns in the subplot grid.
        output_file (str, optional): File path to save the plot. If None, displays the plot.
    """
    n_topics = F_matrix.shape[0]
    top_terms_indices = np.argsort(-F_matrix, axis=1)[:, :n_top_terms]
    top_terms_probs = np.take_along_axis(F_matrix, top_terms_indices, axis=1)

    # Use term names if provided
    if term_names is not None:
        top_terms_labels = np.array(term_names)[top_terms_indices]
    else:
        top_terms_labels = top_terms_indices.astype(str)

    # Color palette
    if color_palette is None:
        colors = plt.cm.tab20(np.linspace(0, 1, n_topics))
    else:
        if isinstance(color_palette, list):
            colors = color_palette
        else:
            colors = color_palette(np.linspace(0, 1, n_topics))

    # Grid layout
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
    axes = axes.flatten()

    for i in range(n_topics):
        ax = axes[i]
        # Get data for topic i
        probs = top_terms_probs[i]
        labels = top_terms_labels[i]

        # Place highest probability terms at the top
        y_pos = np.arange(n_top_terms)[::-1]
        ax.barh(y_pos, probs, color=colors[i % len(colors)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Probability")
        ax.set_title(f"Topic {i}")
        ax.set_xlim(0, top_terms_probs.max() * 1.1)
    # Hide unused subplots
    for j in range(n_topics, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title)

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
