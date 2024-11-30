import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tinytopics.plot import plot_loss, plot_structure, plot_top_terms

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@pytest.fixture
def sample_losses():
    """Fixture providing sample loss values."""
    return [100, 80, 60, 40, 20]


@pytest.fixture
def sample_L_matrix():
    """Fixture providing sample document-topic matrix."""
    return np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])


@pytest.fixture
def sample_F_matrix():
    """Fixture providing sample topic-term matrix."""
    return np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])


def test_plot_loss(sample_losses):
    """Test loss curve plotting."""
    # Test basic plotting
    plot_loss(sample_losses)
    plt.close()

    # Test saving to file
    with pytest.raises(FileNotFoundError):
        plot_loss(sample_losses, output_file="/nonexistent/path/plot.png")

    # Test custom parameters
    plot_loss(
        sample_losses,
        figsize=(4, 3),
        dpi=150,
        title="Custom Title",
        color_palette="#FF0000",
    )
    plt.close()


def test_plot_structure(sample_L_matrix):
    """Test structure plot functionality."""
    # Test basic plotting
    plot_structure(sample_L_matrix)
    plt.close()

    # Test with row normalization
    plot_structure(sample_L_matrix, normalize_rows=True)
    plt.close()

    # Test custom parameters
    plot_structure(
        sample_L_matrix,
        figsize=(6, 4),
        dpi=150,
        title="Custom Title",
        color_palette=["#FF0000", "#00FF00"],
    )
    plt.close()


def test_plot_top_terms(sample_F_matrix):
    """Test top terms plotting."""
    term_names = ["term1", "term2", "term3", "term4"]

    # Test basic plotting
    plot_top_terms(sample_F_matrix, n_top_terms=4)
    plt.close()

    # Test with term names
    plot_top_terms(sample_F_matrix, term_names=term_names, n_top_terms=4)
    plt.close()

    # Test custom parameters
    plot_top_terms(
        sample_F_matrix,
        n_top_terms=2,
        figsize=(4, 3),
        dpi=150,
        title="Custom Title",
        color_palette=["#FF0000", "#00FF00"],
        nrows=1,
        ncols=2,
    )
    plt.close()


@pytest.mark.parametrize("plot_func", [plot_loss, plot_structure, plot_top_terms])
def test_plot_functions_close_figures(
    plot_func, sample_losses, sample_L_matrix, sample_F_matrix, tmp_path
):
    """Test that all plot functions properly close figures when saving."""
    output_file = tmp_path / "test_plot.png"

    # Close any existing figures before starting the test
    plt.close("all")

    if plot_func == plot_loss:
        plot_func(sample_losses, output_file=str(output_file))
    elif plot_func == plot_structure:
        plot_func(sample_L_matrix, output_file=str(output_file))
    else:  # plot_top_terms
        plot_func(sample_F_matrix, n_top_terms=4, output_file=str(output_file))

    # Wait for file to be written
    assert output_file.exists()

    # Check if figures are closed
    assert len(plt.get_fignums()) == 0
