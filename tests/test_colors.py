import pytest
import numpy as np
from matplotlib.colors import ListedColormap

from tinytopics.colors import pal_tinytopics, scale_color_tinytopics


def test_pal_tinytopics_hex():
    """Test hex color format output."""
    colors = pal_tinytopics(format="hex")

    assert len(colors) == 10
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith("#") for c in colors)
    assert all(len(c) == 7 for c in colors)


def test_pal_tinytopics_rgb():
    """Test RGB color format output."""
    colors = pal_tinytopics(format="rgb")

    assert isinstance(colors, np.ndarray)
    assert colors.shape == (10, 3)
    assert np.all((0 <= colors) & (colors <= 1))


def test_pal_tinytopics_lab():
    """Test LAB color format output."""
    colors = pal_tinytopics(format="lab")

    assert isinstance(colors, np.ndarray)
    assert colors.shape == (10, 3)


def test_pal_tinytopics_invalid_format():
    """Test error handling for invalid format."""
    with pytest.raises(ValueError):
        pal_tinytopics(format="invalid")


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_scale_color_tinytopics(n):
    """Test color scale generation with different numbers of colors."""
    colormap = scale_color_tinytopics(n)

    assert isinstance(colormap, ListedColormap)
    assert len(colormap.colors) == n
    assert np.all((0 <= colormap.colors) & (colormap.colors <= 1))


def test_scale_color_tinytopics_interpolation():
    """Test color interpolation for a large number of colors."""
    n_colors = 60
    colormap = scale_color_tinytopics(n_colors)

    # Check interpolated colors length
    colors = colormap.colors
    assert len(colors) == n_colors

    # Check if color transitions are smooth by computing differences
    # between consecutive colors
    color_diffs = np.diff(colors, axis=0)
    max_diff = np.max(np.abs(color_diffs))
    assert max_diff < 0.5
