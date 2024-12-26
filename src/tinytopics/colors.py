from typing import Literal, overload
from collections.abc import Sequence, MutableSequence

import numpy as np
from numpy.typing import NDArray
from matplotlib import colors
from matplotlib.colors import ListedColormap
from skimage import color
from scipy.interpolate import make_interp_spline


ColorFormat = Literal["hex", "rgb", "lab"]


@overload
def pal_tinytopics(format: Literal["hex"]) -> MutableSequence[str]: ...


@overload
def pal_tinytopics(format: Literal["rgb", "lab"]) -> NDArray[np.float64]: ...


def pal_tinytopics(
    format: ColorFormat = "hex",
) -> MutableSequence[str] | NDArray[np.float64]:
    """
    The tinytopics 10 color palette.

    A rearranged version of the original Observable 10 palette.
    Reordered to align with the color ordering of the D3 Category 10 palette,
    also known as `matplotlib.cm.tab10`.
    The rearrangement aims to improve perceptual familiarity and color harmony,
    especially when used in a context where color interpolation is needed.

    Args:
        format: Returned color format. Options are:

            - `hex`: Hex strings (default).
            - `rgb`: Array of RGB values.
            - `lab`: Array of CIELAB values.

    Returns:
        Colors in the requested format:

            - If `format='hex'`, returns a list of hex color strings.
            - If `format='rgb'`, returns an Nx3 numpy array of RGB values.
            - If `format='lab'`, returns an Nx3 numpy array of CIELAB values.

    Raises:
        ValueError: If format is not `'hex'`, `'rgb'`, or `'lab'`.
    """
    TINYTOPICS_10_COLORS: Sequence[str] = (
        "#4269D0",  # Blue
        "#EFB118",  # Orange
        "#3CA951",  # Green
        "#FF725C",  # Red
        "#A463F2",  # Purple
        "#9C6B4E",  # Brown
        "#FF8AB7",  # Pink
        "#9498A0",  # Gray
        "#6CC5B0",  # Cyan
        "#97BBF5",  # Light Blue
    )

    if format == "hex":
        return list(TINYTOPICS_10_COLORS)

    # Convert hex to RGB array
    rgb_colors: NDArray[np.float64] = np.array(
        [colors.to_rgb(color) for color in TINYTOPICS_10_COLORS]
    )

    if format == "rgb":
        return rgb_colors
    elif format == "lab":
        # Convert RGB to CIELAB
        return color.rgb2lab(rgb_colors.reshape(1, -1, 3)).reshape(-1, 3)
    else:
        raise ValueError("Format must be 'hex', 'rgb', or 'lab'.")


def scale_color_tinytopics(n: int) -> ListedColormap:
    """
    A tinytopics 10 color scale. If > 10 colors are required, will generate
    an interpolated color palette based on the 10-color palette in the CIELAB
    color space using B-splines.

    Args:
        n: The number of colors needed.

    Returns:
        A colormap with n colors, possibly interpolated from the 10 colors.
    """
    base_rgb_colors: NDArray[np.float64] = pal_tinytopics(format="rgb")
    base_lab_colors: NDArray[np.float64] = pal_tinytopics(format="lab")

    # If interpolation is NOT needed, return the first n colors directly
    if n <= len(base_rgb_colors):
        return colors.ListedColormap(base_rgb_colors[:n])

    # If interpolation is needed, interpolate in the CIELAB space
    # for perceptually uniform colors
    additional_colors_needed: int = n - 10

    # Original positions of the 10 base colors
    x: NDArray[np.float64] = np.linspace(0, 1, len(base_lab_colors))

    # B-spline interpolator in the CIELAB space
    bspline = make_interp_spline(x, base_lab_colors, k=3)

    # Interpolated positions for new colors
    x_new: NDArray[np.float64] = np.linspace(0, 1, additional_colors_needed + 10)
    interpolated_lab: NDArray[np.float64] = bspline(x_new)

    # Convert interpolated LAB colors back to RGB
    interpolated_rgb: NDArray[np.float64] = color.lab2rgb(
        interpolated_lab.reshape(1, -1, 3)
    ).reshape(-1, 3)

    return colors.ListedColormap(interpolated_rgb)
