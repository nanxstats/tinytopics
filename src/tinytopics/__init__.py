"""
Topic modeling via sum-to-one constrained neural Poisson NMF.
"""

from .colors import pal_tinytopics, scale_color_tinytopics
from .data import NumpyDiskDataset, TorchDiskDataset
from .fit import fit_model
from .fit_distributed import fit_model_distributed
from .loss import poisson_nmf_loss
from .models import NeuralPoissonNMF
from .plot import plot_loss, plot_structure, plot_top_terms
from .utils import (
    align_topics,
    generate_synthetic_data,
    set_random_seed,
    sort_documents,
)

__all__ = [
    "pal_tinytopics",
    "scale_color_tinytopics",
    "NumpyDiskDataset",
    "TorchDiskDataset",
    "fit_model",
    "fit_model_distributed",
    "poisson_nmf_loss",
    "NeuralPoissonNMF",
    "plot_loss",
    "plot_structure",
    "plot_top_terms",
    "align_topics",
    "generate_synthetic_data",
    "set_random_seed",
    "sort_documents",
]
