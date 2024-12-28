"""
Topic modeling via sum-to-one constrained neural Poisson NMF.
"""

from .fit import fit_model
from .fit_distributed import fit_model_distributed
from .models import NeuralPoissonNMF
from .loss import poisson_nmf_loss
from .data import NumpyDiskDataset
from .utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)
from .colors import pal_tinytopics, scale_color_tinytopics
from .plot import plot_loss, plot_structure, plot_top_terms
