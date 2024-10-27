"""
Topic modeling via sum-to-one constrained neural Poisson NMF.

Modules:
    fit: Model fitting and loss calculation.
    models: NeuralPoissonNMF model definition.
    plot: Functions for plotting loss curves, document-topic distributions, and top terms.
    colors: Color palettes.
    utils: Utility functions for data generation, topic alignment, and document sorting.
"""

from .models import NeuralPoissonNMF
from .fit import fit_model, poisson_nmf_loss
from .utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)
from .colors import pal_tinytopics, scale_color_tinytopics
from .plot import plot_loss, plot_structure, plot_top_terms
