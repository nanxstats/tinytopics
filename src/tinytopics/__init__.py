"""
Topic modeling via sum-to-one constrained Poisson non-negative matrix factorization (NMF).

Modules:
    models: NeuralPoissonNMF model definition.
    fit: Model fitting and loss calculation.
    utils: Utility functions for data generation, topic alignment, and document sorting.
    plot: Functions for plotting loss curves, document-topic distributions, and top terms.
"""

from .models import NeuralPoissonNMF
from .fit import fit_model, poisson_nmf_loss
from .utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)
from .plot import plot_loss, plot_structure, plot_top_terms
