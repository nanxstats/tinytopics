# Get started


<!-- `.md` and `.py` files are generated from the `.qmd` file. Please edit that file. -->

!!! tip

    To run the code from this article as a Python script:

    ```bash
    python3 examples/get-started.py
    ```

## Introduction

Fitting topic models at scale using classical algorithms on CPUs can be
slow. Carbonetto et al. (2022) demonstrated the equivalence between
Poisson NMF and multinomial topic model likelihoods, and proposed a
novel optimization strategy: fit a Poisson NMF via coordinate descent,
then recover the corresponding topic model through a simple
transformation. This method was implemented in their R package,
[fastTopics](https://cran.r-project.org/package=fastTopics).

Building on this theoretical insight, tinytopics adopts an alternative
approach by directly solving a sum-to-one constrained neural Poisson
NMF, optimized using stochastic gradient methods, implemented in
PyTorch. Although this approach may not have the same theoretical
guarantees, it does have a few potential practical benefits:

- Scalable: Runs efficiently on both CPUs and GPUs and enables
  large-scale topic modeling tasks.
- Extensible: The model architecture is flexible and can be extended,
  for example, by adding regularization or integrating with other data
  modalities.
- Minimal: The core implementation is kept simple and readable,
  reflecting the package name: **tiny**topics.

This article shows a canonical tinytopics workflow using a simulated
dataset.

## Import tinytopics

``` python
from tinytopics.fit import fit_model
from tinytopics.plot import plot_loss, plot_structure, plot_top_terms
from tinytopics.utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)
```

## Generate synthetic data

Set random seed for reproducibility:

``` python
set_random_seed(42)
```

Generate a synthetic dataset:

``` python
n, m, k = 5000, 1000, 10
X, true_L, true_F = generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)
```

## Fit topic model

Fit the topic model and plot the loss curve. There will be a progress
bar.

``` python
model, losses = fit_model(X, k)

plot_loss(losses, output_file="loss.png")
```

![](images/loss.png)

!!! tip

    The performance of the model can be sensitive to the learning rate.
    If you experience suboptimal results or observe performance discrepancies
    between the model trained on CPU and GPU, tuning the learning rate can help.

    For example, using the default learning rate of 0.001 on this synthetic
    dataset can lead to inconsistent results between devices (worse model
    on CPU than GPU). Increasing the learning rate towards 0.01
    improves model fit and ensures consistent performance across both devices.

## Post-process results

Get the learned L and F matrices from the fitted topic model:

``` python
learned_L = model.get_normalized_L().numpy()
learned_F = model.get_normalized_F().numpy()
```

To make it easier to inspect the results visually, we should try to
“align” the learned topics with the ground truth topics by their terms
similarity.

``` python
aligned_indices = align_topics(true_F, learned_F)
learned_F_aligned = learned_F[aligned_indices]
learned_L_aligned = learned_L[:, aligned_indices]
```

Sort the documents in both the true document-topic matrix and the
learned document-topic matrix, grouped by dominant topics.

``` python
sorted_indices = sort_documents(true_L)
true_L_sorted = true_L[sorted_indices]
learned_L_sorted = learned_L_aligned[sorted_indices]
```

!!! note

    Most of the alignment and sorting steps only apply to simulations
    because we don't know the ground truth L and F for real datasets.

## Visualize results

We can use a “Structure plot” to visualize and compare the
document-topic distributions.

``` python
plot_structure(
    true_L_sorted,
    normalize_rows=True,
    title="True document-topic distributions (sorted)",
    output_file="L-true.png",
)
```

![](images/L-true.png)

``` python
plot_structure(
    learned_L_sorted,
    normalize_rows=True,
    title="Learned document-topic distributions (sorted and aligned)",
    output_file="L-learned.png",
)
```

![](images/L-learned.png)

We can also plot the top terms for each topic using bar charts.

``` python
plot_top_terms(
    true_F,
    n_top_terms=15,
    title="Top terms per topic - true F matrix",
    output_file="F-top-terms-true.png",
)
```

![](images/F-top-terms-true.png)

``` python
plot_top_terms(
    learned_F_aligned,
    n_top_terms=15,
    title="Top terms per topic - learned F matrix (aligned)",
    output_file="F-top-terms-learned.png",
)
```

![](images/F-top-terms-learned.png)

## References

Carbonetto, P., Sarkar, A., Wang, Z., & Stephens, M. (2021).
Non-negative matrix factorization algorithms greatly improve topic model
fits. arXiv Preprint arXiv:2105.13440.
