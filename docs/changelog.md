# Changelog

## tinytopics 0.2.0

### New features

- Add `scale_color_tinytopics()` to support the coloring need for
  arbitrary number of topics (#4).

### Improvements

- Simplify hyperparameter tuning by adopting modern stochastic gradient methods.
  `fit_model()` now uses a combination of the AdamW optimizer (with weight
  decay) and the cosine annealing (with warm restarts) scheduler (#2).

## Bug fixes

- Fix "Structure plot" y-axis range issue by adding a `normalize_rows` argument
  to `plot_structure()` for normalizing rows so that they all sum exactly to 1,
  and explicitly setting the y-axis limit to [0, 1]. (#1).

### Documentation

- Add text data topic modeling example article (#7).

## tinytopics 0.1.3

### Improvements

- Reorder arguments in plotting functions to follow conventions.

## tinytopics 0.1.2

### Improvements

- Reduce the minimum version requirement for all dependencies in `pyproject.toml`.

### Documentation

- Add more details on PyTorch installation in `README.md`.
- Improve text quality in articles.

## tinytopics 0.1.1

### Improvements

- Add `CHANGELOG.md` to record changes.
- Add essential metadata to `pyproject.toml`.

## tinytopics 0.1.0

### New features

- First version.
