# Changelog

## tinytopics 0.7.0

### New features

- Add `fit_model_distributed()` to support distributed training using
  Hugging Face Accelerate.
  See the [distributed training](https://nanx.me/tinytopics/articles/distributed/)
  article for details (#32).

### Improvements

- Use `tqdm.auto` for better progress bar visuals when used in notebooks (#30).
- Move dataset classes and loss functions into dedicated modules to improve
  code structure and reusability (#31).

## tinytopics 0.6.0

### New features

- `fit_model()` now supports using PyTorch `Dataset` as input, in addition
  to in-memory tensors. This allows fitting topic models on data larger than
  GPU VRAM or system RAM. The `NumpyDiskDataset` class is added to read
  `.npy` document-term matrices from disk on-demand (#26).

### Documentation

- Added a [memory-efficient training](https://nanx.me/tinytopics/articles/memory/)
  article demonstrating the new features for fitting topic models on
  large datasets (#27).

## tinytopics 0.5.1

### Documentation

- Add badges for CI tests and mkdocs workflows to `README.md` (#24).
- Add PyTorch management guide link for uv to `README.md` (735fcca).

### Maintenance

- Use hatchling 1.26.3 in `pyproject.toml` to work around `rye publish`
  errors (c56387c).

## tinytopics 0.5.0

### Improvements

- Increased the speed of `generate_synthetic_data()` significantly by using
  direct mixture sampling, which leverages the properties of multinomial
  distributions (#21).

    This change makes simulating data at the scale of 100K x 100K
    more feasible. Although the approaches before and after are mathematically
    equivalent, the data generated with the same seed in previous versions and
    this version onward will be bitwise different.

## tinytopics 0.4.1

### Documentation

- Use `pip` and `python3` in command line instructions consistently.

## tinytopics 0.4.0

### Breaking changes

- tinytopics now requires Python >= 3.10 to use PEP 604 style shorthand syntax
  for union and optional types (#14).

### Typing

- Refactor type hints to use more base abstract classes, making them less
  limiting to specific implementations (#14).

### Testing

- Add unit tests for all functions using pytest, with a GitHub Actions workflow
  to run tests under Linux and Windows (#18).

### Improvements

- Update articles to simplify import syntax using `import tinytopics as tt` (#16).
- Close precise figure handles in plot functions instead of the current figure (#18).

### Bug fixes

- Plot functions now correctly use string and list type color palette inputs
  when specified (do not call them as functions) (#18).

## tinytopics 0.3.0

### Improvements

- Refactor the code to use a more functional style and add type hints
  to improve code clarity (#9).

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
