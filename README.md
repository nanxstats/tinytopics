# tinytopics <img src="https://github.com/nanxstats/tinytopics/raw/main/docs/assets/logo.png" align="right" width="120" />

[![PyPI version](https://img.shields.io/pypi/v/tinytopics)](https://pypi.org/project/tinytopics/)
![Python versions](https://img.shields.io/pypi/pyversions/tinytopics)
[![CI tests](https://github.com/nanxstats/tinytopics/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/nanxstats/tinytopics/actions/workflows/ci-tests.yml)
[![Mypy check](https://github.com/nanxstats/tinytopics/actions/workflows/mypy.yml/badge.svg)](https://github.com/nanxstats/tinytopics/actions/workflows/mypy.yml)
[![Ruff check](https://github.com/nanxstats/tinytopics/actions/workflows/ruff-check.yml/badge.svg)](https://github.com/nanxstats/tinytopics/actions/workflows/ruff-check.yml)
[![Documentation](https://github.com/nanxstats/tinytopics/actions/workflows/docs.yml/badge.svg)](https://nanx.me/tinytopics/)
![License](https://img.shields.io/pypi/l/tinytopics)

Topic modeling via sum-to-one constrained neural Poisson NMF.
Built with PyTorch, runs on both CPUs and GPUs.

## Installation

### Using pip

You can install tinytopics from PyPI:

```bash
pip install tinytopics
```

Or install the development version from GitHub:

```bash
git clone https://github.com/nanxstats/tinytopics.git
cd tinytopics
python3 -m pip install -e .
```

### Using uv (recommended)

For a more robust package management experience, use
[uv](https://docs.astral.sh/uv/) to manage tinytopics as a project dependency.

Add tinytopics and PyTorch to your project:

```bash
uv add tinytopics torch torchvision
```

To install PyTorch with GPU support (for example, Windows with CUDA 13.0),
configure `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Then sync your environment:

```bash
uv sync
```

For other platforms and accelerators (CPU-only, ROCm, Intel GPUs), see
[Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Examples

After tinytopics is installed, try examples from:

- [Getting started guide with simulated count data](https://nanx.me/tinytopics/articles/get-started/)
- [CPU vs. GPU speed benchmark](https://nanx.me/tinytopics/articles/benchmark/)
- [Text data topic modeling example](https://nanx.me/tinytopics/articles/text/)
- [Memory-efficient training](https://nanx.me/tinytopics/articles/memory/)
- [Distributed training](https://nanx.me/tinytopics/articles/distributed/)
