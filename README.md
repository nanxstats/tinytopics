# tinytopics <img src="docs/assets/logo.png" align="right" width="120" />

[![PyPI version](https://img.shields.io/pypi/v/tinytopics)](https://pypi.org/project/tinytopics/)
![Python versions](https://img.shields.io/pypi/pyversions/tinytopics)
[![CI Tests](https://github.com/nanxstats/tinytopics/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/nanxstats/tinytopics/actions/workflows/ci-tests.yaml)
[![mkdocs](https://github.com/nanxstats/tinytopics/actions/workflows/mkdocs.yml/badge.svg)](https://nanx.me/tinytopics/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tinytopics)](https://pypistats.org/packages/tinytopics)
![License](https://img.shields.io/pypi/l/tinytopics)

Topic modeling via sum-to-one constrained neural Poisson NMF.
Built with PyTorch, runs on both CPUs and GPUs.

## Installation

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

## GPU support

The above will install the CPU version of PyTorch by default. To enable GPU support,
follow the [PyTorch official guide](https://pytorch.org/get-started/locally/)
to install the appropriate PyTorch version.

For example, install PyTorch for Windows with CUDA 12.4:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To manage the PyTorch dependency under a project context using virtual
environments, you might want to set up manual sources. For example,
[using Rye](https://rye.astral.sh/guide/faq/#how-do-i-install-pytorch) or
[using uv](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Examples

After tinytopics is installed, try examples from:

- [Getting started guide with simulated count data](https://nanx.me/tinytopics/articles/get-started/)
- [CPU vs. GPU speed benchmark](https://nanx.me/tinytopics/articles/benchmark/)
- [Text data topic modeling example](https://nanx.me/tinytopics/articles/text/)
- [Memory-efficient training](https://nanx.me/tinytopics/articles/memory/)
- [Distributed training](https://nanx.me/tinytopics/articles/distributed/)
