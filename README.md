# tinytopics <img src="docs/assets/logo.png" align="right" width="120" />

[![PyPI version](https://img.shields.io/pypi/v/tinytopics)](https://pypi.org/project/tinytopics/)
![License](https://img.shields.io/pypi/l/tinytopics)
![Python versions](https://img.shields.io/pypi/pyversions/tinytopics)

Topic modeling via sum-to-one constrained Poisson non-negative
matrix factorization (NMF), built on PyTorch and runs on both CPUs and GPUs.

## Installation

First, [install PyTorch](https://pytorch.org/get-started/locally/).
To run on Nvidia GPUs, install a CUDA-enabled version on Linux or Windows.

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

After tinytopics is installed, try the example from the
[getting started guide](https://nanx.me/tinytopics/articles/get-started/)
or see the [speed benchmark](https://nanx.me/tinytopics/articles/benchmark/).
