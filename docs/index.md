# tinytopics <img src="assets/logo.png" align="right" width="120" />

Topic modeling via sum-to-one constrained Poisson non-negative
matrix factorization (NMF), built on PyTorch and runs on GPU.

## Installation

First, install a CUDA-enabled version of
[PyTorch](https://pytorch.org/get-started/locally/) and run with an Nvidia GPU.
At the moment, only the Linux and Windows versions of PyTorch support CUDA.

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

Try [getting started](articles/get-started.md).

## Known issues

- [ ] Running on CPU produces different (and worse) models than on GPU.
