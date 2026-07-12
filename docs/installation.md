---
title: Installation
layout: default
nav_order: 2
---

# Installation
{: .no_toc }

1. TOC
{:toc}

## Requirements

`jaxpe` relies on JAX and its ecosystem for automatic differentiation, GPU acceleration, and vectorization.

- [JAX](https://github.com/google/jax)
- [flowjax](https://github.com/danielward27/flowjax) — Rational-quadratic-spline coupling flows
- standard scientific stack: `numpy`, `scipy`, `matplotlib`

## Installing via Conda (Recommended)

For the most robust installation—especially when dealing with complex dependencies like `lalsuite` and GPU acceleration—we recommend using the provided Conda environment files. These will automatically configure `lalsuite` through `conda-forge` and install `jaxpe` with its optional dependencies (`dev`, `gwdata`, and `surrogate`) using `pip`.

**For CPU:**
```bash
conda env create -f conda.yml
conda activate jaxpe
```

**For GPU (NVIDIA):**
This environment specifically prioritizes the official `jax[cuda12]` binaries to ensure hardware acceleration is properly linked before installing `jaxpe`.
```bash
conda env create -f conda-gpu.yml
conda activate jaxpe-gpu
```

## Installing from source (pip only)

Alternatively, the package can be installed manually via `pip`:

```bash
git clone https://github.com/jaxpe/jaxpe.git
cd jaxpe
pip install -e .
```

Ensure that you have installed the correct JAX version with GPU support (if a GPU is available) by following the [JAX installation instructions](https://github.com/google/jax#installation).

## GPU Memory Allocation

By default, JAX pre-allocates 90% of the available GPU memory. When running heavily vmapped applications or using large batch sizes (like in GW PE), this might need tuning.

If you are running on smaller GPUs or alongside other workloads, you may need to restrict JAX's memory allocation:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

This prevents on-demand allocation fragmentation which can sometimes cause issues mid-run.
