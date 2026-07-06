---
title: sampler
parent: API Reference
layout: default
---

# `jaxpe.sampler`
{: .no_toc }

1. TOC
{:toc}

The `sampler` module provides the orchestration loop for the flowMC-style global-local sampling algorithm.

## `Sampler`

The main class that orchestrates the local kernels and the global normalizing flow proposals.

```python
class Sampler:
    def __init__(self, local_kernel, problem, config):
        """
        Initialize the Sampler.
        """
        pass

    def run(self, key, x0):
        """
        Execute the sampling loop.
        """
        pass

    def to_physical(self, samples):
        """
        Transform the raw unconstrained samples back into the physical space.
        """
        pass
```

## `GlobalLocalConfig`

Configuration object for the `Sampler`.

```python
class GlobalLocalConfig:
    def __init__(self, n_chains):
        """
        Configure the number of chains and other hyper-parameters.
        """
        pass
```

## Initialization

Initialization is critical for highly multimodal posteriors (like GW PE).

### `best_of_prior_init`

Evaluates the log-likelihood over a large number of prior draws and selects the best candidates to seed the chains, preventing mode collapse.

```python
def best_of_prior_init(key, problem, n_chains):
    """
    Return `n_chains` initial positions drawn from the prior, weighted by likelihood.
    """
    pass
```
