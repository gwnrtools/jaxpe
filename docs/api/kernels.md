---
title: kernels
parent: API Reference
layout: default
---

# `jaxpe.kernels`
{: .no_toc }

1. TOC
{:toc}

The `kernels` module provides the gradient-based local transition kernels used in the local phase of the global-local sampling loop.

## Available Kernels

### `MALA`

Metropolis-Adjusted Langevin Algorithm. A gradient-based MCMC algorithm that uses the gradient of the log-posterior to propose moves.

```python
class MALA:
    def __init__(self, step_size):
        """
        Initializes a MALA kernel with a given step size.
        """
        pass
```

### Other Kernels

- HMC (Hamiltonian Monte Carlo)
- mMALA (manifold MALA)
- Underdamped Langevin
- Gradient-free random walk (for fallback or specific use cases)

## Step-size and Mass Adaptation

The local kernels support step-size adaptation and mass matrix tuning to ensure optimal acceptance rates during the warm-up phase of the sampling.
