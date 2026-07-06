---
title: core
parent: API Reference
layout: default
---

# `jaxpe.core`
{: .no_toc }

1. TOC
{:toc}

The `core` module provides the foundational interfaces for defining an inference problem in `jaxpe`, including priors, unconstraining transforms, and the `InferenceProblem` abstract interface.

## Interface

The typical entrypoint for a problem definition is implementing an `InferenceProblem`.

### `InferenceProblem`

An abstract base class that encapsulates the log-likelihood and prior of a specific model.

```python
class InferenceProblem:
    @property
    def ndim(self) -> int:
        """Dimensionality of the unconstrained parameter space."""
        pass

    def log_prob(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate the log-posterior density for a given unconstrained parameter vector.
        """
        pass
```

### Transforms

`jaxpe` operates entirely in an unconstrained parameter space ($\mathbb{R}^D$). The `core` package provides transforms to map bounded physical priors (e.g. uniform $[a, b]$, periodic) into $\mathbb{R}^D$.
