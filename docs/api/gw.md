---
title: gw
parent: API Reference
layout: default
---

# `jaxpe.gw`
{: .no_toc }

1. TOC
{:toc}

The `gw` module builds a frequency-domain likelihood from a user-supplied, JAX-differentiable time-domain waveform model and standard GW priors.

## Components

### `ToyChirp`

A toy time-domain waveform model used for testing and validation.

### `make_injection`

Creates a mock gravitational-wave likelihood from a model, a set of injection parameters, and a noise seed.

```python
def make_injection(model, params, noise_seed=42):
    """
    Creates an injection likelihood for testing.
    """
    pass
```

### GW Priors

The module provides standard BBH priors that are compatible with `jaxpe.core`.

```python
def bbh_priors(geocent_time):
    """
    Returns standard binary black hole (BBH) priors.
    """
    pass
```

## Precision and Floating Point

GW likelihoods are validated in float64 (`jax.config.update("jax_enable_x64", True)`). Any float32 fast path must use segment-relative times. Absolute GPS epochs are unrepresentable at float32 resolution.
