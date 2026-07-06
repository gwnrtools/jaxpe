---
title: gw
parent: jaxpe
layout: default
---

# Sec. II: Gravitational-Wave Physics (`jaxpe.gw`)
{: .no_toc }

1. TOC
{:toc}

In this section, we detail the underlying gravitational-wave physics module (`jaxpe.gw`), including the waveform construction, detector responses, and the frequency-domain likelihood.

## Waveform Construction

The generation of gravitational waves from perturbed black holes relies on evolving the Newman-Penrose scalar $$\Psi_4$$. As the waves propagate to the transverse-traceless (TT) gauge of our detectors on Earth, they manifest as the metric strain components $$h_+(t)$$ and $$h_\times(t)$$.

### `ToyChirp`

A toy time-domain waveform model used for testing and validation. It demonstrates the fundamental quadrupole radiation physics.

## The Frequency-Domain Likelihood

The `gw` module builds a frequency-domain likelihood from a user-supplied, JAX-differentiable time-domain waveform model and standard GW priors. Assuming the detector noise $$n(t)$$ is stationary and Gaussian with a one-sided power spectral density (PSD) $$S_n(f)$$, the likelihood of observing data $$d$$ given parameters $$\boldsymbol{\theta}$$ is governed by the Whittle likelihood in the frequency domain.

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
