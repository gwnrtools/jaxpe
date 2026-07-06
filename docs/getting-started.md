---
title: Getting Started
layout: default
nav_order: 3
---

# Getting Started
{: .no_toc }

1. TOC
{:toc}

## Quick start (GW injection)

This script demonstrates setting up an end-to-end gravitational-wave injection and parameter estimation using `jaxpe`.

```python
import jax
# GW likelihoods are validated in float64
jax.config.update("jax_enable_x64", True)

from jaxpe.gw import ToyChirp, bbh_priors, make_injection
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

# Create a mock likelihood with a toy chirp model
like = make_injection(ToyChirp(20.0), injection_params, noise_seed=42)
problem = like.problem(bbh_priors(geocent_time=t_trigger))

# Configure the Sampler
cfg = GlobalLocalConfig(n_chains=64)          # scale chains to GPU memory
sampler = Sampler(MALA(0.05), problem=problem, config=cfg)
key = jax.random.PRNGKey(0)

# Initialization matters!
# GW posteriors occupy a tiny fraction of the prior volume.
# best_of_prior_init seeds every comparable-likelihood mode.
x0 = best_of_prior_init(key, problem, cfg.n_chains)

# Run the sampler
result = sampler.run(key, x0=x0)

# Transform back to physical parameter space
samples = sampler.to_physical(result.samples)
```

## Practical Notes

- **Two-detector sky ring**: with H1+L1 only, the time-delay ring makes the sky posterior an extended multimodal ridge; give the sampler generous training loops and global steps, and check `result.global_acceptance` (healthy: ~0.1+ for GW, ~0.8 on unimodal toys) and split R-hat.
- See `examples/03_gw_injection.py` for a full script.
- See `examples/validate_injection_vs_dynesty.py` for a cross-check against bilby+dynesty on the identical likelihood.
