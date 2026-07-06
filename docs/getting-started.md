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

## Why JAX over PyTorch?

One might wonder if `jaxpe` would perform faster if rewritten in PyTorch. Based on the architecture of `jaxpe`, the package would likely not perform faster in PyTorch. For this specific type of algorithmic workload, JAX is generally considered the most performant framework available, for several key reasons:

### Zero-Overhead Vectorization (`vmap`)
The `jaxpe` algorithm relies heavily on running many MCMC chains in parallel. JAX was built from the ground up around `jax.vmap`. It effortlessly vectorizes complex scalar operations (like a single MCMC chain step evaluating a gravitational wave likelihood) across thousands of batch dimensions to saturate GPU cores. While PyTorch recently introduced `torch.vmap` via `torch.func`, JAX's implementation is older, universally supported across its ecosystem, and deeply integrated with its compiler.

### XLA JIT Compilation & Kernel Launch Overhead
MCMC sampling is inherently sequential: you must finish step $N$ to calculate step $N+1$. In a standard PyTorch loop, dispatching thousands of sequential operations to the GPU incurs massive Python/C++ kernel launch overhead, often leaving the GPU idle. JAX solves this with `jax.jit` and the XLA compiler. JAX can compile the *entire* MCMC transition step—including the gradient calculation, the likelihood evaluation, and the MALA/HMC proposal—into a single optimized GPU kernel. This nearly eliminates launch overhead, which is critical for MCMC workloads.

### Extremely Fast Gradients
Algorithms like MALA (Metropolis-Adjusted Langevin Algorithm) and HMC (Hamiltonian Monte Carlo) require computing the gradient of the log-likelihood at every single step. JAX's functional automatic differentiation (`jax.grad`) combined with XLA compilation is heavily optimized for this exact use case. PyTorch's standard dynamic autograd graph adds unnecessary overhead when repeatedly computing gradients for fixed-structure likelihoods.

### Ecosystem Synergy
`jaxpe` leverages `flowjax` for its normalizing flows. The interplay between the local MCMC steps and the global normalizing flow proposals benefits from being in the same functional, JIT-compiled ecosystem, allowing for highly fused training and sampling loops.

*(Note: PyTorch 2.0 introduced `torch.compile` and the `torch.func` library, closing the performance gap significantly. However, you would be fighting against PyTorch's default object-oriented/dynamic nature, whereas JAX naturally enforces the exact functional paradigms required to make this specific mathematical workload run as fast as possible on hardware accelerators.)*
