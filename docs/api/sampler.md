---
title: sampler
parent: jaxpe
layout: default
---

# Sec. IV: Global-Local Sampler (`jaxpe.sampler`)
{: .no_toc }

1. TOC
{:toc}

Standard local MCMC kernels (like HMC or MALA) are extraordinarily efficient at exploring the local geometry of a single posterior mode. However, the true gravitational-wave posterior is notoriously multi-modal due to physical degeneracies (e.g., sky position rings, mass-spin correlations, and phase inversions). Local gradients provide no information about these disconnected modes, trapping the chains locally forever.

## Global-Local Orchestration

To circumvent this topological obstruction, the `sampler` orchestrates a "global-local" loop [1]. Once the Normalizing Flow is trained on accumulated MCMC samples, it acts as an analytic proxy for the target posterior: $$q_\phi(\mathbf{x}) \approx \pi(\mathbf{x}|d)$$. 

The sampler operates in two alternating phases:
1. **Local Phase**: The parallel ensemble of chains takes $N$ steps using a local gradient kernel (e.g., HMC), deeply exploring their current local mode.
2. **Global Phase**: The sampler draws independent proposals for every chain directly from the global flow: $$\mathbf{y} \sim q_\phi(\mathbf{y})$$. 

To strictly satisfy detailed balance (reversibility) and guarantee convergence to the exact target posterior, these global jumps are accepted or rejected via the Metropolis-Hastings criterion. For an independence sampler where the proposal $$\mathbf{y}$$ does not depend on the current state $$\mathbf{x}$$, the transition kernel is simply $$T(\mathbf{x} \to \mathbf{y}) = q_\phi(\mathbf{y})$$. The required acceptance probability is:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y}|d) T(\mathbf{y} \to \mathbf{x})}{\pi(\mathbf{x}|d) T(\mathbf{x} \to \mathbf{y})}\right) = \min\left(1, \frac{\pi(\mathbf{y}|d) q_\phi(\mathbf{x})}{\pi(\mathbf{x}|d) q_\phi(\mathbf{y})}\right)
$$

Because the learned density $$q_\phi$$ accurately approximates the global disconnected mode structure, the ratio $$\pi/q_\phi$$ is very close to 1 across the entire parameter space. This ensures that the acceptance probability remains high even when a chain is teleported completely across the parameter space to a disjoint mode.

## `Sampler`

The main class orchestrating the local kernels and the global normalizing flow proposals via JAX's `lax.scan` for highly efficient, compiled looping.

## `GlobalLocalConfig`

Configuration object controlling the number of chains, neural network architecture, adaptation length, and phase scheduling.

## Initialization

Initialization is critical for highly multimodal posteriors (like GW PE). 

### `best_of_prior_init`

Evaluates the log-likelihood over a massive batch (e.g., $$10^6$$) of prior draws and selects the $$N_{\text{chains}}$$ highest-probability candidates to seed the initial chains. This explicitly prevents mode collapse during the initial warm-up phase by ensuring all modes with significant prior support are populated.

### REFERENCES

[1] K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).  
[2] L. Tierney, "Markov Chains for Exploring Posterior Distributions," Ann. Stat. **22**, 1701-1728 (1994).
