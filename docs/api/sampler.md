---
title: sampler
parent: jaxpe
layout: default
---

# Sec. IV: Global-Local Sampler (`jaxpe.sampler`)
{: .no_toc }

1. TOC
{:toc}

Imagine a fleet of intrepid explorers mapping a vast, rugged terrain. If they only follow the local slopes (using HMC or MALA), they will thoroughly map the valley they started in, but they will never realize there is a deeper, richer valley ten miles to the east. This is the multi-modal trap. 

To circumvent this topological obstruction, the `sampler` orchestrates a "global-local" loop [1]. We deploy an omniscient satellite—the Normalizing Flow—that observes the explorers and learns a global map of the entire landscape: $$q_\phi(\mathbf{x}) \approx \pi(\mathbf{x}|d)$$.

## Global-Local Orchestration

The sampler operates in two beautifully synchronized phases:
1. **Local Phase**: The parallel ensemble of explorers takes $N$ steps using a local gradient kernel, deeply investigating the micro-structure of their current valleys.
2. **Global Phase**: The satellite beams down coordinates, proposing independent global jumps for every chain directly from the learned global map: $$\mathbf{y} \sim q_\phi(\mathbf{y})$$. 

But we cannot just teleport our explorers blindly; that would violate the underlying physics (the true probability distribution) of the landscape. To strictly satisfy detailed balance (reversibility) and guarantee convergence to the exact target posterior, these global jumps are governed by the Metropolis-Hastings criterion. 

Recall the exact detailed balance condition, which demands that the flux of probability from state $$\mathbf{x}$$ to state $$\mathbf{y}$$ exactly equals the reverse flux:

$$
\pi(\mathbf{x}|d) T(\mathbf{x} \to \mathbf{y}) = \pi(\mathbf{y}|d) T(\mathbf{y} \to \mathbf{x})
$$

For an independence sampler where the proposal $$\mathbf{y}$$ does not depend at all on the current state $$\mathbf{x}$$, the transition kernel is simply $$T(\mathbf{x} \to \mathbf{y}) = q_\phi(\mathbf{y})$$. To enforce balance, we introduce an acceptance probability $$\alpha$$, yielding the famous Metropolis-Hastings ratio:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y}|d) q_\phi(\mathbf{x})}{\pi(\mathbf{x}|d) q_\phi(\mathbf{y})}\right)
$$

Because our satellite map $$q_\phi$$ is a highly accurate reflection of the true global mode structure, the ratio $$\pi/q_\phi$$ is very close to 1 across the entire parameter space. This ensures that the acceptance probability remains extraordinarily high even when an explorer is teleported completely across the parameter space to a disjoint mode.

## `Sampler`

The main class orchestrating the local kernels and the global normalizing flow proposals. Under the hood, it leverages JAX's `lax.scan` for a highly efficient, statically compiled orchestration loop.

## `GlobalLocalConfig`

The master configuration object controlling the number of chains, neural network architecture, adaptation length, and phase scheduling.

## Initialization

You might ask, "Where do we drop our explorers in the first place?" Initialization is absolutely critical. If we drop them all in the same valley, the satellite will only ever learn about that one valley. 

### `best_of_prior_init`

To prevent initial mode collapse, we evaluate the log-likelihood over a massive batch (e.g., $$10^6$$) of prior draws. We then select the $$N_{\text{chains}}$$ highest-probability candidates to seed the initial chains. This explicitly ensures that all valleys with significant prior support are populated with explorers from step zero.

### REFERENCES

[1] K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).  
[2] L. Tierney, "Markov Chains for Exploring Posterior Distributions," Ann. Stat. **22**, 1701-1728 (1994).
