---
title: sampler
parent: jaxpe
layout: default
---

# Sec. IV: Global-Local Sampler (`jaxpe.sampler`)
{: .no_toc }

1. TOC
{:toc}

Standard local MCMC kernels (like HMC or MALA) are extraordinarily efficient at exploring the local geometry of a single posterior mode. However, the true gravitational-wave posterior is notoriously multi-modal. Local gradients provide no information about disconnected modes, trapping the chains.

## Global-Local Orchestration

To circumvent this topological obstruction, the `sampler` orchestrates a "global-local" loop. Once the Normalizing Flow is trained on accumulated MCMC samples, it acts as an analytic proxy for the posterior: $$q_\phi(\mathbf{x}) \approx \pi(\mathbf{x}|d)$$. 

We then propose independent global jumps $$\mathbf{y} \sim q_\phi(\mathbf{y})$$ and accept them via the Metropolis-Hastings criterion:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y}|d) q_\phi(\mathbf{x})}{\pi(\mathbf{x}|d) q_\phi(\mathbf{y})}\right)
$$

Because the learned density $$q_\phi$$ accurately approximates the disconnected mode structure, this acceptance probability remains high even for inter-modal jumps.

## `Sampler`

The main class that orchestrates the local kernels and the global normalizing flow proposals.

## `GlobalLocalConfig`

Configuration object controlling the number of chains, neural network architecture, and adaptation.

## Initialization

Initialization is critical for highly multimodal posteriors (like GW PE). `best_of_prior_init` evaluates the log-likelihood over a large number of prior draws and selects the best candidates to seed the chains, preventing mode collapse.
