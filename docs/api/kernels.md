---
title: kernels
parent: jaxpe
layout: default
---

# Sec. III: MCMC Kernels (`jaxpe.kernels`)
{: .no_toc }

1. TOC
{:toc}

In this section, we describe the suite of MCMC kernels and the mathematical mechanics of Hamiltonian and Langevin dynamics.

## Hamilton's Equations and MCMC

Standard stochastic random walks scale poorly in high-dimensional spaces. Hamiltonian Monte Carlo (HMC) elevates the statistical problem into a classical mechanics simulation. We treat the negative log-posterior as a potential energy well:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

and assign a conjugate momentum $$\mathbf{p}$$ with kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The system evolves along contours of the Hamiltonian $$H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})$$ according to Hamilton's equations. Because this flow preserves phase-space volume and conserves energy, the deterministic leapfrog integration generates distant proposals across the correlated posterior manifold with near-perfect acceptance rates.

## Available Kernels

### `MALA`

The Metropolis-Adjusted Langevin Algorithm models the overdamped limit of Langevin diffusion. It employs the gradient of the log-posterior to drift toward regions of high probability.

### `HMC`

Hamiltonian Monte Carlo implements the leapfrog symplectic integrator to solve Hamilton's equations of motion.

### `mMALA` & `ULD`

Manifold MALA and Underdamped Langevin dynamics provide alternative differential geometric approaches for diffusion over the posterior landscape.

## Step-size and Mass Adaptation

The local kernels support step-size adaptation and mass matrix tuning to ensure optimal acceptance rates during the warm-up phase of the sampling.
