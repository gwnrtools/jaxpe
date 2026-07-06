---
title: kernels
parent: jaxpe
layout: default
---

# Sec. III: MCMC Kernels (`jaxpe.kernels`)
{: .no_toc }

1. TOC
{:toc}

In this section, we describe the suite of MCMC kernels and the mathematical mechanics of Hamiltonian and Langevin dynamics used to traverse the parameter manifold.

## Hamilton's Equations and MCMC

Standard stochastic random walks scale poorly in high-dimensional spaces. Hamiltonian Monte Carlo (HMC) [1, 2] elevates the statistical problem into a classical mechanics simulation. We treat the negative log-posterior as a potential energy well:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

and assign a conjugate momentum $$\mathbf{p}$$ with kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The system evolves along contours of the Hamiltonian $$H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})$$ according to Hamilton's equations. Because this continuous flow preserves phase-space volume (Liouville's theorem) and conserves energy, the deterministic leapfrog integration generates distant proposals across the correlated posterior manifold with near-perfect acceptance rates.

## Langevin Diffusion and MALA

The Metropolis-Adjusted Langevin Algorithm (MALA) [3] models the overdamped limit of Langevin diffusion. It employs the gradient of the log-posterior to drift toward regions of high probability. The continuous-time stochastic differential equation (SDE) governing the parameter vector is:

$$
d\boldsymbol{\theta}_t = \frac{1}{2} \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}_t|d) dt + dW_t
$$

where $$W_t$$ is a standard Wiener process (Brownian motion). The corresponding Fokker-Planck equation shows that the stationary distribution of this process is exactly the posterior $$p(\boldsymbol{\theta}|d)$$. Discretizing this SDE via the Euler-Maruyama method with step size $$\epsilon$$ yields the proposal density:

$$
\boldsymbol{\theta}' \sim \mathcal{N}\left( \boldsymbol{\theta} + \frac{\epsilon^2}{2} \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d), \epsilon^2 \mathbf{I} \right)
$$

This proposal is then subjected to a standard Metropolis-Hastings acceptance step to correct for time-discretization errors.

## Advanced Differential Geometry Kernels

### Manifold MALA (`mMALA`)

In highly curved posteriors, an isotropic step size $$\epsilon^2 \mathbf{I}$$ is inefficient. Manifold MALA uses a position-dependent preconditioning matrix $$\mathbf{G}(\boldsymbol{\theta})$$ (such as the Fisher Information Matrix or the inverse Hessian) to scale the drift and diffusion terms locally, adapting to the geometry of the target density.

### Underdamped Langevin Dynamics (`ULD`)

ULD interpolates between HMC and MALA by adding a friction term $$\gamma$$ to the Hamiltonian system. The momenta undergo an Ornstein-Uhlenbeck process, continuously refreshing while the positions drift deterministically:

$$
d\boldsymbol{\theta}_t = \mathbf{M}^{-1} \mathbf{p}_t dt
$$
$$
d\mathbf{p}_t = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}_t) dt - \gamma \mathbf{M}^{-1} \mathbf{p}_t dt + \sqrt{2\gamma} \mathbf{M}^{1/2} dW_t
$$

This approach can traverse highly correlated spaces while being robust to the exact choice of integration time.

## Step-size and Mass Adaptation

The local kernels support dual-averaging step-size adaptation to target a specific acceptance rate (e.g., 65% for HMC, 57% for MALA) and mass matrix tuning to pre-condition the parameter space during the warm-up phase.

### REFERENCES

[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[2] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of Markov Chain Monte Carlo **2**, 113 (2011).  
[3] G. O. Roberts and R. L. Tweedie, "Exponential convergence of Langevin distributions and their discrete approximations," Bernoulli **2**, 341 (1996).
