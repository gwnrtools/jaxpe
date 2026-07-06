---
title: kernels
parent: jaxpe
layout: default
---

# Sec. III: MCMC Kernels (`jaxpe.kernels`)
{: .no_toc }

1. TOC
{:toc}

How do we actually explore a 15-dimensional posterior landscape? If we simply guess randomly, we will spend the age of the universe wandering through the deserts of low probability. Instead, in this section, we introduce the MCMC kernels of `jaxpe` that leverage classical mechanics and fluid dynamics to glide through the parameter space.

## Hamilton's Equations and MCMC

Consider Hamiltonian Monte Carlo (HMC) [1, 2]. We stop treating inference as a statistical guessing game and start treating it as a physics simulation. Imagine the negative log-posterior as the physical landscape of a rolling hillside:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

We place a frictionless puck (our parameter vector) on this hill, and give it a random kick. It gains a conjugate momentum $$\mathbf{p}$$ and a kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The puck now traces a deterministic path along the contours of the total energy (the Hamiltonian):

$$
H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})
$$

governed beautifully by Hamilton's equations:

$$
\frac{d\boldsymbol{\theta}}{dt} = \mathbf{M}^{-1} \mathbf{p} \, , \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta})
$$

Because this continuous flow is time-reversible and preserves phase-space volume (thanks to Liouville's theorem), we can numerically integrate this trajectory for a long time—generating distant, independent proposals that are accepted with near certainty.

## Langevin Diffusion and MALA

Now, suppose we don't want a frictionless puck. Suppose we want to describe a particle diffusing through a thick, viscous fluid, constantly buffeted by thermal noise, but gently pulled downhill by gravity. This is the Metropolis-Adjusted Langevin Algorithm (MALA) [3].

MALA models the overdamped limit of Langevin diffusion. It employs the gradient to drift toward regions of high probability while taking random steps. The continuous-time stochastic differential equation (SDE) governing our parameter vector $$\boldsymbol{\theta}_t$$ is:

$$
d\boldsymbol{\theta}_t = \frac{1}{2} \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}_t|d) dt + dW_t
$$

where $$dW_t$$ is a standard Wiener process (a continuous random walk). But how do we know this process actually finds the posterior? We look at the Fokker-Planck equation, which describes how the entire probability density $$\rho(\boldsymbol{\theta}, t)$$ evolves over time:

$$
\frac{\partial \rho}{\partial t} = \nabla_{\boldsymbol{\theta}} \cdot \left[ \frac{1}{2} \nabla_{\boldsymbol{\theta}} \rho - \frac{1}{2} \rho \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d) \right]
$$

If you set $$\partial \rho / \partial t = 0$$, you will find that the stationary distribution is exactly our target posterior $$p(\boldsymbol{\theta}|d)$$. Discretizing the SDE via the Euler-Maruyama method gives us our proposal density:

$$
\boldsymbol{\theta}' \sim \mathcal{N}\left( \boldsymbol{\theta} + \frac{\epsilon^2}{2} \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d), \epsilon^2 \mathbf{I} \right)
$$

## Advanced Differential Geometry Kernels

### Manifold MALA (`mMALA`)

In highly curved posteriors, taking isotropic random steps ($$\epsilon^2 \mathbf{I}$$) is like trying to walk in a straight line on a steep, twisted mountain ridge. Manifold MALA introduces a position-dependent preconditioning matrix $$\mathbf{G}(\boldsymbol{\theta})$$ (such as the Fisher Information Matrix) to scale the drift and diffusion locally, adapting perfectly to the geometry of the target.

### Underdamped Langevin Dynamics (`ULD`)

ULD is the beautiful synthesis of HMC and MALA. We restore the momentum of our frictionless puck, but we add a friction term $$\gamma$$ and continuous thermal kicks. The momenta undergo an Ornstein-Uhlenbeck process, constantly refreshing, while the positions drift:

$$
d\boldsymbol{\theta}_t = \mathbf{M}^{-1} \mathbf{p}_t dt
$$
$$
d\mathbf{p}_t = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}_t) dt - \gamma \mathbf{M}^{-1} \mathbf{p}_t dt + \sqrt{2\gamma} \mathbf{M}^{1/2} dW_t
$$

This allows the chains to traverse highly correlated spaces robustly, without needing to perfectly tune the exact integration time of an HMC trajectory.

## Step-size and Mass Adaptation

Finally, we cannot expect to guess the optimal step size $$\epsilon$$ blindly. The local kernels support dual-averaging step-size adaptation to target an optimal acceptance rate (e.g., 65% for HMC, 57.4% for MALA), ensuring that our hikers are neither taking steps that are too timid nor steps so large they stumble.

### REFERENCES

[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[2] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of Markov Chain Monte Carlo **2**, 113 (2011).  
[3] G. O. Roberts and R. L. Tweedie, "Exponential convergence of Langevin distributions and their discrete approximations," Bernoulli **2**, 341 (1996).
