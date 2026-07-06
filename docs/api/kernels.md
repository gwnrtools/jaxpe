---
title: kernels
parent: jaxpe
layout: default
nav_order: 3
---

# Sec. III: Symplectic Geometry and Langevin Diffusion (`jaxpe.kernels`)
{: .no_toc }

1. TOC
{:toc}

How do we actually explore a 15-dimensional posterior landscape? Standard random-walk Metropolis-Hastings proposals explore space via diffusive Brownian motion, resulting in an exploration radius that scales as \\(R \propto \sqrt{N}\\) after \\(N\\) steps. This is disastrously inefficient in high dimensions. In this section, we formulate the MCMC kernels of `jaxpe` that leverage the differential geometry of symplectic manifolds to glide seamlessly through the parameter space.

## Hamiltonian Monte Carlo on Symplectic Manifolds

We elevate the statistical problem of drawing samples from a posterior \\(\pi(\theta^\mu|d)\\) into a deterministic simulation of classical mechanics [1, 2]. Let our parameter space be a Riemannian manifold \\(\mathcal{M}\\) with coordinates \\(\theta^\mu\\). We consider the cotangent bundle \\(T^*\mathcal{M}\\), which is naturally a symplectic manifold equipped with a closed, non-degenerate 2-form \\(\omega = d\theta^\mu \wedge dp_\mu\\), where \\(p_\mu\\) are the conjugate momenta.

We define the Hamiltonian scalar function \\(H: T^*\mathcal{M} \to \mathbb{R}\\) as the sum of the potential energy (the negative log-posterior) and the kinetic energy:

$$
H(\theta^\mu, p_\mu) = U(\theta^\mu) + K(p_\mu) = -\log \pi(\theta^\mu|d) + \frac{1}{2} g^{\mu\nu} p_\mu p_\nu
$$

where \\(g^{\mu\nu}\\) is the inverse of the mass matrix (which can be interpreted as a flat Riemannian metric on \\(\mathcal{M}\\)). The evolution of any observable \\(F\\) on phase space is governed by the Poisson bracket:

$$
\frac{dF}{dt} = \{F, H\} = \frac{\partial F}{\partial \theta^\mu} \frac{\partial H}{\partial p_\mu} - \frac{\partial F}{\partial p_\mu} \frac{\partial H}{\partial \theta^\mu}
$$

Setting \\(F = \theta^\mu\\) and \\(F = p_\mu\\) recovers Hamilton's equations of motion:

$$
\frac{d\theta^\mu}{dt} = g^{\mu\nu} p_\nu \, , \quad \frac{dp_\mu}{dt} = -\partial_\mu U(\theta)
$$

Because Hamiltonian flow is a symplectomorphism, it preserves the phase space volume form \\(\Omega = \frac{(-1)^{n(n-1)/2}}{n!} \omega^n\\) exactly (Liouville's Theorem).

### Numerical Integration: The Leapfrog Symplectic Integrator

In practice, we cannot integrate these continuous Hamiltonian differential equations exactly. We must discretize time into discrete steps \\(\Delta t\\). However, standard integration schemes like Runge-Kutta do not preserve the symplectic volume form \\(\omega\\), causing the simulated energy to systematically drift and destroying the detailed balance condition of the Markov chain.

Instead, we employ the Leapfrog (Verlet) integrator, which is explicitly symplectic and time-reversible. A single Leapfrog step consists of a half-step update for the momenta, a full-step update for the positions, and a final half-step update for the momenta:

$$
p_\mu(t + \Delta t/2) = p_\mu(t) - \frac{\Delta t}{2} \partial_\mu U(\theta(t))
$$
$$
\theta^\mu(t + \Delta t) = \theta^\mu(t) + \Delta t \, g^{\mu\nu} p_\nu(t + \Delta t/2)
$$
$$
p_\mu(t + \Delta t) = p_\mu(t + \Delta t/2) - \frac{\Delta t}{2} \partial_\mu U(\theta(t + \Delta t))
$$

Because this integrator preserves phase-space volume, the only source of error is the truncation error in the energy conservation \\(\mathcal{O}(\Delta t^2)\\). We correct for this small discretization error by wrapping the trajectory in a final Metropolis-Hastings acceptance step.

## Langevin Diffusion and the Fokker-Planck Equation

Suppose we abandon the frictionless determinism of HMC and instead immerse a particle in a heat bath subject to a potential gradient. This is the Metropolis-Adjusted Langevin Algorithm (MALA) [3], representing the overdamped limit of Langevin diffusion.

The continuous-time stochastic differential equation (SDE) governing the parameter vector \\(\theta^\mu_t\\) on a flat manifold is:

$$
d\theta^\mu_t = -\frac{1}{2} g^{\mu\nu} \partial_\nu U(\theta_t) dt + \sqrt{g^{\mu\nu}} dW_{\nu, t}
$$

where \\(dW_{\nu, t}\\) is a multi-dimensional Wiener process. The macroscopic evolution of the probability density \\(\rho(\theta^\mu, t)\\) is governed by the Fokker-Planck equation:

$$
\frac{\partial \rho}{\partial t} = \partial_\mu \left( \frac{1}{2} g^{\mu\nu} \partial_\nu \rho + \frac{1}{2} \rho g^{\mu\nu} \partial_\nu U \right)
$$

Imposing stationarity (\\(\partial_t \rho = 0\\)), we trivially recover that the equilibrium distribution is exactly the target posterior \\(\rho \propto \exp(-U(\theta)) = \pi(\theta|d)\\).

### Manifold MALA (`mMALA`)

In highly curved posteriors, using a globally constant metric \\(g^{\mu\nu}\\) is severely sub-optimal. Manifold MALA (mMALA) promotes the mass matrix to a position-dependent Riemannian metric tensor \\(g_{\mu\nu}(\theta)\\) (often the Fisher Information Matrix). The generalized Langevin SDE must now include corrections from the Levi-Civita connection (the Christoffel symbols \\(\Gamma^\mu_{\alpha\beta}\\)) to remain covariant:

$$
d\theta^\mu_t = \frac{1}{2} g^{\mu\nu}(\theta_t) \partial_\nu \log \pi(\theta_t) dt + \frac{1}{2} \Gamma^\mu_{\alpha\beta} g^{\alpha\beta} dt + e^\mu_\alpha dW^\alpha_t
$$

where \\(e^\mu_\alpha\\) is the vielbein (tetrad) defined by \\(g^{\mu\nu} = e^\mu_\alpha e^\nu_\beta \delta^{\alpha\beta}\\). This allows the diffusion to adapt perfectly to the local curvature of the target density.

### Underdamped Langevin Dynamics (`ULD`)

ULD restores the conjugate momenta to the diffusion process, adding a continuous friction tensor \\(\Gamma^\mu_\nu\\) to Hamilton's equations. The momenta undergo an Ornstein-Uhlenbeck process while the positions drift:

$$
d\theta^\mu_t = g^{\mu\nu} p_{\nu, t} dt
$$
$$
dp_{\mu, t} = -\partial_\mu U(\theta_t) dt - \Gamma^\alpha_\mu p_{\alpha, t} dt + \sqrt{2 \Gamma_{\mu\alpha}} dW^\alpha_t
$$

This continuous momentum refreshment robustly navigates strongly correlated spaces without the strict requirement of tuning HMC trajectory lengths.

### REFERENCES

[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[2] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of Markov Chain Monte Carlo **2**, 113 (2011).  
[3] G. O. Roberts and R. L. Tweedie, "Exponential convergence of Langevin distributions and their discrete approximations," Bernoulli **2**, 341 (1996).
