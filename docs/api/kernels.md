---
title: kernels
parent: API Reference
layout: default
nav_order: 3
---

# Sec. III: Symplectic Geometry and Langevin Diffusion (`jaxpe.kernels`)
{: .no_toc }

1. TOC
{:toc}

In high-dimensional parameter spaces typical of gravitational-wave inference ($$D \approx 15$$), standard random-walk Metropolis-Hastings proposals driven by isotropic Brownian motion suffer from severe diffusive inefficiency, yielding an exploration radius that scales as $$R \propto \sqrt{N}$$. To circumvent this, the transition kernels in `jaxpe` leverage the differential geometry of symplectic manifolds to generate coherent, directed trajectories across the posterior measure.

## Hamiltonian Monte Carlo on Symplectic Manifolds

### The Classical Hamiltonian Formalism
In classical mechanics, a Hamiltonian $$H$$ is a scalar function that represents the total energy of a physical system. The fundamental objective of Hamiltonian mechanics is to describe the deterministic time evolution of this system. Rather than working solely with spatial coordinates, the Hamiltonian framework operates on the phase space of the system, defined by generalized coordinates (positions) $$\theta^\mu$$ and their conjugate momenta $$p_\mu$$. 

By evaluating the partial derivatives of the Hamiltonian scalar $$H(\theta, p)$$, Hamilton's equations of motion generate a vector field that perfectly dictates the trajectory of the particle over time, conserving total energy and preserving the volume of the phase space.

### Mapping Statistics to Mechanics
Hamiltonian Monte Carlo (HMC) [1, 2] achieves extraordinary sampling efficiency by establishing a strict isomorphism between the purely statistical problem of sampling a probability distribution and the mechanical problem of simulating a physical particle's trajectory. 

We elevate the $$D$$-dimensional parameter space of our gravitational-wave problem to act as the positional manifold $$\mathcal{M}$$. To construct the required phase space, we introduce a set of auxiliary variables $$p_\mu$$ that serve as the conjugate momenta. Geometrically, this combined space is the cotangent bundle $$T^*\mathcal{M}$$, which is naturally a symplectic manifold equipped with a closed, non-degenerate 2-form $$\omega = d\theta^\mu \wedge dp_\mu$$.

The mechanical terms map to the statistical problem as follows:
1. **Position ($$\theta^\mu$$)**: The actual astrophysical parameters we wish to infer (e.g., chirp mass, spin).
2. **Momentum ($$p_\mu$$)**: Auxiliary variables artificially introduced to grant the system "inertia," allowing it to glide coherently through the parameter space.
3. **Potential Energy ($$U(\theta)$$)**: Defined strictly as the negative log-posterior, $$U(\theta) = -\log \pi(\theta|d)$$. The high-probability peaks of the posterior become deep gravitational wells that attract the particle.
4. **Kinetic Energy ($$K(p)$$)**: A quadratic form defining the energy of the auxiliary momenta, parameterized by a positive-definite inverse mass matrix $$g^{\mu\nu}$$.

We define the Hamiltonian scalar function $$H: T^*\mathcal{M} \to \mathbb{R}$$ as the sum of these energies:

$$
H(\theta^\mu, p_\mu) = U(\theta^\mu) + K(p_\mu) = -\log \pi(\theta^\mu|d) + \frac{1}{2} g^{\mu\nu} p_\mu p_\nu
$$

where $$g^{\mu\nu}$$ is the inverse of the mass matrix (which can be interpreted as a flat Riemannian metric on $$\mathcal{M}$$). The evolution of any observable $$F$$ on phase space is governed by the Poisson bracket:

$$
\frac{dF}{dt} = \{F, H\} = \frac{\partial F}{\partial \theta^\mu} \frac{\partial H}{\partial p_\mu} - \frac{\partial F}{\partial p_\mu} \frac{\partial H}{\partial \theta^\mu}
$$

Setting $$F = \theta^\mu$$ and $$F = p_\mu$$ recovers Hamilton's equations of motion:

$$
\frac{d\theta^\mu}{dt} = g^{\mu\nu} p_\nu \, , \quad \frac{dp_\mu}{dt} = -\partial_\mu U(\theta)
$$

Because Hamiltonian flow is a symplectomorphism, it preserves the phase space volume form $$\Omega = \frac{(-1)^{n(n-1)/2}}{n!} \omega^n$$ exactly (Liouville's Theorem).

### Numerical Integration: The Leapfrog Symplectic Integrator

In practice, we cannot integrate these continuous Hamiltonian differential equations exactly. We must discretize time into discrete steps $$\Delta t$$. However, standard integration schemes like Runge-Kutta do not preserve the symplectic volume form $$\omega$$, causing the simulated energy to systematically drift and destroying the detailed balance condition of the Markov chain.

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

Because this integrator preserves phase-space volume, the only source of error is the truncation error in the energy conservation $$\mathcal{O}(\Delta t^2)$$. We correct for this small discretization error by wrapping the trajectory in a final Metropolis-Hastings acceptance step.

The [`HMC`](#hmc) kernel in `jaxpe` encapsulates this deterministic trajectory integration:

```python
from jaxpe.kernels.hmc import HMC

kernel = HMC(step_size=1e-2, n_leapfrog=10)
```

## Langevin Diffusion and the Fokker-Planck Equation

As an alternative to the deterministic integration of Hamilton's equations, the Metropolis-Adjusted Langevin Algorithm (MALA) [3] constructs proposals by simulating the overdamped limit of Langevin diffusion.

The continuous-time stochastic differential equation (SDE) governing the parameter vector $$\theta^\mu_t$$ on a flat manifold is:

$$
d\theta^\mu_t = -\frac{1}{2} g^{\mu\nu} \partial_\nu U(\theta_t) dt + \sqrt{g^{\mu\nu}} dW_{\nu, t}
$$

where $$dW_{\nu, t}$$ is a multi-dimensional Wiener process. The macroscopic evolution of the probability density $$\rho(\theta^\mu, t)$$ is governed by the Fokker-Planck equation:

$$
\frac{\partial \rho}{\partial t} = \partial_\mu \left( \frac{1}{2} g^{\mu\nu} \partial_\nu \rho + \frac{1}{2} \rho g^{\mu\nu} \partial_\nu U \right)
$$

Imposing stationarity ($$\partial_t \rho = 0$$), we trivially recover that the equilibrium distribution is exactly the target posterior $$\rho \propto \exp(-U(\theta)) = \pi(\theta|d)$$.

This overdamped continuous diffusion is provided natively by the [`MALA`](#mala) kernel:

```python
from jaxpe.kernels.mala import MALA

kernel = MALA(step_size=1e-3)
```

### Manifold MALA (`mMALA`)

In highly curved posteriors, using a globally constant metric $$g^{\mu\nu}$$ is severely sub-optimal. Manifold MALA (mMALA) promotes the mass matrix to a position-dependent Riemannian metric tensor $$g_{\mu\nu}(\theta)$$ (often the Fisher Information Matrix). The generalized Langevin SDE must now include corrections from the Levi-Civita connection (the Christoffel symbols $$\Gamma^\mu_{\alpha\beta}$$) to remain covariant:

$$
d\theta^\mu_t = \frac{1}{2} g^{\mu\nu}(\theta_t) \partial_\nu \log \pi(\theta_t) dt + \frac{1}{2} \Gamma^\mu_{\alpha\beta} g^{\alpha\beta} dt + e^\mu_\alpha dW^\alpha_t
$$

where $$e^\mu_\alpha$$ is the vielbein (tetrad) defined by $$g^{\mu\nu} = e^\mu_\alpha e^\nu_\beta \delta^{\alpha\beta}$$. This allows the diffusion to adapt perfectly to the local curvature of the target density.

### Underdamped Langevin Dynamics (`ULD`)

ULD restores the conjugate momenta to the diffusion process, adding a continuous friction tensor $$\Gamma^\mu_\nu$$ to Hamilton's equations. The momenta undergo an Ornstein-Uhlenbeck process while the positions drift:

$$
d\theta^\mu_t = g^{\mu\nu} p_{\nu, t} dt
$$

$$
dp_{\mu, t} = -\partial_\mu U(\theta_t) dt - \Gamma^\alpha_\mu p_{\alpha, t} dt + \sqrt{2 \Gamma_{\mu\alpha}} dW^\alpha_t
$$

This continuous momentum refreshment robustly navigates strongly correlated spaces without the strict requirement of tuning HMC trajectory lengths.

In `jaxpe`, this momentum-restoring process is available via the [`ULD`](#uld) transition kernel:

```python
from jaxpe.kernels.uld import ULD

kernel = ULD(step_size=1e-2, friction=1.0)
```

## API Reference

### `HMC`
**`jaxpe.kernels.hmc.HMC(step_size: float, n_leapfrog: int, scale=None)`**
Hamiltonian Monte Carlo transition kernel. Simulates deterministic Hamilton's equations over `n_leapfrog` steps using the symplectic Leapfrog integrator.

### `MALA`
**`jaxpe.kernels.mala.MALA(step_size: float, scale=None)`**
Metropolis-Adjusted Langevin Algorithm. Simulates overdamped Langevin diffusion in the posterior energy landscape.

### `ULD`
**`jaxpe.kernels.uld.ULD(step_size: float, friction: float, scale=None)`**
Underdamped Langevin Dynamics. Simulates Langevin diffusion while preserving conjugate momenta, governed by a continuous friction coefficient.

---

### REFERENCES

[1] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[2] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of Markov Chain Monte Carlo **2**, 113 (2011).  
[3] G. O. Roberts and R. L. Tweedie, "Exponential convergence of Langevin distributions and their discrete approximations," Bernoulli **2**, 341 (1996).
