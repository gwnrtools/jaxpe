---
title: jaxpe
layout: default
nav_order: 1
has_children: true
---

# `jaxpe`: Accelerating Gravitational-Wave Parameter Estimation via Normalizing-Flow-Enhanced Gradient MCMC

### I. INTRODUCTION

The first direct detection of gravitational waves (GWs) from a binary black hole merger by the Advanced LIGO detectors marked the dawn of observational gravitational-wave astronomy. The ensuing catalogs have since revealed a hidden population of compact object binaries—comprising black holes and neutron stars—that inspiral and coalesce in the distant universe. To extract the astrophysics from these spacetime perturbations, we must confront a rigorous inverse problem: given the noisy interferometric strain data, what are the intrinsic masses, spins, and extrinsic orientations of the progenitor binary?

In the formalism of General Relativity, the generation of gravitational waves from perturbed black holes can be elegantly described by the Teukolsky equation, which governs the dynamics of the Newman-Penrose scalar $$\Psi_4$$. As the waves propagate to the transverse-traceless (TT) gauge of our detectors on Earth, they manifest as the metric strain components $$h_+(t)$$ and $$h_\times(t)$$. The output of an interferometric detector is a single time series:

$$
d(t) = h(t; \boldsymbol{\theta}) + n(t)
$$

where $$n(t)$$ represents the stochastic noise realization and $$\boldsymbol{\theta}$$ denotes the multi-dimensional parameter vector characterizing the source (e.g., chirp mass $$\mathcal{M}_c$$, mass ratio $$q$$, luminosity distance $$d_L$$, inclination $$\iota$$, etc.).

The extraction of $$\boldsymbol{\theta}$$ is formulated within a Bayesian framework. Assuming the detector noise $$n(t)$$ is stationary and Gaussian with a one-sided power spectral density (PSD) $$S_n(f)$$, the likelihood of observing data $$d$$ given parameters $$\boldsymbol{\theta}$$ is governed by the Whittle likelihood in the frequency domain. Defining the noise-weighted inner product:

$$
(a | b) = 4 \Re \int_{0}^{\infty} \frac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} df \, ,
$$

the likelihood function is proportional to:

$$
\exp\left[ -\frac{1}{2} (d - h(\boldsymbol{\theta}) | d - h(\boldsymbol{\theta})) \right]
$$

Combining this with prior knowledge $$p(\boldsymbol{\theta})$$ via Bayes' theorem yields the posterior distribution:

$$
p(\boldsymbol{\theta} | d) \propto \mathcal{L}(d | \boldsymbol{\theta}) p(\boldsymbol{\theta})
$$

However, characterizing this posterior in practice presents a formidable computational challenge. The parameter space is high-dimensional (typically 15 dimensions for a quasicircular binary), and the likelihood surface is pathologically complex. It features strong correlations (e.g., between distance and inclination) and multiple disconnected modes induced by degeneracies in the signal's sky position and phase. Standard stochastic sampling techniques—such as unguided random walk Metropolis-Hastings or nested sampling—scale poorly as the dimensionality increases, often taking days or weeks to converge for a single event. 

The `jaxpe` package introduces a modern, high-performance solution to this bottleneck. By leveraging JAX's auto-differentiation (AD) and just-in-time (JIT) compilation, `jaxpe` exploits the gradient of the posterior, $$\nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d)$$, to drive highly efficient Markov Chain Monte Carlo (MCMC) kernels. Specifically, Hamiltonian Monte Carlo (HMC) elevates the statistical problem into a classical mechanics simulation. We treat the negative log-posterior as a potential energy well:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

and assign a conjugate momentum $$\mathbf{p}$$ with kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The system evolves along contours of the Hamiltonian:

$$
H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})
$$

according to Hamilton's equations:

$$
\frac{d\boldsymbol{\theta}}{dt} = \mathbf{M}^{-1} \mathbf{p} \, , \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d) \, .
$$

Because this flow preserves phase-space volume and conserves energy, the deterministic leapfrog integration can generate distant proposals across the correlated posterior manifold with near-perfect acceptance rates, drastically suppressing random-walk behavior.

Crucially, gradient-directed local exploration is insufficient to traverse the severe multi-modality of gravitational-wave posteriors. To bridge isolated modes, `jaxpe` orchestrates a "global-local" sampling architecture. As thousands of parallel chains explore local modes on the GPU, their accumulated samples are periodically used to train a Normalizing Flow—a deep generative model composed of rational-quadratic-spline coupling layers. Once trained, the flow acts as a learned, global proposal distribution for Metropolis-Hastings steps, capable of instantly teleporting chains between degenerate modes across the parameter space while strictly satisfying detailed balance.

In [Sec. II](docs/api/gw.html) of this documentation, we detail the underlying gravitational-wave physics module (`jaxpe.gw`), including the waveform construction, detector responses, and the frequency-domain likelihood. [Sec. III](docs/api/kernels.html) describes the suite of MCMC kernels (`jaxpe.kernels`) and the mathematical mechanics of Hamiltonian and Langevin dynamics. [Sec. IV](docs/api/flows.html) details the normalizing flow architecture (`jaxpe.flows`) and the orchestrating global-local sampler (`jaxpe.sampler`). Finally, [Sec. V](docs/api/diagnostics.html) provides diagnostic tools and utilities (`jaxpe.diagnostics`, `jaxpe.core`).

This repository is designed not merely as a black-box tool, but as a pedagogically transparent, self-contained textbook in code. By laying bare the mathematics of gravitational-wave inference and the computational techniques required to solve it, we invite the student and the researcher alike to engage directly with the mechanics of modern gravitational-wave astronomy.
