---
title: jaxpe
layout: default
nav_order: 1
has_children: true
---

# `jaxpe`: Accelerating Gravitational-Wave Parameter Estimation via Normalizing-Flow-Enhanced Gradient MCMC

### I. INTRODUCTION

The first direct detection of gravitational waves (GWs) from a binary black hole merger by the Advanced LIGO detectors marked the dawn of observational gravitational-wave astronomy [1]. The ensuing catalogs have since revealed a hidden population of compact object binaries—comprising black holes and neutron stars—that inspiral and coalesce in the distant universe. To extract the astrophysics from these spacetime perturbations, we must confront a rigorous inverse problem: given the noisy interferometric strain data, what are the intrinsic masses, spins, and extrinsic orientations of the progenitor binary?

In the formalism of General Relativity, the generation of gravitational waves from perturbed black holes can be described by the Teukolsky equation, which governs the dynamics of the Newman-Penrose scalar $$\Psi_4$$ [2]. However, realistic compact binary coalescences require sophisticated theoretical machinery. The early inspiral phase is modeled using Post-Newtonian (PN) theory [3], expanding the orbital dynamics in powers of $$v/c$$. For extreme-mass-ratio inspirals, the Self-Force (SF) formalism computes the back-reaction of the small body on the background spacetime [4]. Finally, the highly nonlinear merger phase is simulated using full Numerical Relativity (NR) [5]. Together, these formalisms allow us to construct high-fidelity templates for the metric strain components $$h_+(t)$$ and $$h_\times(t)$$. The output of an interferometric detector is a single time series:

$$
d(t) = h(t; \boldsymbol{\theta}) + n(t)
$$

where $$n(t)$$ represents the stochastic noise realization and $$\boldsymbol{\theta}$$ denotes the multi-dimensional parameter vector characterizing the source (e.g., chirp mass $$\mathcal{M}_c$$, mass ratio $$q$$, luminosity distance $$d_L$$, inclination $$\iota$$, etc.).

The extraction of $$\boldsymbol{\theta}$$ is formulated within a Bayesian framework. Assuming the detector noise $$n(t)$$ is stationary and Gaussian with a one-sided power spectral density (PSD) $$S_n(f)$$, the likelihood of observing data $$d$$ given parameters $$\boldsymbol{\theta}$$ is governed by the Whittle likelihood in the frequency domain [6]. Defining the noise-weighted inner product:

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

### II. GRADIENT-DIRECTED MARKOV CHAIN MONTE CARLO

The `jaxpe` package introduces a modern, high-performance solution to this bottleneck. By leveraging JAX's auto-differentiation (AD) and just-in-time (JIT) compilation, `jaxpe` exploits the gradient of the posterior, $$\nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d)$$, to drive highly efficient Markov Chain Monte Carlo (MCMC) kernels. Specifically, Hamiltonian Monte Carlo (HMC) [7, 8] elevates the statistical problem into a classical mechanics simulation. We treat the negative log-posterior as a potential energy well:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

and assign a conjugate momentum $$\mathbf{p}$$ with kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

where $$\mathbf{M}$$ is the mass matrix (or inverse covariance). The system evolves along contours of the Hamiltonian:

$$
H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})
$$

according to Hamilton's equations:

$$
\frac{d\boldsymbol{\theta}}{dt} = \mathbf{M}^{-1} \mathbf{p} \, , \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d) \, .
$$

Because this flow preserves phase-space volume and conserves energy, the deterministic leapfrog integration can generate distant proposals across the correlated posterior manifold with near-perfect acceptance rates, drastically suppressing random-walk behavior.

### III. GLOBAL-LOCAL SAMPLING WITH NORMALIZING FLOWS

Crucially, gradient-directed local exploration is insufficient to traverse the severe multi-modality of gravitational-wave posteriors. A purely local gradient provides no information about disconnected, widely separated modes (e.g., the exact $\pi$-phase flip degeneracy or disjoint sky positions), trapping the MCMC chains. 

To bridge isolated modes, `jaxpe` orchestrates a "global-local" sampling architecture [9]. As thousands of parallel chains explore local modes on the GPU, their accumulated samples are periodically used to train a Normalizing Flow—a deep generative model parameterized by neural networks. 

A Normalizing Flow [10] constructs a highly complex probability distribution $$q_\phi(\mathbf{x})$$ by applying a sequence of invertible, differentiable transformations (diffeomorphisms) $$f_\phi$$ to a simple, tractable base distribution, typically an isotropic Gaussian $$p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})$$. The parameter vector $$\phi$$ represents the weights and biases of the underlying neural networks. By the fundamental change of variables theorem from multivariate calculus, the exact density of the generated samples $$\mathbf{x} = f_\phi(\mathbf{z})$$ can be evaluated as:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \mathbf{J}_{f_\phi^{-1}}(\mathbf{x}) \right|
$$

where $$\mathbf{J}_{f_\phi^{-1}}$$ is the Jacobian matrix of the inverse transformation. We train this flow to emulate the target posterior by minimizing the Kullback-Leibler (KL) divergence from the empirical distribution of the MCMC samples to the flow distribution.

Once the flow is trained, it acts as a learned, global proposal distribution. We propose global jumps to new parameters $$\mathbf{y} \sim q_\phi(\mathbf{y})$$ independently of the current chain state $$\mathbf{x}$$. To strictly satisfy detailed balance and ensure convergence to the true posterior $$\pi(\boldsymbol{\theta}|d)$$, these independence jumps are subjected to the Metropolis-Hastings criterion. The acceptance probability $$\alpha$$ is given by:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y}|d) q_\phi(\mathbf{x})}{\pi(\mathbf{x}|d) q_\phi(\mathbf{y})}\right)
$$

Because the learned density $$q_\phi(\mathbf{x})$$ accurately approximates the true disconnected mode structure $$\pi(\mathbf{x}|d)$$, the ratio $$\pi/q_\phi$$ is close to order unity everywhere. This allows the sampler to instantly teleport chains between degenerate modes across the parameter space with high acceptance probabilities, completely eliminating the topological bottlenecks of multimodal inference.

### IV. STRUCTURE OF THIS REPOSITORY

In [Sec. II](docs/api/gw.html) of this documentation, we detail the underlying gravitational-wave physics module (`jaxpe.gw`), including the waveform construction, detector responses, and the frequency-domain likelihood. [Sec. III](docs/api/kernels.html) describes the suite of MCMC kernels (`jaxpe.kernels`) and the mathematical mechanics of Hamiltonian and Langevin dynamics. [Sec. IV](docs/api/flows.html) details the normalizing flow architecture (`jaxpe.flows`) and the orchestrating global-local sampler (`jaxpe.sampler`). Finally, [Sec. V](docs/api/diagnostics.html) provides diagnostic tools and utilities (`jaxpe.diagnostics`, `jaxpe.core`).

This repository is designed not merely as a black-box tool, but as a pedagogically transparent, self-contained textbook in code. By laying bare the mathematics of gravitational-wave inference and the computational techniques required to solve it, we invite the student and the researcher alike to engage directly with the mechanics of modern gravitational-wave astronomy.

### REFERENCES

[1] B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Observation of Gravitational Waves from a Binary Black Hole Merger," Phys. Rev. Lett. **116**, 061102 (2016).  
[2] S. A. Teukolsky, "Perturbations of a Rotating Black Hole. I. Fundamental Equations for Gravitational, Electromagnetic, and Neutrino-Field Perturbations," Astrophys. J. **185**, 635 (1973).  
[3] L. Blanchet, "Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries," Living Rev. Relativ. **17**, 2 (2014).  
[4] L. Barack and A. Pound, "Self-force and radiation reaction in general relativity," Rep. Prog. Phys. **82**, 016904 (2018).  
[5] F. Pretorius, "Evolution of Binary Black-Hole Spacetimes," Phys. Rev. Lett. **95**, 121101 (2005).  
[6] P. Whittle, "The analysis of multiple stationary time series," J. R. Stat. Soc. Series B Stat. Methodol. **15**, 125 (1953).  
[7] S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[8] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of Markov Chain Monte Carlo **2**, 113 (2011).  
[9] K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).  
[10] D. Rezende and S. Mohamed, "Variational Inference with Normalizing Flows," ICML (2015).
