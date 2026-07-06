---
title: jaxpe
layout: default
nav_order: 1
has_children: true
---

# `jaxpe`: Accelerating Gravitational-Wave Parameter Estimation via Normalizing-Flow-Enhanced Gradient MCMC

### I. INTRODUCTION

Welcome. Let us imagine for a moment that you are standing far away from a vast, cosmic drum. You cannot see the drum, nor can you see the mallets striking it. All you have is a faint, noisy audio recording of its reverberations reaching you across billions of light-years. Your task is to reconstruct the precise size, shape, and composition of that drum from the echoes alone. 

This, fundamentally, is the challenge of observational gravitational-wave astronomy. The first direct detection of gravitational waves from a binary black hole merger by the Advanced LIGO detectors [1] presented us with exactly this inverse problem: given the noisy interferometric strain data, what are the intrinsic masses, spins, and extrinsic orientations of the progenitor binary?

To solve this, we must first understand the forward problem—the "cosmic symphony" itself. In the formalism of General Relativity, the generation of gravitational waves from perturbed black holes can be mathematically encapsulated by the Teukolsky master equation, which governs the dynamics of the Newman-Penrose scalar:

$$
\Psi_4
$$

However, the reality of two inspiraling compact objects requires a suite of sophisticated mathematical machinery. We use Post-Newtonian (PN) theory [3] for the early, slow-motion inspiral; the Self-Force (SF) formalism [4] when a small body perturbs the spacetime of a massive host; and full Numerical Relativity (NR) [5] to solve the non-linear chaos of the merger itself. These formalisms allow us to construct high-fidelity templates for the metric strain components:

$$
h_+(t) \quad \text{and} \quad h_\times(t)
$$

The output of our interferometer is therefore a single time series:

$$
d(t) = h(t; \boldsymbol{\theta}) + n(t)
$$

Here, \(n(t)\) represents the stochastic background noise, and \(\boldsymbol{\theta}\) denotes the multi-dimensional parameter vector characterizing the source—its chirp mass, its luminosity distance, and so forth.

The extraction of \(\boldsymbol{\theta}\) is a rigorous exercise in Bayesian inference. Assuming the detector noise \(n(t)\) is stationary and Gaussian with a one-sided power spectral density (PSD) \(S_n(f)\), we evaluate the Whittle likelihood in the frequency domain [6]. We define the noise-weighted inner product:

$$
(a | b) = 4 \Re \int_{0}^{\infty} \frac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} df \, ,
$$

which naturally leads to the likelihood function:

$$
\mathcal{L}(d | \boldsymbol{\theta}) \propto \exp\left[ -\frac{1}{2} (d - h(\boldsymbol{\theta}) | d - h(\boldsymbol{\theta})) \right]
$$

Combining this with our prior knowledge \(p(\boldsymbol{\theta})\) via Bayes' theorem yields the posterior distribution we seek:

$$
p(\boldsymbol{\theta} | d) = \frac{\mathcal{L}(d | \boldsymbol{\theta}) p(\boldsymbol{\theta})}{Z}
$$

where \(Z\) is the Bayesian evidence. But here is the rub: this posterior surface is a 15-dimensional landscape fraught with deep valleys, winding ridges, and isolated peaks. Standard stochastic samplers act like blindfolded hikers, randomly guessing where to step next. In such high dimensions, they simply take too long to find the peaks.

### II. GRADIENT-DIRECTED MARKOV CHAIN MONTE CARLO

How do we guide our hikers? The `jaxpe` package introduces a modern, high-performance solution by removing the blindfolds. We leverage JAX's auto-differentiation (AD) to compute the exact gradient of the posterior:

$$
\nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|d)
$$

Now, consider Hamiltonian Monte Carlo (HMC) [7, 8]. HMC elevates our statistical sampling problem into an elegant simulation of classical mechanics. Imagine flipping the posterior landscape upside down, so the peaks become deep gravity wells. We treat this negative log-posterior as a potential energy:

$$
U(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta}|d)
$$

We then place a frictionless puck (our parameter state) on this landscape and give it a random kick of conjugate momentum \(\mathbf{p}\) with kinetic energy:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The system now evolves gracefully along the contours of the total Hamiltonian:

$$
H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})
$$

governed precisely by Hamilton's equations of motion:

$$
\frac{d\boldsymbol{\theta}}{dt} = \mathbf{M}^{-1} \mathbf{p} \, , \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta})
$$

Because this continuous flow preserves phase-space volume and conserves energy perfectly, we can let the puck slide across the entire parameter space for a long time, generating distant, uncorrelated proposals that are accepted with near certainty.

### III. GLOBAL-LOCAL SAMPLING WITH NORMALIZING FLOWS

But what if our landscape has two deep, entirely disconnected valleys separated by an impassable mountain range? A frictionless puck—driven only by local gradients—cannot spontaneously teleport to the other valley. It remains trapped. This is the challenge of multimodal posteriors (like the exact \(\pi\)-phase flip degeneracy in GWs). 

To bridge these isolated modes, `jaxpe` orchestrates a "global-local" sampling architecture [9]. While our local HMC explorers map out the individual valleys, a satellite overhead is watching them. This satellite is a Normalizing Flow [10]. 

Think of a Normalizing Flow as molding a simple, structureless block of clay (a standard Gaussian distribution, \(p(\mathbf{z})\)) into an intricate, multi-modal sculpture (the posterior). It achieves this by applying a sequence of invertible, differentiable transformations (diffeomorphisms) \(f_\phi\), parameterized by a neural network. By the fundamental change of variables theorem from multivariate calculus, the exact probability density of any point \(\mathbf{x} = f_\phi(\mathbf{z})\) is rigorously given by:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \mathbf{J}_{f_\phi^{-1}}(\mathbf{x}) \right|
$$

We train this flow to emulate the target posterior by minimizing the Kullback-Leibler (KL) divergence from the empirical distribution of the MCMC samples to the flow distribution.

Once trained, this flow becomes our teleportation machine. We can instantaneously propose global jumps to new parameters \(\mathbf{y} \sim q_\phi(\mathbf{y})\), completely ignoring where the chain currently is (\(\mathbf{x}\)). To strictly satisfy the principle of detailed balance—ensuring our teleportation doesn't artificially skew the true physics—these jumps are subjected to the Metropolis-Hastings criterion. The acceptance probability \(\alpha\) is:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y}|d) q_\phi(\mathbf{x})}{\pi(\mathbf{x}|d) q_\phi(\mathbf{y})}\right)
$$

Because our flow density \(q_\phi\) is a highly accurate map of the true posterior \(\pi\), the ratio approaches unity. Our explorers can now instantly teleport between isolated degenerate modes with exceptionally high success rates.

### IV. STRUCTURE OF THIS REPOSITORY

In [Sec. II](docs/api/gw.html), we delve deeper into the physical formalisms underlying the gravitational-wave templates. [Sec. III](docs/api/kernels.html) dissects the suite of MCMC kernels and the Fokker-Planck equations governing Langevin dynamics. [Sec. IV](docs/api/flows.html) exposes the inner mathematical workings of the Normalizing Flow architecture and the orchestrating global-local sampler (`jaxpe.sampler`). Finally, [Sec. V](docs/api/diagnostics.html) provides the rigorous convergence diagnostics required to trust our inference.

Welcome to `jaxpe`. We invite you to engage directly with the mathematics and mechanics of modern computational astrophysics.

### REFERENCES

[1] B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Observation of Gravitational Waves from a Binary Black Hole Merger," Phys. Rev. Lett. **116**, 061102 (2016).  
[2] S. A. Teukolsky, "Perturbations of a Rotating Black Hole," Astrophys. J. **185**, 635 (1973).  
[3] L. Blanchet, "Gravitational Radiation from Post-Newtonian Sources," Living Rev. Relativ. **17**, 2 (2014).  
[4] L. Barack and A. Pound, "Self-force and radiation reaction in general relativity," Rep. Prog. Phys. **82**, 016904 (2018).  
[5] F. Pretorius, "Evolution of Binary Black-Hole Spacetimes," Phys. Rev. Lett. **95**, 121101 (2005).  
[6] P. Whittle, "The analysis of multiple stationary time series," J. R. Stat. Soc. **15**, 125 (1953).  
[7] S. Duane et al., "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).  
[8] R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of MCMC **2**, 113 (2011).  
[9] K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).  
[10] D. Rezende and S. Mohamed, "Variational Inference with Normalizing Flows," ICML (2015).
