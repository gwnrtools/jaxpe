---
title: jaxpe
layout: default
nav_order: 1
has_children: true
---

# `jaxpe`: Accelerating Gravitational-Wave Parameter Estimation via Normalizing-Flow-Enhanced Gradient MCMC

### I. INTRODUCTION

The direct observation of gravitational waves from coalescing compact binaries [1] necessitates highly accurate and computationally efficient parameter estimation frameworks. The fundamental objective is to solve an inverse problem: given a noisy interferometric strain dataset $$d(t)$$, compute the posterior probability measure $$\pi (\boldsymbol{\theta} \mid d)$$ over the parameters $$\boldsymbol{\theta} \in \mathcal{M}$$ characterizing the astrophysical source, where $$\mathcal{M}$$ denotes the $$D$$-dimensional differentiable parameter manifold.

The generation of gravitational waves fundamentally stems from the non-linear dynamics of General Relativity (GR), governed by the Einstein Field Equations. For coalescing compact binaries, constructing accurate waveforms requires a progression of comprehensive mathematical frameworks. We utilize Post-Newtonian (PN) expansions [3] to approximate the weak-field, slow-velocity inspiral phase. As the binary enters the highly non-linear, strong-field regime of the merger, we must deploy full Numerical Relativity (NR) [5] to solve the unadulterated Einstein equations. Finally, for simpler situations involving the ringdown of the remnant perturbed black hole, the dynamics can be mathematically encapsulated by the Teukolsky master equation, where the Newman-Penrose Weyl scalar $$\Psi_4$$ dictates the outgoing radiation field. Together, these frameworks yield deterministic models for the cross and plus metric strain polarizations:

$$
h_+(t) \quad \text{and} \quad h_\times(t)
$$

The projection of these strains onto an interferometric detector network yields the predicted time-domain signal $$h(t; \boldsymbol{\theta})$$. The physical output of a given detector is modeled as the linear superposition of this signal and additive stochastic noise:

$$
d(t) = h(t; \boldsymbol{\theta}) + n(t)
$$

Assuming the detector noise $$n(t)$$ is a wide-sense stationary Gaussian stochastic process characterized by a one-sided power spectral density (PSD) $$S_n(f)$$, the likelihood of observing the data $$d$$ is evaluated via the Whittle likelihood formalism in the frequency domain [6]. We define the noise-weighted inner product between two frequency-domain series $$\tilde{a}(f)$$ and $$\tilde{b}(f)$$ as:

$$
(a \mid b) = 4 \Re \int_{0}^{\infty} \frac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} df
$$

The corresponding likelihood functional takes the form of a multivariate Gaussian evaluated over the frequency bins:

$$
\mathcal{L}(d \mid \boldsymbol{\theta}) \propto \exp\left[ -\frac{1}{2} (d - h(\boldsymbol{\theta}) \mid d - h(\boldsymbol{\theta})) \right]
$$

Bayesian inference dictates that this likelihood is updated by the prior probability measure $$p(\boldsymbol{\theta})$$ to yield the posterior distribution:

$$
\pi(\boldsymbol{\theta} \mid d) = \frac{\mathcal{L}(d \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{\mathcal{Z}}
$$

where $$\mathcal{Z} = \int_{\mathcal{M}} \mathcal{L}(d \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta}$$ is the marginal likelihood (Bayesian evidence). Evaluating this integral explicitly is analytically intractable. The posterior support for binary black hole mergers typically forms a highly non-convex geometry within a $$\mathcal{O}(15)$$-dimensional manifold, characterized by severe degeneracies and multimodal domains. Traditional Markov Chain Monte Carlo (MCMC) algorithms employing symmetric random-walk proposals suffer from exponential mixing times in such topologies.

### II. GRADIENT-DIRECTED MARKOV CHAIN MONTE CARLO

To accelerate convergence, `jaxpe` employs gradient-directed MCMC algorithms. By utilizing automatic differentiation (AD) natively supported by the JAX framework, the gradient of the log-posterior target density is computed exactly:

$$
\nabla_{\boldsymbol{\theta}} \log \pi(\boldsymbol{\theta} \mid d)
$$

Hamiltonian Monte Carlo (HMC) [7, 8] maps the statistical sampling problem onto the deterministic integration of classical Hamiltonian dynamics. The negative log-posterior is treated as a scalar potential energy function $$U(\boldsymbol{\theta})$$ on the manifold $$\mathcal{M}$$:

$$
U(\boldsymbol{\theta}) = -\log \pi(\boldsymbol{\theta} \mid d)
$$

An auxiliary momentum vector $$\mathbf{p} \in T^*_{\boldsymbol{\theta}}\mathcal{M}$$ is introduced in the cotangent space, equipped with a kinetic energy metric defined by an inverse mass matrix $$\mathbf{M}^{-1}$$:

$$
K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
$$

The evolution of the state vector $$(\boldsymbol{\theta}, \mathbf{p})$$ is governed by Hamilton's equations of motion along the level sets of the total Hamiltonian $$H = U + K$$:

$$
\frac{d\boldsymbol{\theta}}{dt} = \mathbf{M}^{-1} \mathbf{p} \, , \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta})
$$

Because the Hamiltonian flow constitutes a volume-preserving symplectomorphism, integration of these equations generates distant, uncorrelated proposals that are accepted with probability approaching unity, strictly satisfying detailed balance.

### III. GLOBAL-LOCAL SAMPLING WITH NORMALIZING FLOWS

While HMC is exceptionally efficient at exploring connected, locally convex domains, its deterministic trajectories cannot cross regions of vanishing probability measure. Consequently, HMC is non-ergodic on multimodal posteriors where disconnected modes are separated by extensive energy barriers.

To restore ergodicity across disjoint topological modes, `jaxpe` implements a global-local orchestration architecture [9]. The local HMC transitions are interleaved with global independence proposals generated by a Normalizing Flow [10].

A Normalizing Flow constructs a highly flexible probability measure $$q_\phi$$ by mapping a simple base measure (e.g., an isotropic Gaussian $$p(\mathbf{z})$$) through a sequence of parameterized diffeomorphisms $$f_\phi: \mathcal{Z} \to \mathcal{M}$$. The probability density of the generated coordinates $$\mathbf{x} = f_\phi(\mathbf{z})$$ is dictated by the change of variables formula:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \mathbf{J}_{f_\phi^{-1}}(\mathbf{x}) \right|
$$

The diffeomorphic parameters $$\phi$$ are optimized by minimizing the Kullback-Leibler divergence between the empirical distribution of the local MCMC samples and the flow measure $$q_\phi$$.

During the global sampling phase, independent proposals $$\mathbf{y} \sim q_\phi(\mathbf{y})$$ are drawn. To guarantee the stationarity of the Markov chain with respect to the exact posterior $$\pi$$, these proposals are filtered via the Metropolis-Hastings acceptance probability $$\alpha$$:

$$
\alpha(\mathbf{x} \to \mathbf{y}) = \min\left(1, \frac{\pi(\mathbf{y} \mid d) q_\phi(\mathbf{x})}{\pi(\mathbf{x} \mid d) q_\phi(\mathbf{y})}\right)
$$

Because $$q_\phi$$ closely approximates $$\pi$$, the acceptance ratio approaches unity, allowing the Markov chain to transition seamlessly between disconnected modes without violating detailed balance.

### IV. STRUCTURE OF THIS REPOSITORY

In [Sec. II](docs/api/gw.html), the physical formalisms underlying the gravitational-wave generation and likelihood evaluation are detailed. [Sec. III](docs/api/kernels.html) rigorously formulates the symplectic geometry and stochastic differential equations governing the MCMC kernels. [Sec. IV](docs/api/flows.html) exposes the measure-theoretic mechanisms of the Normalizing Flow architecture. Finally, [Sec. V](docs/api/diagnostics.html) provides the statistical convergence criteria required to validate the ergodic mixing of the generated Markov chains.

### REFERENCES
**[1]** B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Observation of Gravitational Waves from a Binary Black Hole Merger," Phys. Rev. Lett. **116**, 061102 (2016).

**[2]** S. A. Teukolsky, "Perturbations of a Rotating Black Hole," Astrophys. J. **185**, 635 (1973).

**[3]** L. Blanchet, "Gravitational Radiation from Post-Newtonian Sources," Living Rev. Relativ. **17**, 2 (2014).

**[4]** L. Barack and A. Pound, "Self-force and radiation reaction in general relativity," Rep. Prog. Phys. **82**, 016904 (2018).

**[5]** F. Pretorius, "Evolution of Binary Black-Hole Spacetimes," Phys. Rev. Lett. **95**, 121101 (2005).

**[6]** P. Whittle, "The analysis of multiple stationary time series," J. R. Stat. Soc. **15**, 125 (1953).

**[7]** S. Duane et al., "Hybrid Monte Carlo," Phys. Lett. B **195**, 216 (1987).

**[8]** R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of MCMC **2**, 113 (2011).

**[9]** K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).

**[10]** D. Rezende and S. Mohamed, "Variational Inference with Normalizing Flows," ICML (2015).
