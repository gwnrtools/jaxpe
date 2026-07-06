---
title: core
parent: jaxpe
layout: default
---

# Sec. VI: Core Data Structures and Transforms (`jaxpe.core`)
{: .no_toc }

1. TOC
{:toc}

In this section, we detail the underlying continuous support mappings and Bayesian problem abstractions that form the bedrock of `jaxpe`.

## Unconstraining Bijections

Gradient-based MCMC kernels (like HMC or MALA) simulate continuous physical dynamics. Consequently, they require the parameter space to be topologically equivalent to $\mathbb{R}^D$ without hard boundaries or constrained intervals. Because many gravitational-wave parameters are strictly bounded (e.g., masses $m_1, m_2 > 0$, or inclination $\iota \in [0, \pi]$), we must employ bijective transformations to map the physical parameters $\boldsymbol{\theta}$ into an unconstrained latent space $\mathbf{x} \in \mathbb{R}^D$ [1].

### The Log Transformation

For parameters with a strict lower bound $a$ (e.g., luminosity distance $d_L > 0$), we use the natural logarithm:

$$
x = \ln(\theta - a) \quad \iff \quad \theta = \exp(x) + a
$$

### The Logit Transformation

For parameters constrained to a finite interval $[a, b]$ (e.g., spins $a_1, a_2 \in [0, 1]$, or cosine inclination $\cos \iota \in [-1, 1]$), we employ the scaled logit mapping:

$$
x = \ln\left( \frac{\theta - a}{b - \theta} \right) \quad \iff \quad \theta = a + \frac{b - a}{1 + \exp(-x)}
$$

### Target Density Adjustment

When sampling in the unconstrained space $\mathbf{x}$, the target probability density must be adjusted by the absolute value of the determinant of the Jacobian of the inverse transformation. By the change of variables formula:

$$
p_X(\mathbf{x}|d) = p_\Theta(f^{-1}(\mathbf{x})|d) \left| \det \mathbf{J}_{f^{-1}}(\mathbf{x}) \right|
$$

In the log-domain, this requires adding the log-determinant of the Jacobian to the physical log-posterior. The `jaxpe.core` module handles these adjustments automatically and differentiably, ensuring that the gradient $\nabla_{\mathbf{x}} \log p_X(\mathbf{x}|d)$ accurately reflects the warped geometry.

## `InferenceProblem`

The `InferenceProblem` class acts as the central interface connecting the physics (`jaxpe.gw`) to the sampling engines (`jaxpe.sampler`). It encapsulates:
- The log-likelihood function $\ln \mathcal{L}(d|\boldsymbol{\theta})$.
- The joint prior density $p(\boldsymbol{\theta})$.
- The composite bijection $\mathbf{x} \leftrightarrow \boldsymbol{\theta}$.

By abstracting away the domain transformations, it presents a pure, differentiable log-density $U(\mathbf{x}) = -\log p_X(\mathbf{x}|d)$ to the HMC leapfrog integrators.

### REFERENCES

[1] Stan Development Team, "Stan Modeling Language Users Guide and Reference Manual," v2.32 (2023).
