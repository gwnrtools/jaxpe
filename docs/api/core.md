---
title: core
parent: jaxpe
layout: default
---

# Sec. VI: Core Data Structures and Transforms (`jaxpe.core`)
{: .no_toc }

1. TOC
{:toc}

In this final section, we strip away the physics and the samplers to look at the bedrock: the mathematical data structures and transformations that allow `jaxpe` to function.

## Unconstraining Bijections

Gradient-based MCMC kernels, like Hamiltonian Monte Carlo, simulate continuous physical dynamics. They send our parameter "pucks" sliding endlessly across the landscape. Consequently, these kernels demand that the parameter space be topologically equivalent to an infinite, boundless plane, $$\mathbb{R}^D$$. 

But the physical universe is not boundless. The mass of a black hole must be strictly positive ($m > 0$). The dimensionless spin cannot exceed the extremal Kerr limit ($a \in [0, 1]$). The inclination angle of the binary orbital plane is confined to a half-circle ($\iota \in [0, \pi]$). If our HMC puck hits a hard boundary, the simulation breaks. 

To solve this, we employ bijective transformations [1]. Think of these bijections as mathematical funhouse mirrors—stretching a finite, enclosed room into an infinite corridor so our MCMC puck never hits a wall, while strictly preserving the mathematical volume (probability) of the space. We map the bounded physical parameters $$\boldsymbol{\theta}$$ into an unconstrained latent space $$\mathbf{x} \in \mathbb{R}^D$$.

### The Log Transformation

For parameters with a strict lower bound $$a$$ (such as luminosity distance $$d_L > 0$$), we stretch the boundary at $$a$$ out to $$-\infty$$ using the natural logarithm:

$$
x = \ln(\theta - a) \quad \iff \quad \theta = \exp(x) + a
$$

### The Logit Transformation

For parameters trapped in a finite interval $$[a, b]$$ (such as cosine inclination $$\cos \iota \in [-1, 1]$$), we push both boundaries out to infinity using the scaled logit mapping:

$$
x = \ln\left( \frac{\theta - a}{b - \theta} \right) \quad \iff \quad \theta = a + \frac{b - a}{1 + \exp(-x)}
$$

### Target Density Adjustment

However, there is no free lunch in calculus. When we warp the physical space $$\boldsymbol{\theta}$$ into the unconstrained space $$\mathbf{x}$$, we stretch and squeeze the probability density. If we sample naively in $$\mathbf{x}$$, we will get the wrong physics.

We must adjust the target probability density by the absolute value of the determinant of the Jacobian of the inverse transformation. By the change of variables formula:

$$
p_X(\mathbf{x}|d) = p_\Theta(f^{-1}(\mathbf{x})|d) \left| \det \mathbf{J}_{f^{-1}}(\mathbf{x}) \right|
$$

For example, when using the logit transformation, the explicit derivative (Jacobian) of the inverse mapping is:

$$
\frac{d\theta}{dx} = \frac{(b-a)\exp(-x)}{(1 + \exp(-x))^2} = \frac{(\theta - a)(b - \theta)}{b - a}
$$

In the log-domain, this requires adding the log-determinant of this Jacobian to our physical log-posterior. The `jaxpe.core` module handles these adjustments automatically and differentiably under the hood. It ensures that the exact gradient $$\nabla_{\mathbf{x}} \log p_X(\mathbf{x}|d)$$ accurately reflects the warped geometry, feeding perfectly correct physics to the HMC integrators.

## `InferenceProblem`

The `InferenceProblem` class acts as the grand orchestrator connecting the physical world (`jaxpe.gw`) to the sampling engines (`jaxpe.sampler`). It encapsulates:
- The log-likelihood function $$\ln \mathcal{L}(d|\boldsymbol{\theta})$$.
- The joint prior density $$p(\boldsymbol{\theta})$$.
- The composite bijection $$\mathbf{x} \leftrightarrow \boldsymbol{\theta}$$.

By abstracting away the tedious domain transformations, it presents a pure, infinitely smooth, unconstrained log-density $$U(\mathbf{x}) = -\log p_X(\mathbf{x}|d)$$ to the MCMC kernels, allowing the rest of the package to operate in frictionless mathematical elegance.

### REFERENCES

[1] Stan Development Team, "Stan Modeling Language Users Guide and Reference Manual," v2.32 (2023).
