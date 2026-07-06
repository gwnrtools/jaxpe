---
title: core
parent: jaxpe
layout: default
nav_order: 6
---

# Sec. VI: Diffeomorphic Bijections and Target Space Geometry (`jaxpe.core`)
{: .no_toc }

1. TOC
{:toc}

In this section, we strip away the stochastic dynamics to examine the foundational geometrical substrate of `jaxpe`. Hamiltonian mechanics fundamentally assumes the parameter space is a smooth manifold topologically equivalent to \(\mathbb{R}^D\). Boundaries, sharp truncations, and finite intervals break the continuous integration of Hamilton's equations, causing the simulated momentum to reflect pathologically.

## Boundary Removal via Diffeomorphisms

The physical universe dictates strict boundaries: a black hole mass \(m > 0\), a dimensionless spin \(a \in [0, 1]\), an inclination \(\iota \in [0, \pi]\). The physical parameter manifold \(\mathcal{M}_\theta\) is therefore a manifold with boundaries. 

To satisfy the geometrical requirements of the MCMC kernels, we must construct a smooth, bijective, and differentiable mapping—a diffeomorphism \(f: \mathcal{M}_\theta \to \mathcal{X}\)—that maps the bounded physical space onto an unconstrained latent manifold \(\mathcal{X} \cong \mathbb{R}^D\) [1].

### The Logarithmic Diffeomorphism

For parameters bounded strictly below by a threshold \(a\) (e.g., \(d_L \in (0, \infty)\)), we apply the natural logarithm to push the boundary to asymptotic infinity:

$$
x = \ln(\theta - a) \quad \iff \quad \theta = \exp(x) + a
$$

### The Scaled Logit Diffeomorphism

For parameters strictly bounded in a finite interval \(\theta \in (a, b)\) (e.g., cosine inclination \(\cos \iota \in (-1, 1)\)), we utilize the scaled logit function, stretching both boundaries to \(\pm \infty\):

$$
x = \ln\left( \frac{\theta - a}{b - \theta} \right) \quad \iff \quad \theta = a + \frac{b - a}{1 + \exp(-x)}
$$

## Jacobians and Target Density Adjustments

When we transport the MCMC process to the unconstrained manifold \(\mathcal{X}\), the probability density is geometrically distorted. The pushforward of the posterior measure \(\pi_\Theta\) under \(f\) induces a necessary volume correction governed by the Jacobian matrix \(J^\mu_\nu = \partial \theta^\mu / \partial x^\nu\).

By the change of variables theorem, the exact unconstrained target density is:

$$
\pi_X(x) = \pi_\Theta(f^{-1}(x)) \left| \det \left( \frac{\partial \theta^\mu}{\partial x^\nu} \right) \right|
$$

Because our bijections are applied element-wise, the Jacobian matrix is strictly diagonal, reducing the determinant to a simple product of scalar derivatives. For the logit transformation, the explicit derivative element is:

$$
\frac{\partial \theta^i}{\partial x^i} = \frac{(\theta^i - a)(b - \theta^i)}{b - a}
$$

In the log-domain required by the MCMC energy potential \(U(x) = -\log \pi_X(x)\), this equates to adding the log-determinant \(\sum_i \ln | \partial_i \theta^i |\) to the physical log-posterior. The `jaxpe.core` module handles these adjustments automatically, ensuring that the gradient covector \(\partial_\mu U(x)\) perfectly aligns with the warped geometry of \(\mathcal{X}\).

## `InferenceProblem`

The `InferenceProblem` class acts as the grand geometrical orchestrator. It encapsulates:
1. The physical log-likelihood function \(\ln \mathcal{L}(d|\theta^\mu)\).
2. The joint physical prior density \(\pi_{\text{prior}}(\theta^\mu)\).
3. The composite diffeomorphism \(x^\mu = f^\mu(\theta^\nu)\).

By abstracting away the tedious domain transformations, it presents a pure, infinitely smooth, unconstrained log-density \(U(x)\) to the MCMC kernels, allowing the rest of the mathematical machinery to operate in pristine, frictionless elegance.

### REFERENCES

[1] Stan Development Team, "Stan Modeling Language Users Guide and Reference Manual," v2.32 (2023).
