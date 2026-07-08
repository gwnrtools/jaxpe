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

This section formalizes the geometrical substrate of `jaxpe`. Continuous Hamiltonian integration fundamentally assumes that the parameter space is a smooth differentiable manifold topologically equivalent to $$\mathbb{R}^D$$. Boundaries, sharp truncations, and finite intervals break the continuity of Hamilton's equations, inducing pathological reflection of the conjugate momenta.

## Boundary Removal via Diffeomorphisms

The physical universe dictates strict boundaries: a black hole mass $$m > 0$$, a dimensionless spin $$a \in [0, 1]$$, an inclination $$\iota \in [0, \pi]$$. The physical parameter manifold $$\mathcal{M}_\theta$$ is therefore a manifold with boundaries.

To satisfy the geometrical requirements of the MCMC kernels, we must construct a smooth, bijective, and differentiable mapping—a diffeomorphism $$f: \mathcal{M}_\theta \to \mathcal{X}$$—that maps the bounded physical space onto an unconstrained latent manifold $$\mathcal{X} \cong \mathbb{R}^D$$ [1].

### The Logarithmic Diffeomorphism

For parameters bounded strictly below by a threshold $$a$$ (e.g., $$d_L \in (0, \infty)$$), we apply the natural logarithm to push the boundary to asymptotic infinity:

$$
x = \ln(\theta - a) \quad \iff \quad \theta = \exp(x) + a
$$

### The Scaled Logit Diffeomorphism

For parameters strictly bounded in a finite interval $$\theta \in (a, b)$$ (e.g., cosine inclination $$\cos \iota \in (-1, 1)$$), we utilize the scaled logit function, stretching both boundaries to $$\pm \infty$$:

$$
x = \ln\left( \frac{\theta - a}{b - \theta} \right) \quad \iff \quad \theta = a + \frac{b - a}{1 + \exp(-x)}
$$

## Jacobians and Target Density Adjustments

Upon mapping the posterior measure to the unconstrained manifold $$\mathcal{X}$$, the probability density undergoes geometric distortion. The pushforward of the posterior measure $$\pi_\Theta$$ under the diffeomorphism $$f$$ induces a strict volume correction governed by the Jacobian matrix $$J^\mu_\nu = \partial \theta^\mu / \partial x^\nu$$.

By the change of variables theorem, the exact unconstrained target density is:

$$
\pi_X(x) = \pi_\Theta(f^{-1}(x)) \left| \det \left( \frac{\partial \theta^\mu}{\partial x^\nu} \right) \right|
$$

Because our bijections are applied element-wise, the Jacobian matrix is strictly diagonal, reducing the determinant to a simple product of scalar derivatives. For the logit transformation, the explicit derivative element is:

$$
\frac{\partial \theta^i}{\partial x^i} = \frac{(\theta^i - a)(b - \theta^i)}{b - a}
$$

The `jaxpe.core` module handles these adjustments automatically, ensuring that the gradient covector $$\partial_\mu U(x)$$ perfectly aligns with the warped geometry of $$\mathcal{X}$$. In code, this geometry is encapsulated by subclasses of [`jaxpe.core.transforms.Bijection`](#bijection), such as `Interval` and `Identity`:

```python
from jaxpe.core.transforms import Interval

# Transform bounded parameter [0, 1] to unconstrained real line
bijection = Interval(low=0.0, high=1.0)
x = bijection.inverse(theta)
```

## `InferenceProblem`

The `InferenceProblem` class acts as the grand geometrical orchestrator. It encapsulates:
1. The physical log-likelihood function $$\ln \mathcal{L}(d \mid \theta^\mu)$$.
2. The joint physical prior density $$\pi_{\text{prior}}(\theta^\mu)$$.
3. The composite diffeomorphism $$x^\mu = f^\mu(\theta^\nu)$$.

By encapsulating the domain transformations, the `InferenceProblem` yields a globally smooth, unconstrained scalar potential $$U(x)$$ that rigorously satisfies the continuity requirements for symplectic integration.

```python
from jaxpe.core.problem import InferenceProblem

inference_problem = InferenceProblem(
    prior=my_prior,
    log_likelihood=my_likelihood_fn
)
unconstrained_samples = inference_problem.sample_unconstrained(key, n=100)
unconstrained_logp = inference_problem.log_posterior(unconstrained_samples)
```

## API Reference

### `InferenceProblem`
**`jaxpe.core.problem.InferenceProblem(prior, log_likelihood)`**
Encapsulates the physical prior distribution and the likelihood function, handling the automatic calculation of the target log-density in the unconstrained space (including Jacobian log-determinants).

### `Bijection`
**`jaxpe.core.transforms.Bijection`**
The base class for geometric diffeomorphisms (like `Identity`, `Affine`, `Interval`) that seamlessly push parameters to the unconstrained real line while tracking the local volume Jacobian.

---

### REFERENCES
**[1]** Stan Development Team, "Stan Modeling Language Users Guide and Reference Manual," v2.32 (2023).
