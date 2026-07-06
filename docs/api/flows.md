---
title: flows
parent: jaxpe
layout: default
---

# Sec. IV: Normalizing Flows (`jaxpe.flows`)
{: .no_toc }

1. TOC
{:toc}

In this section, we detail the normalizing flow architecture that enables `jaxpe` to leap across disconnected modes.

## The Normalizing Flow Architecture

A Normalizing Flow constructs a highly complex probability distribution $$q_\phi(\mathbf{x})$$ by applying a sequence of invertible, differentiable transformations $$f_\phi$$ to a simple base distribution (e.g., a standard multivariate normal $$p(\mathbf{z})$$). By the change of variables formula, the density of the flow is:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \frac{\partial f_\phi^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

The standard choice in `jaxpe` is a rational-quadratic-spline coupling flow, which can effectively learn multi-modal distributions like the time-delay ring or phase degeneracies.

## Maximum-Likelihood Training

To bridge the isolated modes of a gravitational-wave posterior, we train the flow to emulate the exact target geometry. We minimize the Kullback-Leibler (KL) divergence from the empirical MCMC sample distribution to the flow distribution, which is mathematically equivalent to maximizing the log-likelihood of the buffered samples:

$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \log q_\phi(\mathbf{x}_i)
$$

The `flows` module wraps this machinery using [flowjax](https://github.com/danielward27/flowjax), and provides robust training loops.
