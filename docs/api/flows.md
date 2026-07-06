---
title: flows
parent: jaxpe
layout: default
---

# Sec. IV: Normalizing Flows (`jaxpe.flows`)
{: .no_toc }

1. TOC
{:toc}

Let us pause to consider the central topological bottleneck of our problem: isolated, disconnected modes. If a mountain ridge separates two valleys, a gradient-following algorithm will never cross it. To bridge these modes, `jaxpe` relies on the mathematical elegance of Normalizing Flows.

## The Normalizing Flow Architecture

Think of a Normalizing Flow as molding a simple, structureless block of clay into an intricate, twisting sculpture, without ever tearing the clay or folding it onto itself. 

Mathematically, we start with a base distribution that is trivial to sample from—typically an isotropic standard normal $$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$. We then apply a sequence of invertible, differentiable transformations (diffeomorphisms) $$f_\phi$$, parameterized by a deep neural network with weights $$\phi$$. By the fundamental change of variables theorem from multivariate calculus, the exact density of the generated samples $$\mathbf{x} = f_\phi(\mathbf{z})$$ is rigorously given by:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \mathbf{J}_{f_\phi^{-1}}(\mathbf{x}) \right|
$$

where $$\mathbf{J}$$ is the Jacobian matrix. Because we need to evaluate both the forward transformation and its Jacobian determinant billions of times, flow models restrict the neural network architecture such that the Jacobian is lower-triangular, making the determinant computation $$O(D)$$ instead of $$O(D^3)$$.

### Rational-Quadratic Spline Coupling Layers

The workhorse of `jaxpe` is an autoregressive architecture parameterized by Rational-Quadratic Splines [1]. The parameter vector $$\mathbf{x}$$ is split into two halves: $$\mathbf{x}_{1:d}$$ and $$\mathbf{x}_{d+1:D}$$. The first half is left entirely unchanged, while the second half is transformed element-wise, but conditionally on the first half:

$$
\mathbf{y}_{1:d} = \mathbf{x}_{1:d}
$$
$$
\mathbf{y}_{d+1:D} = g_\theta(\mathbf{x}_{d+1:D} ; \mathbf{x}_{1:d})
$$

Here, $$g_\theta$$ is a monotonically increasing spline function defined piece-wise by rational quadratic segments. The neural network outputs the coordinates of $$K$$ knots $$(x^{(k)}, y^{(k)})$$, along with the exact derivatives at these knots. Inside each bin $$[x^{(k)}, x^{(k+1)}]$$, the transformation takes the analytic form:

$$
y = \frac{\alpha_2 x^2 + \alpha_1 x + \alpha_0}{\beta_2 x^2 + \beta_1 x + \beta_0}
$$

The coefficients are uniquely determined to ensure $$C^1$$ continuity across the knots. This allows the flow to warp space with extreme flexibility, easily molding our Gaussian clay into the "banana" shapes and multimodal rings common in gravitational-wave posteriors.

## Maximum-Likelihood Training

But how do we teach the neural network to mold the clay into the exact shape of our target posterior? We train it by minimizing the Kullback-Leibler (KL) divergence from the empirical distribution of the MCMC samples (the true posterior) to our flow distribution $$q_\phi$$. 

The KL divergence is defined mathematically as the expected logarithmic difference:

$$
D_{KL}(P || Q) = \int p(\mathbf{x}) \log\left( \frac{p(\mathbf{x})}{q_\phi(\mathbf{x})} \right) d\mathbf{x}
$$

Because the target density $$p(\mathbf{x})$$ is independent of our neural network parameters $$\phi$$, minimizing this integral is strictly equivalent to maximizing the log-likelihood of the buffered MCMC samples $$\mathbf{x}_i$$:

$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \log q_\phi(\mathbf{x}_i)
$$

The `flows` module wraps this elegant machinery using [flowjax](https://github.com/danielward27/flowjax), providing robust JAX-native training loops that execute effortlessly on the GPU.

### REFERENCES

[1] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, "Neural Spline Flows," Adv. Neural Inf. Process. Syst. **32** (2019).  
[2] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using Real NVP," ICLR (2017).
