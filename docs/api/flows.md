---
title: flows
parent: jaxpe
layout: default
---

# Sec. IV: Normalizing Flows (`jaxpe.flows`)
{: .no_toc }

1. TOC
{:toc}

In this section, we detail the normalizing flow architecture that enables `jaxpe` to leap across disconnected modes by analytically learning the topology of the posterior.

## The Normalizing Flow Architecture

A Normalizing Flow constructs a highly complex probability distribution $$q_\phi(\mathbf{x})$$ by applying a sequence of invertible, differentiable transformations (diffeomorphisms) $$f_\phi$$ to a simple base distribution (e.g., a standard multivariate normal $$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$). By the change of variables formula, the exact density of the generated samples $$\mathbf{x} = f_\phi(\mathbf{z})$$ is:

$$
q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \mathbf{J}_{f_\phi^{-1}}(\mathbf{x}) \right|
$$

where $$\mathbf{J}$$ is the Jacobian matrix. To ensure that both the forward evaluation and the determinant calculation are fast ($$O(D)$$ instead of $$O(D^3)$$ for dimension $$D$$), flow models restrict the neural network architecture such that the Jacobian is lower-triangular. 

### Rational-Quadratic Spline Coupling Layers

The standard choice in `jaxpe` is an autoregressive architecture parameterized by Rational-Quadratic Splines [1]. The parameter vector $$\mathbf{x}$$ is split into two halves: $$\mathbf{x}_{1:d}$$ and $$\mathbf{x}_{d+1:D}$$. The first half is left unchanged, while the second half is transformed element-wise conditionally on the first half:

$$
\mathbf{y}_{1:d} = \mathbf{x}_{1:d}
$$
$$
\mathbf{y}_{d+1:D} = g_\theta(\mathbf{x}_{d+1:D} ; \mathbf{x}_{1:d})
$$

Here, $$g_\theta$$ is a monotonically increasing spline function defined piece-wise by rational quadratic segments. The neural network (conditional on $$\mathbf{x}_{1:d}$$) outputs the coordinates of $$K$$ knots $$(x^{(k)}, y^{(k)})$$, along with the derivatives at these knots. Inside each bin $$[x^{(k)}, x^{(k+1)}]$$, the transformation takes the analytic form:

$$
y = \frac{\alpha_2 x^2 + \alpha_1 x + \alpha_0}{\beta_2 x^2 + \beta_1 x + \beta_0}
$$

where the coefficients are uniquely determined by the neural network's knot outputs to ensure $$C^1$$ continuity. This allows the flow to warp space with extreme flexibility, easily capturing the "banana" shapes and multimodal rings common in gravitational-wave posteriors, while retaining an exact analytic inverse and determinant.

## Maximum-Likelihood Training

To bridge the isolated modes of a gravitational-wave posterior, we train the flow to emulate the exact target geometry. We minimize the Kullback-Leibler (KL) divergence from the empirical MCMC sample distribution (drawn during the warm-up phase) to the flow distribution. Mathematically, this is equivalent to maximizing the log-likelihood of the buffered samples:

$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \log q_\phi(\mathbf{x}_i)
$$

The `flows` module wraps this machinery using [flowjax](https://github.com/danielward27/flowjax), providing robust JAX-native training loops that execute directly on the GPU.

### REFERENCES

[1] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, "Neural Spline Flows," Adv. Neural Inf. Process. Syst. **32** (2019).  
[2] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using Real NVP," ICLR (2017).
