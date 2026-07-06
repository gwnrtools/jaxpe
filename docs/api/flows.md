---
title: flows
parent: jaxpe
layout: default
nav_order: 4
---

# Sec. IV: Measure Theory and Normalizing Flows (`jaxpe.flows`)
{: .no_toc }

1. TOC
{:toc}

A primary topological challenge in gravitational-wave inference is the existence of disconnected posterior support. A local MCMC transition kernel driven exclusively by covariant gradients will systematically map a single connected domain, exhibiting non-ergodicity when degenerate modes are separated by regions of vanishing probability measure. To bridge these topological barriers, `jaxpe` employs Normalizing Flows to construct a global, continuous proxy measure of the exact target posterior.

## Pushforward Measures and Diffeomorphisms

A Normalizing Flow constructs a highly complex target measure \\(\mu_Q\\) by transporting a simple, tractable base measure \\(\mu_P\\) (typically an isotropic Gaussian) across a smooth manifold via a parameterized diffeomorphism \\(f_\phi : \mathcal{Z} \to \mathcal{X}\\).

In the language of measure theory and differential geometry, we define the base probability density \\(p(z)\\) as a \\(D\\)-form on the base manifold \\(\mathcal{Z}\\):

$$
\omega_P = p(z^1, \dots, z^D) dz^1 \wedge dz^2 \wedge \dots \wedge dz^D
$$

The mapping \\(f_\phi\\) induces a pullback on this differential form, defining the generated density \\(q_\phi(x)\\) on the target manifold \\(\mathcal{X}\\). By the fundamental properties of the wedge product under coordinate transformations, the exact density of the generated samples \\(x^\mu = f_\phi^\mu(z^\nu)\\) is rigorously given by the pushforward density:

$$
q_\phi(x) = \left( f_{\phi*} p \right)(x) = p(f_\phi^{-1}(x)) \left| \det \left( \frac{\partial (f_\phi^{-1})^\mu}{\partial x^\nu} \right) \right|
$$

Because we must evaluate both the forward transformation \\(f_\phi\\) and the Jacobian determinant \\(|\det \partial_\nu (f_\phi^{-1})^\mu|\\) millions of times, flow models strictly constrain the neural network architecture to ensure the Jacobian matrix is lower-triangular. This reduces the determinant computation from \\(O(D^3)\\) to \\(O(D)\\).

### Rational-Quadratic Spline Coupling Layers

The workhorse of `jaxpe` is an autoregressive architecture parameterized by Rational-Quadratic Splines [1]. The state vector \\(x^\mu\\) is partitioned into two sub-spaces: \\(x^A\\) (\\(A = 1 \dots d\\)) and \\(x^I\\) (\\(I = d+1 \dots D\\)). The flow applies an identity map to the first partition, and a conditional element-wise diffeomorphism to the second:

$$
y^A = x^A
$$

$$
y^I = g_\theta(x^I ; x^A)
$$

The conditional mapping \\(g_\theta\\) is a monotonically increasing spline function defined piece-wise by rational quadratic segments. For a given bin defined by knots \\([x^{(k)}, x^{(k+1)}]\\), the transformation takes the analytic form:

$$
y = \frac{\alpha_2 x^2 + \alpha_1 x + \alpha_0}{\beta_2 x^2 + \beta_1 x + \beta_0}
$$

The neural network outputs the knot coordinates and boundary derivatives, fixing the coefficients \\(\alpha_i, \beta_i\\) to ensure exact \\(C^1\\) continuity. The Jacobian of this coupling layer is block-lower-triangular, \\(\partial y^\mu / \partial x^\nu = \begin{pmatrix} \delta^A_B & 0 \\ \partial_B g^I & \partial_J g^I \end{pmatrix}\\), rendering its determinant trivially equal to \\(\prod_I \partial_I g^I\\).

In `jaxpe`, this exact architecture is constructed via the [`make_flow`](#make_flow) function, which returns a parameterized [`FlowProposal`](#flowproposal):

```python
from jaxpe.flows.interface import make_flow

flow = make_flow(n_features=15, n_layers=8, hidden_size=64)
```

### Continuous Normalizing Flows (CNFs)

While discrete coupling layers are computationally cheap, they introduce artificial architectural asymmetries depending on the ordering of the partitions. An elegant, mathematically pure alternative is the Continuous Normalizing Flow (CNF). Instead of a discrete mapping, we define the diffeomorphism as the solution to an Ordinary Differential Equation (ODE) governed by a neural vector field \\(v_\phi(x, t)\\):

$$
\frac{dx(t)}{dt} = v_\phi(x(t), t)
$$

The evolution of the log-density along the trajectory of the particle is governed strictly by the instantaneous divergence of the vector field, via the continuous change of variables formula:

$$
\log q_\phi(x(t_1)) = \log p(x(t_0)) - \int_{t_0}^{t_1} \nabla \cdot v_\phi(x(t), t) dt
$$

While mathematically beautiful, CNFs require the numerical integration of an ODE solver at every forward pass, rendering them computationally heavier than discrete Rational-Quadratic Splines for high-dimensional gravitational-wave inference.

## Variational Training via the Kullback-Leibler Divergence

To optimize the diffeomorphic mapping such that the pushforward measure \\(Q_\phi\\) converges to the exact topology of the true posterior measure \\(P\\), we minimize the Kullback-Leibler (KL) divergence from \\(P\\) to \\(Q_\phi\\). 

The KL divergence is the expectation of the logarithmic Radon-Nikodym derivative between the two measures:

$$
D_{KL}(P || Q_\phi) = \int_{\mathcal{X}} \log\left( \frac{d P}{d Q_\phi} \right) d P = \int_{\mathcal{X}} \pi(x) \log\left( \frac{\pi(x)}{q_\phi(x)} \right) d^Dx
$$

Because the target density \\(\pi(x)\\) is entirely independent of our neural network parameters \\(\phi\\), minimizing this functional is strictly isomorphic to maximizing the empirical log-likelihood over the buffered ensemble of \\(N\\) MCMC samples \\(x_{(i)}^\mu\\):

$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \log q_\phi(x_{(i)})
$$

The `flows` module wraps this rigorous variational machinery using [flowjax](https://github.com/danielward27/flowjax), executing JAX-native training loops that rapidly adapt the diffeomorphic weights \\(\phi\\) on GPU accelerators using stochastic gradient descent optimizers like Adam.

This training loop is robustly executed by the [`fit_flow`](#fit_flow) function:

```python
from jaxpe.flows.trainer import fit_flow

trained_flow, losses = fit_flow(key, flow, samples, learning_rate=1e-3, max_epochs=100)
```

## API Reference

### `make_flow`
**`jaxpe.flows.interface.make_flow(n_features, n_layers, ...)`**
Constructs a Normalizing Flow architecture using Rational-Quadratic Splines. Returns a parameterized `FlowProposal` ready for training.

### `FlowProposal`
**`jaxpe.flows.interface.FlowProposal`**
An `equinox.Module` encapsulating the forward diffeomorphism, inverse mapping, and probability evaluation for the Normalizing Flow.

### `fit_flow`
**`jaxpe.flows.trainer.fit_flow(key, flow, samples, ...)`**
Executes the variational training loop over the buffered MCMC samples using Optax, minimizing the Kullback-Leibler divergence.

---

### REFERENCES

[1] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, "Neural Spline Flows," Adv. Neural Inf. Process. Syst. **32** (2019).  
[2] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using Real NVP," ICLR (2017).
