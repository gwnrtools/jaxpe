r"""Maximum-Likelihood Training for Normalizing Flows.

During the training phase of the Global-Local sampler, we continually update the
Normalizing Flow to match the empirical distribution of the MCMC samples.

Motivation & Math
-----------------
A Normalizing Flow constructs a highly complex probability distribution $q_\phi(\mathbf{x})$
by applying a sequence of invertible, differentiable transformations $f_\phi$ to a simple
base distribution (e.g., a standard multivariate normal $p(\mathbf{z})$). By the change
of variables formula, the density of the flow is:
$$ q_\phi(\mathbf{x}) = p(f_\phi^{-1}(\mathbf{x})) \left| \det \frac{\partial f_\phi^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right| $$

To bridge the isolated modes of a gravitational-wave posterior, we train the flow to
emulate the exact target geometry. We do this by minimizing the Kullback-Leibler (KL)
divergence from the empirical MCMC sample distribution to the flow distribution, which
is mathematically equivalent to maximizing the log-likelihood of the buffered samples:
$$ \mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \log q_\phi(\mathbf{x}_i) $$

By minimizing this negative log-likelihood loss using stochastic gradient descent
(Adam), the flow learns to place high probability mass exactly where the MCMC chains
have explored, establishing a global, data-driven proposal for Metropolis-Hastings
that satisfies detailed balance.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@eqx.filter_jit
def _epoch(flow_params, flow_static, opt_state, optimizer, batches):
    """One pass over pre-batched data of shape (n_batches, batch_size, n_dim)."""

    def loss_fn(params, batch):
        flow = eqx.combine(params, flow_static)
        return -jnp.mean(jax.vmap(flow.log_prob)(batch))

    def step(carry, batch):
        params, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state), loss

    (flow_params, opt_state), losses = jax.lax.scan(
        step, (flow_params, opt_state), batches
    )
    return flow_params, opt_state, jnp.mean(losses)


def fit_flow(
    key,
    proposal,
    samples,
    n_epochs: int = 8,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
):
    """
    Fit ``proposal`` to ``samples`` of shape (n_samples, n_dim) by maximum likelihood.

    The MCMC samples are used to recompute the affine whitening constants (mean and standard
    deviation). The data is then whitened, batched, and the spline flow's parameters are
    updated using the Adam optimizer to minimize the negative log-likelihood.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key for random batch permutation.
    proposal : FlowProposal
        The initial un-trained (or partially trained) flow.
    samples : jax.Array
        A buffer of MCMC samples of shape (n_samples, n_dim).
    n_epochs : int, default=8
        Number of full passes over the sample buffer.
    batch_size : int, default=1024
        Batch size for gradient descent.
    learning_rate : float, default=1e-3
        Learning rate for the Adam optimizer.

    Returns
    -------
    tuple[FlowProposal, list]
        The trained flow proposal and a list of the mean loss per epoch.
    """
    mean = jnp.mean(samples, axis=0)
    std = jnp.maximum(jnp.std(samples, axis=0), 1e-8)
    proposal = eqx.tree_at(lambda p: (p.mean, p.std), proposal, (mean, std))
    z = (samples - mean) / std

    n = z.shape[0]
    batch_size = min(batch_size, n)
    n_batches = n // batch_size

    optimizer = optax.adam(learning_rate)
    flow_params, flow_static = eqx.partition(proposal.flow, eqx.is_inexact_array)
    opt_state = optimizer.init(flow_params)

    losses = []
    for _ in range(n_epochs):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n)[: n_batches * batch_size]
        batches = z[perm].reshape(n_batches, batch_size, -1)
        flow_params, opt_state, loss = _epoch(
            flow_params, flow_static, opt_state, optimizer, batches
        )
        losses.append(float(loss))

    flow = eqx.combine(flow_params, flow_static)
    return eqx.tree_at(lambda p: p.flow, proposal, flow), losses
