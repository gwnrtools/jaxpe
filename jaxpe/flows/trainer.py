"""Maximum-likelihood training of the flow proposal on buffered chain samples."""

from functools import partial

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

    (flow_params, opt_state), losses = jax.lax.scan(step, (flow_params, opt_state), batches)
    return flow_params, opt_state, jnp.mean(losses)


def fit_flow(
    key,
    proposal,
    samples,
    n_epochs: int = 8,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
):
    """Fit ``proposal`` to ``samples`` of shape (n_samples, n_dim) by maximum likelihood.

    The whitening constants are recomputed from ``samples``; the spline flow itself is
    then trained on whitened data with Adam. Returns ``(proposal, mean_losses)``.
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
