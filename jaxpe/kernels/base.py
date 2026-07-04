"""Kernel protocol and the vmapped multi-chain runner.

A kernel is a pure Equinox module: hyperparameters are fields, and
``step(key, state, logp_fn)`` maps one chain state to the next. States cache the
log-density and its gradient at the current position so each MH step costs one new
density (and gradient) evaluation at the proposal only.

``run_chains`` scans a kernel over time and vmaps it over chains; the whole loop jits
once, which is what makes many-chain GPU sampling cheap.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

LogProbFn = Callable[[jax.Array], jax.Array]


class KernelState(NamedTuple):
    x: jax.Array  # (n_dim,) position in unconstrained space
    log_prob: jax.Array  # scalar, cached
    grad: jax.Array  # (n_dim,), cached; zeros for gradient-free kernels
    aux: Any = ()  # kernel-specific carry (e.g. ULD velocity)


class StepInfo(NamedTuple):
    accepted: jax.Array  # 1.0 if the proposal was taken
    log_accept_ratio: jax.Array


class Kernel(eqx.Module):
    """Base class; subclasses implement ``init`` and ``step`` as pure functions."""

    needs_gradient: eqx.AbstractClassVar[bool]
    has_accept_prob: ClassVar[bool] = True  # False for unadjusted (no-MH) kernels

    def init(self, x, logp_fn: LogProbFn) -> KernelState:
        if self.needs_gradient:
            log_prob, grad = jax.value_and_grad(logp_fn)(x)
        else:
            log_prob, grad = logp_fn(x), jnp.zeros_like(x)
        return KernelState(x=x, log_prob=log_prob, grad=grad)

    def step(self, key, state: KernelState, logp_fn: LogProbFn) -> tuple[KernelState, StepInfo]:
        raise NotImplementedError


def mh_accept(key, state: KernelState, proposal: KernelState, log_accept_ratio):
    """Metropolis-Hastings accept/reject between two cached states."""
    log_alpha = jnp.minimum(log_accept_ratio, 0.0)
    accept = jnp.log(jax.random.uniform(key)) < log_alpha
    new = jax.tree.map(lambda p, c: jnp.where(accept, p, c), proposal, state)
    return new, StepInfo(accepted=accept.astype(state.x.dtype), log_accept_ratio=log_alpha)


@partial(jax.jit, static_argnames=("kernel_static", "logp_fn", "n_steps", "thin"))
def _run_chains_jit(kernel_params, kernel_static, key, states, logp_fn, n_steps, thin):
    kernel = eqx.combine(kernel_params, kernel_static)

    def one_step(states, key):
        keys = jax.random.split(key, states.x.shape[0])
        return jax.vmap(lambda k, s: kernel.step(k, s, logp_fn))(keys, states)

    def thinned_block(states, key):
        states, infos = jax.lax.scan(one_step, states, jax.random.split(key, thin))
        last_info = jax.tree.map(lambda a: a[-1], infos)
        mean_acc = jnp.mean(infos.accepted)
        return states, (states.x, states.log_prob, last_info._replace(accepted=mean_acc))

    n_out = n_steps // thin
    states, (xs, logps, infos) = jax.lax.scan(
        thinned_block, states, jax.random.split(key, n_out)
    )
    return states, xs, logps, infos


def run_chains(key, kernel: Kernel, logp_fn: LogProbFn, x0, n_steps: int, thin: int = 1):
    """Run ``n_steps`` of ``kernel`` on chains initialized at ``x0`` of shape (n_chains, n_dim).

    Returns ``(final_states, xs, log_probs, infos)`` where ``xs`` has shape
    ``(n_steps // thin, n_chains, n_dim)`` (positions after every ``thin`` steps) and
    ``infos.accepted`` holds per-block mean acceptance.
    """
    states = jax.vmap(lambda x: kernel.init(x, logp_fn))(x0)
    params, static = eqx.partition(kernel, eqx.is_array)
    return _run_chains_jit(params, static, key, states, logp_fn, n_steps, thin)
