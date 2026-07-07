"""Kernel protocol and the vmapped multi-chain runner.

This module defines the foundational abstractions for Markov Chain Monte Carlo (MCMC)
sampling in `jaxpe`. It establishes what a 'Kernel' is, how the state of a Markov Chain
is represented, and provides the Metropolis-Hastings acceptance logic.

Motivation
----------
In Bayesian Parameter Estimation, we wish to sample from a posterior density $P(\theta|D) \propto e^{\log p(\theta)}$.
Since direct sampling is usually intractable in high dimensions, we use MCMC. MCMC constructs a Markov
chain where each step $\theta_{i+1}$ depends only on the previous step $\theta_i$.

A valid MCMC kernel must satisfy "Detailed Balance" (reversibility):
$$ \pi(\theta) K(\theta \to \theta') = \pi(\theta') K(\theta' \to \theta) $$
where $\pi(\theta)$ is the target density and $K$ is the transition probability. If detailed balance holds,
the chain's stationary distribution is guaranteed to be exactly the target posterior $\pi(\theta)$.

Implementation in JAXPE
-----------------------
A kernel is implemented as a pure Equinox module. The ``step(key, state, logp_fn)`` method
maps one chain state to the next. To avoid re-evaluating the expensive log-likelihood (and its gradient)
unnecessarily, the `KernelState` caches the log-density and gradient at the current position.
Thus, each Metropolis-Hastings step only requires evaluating the target density at the proposed new position.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

LogProbFn = Callable[[jax.Array], jax.Array]


class KernelState(NamedTuple):
    """
    State of a single Markov Chain at a given step.

    In MCMC, we transition from state $\theta_t$ to $\theta_{t+1}$.
    Caching the log-probability and its gradient at $\theta_t$ prevents redundant
    computations during the Metropolis-Hastings acceptance step.
    """

    x: jax.Array  # (n_dim,) position in unconstrained space
    log_prob: jax.Array  # scalar, cached log-probability at `x`
    grad: (
        jax.Array
    )  # (n_dim,), cached gradient of log-probability at `x`; zeros for gradient-free kernels
    aux: Any = ()  # kernel-specific carry (e.g. ULD velocity or HMC momentum)


class StepInfo(NamedTuple):
    accepted: jax.Array  # 1.0 if the proposal was taken
    log_accept_ratio: jax.Array


class Kernel(eqx.Module):
    """
    Base class for MCMC kernels (e.g., MALA, HMC, Random Walk).

    Subclasses must implement the ``step`` method, which takes a ``KernelState`` and proposes
    a new state based on specific transition dynamics (like Langevin diffusion or Hamiltonian mechanics).
    """

    needs_gradient: eqx.AbstractClassVar[bool]
    has_accept_prob: ClassVar[bool] = True  # False for unadjusted (no-MH) kernels

    def init(self, x, logp_fn: LogProbFn) -> KernelState:
        if self.needs_gradient:
            log_prob, grad = jax.value_and_grad(logp_fn)(x)
        else:
            log_prob, grad = logp_fn(x), jnp.zeros_like(x)
        return KernelState(x=x, log_prob=log_prob, grad=grad)

    def step(
        self, key, state: KernelState, logp_fn: LogProbFn
    ) -> tuple[KernelState, StepInfo]:
        raise NotImplementedError


def mh_accept(key, state: KernelState, proposal: KernelState, log_accept_ratio):
    """
    Metropolis-Hastings (MH) accept/reject step.

    Motivation & Math
    -----------------
    To ensure our MCMC chain samples from the true posterior, we must correct for any
    bias in our proposal mechanism. We do this via the MH acceptance probability $\alpha$:
    $$ \alpha = \min\left(1, \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}\right) $$
    where $\pi(\theta)$ is the target density, $q(\theta'|\theta)$ is the probability of proposing
    $\theta'$ from $\theta$, and $q(\theta|\theta')$ is the reverse.

    In log-space, this becomes:
    $$ \log \alpha = \min(0, \log \pi(\theta') - \log \pi(\theta) + \log q(\theta|\theta') - \log q(\theta'|\theta)) $$
    The term `log_accept_ratio` passed to this function corresponds to the term inside the min(0, ...).

    We accept the proposal if a random uniform draw $u \sim U(0,1)$ satisfies $\log u < \log \alpha$.
    If rejected, the chain stays at the current state.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The PRNG key for the random uniform draw.
    state : KernelState
        The current state of the chain.
    proposal : KernelState
        The proposed new state.
    log_accept_ratio : jax.Array
        The computed log acceptance ratio (before taking min with 0).

    Returns
    -------
    tuple[KernelState, StepInfo]
        The next state (either the proposal or the original state) and diagnostic info.
    """
    log_alpha = jnp.minimum(log_accept_ratio, 0.0)
    accept = jnp.log(jax.random.uniform(key)) < log_alpha
    new = jax.tree.map(lambda p, c: jnp.where(accept, p, c), proposal, state)
    return new, StepInfo(
        accepted=accept.astype(state.x.dtype), log_accept_ratio=log_alpha
    )


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
        return states, (
            states.x,
            states.log_prob,
            last_info._replace(accepted=mean_acc),
        )

    n_out = n_steps // thin
    states, (xs, logps, infos) = jax.lax.scan(
        thinned_block, states, jax.random.split(key, n_out)
    )
    return states, xs, logps, infos


def run_chains(
    key, kernel: Kernel, logp_fn: LogProbFn, x0, n_steps: int, thin: int = 1
):
    """
    Run an MCMC kernel across many parallel chains efficiently.

    This function utilizes `jax.vmap` to vectorize the kernel step across multiple chains,
    and `jax.lax.scan` to compile the time-stepping loop. The entire execution is Just-In-Time
    (JIT) compiled into a single optimized block, which makes multi-chain sampling extremely
    fast, especially on GPUs.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random seed for the chains.
    kernel : Kernel
        The instantiated MCMC kernel (e.g., MALA, HMC).
    logp_fn : LogProbFn
        The target log-probability density function.
    x0 : jax.Array
        Initial positions for all chains, with shape (n_chains, n_dim).
    n_steps : int
        Total number of MCMC steps to take per chain.
    thin : int, default=1
        Thinning factor. If thin=10, only every 10th step is saved to memory, reducing
        correlation between saved samples and saving RAM.

    Returns
    -------
    tuple
        (final_states, xs, log_probs, infos)
        - final_states: The `KernelState` of the chains at the very end.
        - xs: The history of positions, shape ``(n_steps // thin, n_chains, n_dim)``.
        - log_probs: The history of log-probabilities at those positions.
        - infos.accepted: The per-block mean acceptance rate.
    """
    states = jax.vmap(lambda x: kernel.init(x, logp_fn))(x0)
    params, static = eqx.partition(kernel, eqx.is_array)
    return _run_chains_jit(params, static, key, states, logp_fn, n_steps, thin)
