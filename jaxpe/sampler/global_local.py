"""flowMC-style global-local sampler.

Training phase, repeated ``n_training_loops`` times:
  1. a block of local kernel steps on all chains (vmapped, jitted scan);
  2. step-size / preconditioner adaptation from the block's acceptance and ensemble;
  3. maximum-likelihood refresh of the flow on the buffered chain history;
  4. a block of Metropolis-Hastings steps using the flow as an independence proposal
     (the *global* moves that hop between posterior modes).

Production phase: adaptation off, flow frozen, alternating local/global blocks whose
samples form the returned posterior chains. With the flow fixed, the production chains
are exactly Markovian with the correct invariant density.
"""

from dataclasses import dataclass


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.problem import InferenceProblem
from ..flows import fit_flow, make_flow
from ..kernels import (
    TARGET_ACCEPTANCE,
    adapted_step_size,
    ensemble_cov,
    ensemble_scale,
    run_chains,
    with_updates,
)


@dataclass
class GlobalLocalConfig:
    n_chains: int = 128
    n_training_loops: int = 12
    n_production_loops: int = 6
    n_local_steps: int = 100
    n_global_steps: int = 50
    local_thin: int = 5  # keep every k-th local sample (buffering and production)
    buffer_size: int = 50_000
    # flow architecture
    flow_layers: int = 8
    knots: int = 8
    interval: float = 5.0
    nn_width: int = 64
    nn_depth: int = 1
    # flow training
    n_epochs: int = 8
    batch_size: int = 1024
    learning_rate: float = 1e-3
    # adaptation
    adapt_step_size: bool = True
    adapt_scale: bool = True
    target_acceptance: float | None = None  # default looked up per kernel type
    use_global: bool = True


@dataclass
class SamplerResults:
    """Production samples in unconstrained space plus run history."""

    samples: np.ndarray  # (n_kept, n_chains, n_dim), unconstrained
    log_prob: np.ndarray  # (n_kept, n_chains)
    local_acceptance: list  # per training/production loop means
    global_acceptance: list
    flow_losses: list  # per training loop mean losses
    flow: object  # trained FlowProposal
    kernel: object  # kernel with final adapted hyperparameters

    def flat(self):
        return self.samples.reshape(-1, self.samples.shape[-1])


@eqx.filter_jit
def _global_block(flow, key, x, log_prob, logp_fn, n_steps: int):
    """n_steps of independence MH with the flow as proposal, vmapped over chains."""
    n_chains = x.shape[0]
    log_q = jax.vmap(flow.log_prob)(x)

    def step(carry, key):
        x, log_prob, log_q = carry
        k_prop, k_acc = jax.random.split(key)
        y = flow.sample(k_prop, (n_chains,))
        log_q_y = jax.vmap(flow.log_prob)(y)
        log_p_y = jax.vmap(logp_fn)(y)
        log_ratio = log_p_y - log_prob + log_q - log_q_y
        accept = jnp.log(jax.random.uniform(k_acc, (n_chains,))) < jnp.minimum(log_ratio, 0.0)
        x = jnp.where(accept[:, None], y, x)
        log_prob = jnp.where(accept, log_p_y, log_prob)
        log_q = jnp.where(accept, log_q_y, log_q)
        return (x, log_prob, log_q), (x, log_prob, jnp.mean(accept))

    (x, log_prob, _), (xs, log_probs, acc) = jax.lax.scan(
        step, (x, log_prob, log_q), jax.random.split(key, n_steps)
    )
    return x, log_prob, xs, log_probs, jnp.mean(acc)


class Sampler:
    """Global-local sampler over an ``InferenceProblem`` or a raw log-density.

    Parameters
    ----------
    kernel
        Local kernel instance (its step size / scale act as initial values when
        adaptation is on).
    problem
        Provides the unconstrained log-posterior and prior-based initialization.
    logp_fn, n_dim
        Alternative to ``problem`` for generic densities: a scalar log-density over
        (n_dim,) unconstrained vectors. Chains must then be initialized explicitly
        via ``run(key, x0=...)``.
    """

    def __init__(self, kernel, *, problem: InferenceProblem | None = None,
                 logp_fn=None, n_dim: int | None = None,
                 config: GlobalLocalConfig | None = None):
        if problem is not None:
            self.logp_fn = problem.log_posterior
            self.n_dim = problem.n_dim
        elif logp_fn is not None and n_dim is not None:
            self.logp_fn = logp_fn
            self.n_dim = n_dim
        else:
            raise ValueError("provide either `problem` or (`logp_fn` and `n_dim`)")
        self.problem = problem
        self.kernel = kernel
        self.config = config or GlobalLocalConfig()

    def _target_acceptance(self):
        if self.config.target_acceptance is not None:
            return self.config.target_acceptance
        return TARGET_ACCEPTANCE.get(type(self.kernel).__name__, 0.574)

    def run(self, key, x0=None) -> SamplerResults:
        cfg = self.config
        key, k_flow, k_init = jax.random.split(key, 3)

        if x0 is None:
            if self.problem is None:
                raise ValueError("x0 is required when no InferenceProblem is given")
            x0 = self.problem.sample_unconstrained(k_init, cfg.n_chains)
        x0 = jnp.asarray(x0)

        flow = make_flow(
            k_flow, self.n_dim, flow_layers=cfg.flow_layers, knots=cfg.knots,
            interval=cfg.interval, nn_width=cfg.nn_width, nn_depth=cfg.nn_depth,
        )
        kernel = self.kernel
        target = self._target_acceptance()

        buffer = None
        local_acc, global_acc, flow_losses = [], [], []

        # ---- training phase ----
        for _ in range(cfg.n_training_loops):
            key, k_loc, k_fit, k_glob = jax.random.split(key, 4)
            states, xs, logps, infos = run_chains(
                k_loc, kernel, self.logp_fn, x0, cfg.n_local_steps, thin=cfg.local_thin
            )
            x0, logp0 = states.x, states.log_prob
            acc = float(jnp.mean(infos.accepted))
            local_acc.append(acc)

            new_samples = xs.reshape(-1, self.n_dim)
            buffer = new_samples if buffer is None else jnp.concatenate([buffer, new_samples])
            buffer = buffer[-cfg.buffer_size:]

            # Unadjusted kernels (ULD) always report acceptance 1: skip step-size targeting.
            if cfg.adapt_step_size and type(kernel).has_accept_prob:
                kernel = with_updates(
                    kernel, step_size=adapted_step_size(kernel.step_size, acc, target)
                )
            if cfg.adapt_scale:
                if hasattr(kernel, "scale"):
                    kernel = with_updates(kernel, scale=ensemble_scale(buffer))
                elif hasattr(kernel, "cov") and getattr(kernel, "metric_fn", None) is None:
                    kernel = with_updates(kernel, cov=ensemble_cov(buffer))

            flow, losses = fit_flow(
                k_fit, flow, buffer, n_epochs=cfg.n_epochs,
                batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
            )
            flow_losses.append(losses[-1])

            if cfg.use_global:
                x0, logp0, _, _, g_acc = _global_block(
                    flow, k_glob, x0, logp0, self.logp_fn, cfg.n_global_steps
                )
                global_acc.append(float(g_acc))

        # ---- production phase: adaptation off, flow frozen ----
        kept_x, kept_logp = [], []
        for _ in range(cfg.n_production_loops):
            key, k_loc, k_glob = jax.random.split(key, 3)
            states, xs, logps, infos = run_chains(
                k_loc, kernel, self.logp_fn, x0, cfg.n_local_steps, thin=cfg.local_thin
            )
            x0, logp0 = states.x, states.log_prob
            local_acc.append(float(jnp.mean(infos.accepted)))
            kept_x.append(xs)
            kept_logp.append(logps)

            if cfg.use_global:
                x0, logp0, ys, ylogps, g_acc = _global_block(
                    flow, k_glob, x0, logp0, self.logp_fn, cfg.n_global_steps
                )
                global_acc.append(float(g_acc))
                kept_x.append(ys[:: cfg.local_thin])
                kept_logp.append(ylogps[:: cfg.local_thin])

        return SamplerResults(
            samples=np.concatenate([np.asarray(a) for a in kept_x]),
            log_prob=np.concatenate([np.asarray(a) for a in kept_logp]),
            local_acceptance=local_acc,
            global_acceptance=global_acc,
            flow_losses=flow_losses,
            flow=flow,
            kernel=kernel,
        )

    def to_physical(self, samples):
        """Map unconstrained samples (..., n_dim) to physical space (requires a problem)."""
        if self.problem is None:
            raise ValueError("no InferenceProblem attached")
        flat = jnp.asarray(samples).reshape(-1, self.n_dim)
        phys = jax.vmap(self.problem.prior.to_physical)(flat)
        return np.asarray(phys).reshape(samples.shape)
