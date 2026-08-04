---
title: sampler
parent: API Reference
layout: default
nav_order: 5
---

# Sec. V: Global-Local Orchestration and Detailed Balance (`jaxpe.sampler`)
{: .no_toc }

1. TOC
{:toc}

With a formally trained pushforward measure $$q_\phi$$ accurately mapping the target posterior $$\pi$$, we now confront the problem of orchestration. How do we rigorously couple the local covariant diffusions (HMC/MALA) with the global topological leaps provided by the Normalizing Flow, without violating the fundamental stationarity of the Markov chain?

## Markov Chain Stationarity and Detailed Balance

The sampler operates as a discrete-time stochastic process parameterized by an alternating sequence of transition kernels $$T_{\text{local}}$$ and $$T_{\text{global}}$$ [1]. For the chain to converge to the exact target measure $$\pi(x)$$, each transition kernel $$T(x \to y)$$ must independently leave the target density invariant:

$$
\int_{\mathcal{M}} \pi(x) T(x \to y) d^Dx = \pi(y)
$$

The strongest and most mathematically elegant sufficient condition to satisfy this integral equation is detailed balance (reversibility), which demands that the probability flux from $$x$$ to $$y$$ exactly balances the reverse flux from $$y$$ to $$x$$:

$$
\pi(x) T(x \to y) = \pi(y) T(y \to x)
$$

## The Independence Metropolis-Hastings Transition

During the global phase, the Normalizing Flow proposes independent coordinates $$y \sim q_\phi(y)$$ drawn entirely independently of the current state $$x$$. The corresponding transition probability is defined strictly by the independence proposal kernel: $$K(x \to y) = q_\phi(y)$$.

To rigorously enforce detailed balance over this independence proposal, we subject it to the Metropolis-Hastings filter. The corrected transition kernel is:

$$
T_{\text{global}}(x \to y) = q_\phi(y) \alpha(x, y) + \delta(x - y) \left[ 1 - \int_{\mathcal{M}} q_\phi(y') \alpha(x, y') d^Dy' \right]
$$

where $$\delta(x-y)$$ is the Dirac delta distribution handling rejections, and the acceptance probability $$\alpha(x, y)$$ is uniquely constrained to:

$$
\alpha(x, y) = \min\left(1, \frac{\pi(y) K(y \to x)}{\pi(x) K(x \to y)}\right) = \min\left(1, \frac{\pi(y) q_\phi(x)}{\pi(x) q_\phi(y)}\right)
$$

Because the trained flow measure $$q_\phi$$ closely approximates the exact posterior $$\pi$$, the ratio $$\pi/q_\phi$$ approaches unity. This guarantees that $$\alpha(x, y) \approx 1$$, allowing the Markov chain to traverse large distances across the parameter manifold with vanishingly small rejection rates.

## Ergodicity and the Law of Large Numbers

When detailed balance is satisfied, the Markov chain is guaranteed to be stationary. If the chain is also irreducible and aperiodic (which it trivially is, given the global independence proposals covering the entire support), it is rigorously ergodic. This permits the application of the Birkhoff Ergodic Theorem, which states that time-averages of any observable $$f(x)$$ strictly converge to the spatial averages over the invariant measure:

$$
\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N f(x_{(i)}) = \int_{\mathcal{M}} f(x) \pi(x) d^Dx
$$

This is the foundational theorem that justifies using the discrete samples of our chains to evaluate complex astrophysical quantities like the mean chirp mass or the variance of the luminosity distance.

## Orchestration Implementation (`Sampler`)

The `Sampler` class rigorously orchestrates these transition kernels in a mathematically synchronized loop. Under the hood, it leverages JAX's `lax.scan` primitive to compile the alternating application of $$T_{\text{local}}$$ and $$T_{\text{global}}$$ into a monolithic XLA graph, resulting in orders of magnitude speedups on TPU/GPU hardware.

This orchestration logic is encapsulated entirely by the [`Sampler`](#sampler) class:

```python
from jaxpe.sampler.global_local import Sampler

sampler = Sampler(
    problem=inference_problem,
    kernel=local_hmc_kernel,
    flow=flow_proposal,
    n_chains=100,
    n_loop_training=50,
    n_loop_production=50
)
results = sampler.run(key, initial_positions)
```

### Initialization and Prior Support

A Markov chain initialized in a vanishingly low probability region (or entirely confined to a single degenerate mode) requires a prohibitively long mixing time to achieve stationarity.

The `best_of_prior_init` subroutine explicitly remedies this by evaluating the log-likelihood over a massive Monte Carlo batch (e.g., $$N=10^6$$) drawn directly from the prior measure $$p(\theta)$$. By seeding the initial chain states $$x_{(0)}$$ with the highest-probability candidates, we ensure that the empirical measure of the ensemble immediately populates all valleys of significant support, effectively nullifying the burn-in phase bottleneck.

In `jaxpe`, you can automate this optimal seeding using [`best_of_prior_init`](#best_of_prior_init):

```python
from jaxpe.sampler.global_local import best_of_prior_init

initial_positions = best_of_prior_init(
    key,
    n_chains=100,
    prior=inference_problem.prior,
    logp_fn=inference_problem.log_prob,
    n_samples=100_000
)
```

## API Reference

### `Sampler`
**`jaxpe.sampler.global_local.Sampler(kernel, *, problem=None, logp_fn=None, n_dim=None, config=None)`**

The orchestrator of the global-local MCMC, alternating between the local transition kernel
and global normalizing-flow independence proposals. Note the signature: `kernel` is the
only positional argument and everything else is keyword-only. Supply **either** a
`problem` (an `InferenceProblem`, from which the log-posterior and dimension are taken) or
a raw `logp_fn` together with `n_dim`. Chain count, loop structure and flow architecture
all live in `config`, not in the constructor.

### `GlobalLocalConfig`
**`jaxpe.sampler.global_local.GlobalLocalConfig`**

Every knob of the run, as a dataclass. The loop structure is three-phase:

| field | default | meaning |
|---|---:|---|
| `n_chains` | 128 | parallel chains |
| `n_prelim_loops` | 2 | local-only warmup; samples are **discarded**, not buffered |
| `n_training_loops` | 12 | local steps → flow fit on the buffer → global block |
| `n_production_loops` | 6 | flow frozen, no adaptation |
| `n_local_steps` | 100 | local kernel steps per chain per loop |
| `n_global_steps` | 50 | flow independence-MH steps per loop |
| `local_thin` | 5 | keep every $$k$$-th local sample |
| `buffer_size` | 50 000 | flow training buffer |
| `flow_layers`, `knots`, `interval` | 8, 8, 5.0 | RQ-spline flow architecture |
| `nn_width`, `nn_depth` | 64, 1 | conditioner network |
| `n_epochs`, `batch_size`, `learning_rate` | 8, 1024, 1e-3 | flow optimisation |
| `adapt_step_size`, `adapt_scale` | `True`, `True` | kernel adaptation during training loops |
| `target_acceptance` | `None` | per-kernel literature default when unset |
| `use_global` | `True` | disable to obtain a pure local sampler |
| `checkpoint_every_n_training`, `checkpoint_every_n_production` | 1, 50 | checkpoint cadence |

Two defaults deserve scrutiny rather than acceptance. `flow_layers = 8` is generous for a
low-dimensional posterior — the global block runs *two* flow passes per proposal
(`sample` and `log_prob`), so layer count is close to linear in global-block cost, and
halving it was a pure win in the [BNS benchmark]({{ site.baseurl }}/docs/bns_ce_pe_benchmark.html).
And `interval` sets the RQ-spline's active range: **outside it the spline is the
identity**, so widening it to reach far tails buys reach at a steep cost in acceptance.

### `SamplerResults`
**`jaxpe.sampler.global_local.SamplerResults`**

Returned by the run. `samples` and `log_prob` are the production draws **in the
unconstrained space** — map them back with `PostProcessor` before interpreting them
physically. `local_acceptance`, `global_acceptance` and `flow_losses` are per-loop
diagnostic series; `flow` and `kernel` are the final adapted states, so a run can be
continued or inspected.

### `PostProcessor`
**`jaxpe.sampler.postprocessing.PostProcessor(problem, raw_samples=None, raw_samples_file=None)`**

Estimates the integrated autocorrelation time, thins to approximately independent draws,
and maps unconstrained samples back to physical parameters through the prior's bijections.
This last step is not cosmetic: a credible interval computed in the unconstrained space and
then transformed is *not* the transformed credible interval, because the map is nonlinear.

### `best_of_prior_init`
**`jaxpe.sampler.global_local.best_of_prior_init(key, n_chains, prior, logp_fn, n_samples)`**
Evaluates `n_samples` from the prior and returns the `n_chains` points with the highest log-posterior density, circumventing long burn-in phases.

---

### REFERENCES
**[1]** K. W. Wong et al., "flowMC: Normalizing flow enhanced sampler in jax," arXiv:2211.06397 (2022).

**[2]** L. Tierney, "Markov Chains for Exploring Posterior Distributions," Ann. Stat. **22**, 1701-1728 (1994).
