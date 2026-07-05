# jaxpe

Normalizing-flow-enhanced gradient MCMC in JAX, with gravitational-wave parameter
estimation (PE) as the flagship application.

The sampling engine is problem-agnostic: it operates on any differentiable log-density
over a flat unconstrained parameter vector. The `jaxpe.gw` subpackage builds a
frequency-domain likelihood from a (user-supplied, JAX-differentiable) time-domain
waveform model and standard GW priors, and hands it to the same engine.

## Algorithm

flowMC-style global-local sampling:

1. Many chains (vmapped on GPU) run a gradient-based local kernel
   (MALA / HMC / mMALA / underdamped Langevin, or gradient-free random walk).
2. A rational-quadratic-spline coupling flow ([flowjax](https://github.com/danielward27/flowjax))
   is trained on the accumulated chain samples.
3. The flow drives Metropolis-Hastings *global* proposals, moving chains between
   posterior modes (sky-position ring, phase/polarization degeneracies, ...).
4. Training loops alternate (1)-(3); production freezes the flow.

## Layout

- `jaxpe.core` — priors, unconstraining transforms, the `InferenceProblem` interface
- `jaxpe.kernels` — local MCMC kernels + step-size/mass adaptation
- `jaxpe.flows` — normalizing-flow wrapper and trainer
- `jaxpe.sampler` — the global-local orchestration loop
- `jaxpe.diagnostics` — R-hat, ESS, corner/trace plots
- `jaxpe.gw` — detectors, PSDs, data handling, FD likelihood, GW priors
- `examples/` — toy problems and end-to-end GW injections / GW150914

## Quick start (GW injection)

```python
import jax
jax.config.update("jax_enable_x64", True)

from jaxpe.gw import ToyChirp, bbh_priors, make_injection
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

like = make_injection(ToyChirp(20.0), injection_params, noise_seed=42)
problem = like.problem(bbh_priors(geocent_time=t_trigger))

cfg = GlobalLocalConfig(n_chains=64)          # scale chains to GPU memory
sampler = Sampler(MALA(0.05), problem=problem, config=cfg)
key = jax.random.PRNGKey(0)
x0 = best_of_prior_init(key, problem, cfg.n_chains)   # seeds degenerate modes
result = sampler.run(key, x0=x0)
samples = sampler.to_physical(result.samples)
```

See `examples/03_gw_injection.py` for the full script and
`examples/validate_injection_vs_dynesty.py` for the cross-check against
bilby+dynesty on the identical likelihood.

## Practical notes

- **Initialization matters**: GW posteriors occupy a tiny fraction of the prior
  volume. `best_of_prior_init` (vmapped lnL over ~2e4 prior draws) seeds every
  comparable-likelihood mode — without it, chains burn in to one arbitrary mode and
  the flow mode-collapses.
- **Two-detector sky ring**: with H1+L1 only, the time-delay ring makes the sky
  posterior an extended multimodal ridge; give the sampler generous training loops
  and global steps, and check `result.global_acceptance` (healthy: ~0.1+ for GW,
  ~0.8 on unimodal toys) and split R-hat.
- **Small GPUs**: set `XLA_PYTHON_CLIENT_MEM_FRACTION` to a fixed fraction that fits
  beside your desktop's usage (on-demand allocation fragments and can kill cuBLAS
  mid-run), and scale `n_chains` accordingly.

## Precision

GW likelihoods are validated in float64 (`jax.config.update("jax_enable_x64", True)`).
On GPUs with weak FP64 (Turing/Ampere consumer parts), generate waveforms in float32
and accumulate inner products in float64; `jaxpe.gw.waveform.mismatch_f32_f64` certifies
whether float32 is safe for a given model across the prior. Any float32 fast path must
use segment-relative times — absolute GPS epochs are unrepresentable at float32
resolution.
