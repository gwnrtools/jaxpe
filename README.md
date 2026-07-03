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

## Precision

GW likelihoods are validated in float64 (`jax.config.update("jax_enable_x64", True)`).
On GPUs with weak FP64 (Turing/Ampere consumer parts), generate waveforms in float32
and accumulate inner products in float64; `jaxpe.gw.waveform.mismatch_f32_f64` certifies
whether float32 is safe for a given model across the prior.
