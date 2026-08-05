# Changelog

Versions are derived from git tags by `setuptools-scm`.

## Unreleased

### Fixed

- **Every injection in a campaign received the same noise realisation.** The CLI
  passed `seeds.noise` through verbatim, so `generate-injections --noise gaussian
  --n-injections N` followed by `run-pe` analysed N copies of one realisation. This
  silently invalidates a PP-plot campaign, which is the main thing such a campaign
  exists to run. `generate-injections` now resolves a per-injection seed with
  `derive_noise_seed(seeds.noise, index)` and records it as `metadata.noise_seed`, so
  the realisation is pinned by the artifact; `run-pe` reads it back, falling back to
  deriving from `metadata.index` for sets already on disk.

- **A detector's noise depended on its position in `detector_names`.**
  `make_injection` advanced a single numpy `Generator` through the detector loop, so
  `("H1","L1")` and `("L1","H1")` gave H1 different data at the same seed — measured
  at a 130% relative change. Streams are now keyed by the detector's *name*.

### Changed

- **Noise generation moved to JAX** (`simulate_noise_fd_jax`), unifying the RNG with
  parameter drawing, which already used PRNGKeys. `simulate_noise_fd` (numpy) remains
  exported and unchanged.

  **This changes which noise realisation a given seed produces.** Nothing in the test
  suite pins noise sample values, but the cached artifacts under `examples/output/`
  were computed against the old realisation and have been regenerated.

  One reproducibility property is worth knowing: JAX's underlying random stream is
  bitwise identical across CPU and GPU, but `jax.random.normal` is not — measured at
  up to 3e-15 between backends, because XLA compiles `erf_inv` differently for each.
  Realisations are therefore bitwise reproducible on a given platform and equal to
  ~1e-15 across platforms. Use `simulate_noise_fd` where bitwise cross-platform
  equality is required.

- `examples/09_validate_injection_vs_dynesty.py` fingerprints the injection data into
  its bilby label. bilby resumes from a checkpoint keyed only by `label`, so a
  checkpoint computed against different data would previously have been resumed
  silently under the new likelihood.

### Performance

- **Post-processing no longer rebuilds an injection to recover the prior.**
  `PostProcessor` only uses `problem.prior`; obtaining it cost a waveform generation,
  projection, FFT and jit compile per samples file — measured at 1.70 s and 1.83 s.
  `InferenceProblem.log_likelihood` now defaults to a stub that raises, so a
  prior-only problem is expressible without lying about the density.

## 0.1.0

First tagged release. Cut to mark a significant performance fix in the sampling
engine.

### Fixed

- **`run_chains` initialised its chains outside the jit** (`jaxpe/kernels/base.py`).
  `jax.vmap(kernel.init)` ran eagerly, dispatching the target's entire gradient graph
  one operation at a time. For a gravitational-wave likelihood (~3600 instructions)
  that cost **~2.2 s per call against ~0.004 s jitted** — roughly 550× — and it was a
  *fixed* cost, paid on every call regardless of `n_steps`.

  Initialisation now happens inside `_run_chains_jit`, which already took `logp_fn` as
  a static argument, so there is no additional jit cache and no extra recompilation.
  Every kernel benefits; the effect is largest for expensive targets and for workflows
  that make many short `run_chains` calls.

  Measured on the BNS/Cosmic-Explorer benchmark (`bin/run_bns_ce_pe.py`), which makes
  ~33 such calls per run: **6.05 → 3.11 min** mean wall clock, and 15.42 → 3.11 min
  against the original reference. The fixed per-call cost measures 0.028 s after the
  fix, down from 2.254 s.

  Guarded by two regression tests in `tests/test_kernels.py`. The primary one asserts
  the structural cause — that the target is never invoked with concrete, untraced
  values during `run_chains` — rather than a wall-clock threshold, so it stays
  meaningful across hardware.

### Added

- `bin/profile_sampler_scaling.py` — fits `T = fixed + marginal × work` for
  `run_chains` and `_global_block` across several sizes, plus gradient cost versus
  chain count and global-block cost versus flow capacity. This is the tool that found
  the bug above; point measurements had missed it. Recommended as the first thing to
  run on new hardware.

### Changed

- Benchmark defaults in `bin/run_bns_ce_pe.py`, each with its measurement recorded at
  the argument: `--equil-rounds` 5 → 3, `--production-steps` 25 → 12, `--flow-layers`
  8 → 4 (new flag), `--max-production-blocks` 40 → 80. The first two had previously
  measured as losses *because* of the fixed cost above; both win once it is removed.
- `--max-production-blocks` raised because 40 turned "needs two more blocks" into a
  reported non-convergence on a 1.35 + 1.25 M☉ source. `--max-minutes` is the real
  budget guard.

### Documentation

- `docs/bns_ce_pe_benchmark.md` rewritten and pruned (848 → 338 lines), reorganised
  around the result rather than the chronology of getting there.
