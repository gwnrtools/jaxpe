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

### Added

- **`make_injections`** builds a whole suite of injections in one `vmap`, keeping
  signal, projection and noise on the accelerator. Measured on 4 s @ 1024 Hz H1+L1:
  the vectorised arithmetic alone is ~4x on CPU and ~6x on GPU at `n=256`, and
  end-to-end against a loop over `make_injection` it is 28x at `n=64` on GPU
  (173.5 s → 6.2 s) — though most of that larger margin is amortising
  `make_injection`'s per-call `jax.jit` compilation rather than parallelism.

  Deliberately scoped: this is for suites that are the product (training sets, banks,
  systematics studies), **not** a speedup for validation campaigns, where injection
  creation is a fraction of a percent of wall time. A batch needs one analysis grid,
  so it cannot mix trigger times or durations, and oversized requests are refused
  with an estimate — a 2048 s BNS segment is 67 MB per injection per detector.

  Compare batched against serial by *mismatch* (~1e-9), not elementwise amplitude:
  raw amplitudes differ at ~5e-5 because XLA fuses `IMRPhenomD` differently between
  jit graphs. That predates this work and is unrelated to batching.

- **`network_snr` and `distance_for_target_snr`**, replacing the
  build-measure-rescale recipe open-coded at three sites. `h ∝ 1/D` exactly, so the
  target is hit in one measurement with no search.

- **`analysis_grid` and `resolve_f_max`**, one definition of the segment grid and of
  the 90%-of-Nyquist convention, replacing five copies.

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

- `bin/run_bns_ce_pe.py`, `bin/run_td_phenomt_pe.py` and
  `bin/profile_sampler_scaling.py` use `jaxpe.gw.lalsim_psd("CE", ...)` instead of a
  `cosmic_explorer_psd` helper duplicated in the first two and loaded by file path in
  the third. Verified bitwise identical at 8 s @ 2048 Hz and 2048 s @ 4096 Hz.

- **`bin/run_td_phenomt_pe.py`: `--target-snr` no longer changes the analysis
  window.** Its rescale rebuild passed `tukey_alpha=0.0` while the first build used
  the 0.1 default, so requesting a target SNR silently altered the Tukey taper as
  well as the distance. Both builds now share one hoisted keyword dict.

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
