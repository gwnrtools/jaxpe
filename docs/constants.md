# Hardcoded constants audit

Written after tracking down why the 300-run PE campaign's `phenomd_gpry` leg
was silently producing wrong results twice over: `PhaseDistanceMarginalLikelihood`
defaulted `dist_bounds` to `(1000, 8000)` Mpc — the class's own generic
default — regardless of the run's actual configured distance prior. For a
campaign whose injections sat as close as 200 Mpc, the distance-marginalization
quadrature grid silently never covered the true distance, producing a wrong
(not just noisier — actively misleading) marginal likelihood over the
intrinsic parameters. It was caught by luck, not by design (see
`campaign/plan.md` amendment 13's `dist_bounds` fix and amendment 16's
`GPryEngine` noise-floor/classifier-threshold fix for the sibling bug it
surfaced next to).

The failure shape is: a numeric constant that functions as a physical or
algorithmic boundary, tolerance, or resolution is baked into library code
instead of being a required input (function argument, constructor argument,
or config-file value with no silent library-side fallback). When the baked-in
default is wrong for the case at hand, the result is not a crash — it's a
plausible-looking wrong answer. That is strictly worse than a loud failure,
because nothing downstream (a corner plot, a PP-plot, a credible interval)
looks obviously broken.

This document is the full audit that followed: every file in `jaxpe/` was
checked for constants of this shape. It is organized by category, most severe
first within each. Entries marked **no override** have no argument or config
path at all — editing source is the only way to change them. Entries marked
**kwarg exists** have an override, but nothing forces a caller (in particular
the CLI, which is supposed to be the single place physical choices are made)
to actually supply it from configuration, so the dangerous default is still
live for any caller that doesn't know to ask for it.

Status legend: 🔴 not fixed · 🟡 patched at one call site, landmine still live
in the library · 🟢 fixed at the source.

## Worse than the bug we fixed

### `IMRPhenomT` / `IMRPhenomTHM` are not real waveform models 🟡 `IMRPhenomT` fixed, `IMRPhenomTHM` still 🔴

- `jaxpe/gw/cbc_models/phenomt.py` — **fixed.** `IMRPhenomT` (the dominant
  (2,2) mode) has been faithfully reimplemented against LALSuite's
  `LALSimIMRPhenomTHM_internals.c`/`_fits.c` (Estelles et al. 2020,
  arXiv:2004.08302), replacing the old `_compute_phenom_coefficients`
  placeholder (fixed dict, ignored its `eta`/`chi1`/`chi2` arguments
  entirely) with the real closed-form construction: 3.5PN TaylorT3 inspiral +
  6 NR-fitted higher-order corrections (6×6 `jnp.linalg.solve`), an
  arcsinh-parametrized merger frequency ansatz (3×3 solve), and Damour&Nagar
  2014's ringdown ansatz, with phase as the exact analytic integral of each
  region (no numerical integration). Verified with real `tests/test_phenomt.py`
  assertions against LALSuite's own `IMRPhenomT` (not shape-only): mismatch
  1.4e-6–4.6e-6 across a parameter grid spanning mass ratio 1:1–5:1, spins to
  0.8, and varying inclination/reference phase — the same precision level as
  the already-correct `IMRPhenomD` (see below). Also fixed along the way: the
  reference-phase convention (LALSuite anchors `phase` at `f_ref`, not at
  merger — omitting this left frequency/amplitude correct but phase off by a
  mass/spin-dependent constant) and the `phi_ref`/inclination mode-combination
  convention (`phi = π/2 - phiRef` fed to the spin-weighted `Ylm`, not a
  separate per-mode `exp(i·m·phiRef)` rotation — the two aren't
  interchangeable away from face-on/edge-on).
- `jaxpe/gw/cbc_models/phenomthm.py:65-77` — **still a placeholder, not yet
  fixed.** Only the HM peak frequency/amplitude (`f_peak`, `a_peak`) come from
  real calibrated fits; the inspiral/merger/ringdown envelope shapes are
  invented constants (`t_meco=-10.0`, `t_ring=10.0`,
  `amp_insp ∝ (-t)^-0.25/10`, `amp_merg ∝ exp(-0.01 t²)`,
  `amp_rd ∝ exp(-0.1 t)`, `omega = f_peak*(1+tanh(t/10))`) that do not match
  the real IMRPhenomTHM ansatz. Because the peak values *are* real, the
  output looks physically grounded while the bulk of it isn't. The
  IMRPhenomT reimplementation above reuses ~90% of what `IMRPhenomTHM` needs
  (the amplitude ansatz is already mode-generic, not (2,2)-specific); the
  remaining work is the 4 higher modes' merger/ringdown frequency+phase
  ansätze and per-mode QNM substitution — see the IMRPhenomT plan's Phase 2.
- `jaxpe/gw/cbc_models/phenomthm.py:62-63` — `fits.get(f"IMRPhenomT_..._{l}{m}",
  0.1)`: a missing/mistyped mode key silently substitutes an arbitrary
  geometric-units constant instead of raising `KeyError`. Currently dormant
  (all keys exist today) but the same shape of landmine as everything else
  here. Still open, tracked for Phase 2.

`tests/test_lalsuite_comparison.py`'s shape-only assertion was also
strengthened into a real mismatch assertion (marked `xfail(strict=True)`
against the still-placeholder `IMRPhenomTHM`, so it will loudly demand the
`xfail` marker's removal once Phase 2 lands rather than silently starting to
pass unnoticed). New `tests/test_phenomd.py` gives `IMRPhenomD` (previously
untested against LAL at all, despite being a careful, correct transcription)
the same real-comparison treatment, and a new standalone `jaxpe/gw/match.py`
(PSD-weighted mismatch/match, `jax.jit`/`jax.vmap`-composable) backs all of
these instead of each test reinventing the comparison.

### The `dist_bounds` bug, unfixed at the source 🟢 fixed

- `jaxpe/gw/likelihood/fd_marginal.py:107` —
  `PhaseDistanceMarginalLikelihood.__init__`'s `dist_bounds` is now a required
  keyword-only argument (no default). Omitting it is an immediate `TypeError`
  at construction, not a silently wrong answer.
- `jaxpe/gw/likelihood/modes.py:291-292,419-420` — the same fix applied to
  the mode-based marginalizer that backs `MarginalizedIntrinsicLikelihood`
  (the ESIGMA+GPry path): `dist_min`/`dist_max` are now required keyword-only
  arguments (no default) on both `log_marginal_likelihood` and
  `log_marginal_likelihood_full`.
- `jaxpe/gw/likelihood/marginalized_intrinsic.py` —
  `MarginalizedIntrinsicLikelihood.__init__` now eagerly validates that its
  `settings` dict contains `dist_min`/`dist_max`, raising `ValueError` at
  construction rather than deferring the (now-mandatory) `TypeError` to the
  first, possibly expensive, `__call__`.
- `jaxpe/cli.py` — the default that used to live inside the library classes
  now lives here instead, as a named, documented constant
  (`DISTANCE_MARGINAL_BOUNDS_FALLBACK_MPC = (100.0, 8000.0)`) behind a small
  helper (`_dist_bounds_for_marginalization`). Both `run_pe` branches
  (`marginalized_phase_distance`, `marginalized_intrinsic`) call it: it
  returns the run's actual resolved `luminosity_distance` prior box whenever
  that box is non-degenerate, and only substitutes the `(100, 8000)` Mpc
  fallback (with a loud `warnings.warn`) for the edge case of a degenerate
  box (e.g. a `"fixed"` distance prior, which is a nonsensical config for a
  distance-marginalized likelihood in the first place) — it never silently
  overrides a real, deliberately-narrow user-configured prior.
- All existing callers (`tests/test_fd_marginal.py`, `tests/test_marginalized.py`,
  `tests/test_surrogate.py`, `examples/07_td_higher_mode_route_comparison.py`,
  `examples/08_fd_dominant_mode_route_comparison.py`, `bin/run_direct_ns.py`)
  already passed these explicitly, so the tightened signatures needed no call-site
  changes there; `tests/test_fd_marginal.py`, `tests/test_marginalized.py`, and
  the non-GPry/ESIGMA-dependent parts of `tests/test_cli.py` were re-run green
  after the change.

## Physics (waveforms / likelihood / data / detectors / priors)

| File:line | Constant | Controls | Exposure | Risk | Status |
|---|---|---|---|---|---|
| `phenomt.py:33,35,38-39` | clip to q∈[0.01,1], η∈[1e-4,.25], χ∈[-.99,.99] | silently clips out-of-domain params to the boundary value instead of erroring | no warning on clip | plausible-but-wrong waveform at the prior edges | 🔴 |
| `esigma.py:109` | self-force spin-table grid `±0.995`, range not exposed (only point count is a kwarg) | valid spin domain for the horizon-flux term | `jnp.interp` silently clamps beyond it | wrong (frozen) horizon flux near-extremal spin, no error | 🔴 |
| `esigma.py:146,192` | `1.5×` safety margin on the Newtonian time-to-merger estimate, no override | ODE grid duration | none | can truncate high-eccentricity inspirals before merger, silently | 🔴 |
| `phenomd.py:79` | `fM_CUT = 0.2`, module-level, zero override | dimensionless FD validity cutoff | none | amplitude forced to exactly 0 above cutoff (a truncation, at least louder than extrapolation) | 🔴 |
| `nrsur7dq4.py` | stated NR-training domain (q≤4, spins≤0.8) in the docstring only | surrogate validity | never checked/clipped/warned in `__call__` | silent extrapolation outside the trained region | 🔴 |
| `jaxpe/gw/priors.py:27-32,69-75` | `bbh_priors`/`ebbh_priors`: `chirp_mass=(20,40)`, `mass_ratio=(0.25,1.0)`, `luminosity_distance=(100,2000)`, `time_width=0.1`, `eccentricity=(0,0.4)` | default prior *support* used for inference | kwargs, but nothing forces config-sourcing | true parameter outside the default box → sampler explores a support that excludes the truth; the closest analog in the codebase to the `dist_bounds` bug | 🔴 |
| `jaxpe/gw/detectors.py:121-128` | GMST linear-fit fallback (`gps_2000`, `gmst_2000`, valid "around 2020" per docstring) used when `lal` isn't importable | Greenwich Mean Sidereal Time → antenna pattern / sky localization in every likelihood call | silent `except ImportError`, no warning which branch ran | silently degrades sky-localization accuracy; the stated validity window is now years stale and unchecked | 🔴 |
| `jaxpe/gw/psd.py:144` | `noverlap = nperseg // 2` inside `welch_psd`, zero override | Welch PSD estimator bias/variance tradeoff | none | biases the noise curve every likelihood evaluation is normalized against | 🔴 |
| `jaxpe/gw/psd.py:38` vs `:90` | `f_low=10.0` (`aligo_zdhp_psd`) vs `f_low=5.0` (`lalsim_psd`), independent defaults | PSD low-frequency floor | separate, no shared source of truth | can leak asymmetric near-`inf` PSD bins across detectors if callers don't align them | 🔴 |
| `jaxpe/gw/data.py` | `f_min=20.0` / `duration=8.0` / `sampling_rate=2048.0` / `post_trigger=2.0` / `tukey_alpha=0.1`, repeated across `make_injection`/`make_injections`/`likelihood_from_strain` | analysis segment/grid and low-frequency cutoff | kwargs, not config-forced | forgetting to pass `f_min` for e.g. a third-generation-detector or low-mass analysis silently drops real signal power below 20 Hz | 🔴 |
| `likelihood/fd_marginal.py:191,206` | `delta = 0.3`, `1e-3 * max\|a\|`, inside `_check_dominant_mode` | the one safety guard that catches a higher-mode model being misused with this closed-form (dominant-mode-only) likelihood | **no override** | if poorly matched to a given waveform, the guard can pass while the closed-form is actually invalid | 🔴 |
| `likelihood/modes.py:288,416` | `n_phi: int = 512` (docstring: covers "network SNR ~ 20") | phase-marginalization integral resolution | kwarg, but no assertion/check tying it to the run's actual SNR | a louder real event silently under-resolves the phase marginal — biased, not obviously wrong | 🔴 |
| `likelihood/modes.py:290,418` | `tc_half_samples: int = 205` ("±0.1s @ 2048 Hz") | time-marginalization window | constant baked to an assumed sampling rate, not derived from it | wrong window width at any other sampling rate | 🔴 |
| `relative_binning_fd.py:124-125`, `relative_binning_td.py:59,143,260,431` | `chi=1.0, epsilon=0.5`, `phase_per_bin=0.5` | heterodyne/relative-binning resolution | kwargs, adequacy never checked against the actual signal | biased likelihood for a fiducial far from truth or fast phase evolution, with no signal anything's wrong | 🔴 |
| `relative_binning_td.py:478` | `1e6 * in_band_max` substituted for `inf` | PSD masking before the Toeplitz/Gohberg-Semencul solve | none | can leak non-negligible correlated-noise weight for PSDs with wider dynamic range than this factor assumes | 🔴 |
| `likelihood/base.py:114` | `tukey_alpha: float = 0.1`, dataclass default shared by every `NetworkLikelihood`/`ModesNetworkLikelihood` | window taper width → spectral leakage | overridable, but duration/f_min-independent | same "kwarg exists but still ships a plausible default" shape as `dist_bounds` | 🔴 |

**Lower severity / largely self-correcting**, noted for completeness:
`conditioning.py:36` `post_trigger=2.0`; `conditioning.py:67` `0.9 × Nyquist`
factor in `resolve_f_max`; `modes.py` importance-sampling internals (`15.0`
e-fold cut, width clip `(0.01, 0.25)`, `defense=0.2`, `qmc_seed=7` — covered by
the escalating-rounds/ESS-target self-healing machinery);
`marginalized_intrinsic.py:90` `effective_sample_size_floor: float = 0.0`
(quality floor defaults *off* unless a caller opts in); `waveform.py`
(`f_start=20.0`, `amp_hm=0.6`) is an explicitly pedagogical toy model, not
used in production. `diagnostics/plots.py:45` `quantiles=[0.16,0.5,0.84]`,
`bins=40`, `smooth=0.9` in `corner_plot` — display-only, fully overridable,
worth noting only because the 1σ quantile choice doubles as a de facto
credible-interval default read off the plot.

**Not flagged** (checked, found benign): true physical/mathematical constants
(`MTSUN_SI`, `MPC_SI`, `C_SI`, `MRSUN_SI`, `c`, `G`, `π`); trivial
indexing/loop literals; PN-series and NR-fit coefficients, which are the
physics itself, not tunable boundaries; `core/priors.py`'s `Sine`/`Cosine`
defaults (`0..π`, `-π/2..π/2`) are the full physical range, not a truncation;
`core/problem.py`/`core/transforms.py` have no meaningful numeric literals;
`diagnostics/stats.py` never applies a convergence threshold in code (only in
docstring commentary); `flows/trainer.py` hyperparameters (`n_epochs=8`,
`batch_size=1024`, `lr=1e-3`) are ordinary, fully-exposed ML defaults with no
physics-correctness failure mode; `esigma.py`'s `inspiral_end_radius=4.0`,
`ode_eps=1e-8`, `taper_on/off_seconds`, `f_lower=20.0` and
`pyseobnr_model.py`/`teobresums_model.py`'s `d_ref_mpc=1000.0`, `f_low=20.0`,
`taper_seconds` are all constructor kwargs, not silent in-body defaults.

## Sampler / other algorithms (MCMC, GP surrogate, normalizing flow)

| File:line | Constant | Controls | Exposure | Risk | Status |
|---|---|---|---|---|---|
| `cli.py` `--gpry-noise-level` / `--gpry-svm-threshold` / `--gpry-trust-region-threshold` | `default=None` | the amendment-16 GPry high-SNR fix (regressor noise floor, infinities-classifier thresholds) | **opt-in only** — when unset, the flags are simply omitted rather than computed from the run's SNR, so `GPryEngine` falls back to GPry's own `noise_level=1e-2` / ~3σ thresholds | any future invocation (a new campaign variant, a script, a forgotten flag) silently reproduces the exact `GPAcquisitionError` crash amendment 16 spent a day diagnosing | 🟡 (fix exists, not a forcing function) |
| `kernels/adaptation.py:37` | `gamma: float = 1.0` (Robbins-Monro step-size adaptation rate) | how aggressively the HMC/MALA step size adapts | kwarg exists, but the only caller (`global_local.py:414`) never threads it from config/CLI | can oscillate rather than converge under noisy short-window acceptance-rate estimates, with no diagnostic surfaced | 🔴 |
| `kernels/adaptation.py:38-39,68,78` | `lo=1e-8, hi=1e3` step clip; `1e-8` ensemble-scale floor; `1e-6` ensemble-cov jitter | HMC/MALA adaptation numerical stability | same pattern — unreachable from config/CLI | usually fails loud (degenerate Cholesky) rather than silently, but still hardcoded in every real run | 🔴 |
| `sampler/global_local.py:137`, `flows/interface.py:61` | `interval: float = 5.0` (normalizing-flow rational-quadratic-spline domain) | the whitened-space range the flow can nonlinearly reshape; effectively linear/identity outside `[-interval, interval]` | overridable kwarg, never tuned per-problem or from the actual buffered-sample spread | GW posteriors are routinely multimodal/heavy-tailed (sky ring, distance-inclination degeneracy); a mode or tail beyond ±5σ silently gets a bad global proposal, no error | 🔴 |
| `sampler/postprocessing.py:92` | `max_tau = 100` fallback when autocorrelation-time estimation fails | burn-in/thinning basis | warns, but continues using this arbitrary number | plausible but unjustified effective-sample-size accounting | 🔴 |
| `sampler/global_local.py:502` | `batch_size = 100_000` inside `to_physical`, zero override (contrast `postprocessing.py:122`, which has the same constant as an overridable kwarg) | GPU-memory chunking of the vmapped prior transform | none | OOM risk on smaller GPUs; not a correctness bug (chunking is exact), just inflexible | 🔴 |
| `surrogate/jax_acquisition.py:166` | `_bucket_size(n, bucket=64)`, no override at its call site | JIT-cache padding granularity for the JAX NORA predictive | none | performance/recompilation only — padding is proven exact, not a correctness risk | 🟢 (benign) |

**Clean:** `jaxpe/kernels/hmc.py`, `mala.py`, `mmala.py`, `grw.py`, `uld.py`,
`base.py` — `step_size`, `scale`, `n_leapfrog`, `friction`, `metric_fn`, `cov`
are all properly required or config-threaded constructor arguments; no
orphaned algorithmic literals found. `surrogate/gpry_backend.py` and
`multifidelity.py` pass `options`/`**kwargs` through to `gpry.Runner`
verbatim with no hardcoded GP defaults of their own — the landmine above
lives entirely in the CLI wiring around the surrogate library, not in the
surrogate wrapper itself.

## Hardware limits

| File:line | Constant | Controls | Exposure | Status |
|---|---|---|---|---|
| `bin/condor/generate_campaign_dag.py:47` | `PE_REQUEST_CPUS = 8`, module constant | condor `request_cpus` + the `taskset` core mask for all 300 jobs | no CLI flag | 🔴 |
| `bin/condor/generate_campaign_dag.py` (per-variant dict) | `request_memory = "16GB"` (HMC) / `"4GB"` (GPry variants) | condor memory request | hardcoded per variant, no override | 🔴 |
| `bin/condor/generate_campaign_dag.py:169` | `request_cpus = 2` for the `POSTPROCESS` job | condor CPU request for the report-generation node | hardcoded | 🔴 |
| `sampler/global_local.py:502` | `batch_size = 100_000` (listed above too) | GPU chunking | no override | 🔴 |

`--maxjobs`, `--request-disk`, `--n-injections`, `--config` are already
proper CLI flags on `generate_campaign_dag.py` — only the CPU/memory sizing
constants above are not.

## How to use this document

Each 🔴 row is a candidate for one of:
1. **Make the argument required** (no default) so a missing value is a loud
   `TypeError`/`ConfigError` instead of a silent wrong answer — the right fix
   for anything that gates correctness (distance/mass/spin bounds, PSD
   floors, resolution parameters tied to SNR).
2. **Source it from `cfg`/CLI with no library-side fallback** — the right fix
   for anything that's legitimately run-dependent but currently only
   reachable by editing source (hardware sizing, `gamma`/adaptation
   constants).
3. **Add a runtime domain check** (raise or warn) rather than silently
   clipping/extrapolating — the right fix for calibration-range constants
   (`phenomt.py` clipping, `nrsur7dq4.py`'s unchecked NR domain,
   `esigma.py`'s spin-table clamp).

Update the status column (🔴 → 🟡 → 🟢) as each is addressed, and note the
call site if a fix only patches one caller rather than the library source
(as `dist_bounds` currently is at `cli.py:363`).
