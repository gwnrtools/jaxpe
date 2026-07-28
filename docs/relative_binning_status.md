---
layout: default
title: Status — Relative binning implementation
nav_order: 103
---

# Relative Binning — Implementation Status

Companion to the [design note](relative_binning_design.md). This page tracks what has
been **implemented, tested, committed, and pushed** for the Fourier- and time-domain
relative-binning (heterodyned) likelihoods, and what remains.

All code lives in [`jaxpe/gw/likelihood/`](../jaxpe/gw/likelihood/); every stage was
committed only after its unit tests **and** its performance-comparison test passed, and
pushed to `origin/master`.

---

## Stage ledger

| Phase | Commit | Content | Validation | Speedup (compile excluded) |
|---|---|---|---|---|
| **RB-1** FD dominant | `4f4cb40` | `RelativeBinningFDLikelihood` (detector network) | exact-at-fiducial; parity vs `FDNetworkLikelihood` | ~12× vs full FD |
| **RB-2** FD higher modes | `96b26fe` | `RelativeBinningFDLikelihoodHM` (per-mode/pair, diagonal covariance) | parity vs dense multi-mode FD (`fd_dense_loglikelihood_modes`) | — |
| **RB-3** Toeplitz infra | `2f034bc` | `toeplitz.py`: ACF, matvec, Gohberg–Semencul `C⁻¹v` (JAX) | vs dense to 1e‑9…1e‑11; round-trip; jit+vmap | ~3300× vs dense solve |
| **RB-4** TD dominant | `15269dd` | `RelativeBinningTDLikelihood` + dense reference | exact-at-fiducial; parity (intrinsic/extrinsic/noisy) | ~370× vs dense |
| **RB-4** TD higher modes | `3406365` | `RelativeBinningTDLikelihoodHM` (Appendix-A cross-mode tensors) | parity; single-mode HM == dominant (`<1e-9`) | ~74× (2 modes) |
| **RB-4** TD network | `007e37b` | `RelativeBinningTDNetwork` (block-diagonal sum) | vs summed dense | — |
| **RB-4** t_c | `28e1b79` | `edge_times` + shifted-sampling support | parity vs dense at shifted t_c | — |
| **RB-5** posterior-level | `8705cab` | grid posterior-recovery validation | **JS divergence < 1e-3**, matching mean/width, peaks at truth | — |
| design note | `398c770`, `abc11cc`, … | full FD+TD design, verified vs both source papers | — | — |

---

## Validation method

Every accuracy test compares against the **exact** reference on the same data:

- **exact-at-fiducial** (`<1e-6`) — the tight anchor: with the trial equal to the
  fiducial, the ratio is identically 1 and the heterodyned likelihood must equal the
  full/dense likelihood to machine precision;
- **parity to the Zackay error model** `β·(1+|lnL|)` — the discrepancy grows with
  distance from the fiducial in *likelihood* units, so the principled tolerance is
  proportional to `(1+|lnL|)`, not a fixed absolute; checked across intrinsic,
  extrinsic, noisy, higher-mode, and network cases;
- **convergence** — finer bins reduce the error;
- **performance-comparison tests** — identical likelihood computed with and without
  relative binning; the two match numerically and the binned version is much faster,
  with JAX compile time excluded (warm-up before timing);
- **posterior-level** (RB-5) — the heterodyned and dense posteriors over an intrinsic
  parameter agree in mean/width and have Jensen–Shannon divergence `< 1e-3` (the
  paper's acceptance criterion is agreement at the posterior level, not just pointwise).

References: FD — Zackay, Dai & Venumadhav 2018 (arXiv:1806.08792), bilby
`RelativeBinningGravitationalWaveTransient`; TD — Sharma, Vijaykumar & Kumar 2026
(arXiv:2601.11239), incl. Appendices A/B/C.

---

## Public API (`jaxpe.gw.likelihood`)

- **Fourier domain:** `RelativeBinningFDLikelihood`, `RelativeBinningFDLikelihoodHM`,
  `frequency_bin_edges`, `fd_dense_loglikelihood_modes`.
- **Time domain:** `RelativeBinningTDLikelihood`, `RelativeBinningTDLikelihoodHM`,
  `RelativeBinningTDNetwork`, `time_bin_edges`, `td_dense_loglikelihood`,
  `td_dense_loglikelihood_hm`, `extrinsic_coefficient`.
- **Covariance:** `toeplitz.autocorrelation_from_psd`, `toeplitz.toeplitz_matvec`,
  `toeplitz.inverse_generator`, `toeplitz.inverse_matvec`.

Tests: [`test_relative_binning_fd.py`](../tests/test_relative_binning_fd.py),
[`test_toeplitz.py`](../tests/test_toeplitz.py),
[`test_relative_binning_td.py`](../tests/test_relative_binning_td.py) — 44 tests.

---

## What remains

- **Full production end-to-end PE.** The remaining work is *not* a unit-testable
  commit-after-tests increment: it requires wiring a production higher-mode waveform
  (e.g. `NRSur7dq4`/JaxNRSur) to generate modes at the (non-uniform) bin edges,
  integrating a sampler / GPry surrogate, and running a percentile–percentile injection
  campaign (hours–days of compute). The method itself is already validated at the
  posterior level on a synthetic problem (JS `< 1e-3`), the strongest proxy short of a
  full run. This belongs to a dedicated effort with its own design choices (which
  waveform model, which sampler).

Also noted in the design: the **summary-data setup** is the real cost for long TD
signals (the `N_bins × N_modes` Gohberg–Semencul solves — amortised over ~10⁷
evaluations), and the speedup assumes the waveform can be evaluated at arbitrary
(bin-edge) times; EOB-type models that cannot must generate on a dense grid and
downsample, keeping the covariance speedup but not the waveform one.
