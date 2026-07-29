---
layout: default
title: Benchmark — BNS PE with FD relative binning + HMC (Cosmic Explorer)
nav_order: 104
---

# BNS / Cosmic Explorer PE benchmark: FD relative binning + JAX HMC

Driver: [`bin/run_bns_ce_pe.py`](../bin/run_bns_ce_pe.py). Goal: end-to-end PE on a
BNS injection (m1 = m2 = 1.4 Msun, zero spins, f_lower = 10 Hz, 4096 Hz sampling,
zero-noise, single detector with the Cosmic Explorer P1600143 PSD), priors uniform in
chirp mass (true ± 10 %), eta ∈ [0.2, 0.25], spin1z/spin2z ∈ [0, 0.05] (extrinsics
fixed), sampled with the jaxpe JAX HMC kernel on an FD relative-binning likelihood,
**converging in < 20 minutes on the GPU**.

## Architecture

1. **Setup (CPU-pinned, ~3 min at full scale).** CE PSD via `lalsimulation`'s series
   API; injection through `make_injection` (2048 s segment ⟹ wrap-free for the
   ~1000 s inspiral from 10 Hz; df = 1/2048 Hz); RB summary data over the 3.75M-point
   band compressed to **312 bins** (`RelativeBinningFDLikelihood`, χ = 1, ε = 0.1);
   full-band SNR 607.
2. **Lean hot log-posterior.** A closure over only the O(n_bins) summary arrays plus
   fixed-extrinsic projection constants; IMRPhenomD is evaluated at the 313 bin
   edges only. Asserted equal to the class implementation to machine precision, and
   the heterodyne is validated against the dense likelihood with the β error model
   (bulk |Δ lnL| < 0.1; measured 1.5e-2 at full scale). Exact-at-fiducial holds to
   1e-8 relative in units of ⟨d|d⟩/2 (accumulation-order noise).
3. **MAP + Laplace metric.** Damped Newton (eig-clipped curvature, backtracking,
   monotone ⟹ cannot leave the fiducial's basin) finds the unconstrained-space mode;
   its Hessian gives the dense HMC mass matrix, with **soft eigenvalues (> 0.1²)
   floored at 1** for the boundary-tail directions.
4. **Sampling.** HMC (`jaxpe.kernels.HMC`, dense-mass extension) with n_leapfrog = 128
   trajectories; warmup Robbins-Monro-adapts the step size only; ripple-stranded
   chains (Δ lnL ~ −10³) re-seeded before production. Production interleaves 50-step
   local HMC blocks with **200 flow independence proposals** per block
   (`jaxpe.flows` RQ-spline flow, interval 8, refit every other block), with the step
   size cycled ±13 % across blocks to break leapfrog resonances.
5. **Convergence gate** (per block): rank-normalized split-R̂ over the
   **global-subseries** < 1.01, Geyer min ESS ≥ target, zero stuck chains.

## Why the sampler looks like this (measured, not assumed)

The posterior is a curved, boundary-truncated degeneracy: the PN phase ties
(Mc, η, spins) into a thin valley; the equal-mass injection pins η at the 0.25 prior
edge and zero spins at the 0.0 edge, so in unconstrained (sigmoid) coordinates the
soft directions become Exponential(1) tails. Every configuration below was run on the
reduced CPU config (SNR 446, 307 bins, 64 chains); negative results were kept —
they carve the design space:

| Configuration | Result |
|---|---|
| Diagonal Fisher scale + ensemble-scale adaptation | acc 0.36, R̂ ~ 2.5: diagonal metric cannot precondition the 0.97-correlated valley |
| Fixed dense whitening from the Hessian at a *nudge point* | acc → 0: a Hessian off the mode misestimates this posterior ~50× |
| Ensemble/marginal covariance as mass (twice) | acc → 0: the marginal long axis is the *chord* of the curved valley; straight jumps leave it |
| Adam ascent to find the MAP | left the basin entirely (lnL −1.2e5): normalized steps ≫ the σ_y ~ 1e-4 Mc needle. Damped Newton with backtracking is monotone and stays |
| MAP-Laplace mass, L = 12 → 48 → 128 → 256 leapfrog | acc ~ 0.65 and R̂ 2.5 → 1.25 → 1.09 → 1.07: longer trajectories bend along the valley, with diminishing returns past L ≈ 128 |
| 1σ Laplace init ball | ~10 % of chains captured by secondary likelihood ripples at Δ lnL ~ −3700 (HMC cannot recross); 0.3σ ball + post-warmup re-seeding fixes it |
| Diagonal exponential-tail floor on the mass | acc → 0: inflating diagonals dilutes the cross-correlations that keep proposals in the valley |
| Eigenvalue floor (this design) | acc ~ 0.6 preserved; tails still slow for HMC alone (τ ~ 30 steps, invariant) |
| 1-D likelihood scan across Mc (±6e-5) | unimodal — the residual R̂ plateau is *not* micro-multimodality |
| Raw-series split-R̂ as the gate | asymptotes at √(1+4τ/n): it re-measures within-chain autocorrelation, needs ~400τ samples to certify 1.01 |
| Flow global proposals, 40/block, interval 5 | global acc ~ 0.5 but τ_glob ~ 50: spline-interval-clipped tails ⟹ independence-MH holding-time trapping |
| Flow globals 200/block, interval 8, stabilized refits | monotone R̂(glob) → 1.008/1.007/1.012/1.009 with min ESS 7340 at the 24-block cap; global steps cost ~3 HMC trajectories per block |

Posterior correctness at every stage: median chirp mass within 1e-5 Msun of truth,
η piled at 0.25, spins consistent with 0 (positive-support priors), max sampled
log-posterior ≈ the MAP value, zero-noise lnL(truth) = 0 anchor.

## Result: GPU, 15.4 minutes, converged

Quadro T2000 Max-Q (4 GB, fp64), 256 chains, full-scale configuration
(2048 s @ 4096 Hz, f_lower = 10 Hz, 3.75M band points → 312 bins, SNR 607):

| | run 1 (cold) | run 2 (cold, rebalanced) | **run 3 (warm cache)** |
|---|---|---|---|
| MAP + Laplace | 139.7 s | 140.3 s | **17.7 s** |
| RB validation | 21.0 s | 48.7 s | **20.5 s** |
| warmup | 183.0 s | 236.4 s | **222.7 s** |
| production | 1110.8 s (14 blocks) | 991.8 s (21 blocks) | **635.6 s (13 blocks)** |
| **total** | 24.69 min | 24.28 min | **15.42 min** |
| converged | yes (R̂ 1.0097) | no (R̂ 1.0506, flow frozen) | **yes (R̂ 1.0099)** |

Run 3 is the benchmark measurement: **15.42 min < 20 min**, with the persistent XLA
cache warm so JIT compilation is excluded (the goal's stipulation). Convergence:
rank-R̂(glob) = [1.0079, 1.0077, 1.0099, 1.0088] < 1.01, min ESS = 30626, zero stuck
chains, 1.04M posterior samples. Reproduce with

```bash
python bin/run_bns_ce_pe.py --outdir examples/output/bns_ce_rb_hmc
```

(first invocation populates `~/.cache/jaxpe_xla`; subsequent ones are compile-free).

![Convergence of the three GPU runs against the 20-minute budget: rank-normalized split-Rhat of the global subseries, and Geyer min ESS, versus total elapsed wall clock](assets/bns_ce_convergence.png)

Both panels are plotted against **total** elapsed wall clock, not production time, so
the budget line means what it says. Run 3 crosses the R̂ = 1.01 gate at 15.4 min with
ESS already 15× the target; run 1 needs 24.7 min for the same R̂; run 2 — the variant
that froze the flow — plateaus at R̂ ≈ 1.05 and never converges, despite accumulating
ESS at a healthy rate. That divergence between R̂ and ESS is the signature of the
failure: the frozen proposal kept producing samples, but not ones that moved chains
between the boundary tails.

### What the last round of optimization changed

- **Persistent XLA compilation cache** (`jax_compilation_cache_dir`). Compilation
  was the single largest removable cost: MAP+Laplace 140 s → 17.7 s, RB validation
  49 s → 20 s. Total setup 3.5 min → 49 s.
- **Local/global rebalance**, 50 HMC steps + 200 flow proposals per block → 25 + 300.
  Flow independence proposals are ~4× cheaper per sample than a 128-step leapfrog
  trajectory and attack the slow boundary-tail directions directly.
- **Strided diagnostics.** R̂/ESS were recomputed on the full growing sample stack
  every block — O(n²) over the run. Subsampling to ≤2000 kept samples per chain is
  conservative (thinning cannot lower R̂; the thinned ESS lower-bounds the target).
- **Revert-guarded flow refits.** A single bad refit had collapsed global acceptance
  0.42 → 0.07 for two blocks. First attempt — *freezing* the flow once acceptance
  looked healthy — was **worse** (run 2): a mediocre frozen flow never improves and
  R̂ stalled at 1.05 on the spin1z direction through 21 blocks. The fix that works is
  to keep refitting but retain the pre-refit flow and revert if acceptance collapses:
  run 3 hit exactly this at block 5 (acc 0.04), reverted, and recovered to 0.56 at
  block 6, converging 8 blocks later.

### Posterior

![Corner plot of the BNS/CE posterior over chirp mass, eta, spin1z and spin2z, with the injection marked in orange](assets/bns_ce_corner.png)

Median chirp mass 1.2187730 M☉ (truth 1.2187707), η piled against its 0.25 prior
edge, both spins consistent with zero. Chirp mass is drawn as an offset from the
injected value because its posterior is only ~10⁻⁶ M☉ wide — a 1-part-in-10⁶
measurement, which is what SNR 607 over a ~1000 s inspiral buys.

The chirp-mass marginal is **one-sided about the truth by construction, not
biased**: the injection sits exactly on the η = 0.25 prior boundary, and Mc–η are
~97 % anti-correlated (visible as the thin diagonal ridge), so truncating η from
above truncates Mc from below. The spin panels show the expected χ_eff degeneracy
triangle — only the mass-weighted spin combination is measured, so the individual
spins slide along the anti-diagonal until the positive-support priors cut them off.

### Regenerating the figures

```bash
python bin/make_bns_ce_figures.py     # reads examples/output/bns_ce_rb_hmc/
```

[`bin/make_bns_ce_figures.py`](../bin/make_bns_ce_figures.py) rebuilds both figures
from the run artefacts, so every number on the axes traces back to a measurement
rather than to this page. The convergence series is cached to
[`docs/assets/bns_ce_convergence.csv`](assets/bns_ce_convergence.csv) — the raw
stdout logs fall under the repo's `*.log` ignore, so that CSV is the committed
evidence and the left-hand figure regenerates from a clean checkout. The corner plot
needs `samples.npz` (1.04M × 4 float64, ~40 MB, deliberately untracked); rerun the PE
to regenerate it.

## Notes / known benign details

- Max sampled log-posterior (−15.48) exceeds the damped-Newton "MAP" (−18.14). There
  is no sharp finite mode: the boundary directions are flat until the sigmoid
  Jacobian turns over, so Newton stops early in a plateau. This affects only the
  preconditioner's centring, and acceptance of 0.7–0.8 shows the metric is fine.
- The 580.159.03-userspace workaround (`bin/run_gpu_with_matched_driver.sh`) is no
  longer needed — the machine was rebooted and driver 580.173.02 is now loaded and
  matched. The script is kept for the next time an upgrade lands before a reboot.
