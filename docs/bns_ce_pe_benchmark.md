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

## Status

- **Reduced CPU config: CONVERGED.** Gate closed at production block 27:
  rank-R̂(glob) max = 1.0091 < 1.01, min ESS = 8543, zero stuck chains; posterior
  correct (Mc median within 1e-5 Msun of truth, η piled at 0.25, spins consistent
  with 0, max sampled log-posterior = the MAP value). 42.6 min wall on CPU — the
  CPU validates the machinery; the 20-minute target is for the GPU.
- Full-scale setup (2048 s, 4096 Hz, 10 Hz): **2.9 min on CPU**, RB parity passes
  (312 bins, SNR 607). The production wall-clock guard now budgets the *whole* run.
- **GPU**: the machine's NVIDIA userspace (580.173.02) was upgraded past the loaded
  kernel module (580.159.03; reboot pending), so CUDA cannot initialize against the
  system libraries. The exact-match 580.159.03 userspace was retrieved from the
  official Ubuntu Launchpad archive and extracted (no root, reversible);
  `LD_LIBRARY_PATH` pointing at it restores CUDA — **verified**: JAX sees
  `CudaDevice(id=0)` and computes. The sandboxed agent session is not permitted to
  launch runs with injected driver libraries, so the final GPU benchmark needs a
  human to start it:

  ```bash
  bash bin/run_gpu_with_matched_driver.sh --outdir examples/output/bns_ce_rb_hmc
  ```

  (or, after a reboot, simply
  `python bin/run_bns_ce_pe.py --outdir examples/output/bns_ce_rb_hmc`).
  Budget projection: ~3 min CPU-pinned setup, ~2.4 min MAP+Laplace
  (compile-dominated), warmup + compile ~2 min, and ≲ 25 production blocks (256
  chains halve the R̂ estimator noise vs the 64-chain CPU validation) — inside the
  20-minute budget if the T2000 (fp64) sustains ≲ 20 s per block, to be measured.
