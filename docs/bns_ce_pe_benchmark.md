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

## Speed round 2: chasing a 4-minute budget

A follow-up goal asked for the same PE inside **4 minutes**. That was **not
achieved**: the best configuration lands at ~7–10 min wall clock (warm compile
cache), against 15.4 min for the reference. This section records what moved the
number, what did not, and the measurement chain that bounds the floor — the
negative results are the more useful half.

### Where the time actually goes

Profiling the hot loop first (all on the T2000, f64):

| measurement | value | consequence |
|---|---|---|
| per-leapfrog cost, 64 chains / 125 bins | ~6–7 ms | production needs ~30k of these |
| same, 16 → 512 chains | 1.86 → 8.32 ms | **sublinear**: ~1.7 ms is fixed per step |
| waveform grad, n_freq 1 → 157 | 2.84 → 5.86 ms | cost is mostly **independent of bin count** |
| IMRPhenomD grad graph | 3588 HLO instr., 5 fusions | a long serial chain, not a big array op |

So the cost is a fixed serial critical path through IMRPhenomD's phenomenological
coefficient algebra, once per leapfrog step. Fewer bins and fewer chains barely
help; only **fewer gradient evaluations** or a cheaper waveform would.

### What worked

- **Persistent XLA cache** — setup 3.5 min → 49 s (already in round 1).
- **Fewer chains.** Since cost is sublinear in chains but R̂ depends on *per-chain*
  length, 64 chains give ~2.4× more per-chain steps/second than 256.
- **Coarser bins**, ε 0.1 → 0.25 (312 → 125 bins), validated to still pass the same
  parity tolerance, and separately validated at the posterior level (below).
- **Fixing the step-size adaptation.** The Robbins–Monro gain of 1.0 could not reach
  the acceptance target in a short warmup (acceptance stuck at 0.98, i.e. ε ~4× too
  small). Raising the gain fixed that but introduced *overshoot*: across identical
  runs ε landed at 0.33 / 0.36 / 0.15 and production took 13 / 34 / 28 blocks —
  7–14 min of spread from warmup noise alone. Averaging log ε over the
  post-transient warmup blocks (Polyak averaging, as inside Stan's dual averaging)
  is the fix.
- **Equilibration before the kept series.** Discarded flow rounds spread the chains
  and bootstrap the flow, so production starts at stationarity instead of burning
  blocks on a transient that R̂ cannot distinguish from non-convergence.
- **O(n) diagnostics.** R̂/ESS were recomputed over the whole growing sample stack
  every block — O(n²) across a run. Each block is now decimated on arrival.
- **Forward-mode C¹ matching in IMRPhenomD** (`_value_and_deriv`). PhenomD used six
  internal `jax.value_and_grad` calls to match its inspiral/intermediate/ringdown
  pieces; under a sampler's own reverse-mode gradient those became
  reverse-over-reverse AD. For a scalar argument the JVP with unit tangent *is* the
  derivative, so this is mathematically identical — verified: waveform values agree
  to 2e-18…2e-15 and parameter gradients to 6e-15…3e-13 across BNS and spinning-BBH
  points. Worth ~4% of the gradient graph, not the 4× first hoped.

### What did NOT work (measured, do not retry)

- **float32.** 3× faster, and *wrong*: a BNS inspiral from 10 Hz accumulates ~10⁵
  radians of phase, and f32's ~7 significant digits leave ~0.01 rad — the exact-at-
  fiducial check returned lnL = −34 instead of 0. The coalescence-time phase can be
  cancelled analytically (t_c is fixed, so it divides out of the heterodyne ratio),
  but the *intrinsic* phase cannot. Single precision is unusable for long inspirals.
- **Shorter trajectories + more flow proposals.** A flow proposal costs ~1/300 of a
  128-step trajectory, so this looked like free mixing. It is not: cutting L 128 →
  32 starved the flow's training data and took 40 blocks without converging.
  Long trajectories do work the flow cannot replace.
- **Longer trajectories at fixed ε** (L = 192). Acceptance collapsed to 0.03–0.19:
  ε adapted at one trajectory length does **not** transfer to another, because
  leapfrog energy error accumulates along the trajectory.
- **Re-tuning ε to 0.75 acceptance after equilibration.** Raised acceptance to 0.8
  and made convergence *worse* (20 blocks vs 13): for this geometry the
  larger-step/lower-acceptance setting explores better.
- **Trimming the MAP search** (24 → 12 Newton steps). Saved 5 s of setup and cost
  ~350 s of sampling: a worse mode gives a worse Laplace metric, and production
  acceptance fell from ~0.45 to ~0.17.

### The floor

Convergence needs ~30k leapfrog steps; each has ~1.7 ms of irreducible serial cost;
that is ~50–90 s of production before any overhead, and ~7 min end-to-end in
practice. Getting to 4 minutes needs one of: a GPU with real f64 throughput (this is
a consumer part at 1/32), a cheaper waveform gradient, or a weaker convergence gate.
The remaining in-code lever is that `Phase`/`Amp` evaluate **all three** regions at
every frequency and select with heaviside masks — JAX evaluates both sides of an
elementwise `where`, so every bin pays 3× (grad Phase 13,038 HLO lines vs 3,121 for
inspiral-only). Skipping regions needs a static decision, but the transition
frequencies depend on the sampled masses; specialising on "the whole band is
inspiral" would be true for BNS and false in general, so it is a design change, not
a tweak.

### Generality (no per-source tuning)

Everything above is derived at runtime from the data and priors, not fitted to the
1.4+1.4 source. The equal-mass hard-coding that did exist was removed: component
masses are now `--mass1/--mass2`, η's truth comes from them, and the optimiser start
is the fiducial point inset from whichever prior edges it happens to touch. Verified
by rerunning a **1.35 + 1.25 M☉** injection with no retuning: same bin count (125),
relative-binning parity passes on its own waveform (6.2e-2 < 0.1), and warmup adapts
to ε = 0.301 against 0.298 for the equal-mass source.

### Posterior validation, and calibrating the JS threshold

Halving the bins could in principle bias the posterior, so the fast configuration is
compared to the 15.4-minute reference sample-set to sample-set
([`bin/compare_bns_posteriors.py`](../bin/compare_bns_posteriors.py)):

| comparison | worst JS | worst median shift |
|---|---|---|
| 312-bin reference vs 125-bin fast | 2.1e-3 | 0.018 σ |
| **control:** two independent 125-bin runs | **4.6e-3** | 0.047 σ |

The control is the point: re-running the sampler moves the posterior *more* than
halving the bins does. The binning change is therefore below the Monte Carlo noise
floor, and the 1e-3 threshold quoted in the relative-binning status page — which
belongs to a deterministic grid comparison — is simply not applicable to
sample-based comparisons at these sizes.

## Notes / known benign details

- Max sampled log-posterior (−15.48) exceeds the damped-Newton "MAP" (−18.14). There
  is no sharp finite mode: the boundary directions are flat until the sigmoid
  Jacobian turns over, so Newton stops early in a plateau. This affects only the
  preconditioner's centring, and acceptance of 0.7–0.8 shows the metric is fine.
- The 580.159.03-userspace workaround (`bin/run_gpu_with_matched_driver.sh`) is no
  longer needed — the machine was rebooted and driver 580.173.02 is now loaded and
  matched. The script is kept for the next time an upgrade lands before a reboot.
