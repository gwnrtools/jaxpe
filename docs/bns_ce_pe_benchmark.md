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

[`bin/make_bns_ce_figures.py`](../bin/make_bns_ce_figures.py) rebuilds all three
figures from the run artefacts, so every number on the axes traces back to a
measurement rather than to this page. The convergence series is cached to
[`docs/assets/bns_ce_convergence.csv`](assets/bns_ce_convergence.csv) and the
per-configuration stage timings to
[`docs/assets/bns_ce_speed_stages.csv`](assets/bns_ce_speed_stages.csv) — the raw
stdout logs fall under the repo's `*.log` ignore, so those CSVs are the committed
evidence and both figures regenerate from a clean checkout. The corner plot needs
`samples.npz` (1.04M × 4 float64, ~40 MB, deliberately untracked); rerun the PE to
regenerate it.

## Speed round 2: chasing a 4-minute budget

A follow-up goal asked for the same PE inside **4 minutes**. The best *sound*
configuration lands at **6.16 min** wall clock on the 1.4 + 1.4 M☉ source (warm
compile cache), against 15.42 min for the reference — so 4 minutes is not reached,
but the earlier claim on this page that it was unreachable has been **retracted**;
see [Retraction](#retraction-the-floor-argument-was-wrong) below. Note that 6.16 min
is this configuration *on this source*: a 1.35 + 1.25 M☉ injection runs the same
settings with no retuning but needs more blocks
([Generality](#generality-no-per-source-tuning)). This section records what moved the
number and what did not — the negative results are the more useful half.

![Wall-clock breakdown by pipeline stage for five configurations, from the 15.4-minute round-1 reference down to 6.16 minutes, with a rejected 4.74-minute run hatched](assets/bns_ce_speed_stages.png)

Every bar is one run's `timings:` line, stacked to its own total; the numbers are
regenerated from [`docs/assets/bns_ce_speed_stages.csv`](assets/bns_ce_speed_stages.csv)
by `bin/make_bns_ce_figures.py`. Two things are readable directly off it. First,
**production dominates** in every configuration — setup is ~40 s and flat, so the
optimization target is the sampler, not the pipeline around it. Second, the
hatched bar is the one that hit 4.74 min: it is hatched because its local HMC
acceptance was exactly 0.00 for all 13 blocks, i.e. it converged as a
*flow-only* independence sampler with a dead gradient kernel. It is recorded, not
claimed.

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
- **Re-tuning ε *after* equilibration, against the equilibrated positions** — the
  single largest sampler-side win of round 2 (10.56 → 6.16 min). Warmup adapts the
  step size while the chains are still inside a 0.3σ ball, where the curvature is
  not the curvature they will actually see; three discarded local blocks after
  equilibration fix that, and are what make L = 32 viable. Note this is *not* the
  same as the failed "re-tune to 0.75 acceptance" experiment below: the re-tune
  targets the same acceptance as warmup, it just does so at the right positions.
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
- **Shorter trajectories + more flow proposals,** *at fixed ε*. A flow proposal
  costs ~1/300 of a 128-step trajectory, so this looked like free mixing. Cutting
  L 128 → 32 took 40 blocks without converging. **Superseded:** the cause was the
  inherited step size, not the trajectory length — with a post-equilibration ε
  re-tune, L = 32 converges in 25 blocks and 6.16 min. See
  [Trajectory length is coupled to the step size](#trajectory-length-is-coupled-to-the-step-size).
- **Longer trajectories at fixed ε** (L = 192). Acceptance collapsed to 0.03–0.19:
  ε adapted at one trajectory length does **not** transfer to another, because
  leapfrog energy error accumulates along the trajectory.
- **Re-tuning ε to 0.75 acceptance after equilibration.** Raised acceptance to 0.8
  and made convergence *worse* (20 blocks vs 13): for this geometry the
  larger-step/lower-acceptance setting explores better.
- **Trimming the MAP search** (24 → 12 Newton steps). Saved 5 s of setup and cost
  ~350 s of sampling: a worse mode gives a worse Laplace metric, and production
  acceptance fell from ~0.45 to ~0.17.

### Retraction: the floor argument was wrong

An earlier revision of this page argued that ~30k leapfrog steps × ~1.7 ms of
irreducible serial cost put a ~7-minute end-to-end floor on this problem, and
concluded that 4 minutes was unreachable without different hardware. **The
arithmetic was right and the premise was not.** It took the ~128-step trajectory
length as given, when that length was itself an artefact of a mis-scaled metric —
so it bounded the cost of the wrong sampler. The corrected statement is that the
step count is a *tuning outcome*, not a property of the posterior, and the honest
floor is unknown until the tuning is exhausted.

What made the difference is below; it is the reason the number moved from ~10.5 min
to 6.16 min after that paragraph was written.

### Trajectory length is coupled to the step size

Round 1 established that L = 12 → 48 → 128 monotonically improved R̂ and that
cutting L to 32 "starved the flow" (40 blocks, no convergence). Both observations
are real, and together they are misleading, because **ε was never re-tuned at the
new L**. This page already records that ε does not transfer across trajectory
lengths (L 96 → 192 at fixed ε collapsed acceptance to 0.03–0.19); the same
non-transfer applies downward, and the short-L test inherited a step size adapted
for a long one.

Adding a short **ε re-tune after equilibration** — three discarded local blocks that
re-run Robbins–Monro against the current metric and the *equilibrated* chain
positions, rather than the initial ball — makes L = 32 work:

| configuration | L | ε after tuning | production acc | blocks | total |
|---|---|---|---|---|---|
| round-2 committed | 96 | 0.153 (warmup only) | 0.94–0.96 | 28 | 10.56 min |
| + post-equilibration ε re-tune | 32 | 0.255 | 0.69–0.86 | 25 | **6.16 min** |

Note the acceptance column: the 10.56-min configuration was running at 0.95
acceptance, which for HMC means the step size was far too small and most of the
gradient work was buying almost no displacement. The re-tune is what exposed that,
because it adapts where the chains actually are. Block count barely changed
(28 → 25); the whole gain is that each block now costs 7.5 s instead of 15 s.

Reproduced twice back to back (6.14 / 6.16 min, identical R̂ 1.0099 and min ESS
19919), so this is not step-size lottery.

**L ≈ 32 is the knee, and so is the local/global split.** Both halvings were then
tested back to back against it, each changing one thing:

| change from the 6.16-min configuration | steady-state block cost | blocks | total |
|---|---|---|---|
| — | 8.4 s | 25 | **6.16 min** |
| L 32 → 16 (400 fewer leapfrog steps/block) | 7.0 s | 26* | 7.01 min* |
| flow globals 1200 → 600 | 6.8 s | 35 | 6.54 min |

(*measured on the local-blocks-in-equilibration variant, so compare its production
against that variant's 33 blocks / 311 s, not against the 25 / 213 above.)

Each halving buys ~1.5 s per block and costs 8–10 extra blocks, so each is a net
loss. The two samplers are close to balanced, which is why this configuration
behaves like an optimum rather than an arbitrary point: **re-balancing local against
global is exhausted as a lever.**

**Equilibration is also at its knee.** Cutting it 5 rounds → 3 saves 17 s and costs
107 s: production ran 37 blocks instead of 25. The log says why — global acceptance
in production block 1 was 0.38 against 0.59 with five rounds. The last two rounds
are not spreading chains that are already spread; they are *training the flow*, and
an undertrained flow costs a dozen production blocks. `--n-chains` was not swept:
round 1 measured per-step cost as sublinear in chains (16 → 512 chains, 1.86 → 8.32
ms), so 64 → 32 buys ~20 % per step while halving the samples per block.

Subtracting both local/global deltas from the 8.4 s block leaves ~2 s that is
neither sampler.
That residual was then measured directly rather than inferred, and it is diffuse,
not one hotspot: the decimated diagnostics cost 0.05 s per block early and 0.67 s at
block 25 (rank-normalized split-R̂ on the global subseries and on all samples, plus
Geyer ESS), and the flow refit every 4th block amortizes to ~0.75 s. The remainder
is `to_phys` over the accumulated decimated stack, the device→host transfer of the
global block (~2.4 MB, fetched more than once), and host sync. Removing all of it
optimistically returns ~30 s over a run, ~8 % — worth doing, but not a route to
4 minutes on its own.

### The metric: measured, and a smaller lever than it looked

The MAP-Laplace covariance is the curvature *at the mode*, and this posterior is
badly non-Gaussian there — the η → 1/4 and χ → 0 pileups give heavy tails, so the
mode is much sharper than the actual spread. Measured against 1.94M production
samples:

| direction | posterior σ | Laplace σ | ratio |
|---|---|---|---|
| chirp mass | 1.002e-06 | 2.61e-07 | 3.84 |
| η | 1.494e-04 | 2.98e-05 | 5.01 |
| spin1z | 6.017e-04 | 1.18e-04 | 5.10 |
| spin2z | 6.381e-04 | 1.19e-04 | 5.36 |

(physical marginals, reproducible from `samples.npz` against the `sigma_phys` line
the run prints; the corresponding unconstrained-space eigen-spectra differ by
11–21×, but that comparison sorts two spectra independently and is only indicative).

The metric is therefore genuinely too narrow — but **the ratio is nearly uniform
across directions**, and a uniform under-scaling of the mass matrix is exactly what
the step size absorbs. Only the *anisotropy* matters for preconditioning, and that
spans less than 1.4× here. This is the correction to the hypothesis that drove three
runs: the mis-scaled metric was real, was worth fixing, and was **not** the dominant
cost — re-tuning ε against it (previous section) captured essentially all of the
available gain.

The three attempts to replace the metric outright, all measured:

| attempt | metric σ_y (chirp mass) | result |
|---|---|---|
| ensemble covariance over all equilibration rounds | 6.4e-4 (true ~1.6e-5) | **4.74 min but local acc 0.00** — a flow-only sampler; rejected |
| + log-posterior outlier cut, final round only, revert guard | 7.0e-4 | guard fires, reverts to Laplace, 6.14 min |
| + local HMC blocks interleaved into equilibration | 6.6e-4 | no improvement; +35 s equilibration, 8.39 min |

The failure mode is consistent across all three: the equilibration ensemble is
~40× **over-dispersed in the chirp-mass direction**, which is five orders of
magnitude tighter than the others. A covariance estimated from it proposes
trajectories that leave the Mc ridge immediately, and acceptance goes to exactly
zero. Filtering the ensemble by log-posterior changed the estimate by 0.03 %
(7.019e-4 vs 7.021e-4), which rules out "a few stranded chains" and says the whole
ensemble is over-dispersed — the flow's independence proposals cannot resolve a
1.6e-5-wide ridge, and interleaving local moves (attempt 3) does not pull them back
fast enough to matter. The revert guard — drop back to the Laplace metric if local
acceptance collapses after the re-tune — is what keeps this safe rather than silent.

**Soundness criterion adopted here:** a configuration counts only if the local HMC
kernel has non-zero acceptance in production. A flow-only sampler that passes R̂ and
ESS is not the thing this benchmark is measuring, and its wall time is not
comparable to the others.

### What is left

The sampler-configuration sweep is exhausted: every knob moved off the 6.16-minute
configuration made it worse, in both directions where both were tried.

| knob | values tried | best alternative |
|---|---|---|
| trajectory length L | 16, 32, 96 | 7.01 min (L = 16) |
| flow proposals per block | 600, 1200 | 6.54 min (600) |
| equilibration rounds | 3, 5 | 7.68 min (3) |
| mass matrix | Laplace, 3 ensemble variants | 8.39 min, or unsound |

That is what makes 6.16 min a defensible number rather than a lucky draw, and it
also means the remaining distance to 4 minutes is not in the sampler's settings.

#### The 3-region evaluation is worth ~6 %, not 3× (measured)

Earlier revisions of this page named the obvious remaining lever: `Phase`/`Amp`
evaluate **all three** frequency regions at every point and select with heaviside
masks, and JAX evaluates both sides of an elementwise `where`, so grad `Phase` is
13,038 HLO lines against 3,121 for inspiral-only. That 4× line-count ratio does not
survive being timed.

Measured end to end on the benchmark's own relative-binning log-posterior, with
`Phase`/`Amp` monkeypatched to their inspiral-only branches as an upper bound:

| | gradient, 64 chains |
|---|---|
| full 3-region PhenomD | 2.657 ms |
| inspiral-only (unreachable bound) | 1.613 ms |

So the *ceiling* is 39 % of the gradient. But a production block spends only
800 × 2.657 ms ≈ 2.1 s of its 8.4 s on gradients, so the whole partition is worth
**~21 s of a 371 s run, ~6 %** — and that is the unreachable bound, not the
achievable figure. The line count misled because those instructions are cheap
elementwise work over 126 points; HLO instruction count is not time.

Two further facts kill the "it's all inspiral for a BNS anyway" shortcut: of the 126
bin edges, only 93 lie below the phase transition f1 = 1305 Hz and 79 below the
amplitude transition f3 = 1015 Hz. A correct partition therefore has to evaluate two
regions with a gather/scatter, not one.

**Ruled out, not deferred.** Optimizations may not alter the physics of the
templates — no dropping waveform content, no reduced-frequency-content
approximations, no reduced precision in the template evaluation. The partition
sketched above would have been numerically exact (it skips branches whose heaviside
weight is identically zero, and the region membership is derived from the prior's
mass support, not from the source), but it edits the waveform module for ~6 %, and
that is not a trade worth making against this constraint. The same constraint is why
[float32](#what-did-not-work-measured-do-not-retry) is rejected on correctness
grounds rather than merely noted as risky. **Every timing on this page evaluates the
full three-region IMRPhenomD in float64.**

The speedups that did land are all *sampler-side* — where the chains are, how far the
proposals reach, what the step size is adapted against — and none of them touch the
likelihood or the waveform. In particular the flow supplies Metropolis–Hastings
proposals, so the target density is preserved exactly however good or bad the flow
is; a poor flow costs efficiency, never correctness.

#### Where the time actually goes

Production is ~58 % of the run. Per-chain gradient cost falls steeply with chain
count, i.e. at 64 chains the GPU is launch-overhead-bound rather than compute-bound:

| chains | gradient | per chain |
|---|---|---|
| 32 | 2.713 ms | 84.8 µs |
| 64 | 3.228 ms | 50.4 µs |
| 256 | 6.047 ms | 23.6 µs |
| 1024 | 13.643 ms | 13.3 µs |

Round 1 moved 256 → 64 chains and that helped, but it was decided at L = 128 against
a mis-tuned step size, so it was re-measured under the corrected configuration. The
predicted mechanism was real and still did not pay:

| | 64 chains | 256 chains |
|---|---|---|
| warmup | 33.7 s | 55.2 s |
| equilibration | 72.7 s | 100.3 s |
| production | 213 s (25 blocks) | 223 s (17 blocks) |
| **total** | **6.19 min** | 7.17 min |

Four times the chains genuinely bought a third off the block count (25 → 17) — more
flow training data per block, and R̂ on the spin directions is what gates this run.
But per-block cost rose in step, so production came out a wash, and the whole 59 s
loss is *preamble*: warmup and equilibration run a fixed number of steps, so their
cost scales with chains while their benefit does not.

### Flow proposal *reach*: a large mean gain that buys unacceptable variance

The sweep above tested how *many* flow proposals to make, in both directions, and
never how far they *reach*. The flow is an RQ-spline coupling flow, and outside its
`interval` the transform is the **identity** — so with the round-1 value of 8.0 the
far tails were proposed as if Gaussian, while the η → 1/4 and χ → 0 pileups make them
Exponential(~1) in unconstrained space. That mismatch is visible in every log: R̂
reaches 1.0121 by block 2 and then grinds ~18 blocks on spin1z alone.

Widening it works, and dramatically — on one seed:

| interval | blocks | production | total (seed 42) |
|---|---|---|---|
| 8 | 25 | 213 s | 6.19 min |
| 16 | 14 | 122 s | 4.78 min |
| 32 | 8 | 68 s | **3.87 min** |

**That 3.87 min does not survive a change of random seed.** The identical
configuration at seed 7 takes **55 blocks and 10.71 min** — 2.8× the reference. The
two-seed mean (7.3 min) is *worse* than the interval-8 configuration it appeared to
beat, so the headline number was a lottery win, not a result:

| `--flow-interval 32` | blocks | total |
|---|---|---|
| seed 42 | 8 | 3.87 min |
| seed 7 | 55 | 10.71 min |

Interval 16 behaves the same way — 4.78 min at seed 42, **9.58 min (48 blocks) at
seed 7** — so the full picture across two intervals and two seeds is:

| interval | seed 42 | seed 7 | mean |
|---|---|---|---|
| 16 | 4.78 min | 9.58 min | 7.2 min |
| 32 | 3.87 min | 10.71 min | 7.3 min |

Neither beats interval 8's 6.16 min on the mean, and both have a worst case around
10 min. The mechanism is the same one that produced the apparent gain: widening the
interval lets the flow reach genuinely distant tail states, so an accepted global
move is worth far more — but acceptance collapses with it (0.55 at interval 8 → 0.36
at 16 → 0.09–0.14 at 32), and at ~0.1 acceptance whether the run converges in 8
blocks or 55 depends on whether useful tail jumps land early.

**The control, and the matched-seed result.** The interval-8 runs quoted at
6.14/6.16/6.19 min were all at the *same* seed (42) and all converged at block 25, so
that spread measures GPU/timing nondeterminism, not seed robustness. Running the
default at seed 7 supplies the missing control — 7.12 min, 32 blocks — and makes the
comparison matched:

| interval | seed 42 | seed 7 | mean | spread |
|---|---|---|---|---|
| **8 (default)** | 6.19 min | 7.12 min | **6.66 min** | **1.15×** |
| 16 | 4.78 min | 9.58 min | 7.18 min | 2.00× |
| 32 | 3.87 min | 10.71 min | 7.29 min | 2.75× |

The default wins on the mean *and* has a 1.15× spread against 2.0–2.75×. So the
variance belongs to the widened interval, not to the problem: at healthy acceptance
(0.48–0.58 at interval 8, seed 7) the run is stable across seeds, and it is the
collapse to ~0.1 acceptance that turns the run into a lottery. Widening the interval
is therefore a genuine negative result, not merely an unlucky measurement.

**This also corrects the headline.** "6.16 min" is a seed-42 number. Stated
honestly across the two seeds measured, the final configuration converges in
**6.2–7.1 min (mean 6.66)**, and every configuration comparison earlier on this page
that rests on single-seed timings inherits that same caveat — differences smaller
than ~15 % between two single-seed runs are not resolved by the data.

**Cycling a narrow and a wide kernel does not recover the best case.** The obvious
repair is to stop choosing an interval and instead alternate two independence-MH
kernels per block — half the proposals from the narrow flow (healthy acceptance,
reproducible), half from a wide one (rare long jumps). This is exact: each sub-block
leaves the posterior invariant, so their composition does too, with no mixture
density or reweighting. It is available as `--flow-interval-wide` (default off).

Measured at seed 42: **6.56 min, 26 blocks**, against 6.19 min / 25 blocks for the
narrow flow alone. The per-block log explains it — the wide component's acceptance is
**0.02**, rising only to 0.12 by the final block. It contributes almost nothing while
consuming half the global proposals and an extra flow fit (9.5 s → 14.7 s), and
block-1 R̂ degrades to 1.15–1.27 (from ~1.10) because the effective global sample
count is halved.

That result also reinterprets the interval-32 run. "8 blocks at seed 42" looked like
evidence that *long jumps help*; if that were the mechanism, supplying those same
jumps alongside a healthy narrow kernel would have reproduced the benefit. It did
not. The 8-block run was a lucky trajectory, not a transferable mechanism — which is
consistent with its 10.71 min counterpart at seed 7.

Two lessons, both general:

- **A single-seed timing is not a measurement** when the sampler's acceptance is low.
  This page's headline number was committed before its own queued reproduction
  returned, and the check overturned it.
- **Mean and variance trade against each other here.** The useful framing is not
  "which interval is fastest" but "which is fastest *at an acceptable worst case*",
  and a configuration whose spread spans 3.9–10.7 min has no usable worst case even
  though its best case is the fastest ever measured on this problem.

### The flow was oversized: 8 → 4 coupling layers, 6.59 → 5.31 min

Profiling `_global_block` *warm* exposed a cost pool that had been mis-attributed all
round. A 1200-step global block costs **3.38 s**, of which only **0.61 s** is the
likelihood — the rest is the flow's *two* passes per step (`sample` and `log_prob`).
Earlier accounting costed that block at 0.72 s from the likelihood figure alone, which
is exactly the ~45 % of each production block that kept coming up unaccounted for and
was twice written off as kernel-launch overhead.

(The first attempt at this profile was wrong and is worth recording: `n_steps` is a
static argument, so warming up at 10 steps does not warm the 1200-step trace, and the
numbers carried ~1.3 s of compilation each.)

| flow | warm 1200-step block | per step |
|---|---|---|
| 8 layers / width 64 | 3.384 s | 2.820 ms |
| 4 / 64 | 2.000 s | 1.667 ms |
| 4 / 32 | 1.639 s | 1.366 ms |
| 2 / 64 | 1.307 s | 1.089 ms |
| *likelihood alone* | *0.612 s* | *0.510 ms* |

Eight coupling layers is generous for a **4-dimensional** posterior, and halving them
is the only change in this whole round that is a *pure win* rather than a trade —
every other knob bought cheaper blocks and paid for it in more of them:

| flow | seed 42 | seed 7 | mean | spread |
|---|---|---|---|---|
| 8 layers | 6.05 min (25 blocks) | 7.12 min (32) | 6.59 min | 1.18× |
| **4 layers (shipped)** | **4.97 min (24 blocks)** | **5.65 min (31)** | **5.31 min** | 1.14× |
| 2 layers / width 32 | 5.41 min (33 blocks) | — | — | — |

Faster at *both* seeds, block count unchanged, global acceptance actually improving
(0.40 → 0.59), and no variance inflation — unlike the interval widening, which bought
its best case with a 2.75× spread. **4 layers is the knee:** at 2 layers/width 32 the
capacity genuinely degrades and the block count jumps 24 → 33, so the
cheaper-block/more-blocks trade returns on the far side.

#### One stale negative retracted, one non-result

The equilibration measurement above (5 rounds → 3 costs 107 s) was taken under the
**8-layer** flow, so it was re-run under the 4-layer one. The old failure mechanism is
gone: block-1 global acceptance is now **0.60**, against 0.38 before. A smaller flow
trains adequately in 3 rounds, and the 107 s penalty does not reproduce.

| | 5 rounds | 3 rounds |
|---|---|---|
| equilibration | 59.2 s | 44.0 s |
| production | 157.2 s (24 blocks) | 165.1 s (25 blocks) |
| total | 297.9 s | 289.7 s |

But the *net* is 8 s — 2.7 %, well inside the ~15 % resolution limit for single-seed
comparisons established above — because the extra production block eats most of the
equilibration saving. So: the old negative is **retracted** (a 100+ s effect, clearly
resolvable), while "3 rounds is better than 5" is **not resolved** by this data. The
default stays at 5, which is the value verified at two seeds.

This is the third stale negative overturned by re-measuring under changed conditions
(after "short trajectories starve the flow", and the flow-capacity assumption itself).
The recurring error is treating a measurement as a property of the *problem* when it
was a property of the *configuration it was taken in* — worth stating because it is
the single most productive check applied in this whole exercise.

#### The production block budget, closed

Repeated attempts to find a hidden overhead pool ended by closing the budget instead.
Measured warm and in situ, a 6.6 s production block at the shipped configuration is:

| component | measured |
|---|---|
| local HMC — `run_chains`, 25 steps × L=32 = 800 gradients | **3.597 s** |
| global flow block — 1200 independence-MH steps | 2.000 s |
| diagnostics — rank-R̂ ×2 + Geyer ESS, on the real backend | ~0.30 s |
| flow refit, amortized over 4 blocks | ~0.4 s |
| **sum** | **~6.3 s** (measured block: 6.6 s) |

The budget closes to within ~5 %, so **there is no unaccounted overhead left to
remove**. The earlier "~2 s per block unexplained" was an artefact of costing the
local block with a standalone `vmap(grad(loglike))` at 3.228 ms/gradient; the sampler
differentiates `log_posterior` (likelihood + prior + sigmoid Jacobian) *inside a
scan*, which measures 4.496 ms/gradient. Two lessons repeat here: a component must be
timed in the context that actually runs it, and diagnostics measured under
`JAX_PLATFORMS=cpu` do not bound diagnostics running on the GPU.

The consequence is structural: **local HMC is 55 % of every block, and it is the
waveform gradient itself** — not machinery around it. With trajectory length already
at its measured knee (L = 32; 16 and 96 are both worse) and the template physics
fixed by constraint, that 3.6 s is irreducible here.

### Why 4 minutes was not reached

Unlike the retracted argument above, this is an accounting of measured levers rather
than an extrapolation from an assumed trajectory length. Starting from 371 s:

Every sampler-side knob has now been moved in both directions where both exist, and
every one is worse than the shipped configuration:

| knob | tried | best alternative | why it loses |
|---|---|---|---|
| trajectory length L | 16, 96 | 7.01 min | block cost falls less than block count rises |
| flow proposals/block | 600, 2400 | 6.54 min | same trade, both directions |
| equilibration rounds | 3 | 7.68 min | undertrained flow costs 107 s to save 17 s |
| chains | 256 | 7.17 min | preamble scales with chains, benefit does not |
| mass matrix | 3 ensemble variants | 8.39 min | ensemble is 40× over-dispersed in Mc |
| flow spline interval | 16, 32 | 7.18 min mean | best case 3.87 min, worst 10.71 — a lottery |
| cycled narrow+wide kernels | interval 8+32 | 6.56 min | wide component accepts at 0.02 |
| warmup blocks | 2 | 8.32 min | saves 11 s, costs 140 s via a worse initial flow |

The last row is the pattern in miniature: **both halves of the preamble are
load-bearing.** Cutting equilibration starves the flow of refits; cutting warmup
starves it of its initial training set (fit loss 0.206 → 0.681, step size stranded at
0.226 with acceptance 0.94). The preamble is not overhead to be trimmed — it is what
makes production converge in 25 blocks instead of 41.

That leaves only the code-level levers, both measured:

| lever | measured value | status |
|---|---|---|
| sampler configuration | 0 s | swept exhaustively, table above |
| PhenomD 3-region partition | ≤ 21 s | ~6 %, and **ruled out** — alters the waveform module |
| per-block overhead (diagnostics, refit, transfers) | **8 s, implemented** | estimated at ~30 s; the estimate was 4× too high |
| **shipped, all permitted levers applied** | **363 s = 6.05 min** | still > 4 min |

With the template-physics constraint in force, the only permitted code lever was the
per-block overhead. It is now implemented -- one device->host transfer per array per
block instead of three, each block mapped to physical space once on arrival instead
of re-transforming the whole accumulated stack every block (an O(n^2) over the run),
and the global block decimated once rather than twice. All three are
identity-preserving, and the run still converges at exactly block 25 with Rhat 1.0099
and min ESS 19919, which is the check that the refactor changed only speed. It is
worth **8 s** (371 -> 363 s), not the ~30 s estimated: the transfers and the repeated
transform were real but were about a quarter of the residual, the rest being
kernel-launch and `scan` overhead inside the sampling loops. Closing the remaining 130 s would
have to come out of equilibration (72.7 s) or warmup (33.7 s), and both are now
measured load-bearing in both directions. So on this hardware, with this waveform and
this convergence gate, **4 minutes is not reachable** by configuration or by any code
change that respects the constraint.

What would change that, in rough order of leverage: a GPU with real f64 throughput
(the T2000 is a consumer part at 1/32 rate, and this problem cannot use f32 — see
above); a cheaper waveform gradient than IMRPhenomD's coefficient chain (a
reduced-order or surrogate model); or a sampler whose preamble does not have to
equilibrate 64 chains before the kept series starts. Each of those is a different
project, not a tuning pass — which is the honest place to stop.

### Generality (no per-source tuning)

Everything above is derived at runtime from the data and priors, not fitted to the
1.4+1.4 source. The equal-mass hard-coding that did exist was removed: component
masses are now `--mass1/--mass2`, η's truth comes from them, and the optimiser start
is the fiducial point inset from whichever prior edges it happens to touch. Rerunning
a **1.35 + 1.25 M☉** injection with no retuning exercises that: same bin count (125),
relative-binning parity passes on its own waveform (6.2e-2 < 0.1), warmup adapts to
its own ε (0.325 vs 0.322), and the post-equilibration re-tune to its own (0.283 vs
0.255). Nothing needed a per-source constant.

**But the wall clock does not carry over, and that distinction matters.** The
*configuration* is source-independent; the *6.16-minute number* is not a property of
the configuration alone, it is a property of the configuration applied to this
source. Measured under the L = 32 setup:

| source | network SNR | blocks | total | R̂ | min ESS |
|---|---|---|---|---|---|
| 1.4 + 1.4 M☉ | 607 | 25 | 6.16 min | 1.0099 | 19919 |
| 1.35 + 1.25 M☉ | 571 | 42 | 8.69 min | 1.0099 | 12408 |

The unequal-mass source is simply a harder posterior — it also needed 35 blocks under
the older L = 96 configuration, against 28 for equal mass, so this is not something
the speed work introduced. Flow acceptance is comparable between the two (mean 0.48
vs 0.55), so the flow is not the discriminator; the likely driver is that η's truth
sits just *inside* its prior edge (0.24963) rather than exactly on it, which trades
a pinned corner for a genuinely curved two-dimensional ridge.

One real defect surfaced here and is fixed: `--max-production-blocks` defaulted to
**40**, which the unequal-mass run hit at R̂ = 1.0107 and therefore reported as
`converged: False`. Re-run with the cap raised, the identical configuration
converges at block 42 — it was two blocks short. A cap that converts "needs slightly
longer" into "failed" is measuring the cap, not the sampler; `--max-minutes` is the
real budget guard, so the default is now 80.

The unequal-mass posterior also reproduces the *shape* effects documented for the
equal-mass one, which is the more useful part of this check: chirp mass recovered to
the printed precision, both spins piled above their zero-support prior floor
(median ~6e-4, 90 % CI excluding 0), and η's median sitting ~2σ *below* its truth
(0.24935 vs 0.24963) for exactly the reason given under
[Posterior](#posterior) — forcing χ ≥ 0 pushes η down through the χ_eff–η
anti-correlation. Seeing the same signature at a source whose η truth is *not* on
the prior edge is evidence that it is prior geometry rather than anything
source-specific.

### Posterior validation, and calibrating the JS threshold

Halving the bins, shortening the trajectories and re-tuning the step size could all
in principle bias the posterior, so the fast configuration is compared to the
15.4-minute reference sample-set to sample-set
([`bin/compare_bns_posteriors.py`](../bin/compare_bns_posteriors.py)):

| comparison | worst JS | worst median shift |
|---|---|---|
| 312-bin reference vs **final 6.16-min configuration** | **2.1e-3** | 0.022 σ |
| 312-bin reference vs the earlier 125-bin config | 2.1e-3 | 0.018 σ |
| **control:** two independent 125-bin runs | **4.6e-3** | 0.047 σ |

The control is the point: re-running the sampler moves the posterior *more* than any
of these changes do. They are therefore below the Monte Carlo noise floor, and the
1e-3 threshold quoted in the relative-binning status page — which belongs to a
deterministic grid comparison — is simply not applicable to sample-based comparisons
at these sizes.

Truth recovery under the final configuration is +1.80 σ in chirp mass and −1.82 σ in
η, with both spins ~+1 σ. Those are not small numbers and they are not noise: they
are the prior-boundary signature described under [Posterior](#posterior), and they
appear at the same magnitude in the 15.4-minute reference. A change that left the
sampler unbiased would have to reproduce them, and it does — which is what this
comparison is testing.

Reproducibility of the headline number, three consecutive runs of the final
configuration: **6.14 / 6.16 / 6.19 min**, each converging at block 25 with R̂ 1.0099
and min ESS 19919.

## Notes / known benign details

- Max sampled log-posterior (−15.48) exceeds the damped-Newton "MAP" (−18.14). There
  is no sharp finite mode: the boundary directions are flat until the sigmoid
  Jacobian turns over, so Newton stops early in a plateau. This affects only the
  preconditioner's centring, and acceptance of 0.7–0.8 shows the metric is fine.
- The 580.159.03-userspace workaround (`bin/run_gpu_with_matched_driver.sh`) is no
  longer needed — the machine was rebooted and driver 580.173.02 is now loaded and
  matched. The script is kept for the next time an upgrade lands before a reboot.
