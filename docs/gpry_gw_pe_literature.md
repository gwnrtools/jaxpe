# GP-surrogate active-learning samplers in GW parameter estimation: a literature survey

Written while diagnosing a `gpry.gp_acquisition.GPAcquisitionError` on the
300-run PE campaign's `phenomd_gpry` leg (see `campaign/plan.md` amendment 15).
Two questions motivated this survey: has GPry itself been used for
gravitational-wave PE before, under what settings, and did anyone report the
kind of high-SNR acquisition failure we're seeing? And more broadly, has
*anyone* applied a GP-surrogate/active-learning approach — GPry or otherwise —
to LVK, Cosmic Explorer, or Einstein Telescope parameter estimation?

**Short answer:** GPry itself has exactly one published GW application, and
it's LISA (space-based, mHz band) — not LVK/CE/ET. But a *structurally very
similar* technique is in actual LVK production use today: **RIFT**, which
fits a Gaussian Process to a marginalized intrinsic-parameter likelihood, the
same shape of problem our `marginalized_phase_distance` and
`marginalized_intrinsic` likelihoods pose to GPry. RIFT's fifteen-year
publication record has documented, in detail, exactly the class of GP
robustness problem we hit — just from a different entry point (the regression
step, not GPry's acquisition step) — and the fixes that worked are concrete
and transferable.

## 1. GPry itself

### 1.1 The core algorithm (cosmology-native, not GW)

**El Gammal, Schöneberg, Torrado, Fidler (2022/2023), "Fast and robust
Bayesian inference using Gaussian processes with GPry,"** JCAP 10 (2023) 021,
[arXiv:2211.02045](https://arxiv.org/abs/2211.02045),
[GitHub](https://github.com/jonaselgammal/GPry).

This is the package we depend on (`jaxpe/surrogate/gpry_backend.py` wraps
`gpry.Runner` directly). It was built for **cosmological** likelihoods
(CMB/Planck-style), not GW ones — moderate dimensionality (their validation
problems are low-tens-of-parameters at most), likelihoods expensive enough
that "two orders of magnitude fewer evaluations than MCMC" matters, and — this
is the load-bearing assumption for our failure — likelihoods that are *smooth
up to a small amount of deterministic numerical noise, "less than 0.1 in log
posterior"* per the package documentation. A zero-noise CBC likelihood at
SNR≳50 violates that assumption badly: local log-likelihood curvature scales
as ~SNR², so the log-posterior can swing by hundreds of nats over a length
scale the GP's default hyperparameter search has no reason to expect.

Mechanically, GPry builds a GP surrogate of the log-posterior (default kernel:
anisotropic RBF, i.e. squared-exponential, `noise_level` controlling the
regressor's own noise floor), guards it with an SVM classifier that learns to
discard "uninteresting" (very-low or non-finite likelihood) regions, and
actively proposes new training points via an acquisition step — by default
**NORA** (Nested-sampling Optimized Rank Acquisition,
[arXiv:2305.19267](https://arxiv.org/abs/2305.19267)), which runs an internal
nested sampler (UltraNest, in our stack) *on the GP's current predicted
surface* to rank acquisition candidates. This is precisely the step our
`GPAcquisitionError: Acquisition returning no values after 2 re-tries` comes
from — UltraNest's live-point region construction failing on the GP's own
belief about the likelihood, not on the true likelihood itself.

### 1.2 Runner settings that matter (from package docs)

From `gpry.Runner`'s documented `options` and constructor arguments
(`https://gpry.readthedocs.io`, `en/balrog` branch — see §2 for why that
branch name is not a coincidence):

| Setting | Default | What it does |
|---|---|---|
| `n_initial` | `3 × n_dim` | truth evaluations before the active-learning loop starts |
| `max_initial` | `30 × n_dim^1.5` | budget to *find* `n_initial` finite points |
| `max_total` | `70 × n_dim^1.5` | hard stop on total evaluations |
| `ref_bounds` | full prior if unset | box the **initial** training points are drawn from (`initial_proposer="reference"`) — "should be sized relative to your posterior's expected width" |
| `surrogate.regressor.noise_level` | `1e-2` | GP's own noise floor; the docs say to raise it for "noisy... or numerical instabilities" |
| `surrogate.regressor.kernel` | anisotropic RBF | swappable to `"Matern"` "if smoothness assumptions fail" |
| `gp_acquisition` | NORA (jaxpe's choice) | swappable to `BatchOptimizer` (multi-start L-BFGS on the analytic acquisition gradient) |
| `convergence_criterion` | `CorrectCounter` + `GaussianKL` for NORA | can converge on a poorly-conditioned GP without ever resolving the true peak |

Two of these — `noise_level` and `kernel` — are **not currently exposed as
CLI flags in `jaxpe run-pe`** (only `ref_bounds_rel/abs`, `n_initial`,
`max_initial`, `max_total`, and `acquisition` are, as of this campaign's
work). Given §3 below, `noise_level` is the more promising untested lever.

## 2. GPry applied to gravitational waves: LISA (the only published case)

**"Accelerating LISA inference with Gaussian processes,"**
[arXiv:2503.21871](https://arxiv.org/abs/2503.21871), Phys. Rev. D (2025);
also presented at the CERN "AI for Gravitational Waves" workshop
([indico](https://indico.cern.ch/event/1640502/contributions/7028138/)).
This is what the `balrog` doc branch above is named for — Balrog is the LISA
data simulator this paper injects into.

### 2.1 What it tested

Three LISA source types, benchmarked against the nested sampler `nessai`:

| Source | Dim | SNR | Likelihood cost | Result vs `nessai` |
|---|---|---|---|---|
| DWD (double white dwarf) | 8 | 23.6 | ~10⁻³ s | D_JS ≲ 0.01 (199/200 realizations), but ~2× *slower* wall-clock — GPry's overhead dominates when the likelihood is this cheap |
| stBHB (stellar-mass BHB) | 10 | 16.8 | ~10⁻¹ s | D_JS ~0.2 (target was <0.05) — **underperformed**, see below |
| SMBHB (supermassive BHB) | 10 | **1944.8** | ~1 s | D_JS ≲0.05, ~10⁵→10³ evaluations, ~11 days→~3 hours |

### 2.2 Why this is directly relevant despite being LISA, not LVK/CE/ET

The SMBHB case is an SNR~1945 signal — two orders of magnitude louder than
LVK's typical few-tens SNR, but the *shape* of the problem (a GP trying to
resolve an extremely sharp peak relative to a much wider prior box) is exactly
our `phenomd_gpry_0` failure at SNR~73, just further along the same axis. The
paper is explicit that this required **non-default tuning**, not just running
GPry out of the box:

- **Larger GP noise scale (σₙ)** for stBHB/SMBHB specifically, "to prevent
  overfitting" — i.e. the exact `noise_level` knob flagged in §1.2 as
  untested in our own pipeline.
- **Trust regions** restricting exploration to stay near the accumulated
  training set (`infinities_classifier: {"trust_region": ...}` in GPry's
  surrogate spec) — a guardrail we have not enabled.
- **Raised SVM classifier cutoffs** to exclude low-valued regions more
  aggressively.
- **Pre-conditioning**: a Particle Swarm Optimization pre-search (stBHB) or
  Monte-Carlo sampling of the noiseless posterior to seed 35 initial points
  from a multivariate-Gaussian approximation (SMBHB) — i.e., *not* relying on
  GPry's generic `ref_bounds`-around-injected-truth seeding at all for the
  hardest cases.
- A **shrunken prior** for stBHB, explicitly "to eliminate the need to
  explore the parameter space" broadly.

Even with all of that, stBHB — the case with a genuinely non-Gaussian,
"heavy-tailed... large curving degeneracy" posterior — still underperformed
its target accuracy, and the authors' own conclusion is candid: GPry has
"insufficient characterization of the posterior mode" for strongly
non-Gaussian cases, with "ongoing development... addressing more robust
inference in highly non-Gaussian distributions." They flag "overhead costs
start making GPry an impractical approach for dimensionalities larger than a
few tens" as a separate, orthogonal limit.

### 2.3 Applicability to LVK / CE / ET

**None, directly.** LISA is a space-based, mHz-band detector; the paper's
three source classes (a slowly-evolving Galactic DWD, a long-duration
stellar-mass BHB inspiral years before merger, a supermassive BHB) have no
ground-based CBC analog, and nothing in the paper touches LVK data, an aLIGO
PSD, or ground-based detector response. **We could not find a single
published paper applying GPry to LVK, Cosmic Explorer, or Einstein Telescope
parameter estimation.** Our campaign's `esigma_gpry`/`phenomd_gpry` legs
appear to be a genuinely novel application context for this package.

What *does* transfer is the general engineering lesson: **GPry's out-of-the-box
settings are validated on cosmological likelihoods and on LISA sources whose
SNR tops out in the tens (DWD, stBHB) — the one LISA case in the same SNR
decade as our problem (SMBHB, SNR~1945) required hand-tuning `noise_level`,
trust regions, SVM cutoffs, and non-`ref_bounds` pre-conditioning to work at
all.** Our SNR~73 `phenomd_gpry_0` failure sits well within the regime this
paper had to work to support, which is a reasonable explanation for why our
five default/near-default fix attempts (see `campaign/plan.md` amendment 15:
tighter/wider `ref_bounds`, `BatchOptimizer` acquisition, `n_initial` up to
512) all failed identically — none of them touch `noise_level`, trust
regions, or pre-conditioning, which is exactly where the one directly
comparable published case put its effort.

## 3. RIFT: the closest thing to a production analog, already in LVK use

**Primary references:**
- Pankow et al. (2015), the `rapid_pe` precursor.
- Lange, O'Shaughnessy, Rizzo (2018), **"Rapid and accurate parameter
  inference for coalescing, precessing compact binaries,"**
  [arXiv:1805.10457](https://arxiv.org/abs/1805.10457) — the main RIFT paper.
- Wysocki, O'Shaughnessy et al. (2023), **"Improving performance for
  gravitational-wave parameter inference with an efficient and
  highly-parallelized algorithm,"**
  [arXiv:2210.07912](https://arxiv.org/abs/2210.07912), Phys. Rev. D 107,
  024040 — documents the actual settings used for LVK O3 analyses.
- Wofford et al. (2023), **"Low-latency parameter inference enabled by a
  Gaussian likelihood approximation for RIFT,"**
  [arXiv:2301.01337](https://arxiv.org/abs/2301.01337).
- Rizzo et al. (2025), **"Narrowing RIFT: Focused simulation-based-inference
  for interpreting exceptional GW sources,"**
  [arXiv:2505.11655](https://arxiv.org/abs/2505.11655).
- Documentation: [rift-documentation.readthedocs.io](https://rift-documentation.readthedocs.io/en/latest/).

### 3.1 Why this is the right comparison, not GPry's LISA paper

RIFT's architecture is structurally the closest published thing to what our
`marginalized_phase_distance`/`marginalized_intrinsic` + GPry pipeline is
doing, just with a different regressor-fitting loop instead of GPry's
integrated acquisition loop:

1. **ILE (Integrate Likelihood Extrinsic):** at a grid of candidate
   *intrinsic*-parameter points, Monte-Carlo-marginalize the likelihood over
   extrinsic parameters (distance, time, sky location, polarization) —
   exactly the role our `PhaseDistanceMarginalLikelihood` /
   `MarginalizedIntrinsicLikelihood` play.
2. **CIP (Construct Intrinsic Posterior):** fit a **Gaussian Process** (or
   random forest) to those marginalized-likelihood evaluations over the
   *intrinsic* parameters (chirp mass, symmetric mass ratio, spins) — exactly
   our GPry surrogate's job, over the same four-ish intrinsic dimensions.
3. Iterate: sample the fitted posterior, refine the grid, repeat.

So RIFT is answering, in a battle-tested production pipeline, almost exactly
our question: *how do you fit a GP surrogate to a marginalized CBC
intrinsic-parameter likelihood robustly, across the SNR range LVK actually
sees?*

### 3.2 GP settings and what actually broke

RIFT's kernel (Lange et al. 2018, their Eq. for `k(x,x')`) is the **same
family** as GPry's default:

```
k(x, x') = σ_o² exp(-(x - x') Q (x - x') / 2) + σ_n² δ_{x,x'}
```

— anisotropic squared-exponential plus a noise (nugget) term, hyperparameters
fit via scikit-learn. Validated robustly to d≤6 intrinsic dimensions,
demonstrated to d=8; computational cost (not accuracy) caps training-set size
at O(10⁴) points in 8D since GP fit/predict cost scales as D³/D².

**Documented failure modes, from the O3-operations paper
([arXiv:2210.07912](https://arxiv.org/abs/2210.07912)), map directly onto our
situation:**

- *"Gaussian-process fitting... became almost unusably slow when trained
  with many inputs"* — an operational-cost failure, addressed with
  subsampling workarounds, not what we're hitting.
- *"[Fits] prone to misidentify suitable length scales,"* yielding **"patchy
  and irregular posteriors"** — this is a GP-conditioning failure, and it's
  the low-mass (not high-SNR) end of their parameter space that triggers it.
  Their fix was **not** more points or a different acquisition strategy — it
  was **reparameterizing the input coordinates**: adopting "Rotated
  Inspiral-Phase (RIP)" coordinates and **pseudo-cylindrical coordinates for
  spins**, specifically to stop constrained and unconstrained spin components
  from mixing under the GP's (implicitly close-to-isotropic) length-scale
  assumptions. This is a lever we have not tried at all: our
  `marginalized_intrinsic`/`marginalized_phase_distance` likelihoods sample
  directly in `(chirp_mass, mass_ratio, spin1z, spin2z)`, a physically
  natural but not GP-kernel-natural coordinate system (wildly different
  physical scales and, at high SNR, strong curved correlations between
  chirp_mass and mass_ratio in particular).
- **SNR-dependent box sizing is explicit and load-bearing in RIFT**: their
  initial mass-grid half-width scales as `Δln(Mc) ∝ 1/ρ` (ρ = search SNR) —
  tighter boxes at higher SNR, by design, not an afterthought. This
  *validates* the instinct behind our own `ref_bounds` experiments — but note
  it's RIFT's **initial grid for the ILE stage**, analogous to GPry's
  `ref_bounds`/`initial_proposer`, not the acquisition step where *our*
  failure actually occurs. That's a useful negative result in itself: it's
  consistent with our tighter/wider `ref_bounds` tests both failing
  identically — `ref_bounds` genuinely isn't the layer our problem lives in.
- The Narrowing RIFT paper ([arXiv:2505.11655](https://arxiv.org/abs/2505.11655))
  states plainly that RIFT's default operating point was **"conservatively
  optimized for robust inference about poorly constrained
  observations"** — i.e. tuned for the *low*-information end of the LVK
  population. High-SNR, sharply-peaked, well-constrained events are
  explicitly the edge case default settings were never optimized for. This
  matches both the GPry/LISA experience (§2.2) and our own.

### 3.3 A lower-cost alternative worth knowing about: skip the GP entirely near the peak

Wofford et al.'s Gaussian-likelihood variant of RIFT
([arXiv:2301.01337](https://arxiv.org/abs/2301.01337)) sidesteps GP-fitting
cost (not robustness — the paper is explicit this is a speed optimization,
not a robustness fix) by fitting a **quadratic log-likelihood** (equivalently,
a Gaussian likelihood) directly, using only points within 20% of the peak
likelihood, then drawing/rejecting/reweighting samples from that closed-form
Gaussian. Runs in ~50s across 5 iterations on 400 GPU cores. It works well
near a well-behaved peak — the authors' own caveat is it "will behave most
poorly for the lowest-amplitude event candidates," i.e. it's a high-SNR-shaped
tool, the mirror image of GPry's apparent sweet spot. Not directly reusable
in our GPry-based pipeline, but a useful conceptual pointer: at high SNR,
where a full nonparametric surrogate becomes hard to fit robustly, a
much simpler local quadratic/Laplace approximation to the peak may be *more*
appropriate, not less.

### 3.4 GPU support: ILE only, not CIP

RIFT does support GPU execution, but only for the **ILE** stage (the
Monte-Carlo extrinsic-parameter marginalization — naturally
embarrassingly-parallel), not **CIP** (the GP-fitting stage, the one
actually analogous to our GPry use). Established in Wysocki, O'Shaughnessy,
Lange, Fang (2019), **"Accelerating parameter inference with graphics
processing units,"** [arXiv:1902.04934](https://arxiv.org/abs/1902.04934),
Phys. Rev. D 99, 084026; mechanism is `cupy` (NumPy/SciPy-compatible GPU
arrays), confirmed as a soft-optional dependency in the current mainline
codebase's `INSTALL.md`
([github.com/oshaughn/research-projects-RIT](https://github.com/oshaughn/research-projects-RIT),
mirrored from the LVK production repo `git.ligo.org/rapidpe-rift`): *"The
code uses cupy to access GPUs. If you don't have one, the code will still
work."* We found no GPU/GPyTorch/JAX port of RIFT's CIP (GP-regression) step
anywhere in current searches — it remains scikit-learn-based, same as
described in the 2018 paper. This mirrors our own situation: GPry's GP
fitting (via its scikit-learn backend) is likewise CPU-bound regardless of
JAX/GPU availability elsewhere in `jaxpe`'s stack, so RIFT's GPU story
doesn't offer a shortcut around the failure mode we're chasing.

### 3.5 RIFT and third-generation detectors

We found one demonstration of RIFT run against a **Cosmic Explorer**
sensitivity curve (single detector, LIGO Hanford site, from the "parameter
estimation of gravitational waves from hyperbolic black hole encounters"
line of work), reaching **SNR≈42** with a source placed at a few Mpc to
achieve that SNR at CE sensitivity — i.e., a genuine but limited, single-event
demonstration, not a systematic CE/ET population study, and still well below
the SNR~73–124 range our campaign's PhenomD legs land in. We found no
RIFT+Einstein Telescope study, and no systematic RIFT (or GPry) study of the
full CE/ET SNR distribution (hundreds, routinely, for a 3G network). Multiple
independent papers we surveyed note in passing that 3G detectors will push
essentially every accelerated-PE technique into unfamiliar high-SNR territory
— e.g. a completely unrelated metric-tiling-based rapid-PE paper
("Robust, Rapid, and Simple Gravitational-wave Parameter Estimation,"
[arXiv:2410.05190](https://arxiv.org/html/2410.05190)) reports its own
efficiency degrading at high SNR for structurally analogous reasons (fixed
tiling coarseness becomes visible as the likelihood sharpens). **High-SNR
robustness looks like an open problem across the GW rapid-PE literature
broadly, not a GPry-specific defect** — which is some reassurance that we're
not missing an obvious, already-published fix, but also means there isn't
one to borrow wholesale.

## 4. Other GP-adjacent techniques (noted, but out of scope)

Two other GW+GP lines of work turned up repeatedly in search results; neither
is the same class of tool as GPry/RIFT (an active-learning likelihood
surrogate that *replaces* likelihood evaluations during sampling), so they're
noted here only to avoid confusion:

- **Moore & Gair (2015), "Improving gravitational-wave parameter estimation
  using Gaussian process regression,"**
  [arXiv:1509.04066](https://arxiv.org/abs/1509.04066), Phys. Rev. D 93,
  064001. Uses a GP to marginalize over *waveform-model systematic error*
  (fitting GP to the difference between an accurate and an approximate
  waveform), folded in as an additional term in the likelihood itself. Not a
  likelihood surrogate/sampler accelerant — orthogonal to our problem.
- **Alvey et al., "Density estimation with Gaussian processes for
  gravitational-wave posteriors,"**
  [arXiv:2104.05357](https://arxiv.org/abs/2104.05357). Post-processes
  *already-obtained* posterior samples into a smooth density estimate (for
  resampling/hierarchical-population use), not an inference accelerant during
  sampling. Also orthogonal.

## 5. Takeaways for this campaign

1. **There is no published precedent for GPry (or any GP-surrogate active-learning
   sampler) at LVK/CE/ET ground-based CBC SNRs.** The one GW-specific GPry
   study is LISA-only; the one production ground-based GP-surrogate pipeline
   (RIFT) has real O3 mileage but its documented tuning knowledge is about
   *low*-SNR robustness and coordinate parameterization, not the high-SNR
   acquisition failure we hit, and its one CE demonstration (SNR≈42) is a
   single event, not a systematic high-SNR characterization.
2. **`ref_bounds` genuinely looks like the wrong layer to keep tuning** — RIFT's
   own SNR-scaled grid-sizing formula is at the *ILE/initial-grid* stage
   (GPry's analog: `ref_bounds`/`initial_proposer`), and we already tested
   both directions of that knob without effect, consistent with RIFT's
   experience that grid sizing and GP-conditioning are separate failure
   modes.
3. **Two concrete, untested levers directly motivated by this survey:**
   - GPry's `surrogate.regressor.noise_level` (not currently exposed as a
     `jaxpe run-pe` flag) — the one thing the LISA paper's high-SNR SMBHB
     case changed that we haven't tried.
   - A coordinate reparameterization for the intrinsic parameters, following
     RIFT's RIP/pseudo-cylindrical-spin precedent — plausible given
     `chirp_mass`/`mass_ratio`/`spin1z`/`spin2z` is a physically natural but
     not GP-kernel-natural basis, and our failure, like RIFT's length-scale
     misidentification, could be a genuine GP-conditioning problem the
     acquisition-layer error is just the visible symptom of.
   - Also worth trying, following the LISA paper directly: raising the SVM/
     `infinities_classifier` cutoff, and/or a `TrustRegion` classifier to keep
     acquisition from wandering into GP-extrapolation territory far from the
     training set.
4. **A Laplace/Gaussian-likelihood fallback near the peak (§3.3) is a
   reasonable design escape hatch** if GP-based approaches keep proving
   fragile at the high-SNR end of this campaign's injection set — conceptually
   simple, and explicitly validated in the regime (high SNR, well-behaved
   peak) where GPry is struggling.
