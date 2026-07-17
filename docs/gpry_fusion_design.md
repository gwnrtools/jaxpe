---
layout: default
title: Design — Fusing GPry into jaxpe
nav_order: 101
---

# Design Note: Fusing GPry into jaxpe for Expensive Non-JAX Waveform Models

**Status:** design accepted in discussion (2026-07); implementation not started.
**Companion notes:** [`under_construction.md`](under_construction.md) (ESIGMA gradients),
[`under_construction_esigma.md`](under_construction_esigma.md) (ISCO).

---

## How to read this note

*This note is written to be read by section: each section restates the terms it
needs, so you can jump straight to the one you care about without having read the
others. Some repetition between sections is deliberate. The shared vocabulary —
defined once here and reintroduced where it is used — is:*

- **PE (parameter estimation)** — Bayesian inference of a compact binary's
  parameters from gravitational-wave detector strain data.
- **jaxpe** — this library: JAX-based, GPU-capable, gradient-based GW PE.
- **GPry** — an external, published active-learning package (local checkout at
  `~/src/GPry`) that builds a **Gaussian-process (GP) surrogate** of a likelihood
  and uses an acquisition rule (its "NORA" strategy) to decide where to evaluate
  the likelihood next. Because it spends evaluations only where they matter, it
  converges in far fewer likelihood calls than MCMC or nested sampling. We
  **interface** with GPry as a dependency; we do not reimplement it (decision D4).
- **Case (1) vs case (2) waveform models** — **(1)** models implemented in JAX
  (`IMRPhenomD`, `ESIGMA`, `NRSur7dq4`): cheap (~milliseconds), differentiable,
  GPU-batchable, and sampled directly. **(2)** external non-JAX models
  (**TEOBResumS**, **SEOBNRv6EHM**): **0.5–10 minutes per call**, non-differentiable
  black boxes, but carrying leading-edge physics we want. **Case (2) is the entire
  reason this note exists.**
- **Intrinsic (θ_int) vs extrinsic (θ_ext) parameters** — **intrinsic**: what the
  expensive waveform actually depends on (chirp mass 𝓜, mass ratio q, spins χ,
  eccentricity e, mean anomaly ℓ). **extrinsic**: the comparatively cheap geometry
  (luminosity distance D_L, sky position α/δ, inclination ι, polarization ψ,
  coalescence phase φ_c and time t_c) — 7 parameters jaxpe can handle analytically.
- **Marginalized intrinsic likelihood 𝓛(θ_int)** — the likelihood with *all seven*
  extrinsic parameters integrated out, leaving a scalar function of θ_int only.
  This is the function the GP surrogate learns; computing it cheaply and correctly
  (§3) is the technical core of the whole design.
- **Importance sampling (IS) / effective sample size (ESS)** — a reweighting scheme
  that corrects an estimate to be asymptotically exact, together with its built-in
  quality gauge ESS = (Σw)²/Σw². Low ESS means the estimate is unreliable — so
  failure is **visible**, never silent.
- **Multifidelity mean** — giving the GP a head start by using a cheap case-(1)
  model's 𝓛(θ_int) as its prior mean, so the GP only has to learn the (hopefully
  small, smooth) residual δ = 𝓛_expensive − 𝓛_cheap (§4).
- **Route A vs Route B** — the two end-to-end PE methods now cross-validated against
  each other on case-(1) models: **Route A** samples the full likelihood directly
  with jaxpe's gradient sampler; **Route B** is the GPry surrogate over 𝓛(θ_int).

---

## 1. Problem statement

*In one line: some waveform models are so slow (minutes per call) that ordinary
sampling — which needs millions of calls — is infeasible; for those models we learn
a cheap surrogate of the likelihood instead of calling the model directly millions
of times. This section states the two model classes jaxpe must serve and quantifies
what "infeasible" means.*

jaxpe must support PE with two classes of waveform models:

1. **Case (1) — JAX-implemented models** (frequency- and time-domain): `IMRPhenomD` (ripple),
   `ESIGMA`, `NRSur7dq4` (JaxNRSur) — cheap (ms-scale), differentiable, GPU-batched. Sampled
   directly with jaxpe's gradient/NF samplers ([`jaxpe/sampler/global_local.py`](../jaxpe/sampler/global_local.py),
   [`jaxpe/kernels/`](../jaxpe/kernels/)).
2. **Case (2) — non-JAX models**, mostly time-domain, carrying leading-edge physics:
   **TEOBResumS** and **SEOBNRv6EHM**. Per-call cost **0.5–10 minutes**. Non-differentiable
   black boxes. **Precession is in scope from day one.**

Goal: one PE system where case (2) is handled by a **GPry-driven active-learning surrogate of
the likelihood**, sharing jaxpe's data, PSD, detector-projection, and sampling infrastructure,
with **exact-posterior guarantees** and minimal new code.

### 1.1 Why not brute force, quantitatively

*Context: "brute force" = just running a standard sampler (nested sampling or
parallel MCMC) that calls the expensive waveform model directly for every
likelihood evaluation. The arithmetic below is why that is off the table for
case-(2) models except at the very cheap end — and why it still serves as our
gold-standard validation baseline (§7).*

A nested-sampling / parallel-MCMC run needs ~10⁶–10⁷ likelihood calls. At 0.5–10 min/call:

| per-call | 10⁶ calls, serial | on 500 cores |
|---:|---:|---:|
| 0.5 min | ~1 year | ~17 h — borderline feasible (parallel-bilby regime) |
| 5 min | ~10 years | ~1 week |
| 10 min | ~19 years | ~2 weeks |

So at the cheap end brute force is *possible* on a large cluster (and is our validation
baseline, §7); the surrogate's value is a **100–1000× reduction in expensive calls**, turning
per-event PE into hours on a workstation + small pool of waveform workers, and making
multi-event studies affordable.

### 1.2 Why not amortized NPE (DINGO-style)

*Context: amortized neural posterior estimation (NPE) — e.g. DINGO — trains a neural
network once so that inference on any future event is near-instant. The catch for
case-(2) models is that the one-time training itself needs the same impossible
number of expensive waveform evaluations, just paid up front.*

Needs 10⁶–10⁷ *training* waveforms up front — the same infeasible budget, paid before the first
event. Rejected for case (2). (It remains attractive for case (1), orthogonal to this note.)

### 1.3 Prior art: this architecture is RIFT-shaped

*Context: we are not proposing a new paradigm. RIFT is an established GW PE method
that already marginalizes the likelihood over the extrinsic parameters at each
intrinsic point and interpolates the result over intrinsic space. This section
names what we reuse from that skeleton and the four things we do differently — the
four differences are the actual novelty surface of this work.*

RIFT (Lange, O'Shaughnessy et al.) computes an extrinsic-**marginalized** likelihood per
intrinsic point and interpolates it (GP/rbf) over intrinsic space with iterative refinement.
Our plan is the same skeleton with four substitutions, which are the novelty surface:

- principled, uncertainty-driven **active learning** (GPry's NORA acquisition) instead of
  ad-hoc grid refinement;
- a **multifidelity mean function** from a differentiable case-(1) jaxpe model (§4);
- **GPU, JAX-differentiable extrinsic marginalization** (§3);
- **exactness via importance-sampling reweighting** with a built-in ESS diagnostic (§5).

*Due diligence (to-verify):* re-check RIFT's current interpolator and refinement scheme before
publishing comparisons; run one RIFT baseline on a golden event if practical.

---

## 2. The four design decisions (D1–D4), as debated

*Context: four decisions fix the shape of the whole design, and everything later in
the note follows from them. **D1** — which function the Gaussian-process (GP)
surrogate should represent. **D2** — whether to surrogate the likelihood or the
waveform. **D3** — how the surrogate stays exact and gives back the extrinsic
parameters. **D4** — whether to interface with the existing GPry package or rewrite
its loop in JAX. Each is presented with the options that were weighed and the reason
for the choice.*

### D1. What function does the GP learn?

*Context: recall θ_int = the intrinsic parameters the expensive waveform depends on
(masses, spins, eccentricity, mean anomaly); θ_ext = the cheap extrinsic geometry
(distance, sky, inclination, polarization, coalescence phase and time). The question
here is which function of these the GP should model.*

Options considered:

- **Full `ℒ(θ_int, θ_ext)` at 15–17D** — over GPry's own "< 20 as a rule of thumb" ceiling,
  and GP sample complexity grows steeply with dimension. **Rejected.**
- **Fix extrinsic parameters at max-likelihood** — biased posteriors. **Rejected.**
- **GP over the extrinsic-marginalized intrinsic likelihood** ✅

$$ \mathcal{L}(\theta_{\rm int}) \;=\; \log \int L(d \mid \theta_{\rm int}, \theta_{\rm ext})\,
   \pi(\theta_{\rm ext})\, d\theta_{\rm ext} $$

with $\theta_{\rm ext} = (D_L, \alpha, \delta, \iota, \psi, \phi_c, t_c)$ (7D) handled
analytically / by FFT / by low-D quadrature in jaxpe (§3), and $\theta_{\rm int}$ the GP's
domain:

| target model | assumed physics | θ_int | dim |
|---|---|---|---:|
| SEOBNRv6EHM | eccentric, aligned-spin *(to-verify: v5EHM was aligned-spin eccentric; confirm v6EHM spin content)* | $(\mathcal{M}, q, \chi_{1z}, \chi_{2z}, e, \ell)$ | 6 |
| TEOBResumS (precessing, quasi-circular) | precessing | $(\mathcal{M}, q, \vec\chi_1, \vec\chi_2)$ | 8 |
| TEOBResumS (eccentric + precessing, if used) | both | $(\mathcal{M}, q, \vec\chi_1, \vec\chi_2, e, \ell)$ | 10 |

All within GPry's ceiling; 10D is the stress case.

**A subtle but important bonus:** marginalizing over $(t_c, \phi_c)$ quotients out time/phase
**alignment conventions** between models. This both smooths $\mathcal{L}(\theta_{\rm int})$
itself and — critically for D4/§4 — smooths the *discrepancy* between the expensive model and
the cheap multifidelity mean, since overall alignment offsets between the two models no longer
contribute.

### D2. Surrogate the likelihood, or the waveform (ROM)?

*Context: there are two things one could approximate to avoid calling the expensive
model. A **reduced-order model (ROM)** approximates the waveform itself and is
reusable across all future events; a **likelihood surrogate** approximates only
*this* event's likelihood. This decides between them — and keeps a cheap option open
by caching waveform modes regardless.*

- A **waveform ROM** amortizes across events but demands a prior-wide offline training
  campaign against a still-evolving leading-edge model — the wrong trade for per-event use of
  frontier physics.
- A **likelihood surrogate** is per-event, data-adaptive: the acquisition spends expensive
  calls only where *this event's* posterior has support.
- **Decision: likelihood surrogate, but cache all modes.** Every expensive call stores
  $h_{\ell m}(t;\theta_{\rm int})$ (~MB each × few×10³ calls — trivial). The cache (a) enables
  extrinsic-conditional sampling and mode-level diagnostics, and (b) doubles as free,
  posterior-concentrated ROM training data if we later want to amortize. Optionality at zero
  cost.

### D3. Exactness and extrinsic recovery — one mechanism for both

*Context: two problems share one solution here. A GP surrogate is only an
*approximation* of the likelihood, and by construction 𝓛(θ_int) has already
integrated the extrinsic parameters away, so the surrogate alone yields neither an
exact posterior nor any extrinsic estimates. Both are recovered in a single
post-processing pass: spend a fixed budget of genuine expensive-model calls at
surrogate-drawn points, use them as importance-sampling (IS) weights to restore
exactness, and reuse the same calls' waveform modes to sample the extrinsics. The IS
effective sample size (ESS) doubles as the convergence diagnostic.*

The GP posterior is an *estimate*; we restore exactness and recover $\theta_{\rm ext}$ with a
single post-processing budget of exact calls:

1. Sample $\{\theta_{\rm int}^{(k)}\}$ from the surrogate posterior (GPry's final MC step /
   BlackJAX interface).
2. Spend a fixed budget (~500–2000) of **exact** expensive calls at (a thinned subset of)
   those draws. Each call yields:
   - the true $\mathcal{L}$ → **importance weights**
     $w_k = \exp[\mathcal{L}_{\rm true}(\theta^{(k)}) - \mathcal{L}_{\rm GP}(\theta^{(k)})]$,
     restoring asymptotic exactness; the effective sample size
     ${\rm ESS} = (\sum w)^2/\sum w^2$ is the **built-in convergence diagnostic** — low ESS
     means "acquire more and repeat", so failure is *visible*, never silent;
   - the modes $h_{\ell m}$ → jaxpe samples the conditional
     $p(\theta_{\rm ext} \mid \theta_{\rm int}^{(k)}, d)$ with its gradient sampler (7D,
     differentiable given fixed modes, cheap). Joint posterior assembled hierarchically.

*Rejected alternative:* delayed-acceptance MCMC — puts the expensive model back in the
sampling loop; IS-reweighting keeps it embarrassingly parallel and post-hoc.

### D4. Interface with GPry, or rewrite it in JAX inside jaxpe?

*Context: GPry is an existing ~16k-line Python package. The tempting alternative is
to reimplement its GP + acquisition loop natively in JAX for speed and integration.
This decision weighs where wall-clock actually goes (the expensive waveform, not the
GP) and the risk of re-validating a from-scratch loop, and comes down on interfacing
— with one measured escape hatch.*

**Decision: interface (pin GPry as an optional dependency behind a thin engine protocol); do
not rewrite.** The debate, condensed:

- **Where wall-clock goes:** the expensive waveform dominates by construction; the GP
  fit/acquisition is subdominant. JAX-accelerating the subdominant term is speculative
  optimization.
- **The outer loop is host-side Python regardless:** acquire → evaluate black box → refit
  cannot be jitted; a rewrite accelerates only inner linear algebra.
- **float64:** GP Cholesky needs it; consumer GPUs have poor fp64 throughput — a "GPU GP" may
  be *slower* than sklearn/LAPACK on CPU at N ≲ few×10³ training points.
- **Correctness > performance:** GPry's 16k lines are mostly *heuristics* (SVM infinities
  classifier, trust regions, dimension-scaled acquisition, convergence criteria,
  preprocessing, checkpointing) encoding failure modes already found and fixed. Surrogate-PE
  failures are silent (wrong posteriors, not crashes); re-validating a from-scratch loop is
  strictly riskier than pinning a published one.
- **Insurance:** a minimal `SurrogateEngine` protocol in jaxpe so a JAX GP backend (tinygp)
  *could* replace a component later. **Profiling checkpoint:** after the Phase-1 pilot,
  measure the wall-clock split (waveform / GP fit / NORA exploration / final MC); revisit
  porting *a single component* only if the non-waveform share exceeds ~30%.

### Verified GPry integration seams (code read, 2026-07; GPry checkout at `~/src/GPry`)

*Context: the "interface, don't rewrite" decision (D4) is only safe if the exact
entry points we need actually exist and behave as assumed. These were verified by
reading GPry's source directly. The final row is the single place where a plain
interface is insufficient — the multifidelity mean (§4) requires subclassing GPry's
GP regressor, because its preprocessing hook is y-only and cannot carry an
X-dependent mean.*

| seam | location | note |
|---|---|---|
| Plain-callable entry | `gpry/run.py::Runner(loglike=<callable>, bounds=<dict>)` | Cobaya fully optional |
| True-model wrapper | `gpry/truth.py::Truth` | alternative injection point |
| GP regressor | `gpry/gpr.py::GaussianProcessRegressor(sk_GPR)` | supports `predict(..., return_mean_grad=True, return_std_grad=True)` — analytic surrogate gradients already exist |
| Batch acquisition | `gpry/gp_acquisition.py::NORA` (also `BatchOptimizer`) | batch size ↔ number of parallel waveform workers |
| Parallel truth evals | `gpry/mpi.py` | MPI batch evaluation built in |
| Surrogate exploration | `gpry/ns_interfaces.py` — PolyChord/nessai/UltraNest/**BlackJAX** | JAX-adjacent already |
| Robustness | `gpry/infinities_classifier.py` (SVM + trust region), `gpry/convergence.py` (CorrectCounter, GaussianKL) | do not reimplement |
| **Multifidelity seam** | `gpry/preprocessing.py::PipelineY.transform(y)` is **y-only** (verified) — an X-dependent mean cannot go there | ⇒ subclass `GaussianProcessRegressor` with explicit prior mean $m(X)$ (§4); ~100 lines |

**Trap (verified reasoning):** do *not* feed the discrepancy $\delta$ directly as GPry's
`loglike` — the acquisition and SVM target high-*posterior* regions of whatever the GP models,
and high-$\delta$ ≠ high-posterior. The mean-function subclass is the correct route.

---

## 3. The marginalized intrinsic likelihood (Phase 0 core)

*Context: this is the technical heart of the whole design. Everything upstream
decided that the GP should learn 𝓛(θ_int), the likelihood with all seven extrinsic
parameters (distance D_L, sky α/δ, inclination ι, polarization ψ, coalescence phase
φ_c and time t_c) integrated out. This section is how that integral is actually
computed — cheaply, in float64, reusing jaxpe's existing detector-projection, PSD,
and Whittle-inner-product code. Each extrinsic parameter is integrated by the method
its structure rewards (closed form, FFT, low-D quadrature, or importance sampling);
two of them — φ_c and the sky angles — turned out to need methods stronger than
naive quadrature, a Phase-0 discovery flagged inline below.*

New module `jaxpe/gw/marginalized.py`, sibling of
[`jaxpe/gw/likelihood.py`](../jaxpe/gw/likelihood.py) and reusing its detector projection
(`project_to_detector`), PSD, and Whittle inner products:

```
(h_lm modes on model time grid, event data, PSDs)  →  ℒ(θ_int)   [scalar]
```

Structure of the marginalization (all in float64):

- **modes → detector**: condition (taper/resample to uniform Δt — reuse
  [`jaxpe/gw/conditioning.py`](../jaxpe/gw/conditioning.py)), FFT once per mode; polarizations
  from $\sum_{\ell m} h_{\ell m}\, {}_{-2}Y_{\ell m}(\iota, \phi_c)$.
- **$D_L$**: Gaussian in $u = d_{\rm ref}/D_L$ given $\langle d|h\rangle, \langle h|h\rangle$;
  power-law prior via **adaptive Gauss-Legendre on a peak-tracking window** (fixed nodes
  under-resolve the width-$1/\sigma$ peak at realistic SNR; validated against the erf closed
  form of the flat-in-$u$ prior).
- **$t_c$**: FFT of the overlap integrand — all integer-sample shifts at once; uniform prior
  over a node window; GMST frozen at the window center (µrad-exact).
- **$\phi_c$**: *(amended in Phase 0)* naive quadrature is **wrong** here — the integrand
  $e^{\ln L(\phi_c)}$ carries harmonics up to $\sim \max|m|\,{\rm SNR}^2/2$, so ~32–64 nodes
  fail already at SNR 10 (measured: 0.04 shift between 16 and 32 nodes). Instead the strain's
  **exact azimuthal decomposition** $h_{\rm det} = \sum_M e^{iM\phi_c} G_M(f)$ (one FFT per
  distinct $m$, not per node) makes dense $O({\rm SNR}^2)$-node grids essentially free.
- **$(\alpha, \delta, \psi, \iota)$**: *(amended in Phase 0)* plain QMC/product grids are
  **hopeless**, not merely slow: at network SNR 11 the measured ESS was 1.5/8192 (mass
  concentrated in ~10⁻⁴ of the extrinsic space) with seed-to-seed spreads of ~7 in log.
  Adopted: **defensive adaptive importance sampling** (pilot Sobol scan → wrapped/reflected
  Gaussian-KDE proposal around the discovered mass, mixed with a uniform defensive floor →
  IS estimate with ESS as the built-in convergence diagnostic) — RIFT-style in spirit,
  host-side by construction, each batch a compiled `lax.map`.

Cost per intrinsic point: measured ~7 ms per extrinsic node (CPU, reduced inner settings) ×
a few×10³ adaptive-IS nodes; production inner settings and the GPU figure are a Phase-1
profiling item — if the extrinsic layer ever rivals a 0.5–10 min waveform call, the
RIFT-style per-detector time-series factorization is the upgrade path. The fixed-extrinsic
likelihood and the 3D marginal are **differentiable given modes** (used by the
extrinsic-conditional sampler in D3 and the multifidelity mean in §4); the adaptive-IS full
marginal is deliberately not differentiable end-to-end (GPry needs no gradients).

**Precession notes:** modes are inertial-frame at a fixed `f_ref`; spins in $\theta_{\rm int}$
are defined at `f_ref`. The $\iota$-dependence enters only through ${}_{-2}Y_{\ell m}$ — exact
under quadrature. Convention alignment of `f_ref` between TEOBResumS/SEOBNRv6EHM and any
cheap mean model must be checked explicitly (§4).

---

## 4. Multifidelity mean (Phase 2)

*Context: "multifidelity" here means giving the GP a running start. Instead of
learning 𝓛(θ_int) from zero, the GP's prior mean is set to a cheap case-(1) model's
own marginalized likelihood 𝓛_cheap(θ_int), so the GP only has to learn the residual
δ = 𝓛_expensive − 𝓛_cheap. If δ is small and smooth, this cuts the number of
expensive calls sharply. The honest complication, stated below, is that no current
cheap (case-1) model covers eccentricity **and** precession at once — so
multifidelity is a per-target opt-in, gated by an explicit check that δ is actually
easier to learn than 𝓛_expensive itself.*

Subclass `gpry.gpr.GaussianProcessRegressor` with prior mean
$m(\theta_{\rm int}) = \mathcal{L}_{\rm cheap}(\theta_{\rm int})$ (a case-(1) jaxpe model run
through the *same* §3 marginalization): fit GP on residuals $y - m(X)$, add $m(X)$ back in
`predict`, add $\nabla m$ (exact, from JAX) into `return_mean_grad`. Downstream acquisition /
SVM / convergence see the composite prediction unchanged.

**The mean-model gap — stated honestly.** No current case-(1) model covers eccentricity *and*
precession simultaneously:

| expensive target | best available mean | what the GP must absorb in δ |
|---|---|---|
| SEOBNRv6EHM (ecc, aligned) | **ESIGMA** (ecc, aligned) — natural pairing, 6D | EOB-vs-PN dephasing only — expect δ small/smooth |
| TEOBResumS precessing (quasi-circ) | **NRSur7dq4** (precessing) or IMRPhenomD (aligned) | model differences / (+ precession if PhenomD) |
| TEOBResumS ecc+precessing | ESIGMA *or* NRSur7dq4 — neither covers both | the physics the mean lacks |

Rule: **multifidelity is per-pair opt-in.** Before adopting a mean for a given target, evaluate
$\delta = \mathcal{L}_{\rm exp} - \mathcal{L}_{\rm cheap}$ at ~20 scattered points; adopt only
if δ is materially smaller/smoother than $\mathcal{L}_{\rm exp}$ itself. Otherwise run
single-fidelity (zero mean) — the architecture supports both per run. Recall §D1: the
$(t_c,\phi_c)$ marginalization already removes alignment-convention roughness from δ.

---

## 5. Exactness, extrinsic recovery, and diagnostics

*Context: this turns decision D3 into concrete deliverables. Recall the two jobs it
does at once — restore exactness (the GP is only an approximation) and recover the
extrinsic parameters (𝓛(θ_int) integrated them away) — both from one budget of exact
expensive-model calls via importance-sampling (IS) reweighting. This section also
fixes what a production run's report must always contain, so that reliability is
auditable rather than assumed.*

As decided in D3. Deliverables: an IS-reweighting post-processor with ESS reporting, and the
hierarchical extrinsic-conditional sampler (jaxpe gradient kernel over 7D with cached modes).
Acceptance rule for a production run: report is incomplete without (ESS/N, number of
acquisition rounds, δ-diagnostics if multifidelity, and the convergence-criterion trace).

---

## 6. Architecture / new code layout

*Context: where the new code lives, and the one boundary that must never be crossed.
The expensive case-(2) models are non-JAX black boxes; the discipline below keeps
them strictly in plain Python (with MPI for parallel evaluation) and keeps JAX for
everything differentiable (marginalization, the mean model, extrinsic sampling,
diagnostics). The black box must never enter a `jit`/`grad` trace.*

```
jaxpe/
  gw/
    marginalized.py          # §3: modes + data → ℒ(θ_int); differentiable given modes
    external_models/         # case-(2) wrappers — plain Python, NEVER inside jit/grad
      base.py                #   ExternalModeModel protocol: params → {(l,m): h_lm(t)}, t
      teobresums.py
      seobnrv6ehm.py
  surrogate/
    engine.py                # SurrogateEngine protocol: fit / predict / acquire / sample
    gpry_backend.py          # Runner wiring, bounds/prior translation, checkpointing
    multifidelity.py         # mean-function GPR subclass (§4)
    reweight.py              # IS weights, ESS, hierarchical extrinsic recovery (§5)
    cache.py                 # mode cache (θ_int-keyed, HDF5)
bin/
  run_gpry_pe.py             # driver: config → Runner loop → reweight → posterior
```

- Boundary discipline: expensive calls live in plain Python/MPI (GPry's `mpi.py`); JAX owns
  marginalization, mean model, extrinsic sampling, diagnostics. The black box **never** enters
  a `jit`/`grad` trace (no `pure_callback`).
- `pyproject.toml`: new optional group `surrogate = ["gpry"]` (pin the version; TEOBResumS /
  SEOBNRv6EHM bindings documented per-model, not hard dependencies).
- `ExternalModeModel` is deliberately **not** a
  [`WaveformModel`](../jaxpe/gw/cbc_models/base.py) subclass — that ABC promises JAX
  traceability; these wrappers must not.

---

## 7. Validation ladder (cheap → expensive)

*Context: how we convince ourselves the system is correct, cheapest test first.
Because surrogate-PE failures are silent (they produce a wrong posterior, not a
crash), validation is not optional. The decisive rung is step 2: run the *complete*
GPry loop on a cheap case-(1) model that we can *also* sample directly, so exact
ground truth is available at zero expensive-model cost. The final rung is the one
brute-force run on the real model (the very thing §1.1 said we cannot afford
routinely), done once per model family and kept as the reference.*

1. **Marginalization correctness** (Phase 0): $\mathcal{L}(\theta_{\rm int})$ vs direct
   high-resolution numerical integration over θ_ext on a case-(1) model; and joint-posterior
   consistency vs jaxpe direct gradient sampling. Exact ground truth, minutes.
2. **Pseudo-black-box end-to-end** (Phase 1): run the *full* GPry loop treating a case-(1)
   model (PhenomD, then ESIGMA) as an opaque callable; compare surrogate posterior vs direct
   sampling truth (PP-style + KL + credible-interval coverage). Exact ground truth, hours.
   This is the decisive correctness test of the whole fusion, at zero expensive-model cost.
3. **ESS on every production run** (built-in, §5).
4. **One golden-event brute-force cross-check per model family** (parallel nested sampling on
   the true model, ~1 week on ~500 cores at 5 min/call, per §1.1) — run once, kept as the
   reference.

---

## 8. Budget estimate (to be measured, not trusted)

*Context: order-of-magnitude wall-clock estimates for the expensive part of a run,
under stated assumptions. The title is the point — these are feasibility figures to
be replaced by Phase-1 measurements, not commitments. "NORA batch = worker count"
means GPry proposes one batch of points per round and we evaluate them in parallel
across that many waveform workers.*

Assumptions: NORA batch = worker count; single-fidelity 10D needs ~1–3k acquisitions
(GPry's published low-D counts are hundreds; 10D is extrapolation — *measure in Phase 1*);
multifidelity target ~3–10× fewer for the SEOBNRv6EHM/ESIGMA pairing.

| scenario | calls | per-call | workers | expensive wall-clock |
|---|---:|---:|---:|---:|
| 6D ecc-aligned, multifidelity | ~500 | 2 min | 32 | ~30 min |
| 8D precessing, single-fid | ~2000 | 5 min | 64 | ~2.6 h |
| 10D ecc+precessing, single-fid | ~5000 | 10 min | 64 | ~13 h |
| + IS reweight budget | 500–2000 | — | 64 | ≤ 5 h |

GP-side overhead (fit + NORA exploration) at N ≲ 5k training points: minutes–tens of minutes
per round on CPU — the §D4 profiling checkpoint guards this assumption.

---

## 9. Work plan

*Context: the phased implementation plan. Phases are ordered by dependency and each
ends in an acceptance gate (G0–G3) that must pass before the next begins.
Interleaved with the plan are dated **"outcome" blocks** recording what actually
happened when each phase was built, including measured findings and course
corrections — this is the project's audit trail and is kept verbatim. If you only
want the current status, read the outcome blocks; if you want the intended path,
read the task tables and gates.*

Phases are strictly ordered by dependency; each has an acceptance gate. Estimates are
working-days of focused effort and are guesses, not commitments.

### Phase 0 — Marginalized intrinsic likelihood (critical path; ~5–8 d)

| # | task | deliverable / test |
|---|---|---|
| 0.1 | `ExternalModeModel` protocol + mode cache (`surrogate/cache.py`) | round-trip test; θ-keyed HDF5 |
| 0.2 | `gw/marginalized.py`: modes→detector conditioning path reusing `conditioning.py` + `likelihood.py` internals | vs `TDNetworkLikelihood` at fixed θ_ext (must agree to ~1e-10 before any marginalization) |
| 0.3 | $D_L$, $t_c$ (FFT), $\phi_c$ (quadrature) marginalization | vs brute-force numerical integration on PhenomD, rel. err < 1e-4 |
| 0.4 | $(\alpha,\delta,\psi,\iota)$ quadrature/QMC layer, `vmap`-ed; convergence study of node counts | grid-refinement convergence plot; per-point cost benchmark on GPU |
| 0.5 | differentiability-given-modes + extrinsic-conditional sampler prototype | grad vs FD; posterior $p(\theta_{\rm ext}\vert\theta_{\rm int},d)$ vs direct sampling |

**Gate G0:** marginalized ℒ validated against direct integration and direct sampling on
case-(1) models; per-point cost ≤ 1 s on GPU.

**Phase 0 outcome (2026-07, done).** Landed: `jaxpe/gw/external_models/` (protocol +
`ModesData` + npz `ModeCache`), `jaxpe/gw/marginalized.py`
(`ModesNetworkLikelihood` subclassing `TDNetworkLikelihood` — Whittle sum/projection
inherited, parity structural), 14 tests in `tests/test_marginalized.py`. Validation:
fixed-extrinsic parity with the TD path at float64 round-off (incl. exact
integer-sample $t_c$ shifts); $(\phi_c, t_c, D_L)$ marginal against an independent
semi-brute-force reconstruction (per-node $(z, \sigma^2)$ solved from three
fixed-parameter lnL evaluations + erf distance integral) to <1e-6; adaptive-IS full
marginal self-consistent across disjoint-randomness runs within MC error, ESS > 100;
MALA extrinsic-conditional prototype recovers injected extrinsics from
adaptive-IS-style initialization. **Two design amendments** (see §3): exact azimuthal
decomposition for $\phi_c$; defensive adaptive IS for the sky layer. **Open for
Phase 1 profiling:** production-settings per-point cost on GPU (the 1 s gate was met
only at reduced inner settings on CPU), and the t_c node spacing (one sample; refine
by zero-padded upsampling if posterior structure demands).

### Phase 1 — Single-fidelity GPry loop, pseudo-black-box (~5–8 d)

| # | task | deliverable / test |
|---|---|---|
| 1.1 | `surrogate/engine.py` protocol (only the four methods Phase 1 calls) + `gpry_backend.py` (`Runner` wiring, bounds/prior translation, checkpoint/resume) | GPry `introductory_example.py` reproduced through our wrapper |
| 1.2 | `bin/run_gpry_pe.py` driver; pin GPry version; `surrogate` extra in `pyproject.toml` | end-to-end on a 2D toy |
| 1.3 | **Pseudo-black-box PhenomD** (aligned, ~4–6D intrinsic): full loop vs direct-sampling truth | PP/coverage + KL acceptance thresholds |
| 1.4 | **Pseudo-black-box ESIGMA** (6D, eccentric): same | same; also records acquisition-count datum for §8 |
| 1.5 | Profiling harness: wall-clock split waveform / GP fit / NORA / final MC | the §D4 checkpoint report |

**Gate G1:** surrogate posterior statistically indistinguishable from truth on two case-(1)
models; profiling report produced; **decision point on any JAX component port**.

**Phase 1 outcome (2026-07-13, done except 1.4).** Landed: `jaxpe/surrogate/`
(`SurrogateEngine` protocol + `SurrogateSamples`; `GPryEngine` wrapping `gpry.Runner`
— GPry 4.0 pinned via the `surrogate` extra, installed editable from `~/src/GPry`),
`MarginalizedIntrinsicLikelihood` + `ModesNetworkLikelihood.marginal_eval_fn` /
`modes_fd_arrays` (mode arrays as *traced arguments* to one per-event jit-compiled
evaluator — a fresh instance per intrinsic point would re-trace at ~1.6 s/eval, which
would have dominated every surrogate run), `bin/run_gpry_pe.py` (driver + profiling
harness), `tests/test_surrogate.py`. **Gate G1 (CI form) passed:** a 2D-intrinsic
pseudo-black-box (synthetic chirp modes -> 3D-marginal lnL) active-learned by GPry
matches the dense two-stage-grid posterior of the same callable in mean (< 1 cell),
width (< 30%), and lnL shape near the peak (< 0.5), converging in ~80-160 truth
evaluations. Demo-driver profile (2D, ms-scale waveform): acquisition (NORA/UltraNest)
dominates at ~83% of 271 s; **extrapolated to a 2 min/call production waveform the
non-truth share is ~3% — far below the 30% D4 port trigger** (measured, not assumed;
re-check at 6-10D where NORA cost grows). *(The acquisition-dominance held up on the real
FD Route B — measured ~77% at §9; but the 2 min/call premise did **not**: real aligned-spin
EOB is 13–800 ms/call across stellar-mass BBH, so that regime is GPry-dominated, not
waveform-dominated — see the §9 EOB-cost block and Phase 2.5.)* Notes: jaxpe requires `jax_enable_x64`
(float32 GPS times silently NaN — the driver sets it; scripts must too);
GPry's `logp_truth` is single-point (wrapped with a loop); strict-editable installs
need `pip install -e . --no-deps` re-run when a new subpackage is added.
**Remaining for G1-full:** task 1.4 (ESIGMA as pseudo-black-box — needs a mode-level
adapter since `ESIGMAInspiral` exposes only polarizations) and a full-marginal
(adaptive-IS) end-to-end run vs direct sampling; then the G2 multifidelity work.

**Task 1.4 + full-marginal outcome (2026-07-13, done).**
`ESIGMAInspiral.mode_dict` now exposes tapered modes at 1 Mpc (pure refactor of the
heavy math into `_hlms_window`/`_hlms_with_adjoint`; ESIGMA regression suite 4/4
unchanged, including gradient-vs-FD). The full-marginal (adaptive-IS) end-to-end run
converged in 58 truth evaluations with f0/span recovered within 1σ, consistent with
the fixed-sky run — at ~12 s per ℒ(θᵢ) call the non-truth share was 0.16
(posterior overlay: [`examples/output/gpry_full_vs_fixed_corner.png`](../examples/output/gpry_full_vs_fixed_corner.png)).
The ESIGMA pseudo-black-box test passes with the D3 IS-reweighting exactness check
(ESS/N ≈ 0.25 on its multi-lobed surface; catastrophic-failure line at 0.05).

*The two findings below emerged from the Phase-1 pseudo-black-box runs (running the
full GPry loop on a cheap model as if it were opaque) and directly shape the later
phases. Both are recorded as measured, not predicted.*

**Two measured findings that shape Phases 2–3:**
1. **Intrinsic GW likelihood surfaces are brutally anisotropic and multi-lobed.** At
   network SNR 50 (ESIGMA 0PN, (Mc, e) space): σ_e ≈ 6×10⁻⁴ with *physical* lobes
   every ~0.005–0.01 in e (e–phasing degeneracy; converged in the φ_c quadrature,
   so not an artifact) and ~60-e-fold lobe contrast. GPry over naive wide bounds
   (Mc 28–32, e 0–0.2): 197 evaluations, no convergence, then an UltraNest MLFriends
   degeneracy. At SNR ~ 12 the same structure has few-e-fold contrast and is
   learnable (converged, truth recovered). **Consequence: cheap-model (case-1)
   posteriors must set the surrogate prior/ref bounds — the Phase-2
   multifidelity/ref-bounds step is not an optimization, it is a requirement for
   loud events;** expect the required evaluation count to grow with SNR² unless the
   e-ripple is absorbed (e.g., marginalizing mean anomaly ℓ into the fast stage the
   way φ_c was — a candidate Phase-3 refinement).
2. **The t_c window must be wide enough to absorb timing degeneracies:** with a
   ±3-sample window, integer-sample t_c quantization imprints ~2-e-fold ripples on
   the Mc marginal that the GP then chases; ±20 samples suffices in the tests
   (production: the full ±0.1 s prior window).

*Context for this block: 𝓛(θ_int) is not computed exactly — its inner extrinsic
integral is an importance-sampling estimate whose quality (ESS) varies from one
intrinsic point to the next. GPry, built for deterministic likelihoods, treats every
value it is handed as exact. That mismatch is a silent-failure channel. The
four-part package below (record → self-heal → recycle → gate) closes it, and is the
reliability contract of §5 made concrete and measured.*

**Reliability of the inner extrinsic marginal — the diagnostics/gate/recycling
package (2026-07-14).** The full-marginal `L(θ_int)` is itself an adaptive
importance-sampling estimate whose quality (effective sample size) varies with θ.
GPry's convergence criterion treats every value it is given as exact — it was built
for deterministic likelihoods — so a run can converge on *systematically biased*
training values with no visible symptom. This is a genuine silent-failure channel; we
closed it in three layers, all in `MarginalizedIntrinsicLikelihood` /
`BalanceHeuristicAccumulator` / `bin/run_gpry_pe.py`:

- **Record** every call's importance-sampling diagnostics (`importance_sampling_history`
  in memory + incremental `importance_sampling_history.jsonl` on disk, so an upstream
  crash — see UltraNest note below — cannot destroy the evidence).
- **Self-heal**: below an `effective_sample_size_floor`, add escalating extra rounds
  (each 2× the previous, up to `max_extra_importance_sampling_rounds`) until the floor
  is met — replacing the naive discard-and-restart retry.
- **Recycle** (`BalanceHeuristicAccumulator`): every batch, pilot included, contributes
  to the estimate and the effective sample size via the balance heuristic
  `w_i = e^{lnL_i}/q̄(u_i)`, `q̄ = Σ (n_j/N) q_j` (Veach–Guibas; adaptive-MIS caveat —
  consistent, not strictly unbiased, bias ≪ discarded-batch variance). The old
  worst-case cost 1×+2×+4× (only the final 4× used) becomes ≤5.75× *all used*, and the
  quality target is typically reached a round earlier. Validated against a closed-form
  4-cube integral in `tests/test_marginalized.py`.
- **Gate**: after the run, `importance_sampling_summary(..., peak_efolds=)` counts
  unhealthy calls *within a few e-folds of the peak* (measured: tail low-ESS calls are
  harmless, peak ones perturb the fit); the driver fails the run on any such call —
  banner, `reliable: false`, `UNRELIABLE` sentinel, exit code 2 — unless `--strict`
  raises `LowEffectiveSampleSizeError` mid-run instead (pairs with GPry checkpointing).

**Measured on the 2-D chirp demo (noise seed 1234; the caveat this resolves).** The
inner-marginal quality directly controls the posterior, so bad estimates are not
cosmetic:

| noisy run | f0 | span | ESS median | calls < floor | gate |
|---|---|---|---|---|---|
| zero-noise reference | 37.00 ± **1.12** | 55.27 ± **2.65** | — | — | — |
| lean, unhealed | 36.60 ± **0.60** | 55.47 ± **0.89** | 77 | **45/70** | — |
| 4× budget (brute) | 36.62 ± 0.84 | 55.61 ± 1.81 | 252 | 12/88 | — |
| lean + targeted healing | 36.38 ± 0.87 | 55.72 ± 1.17 | 175 | **1/120** | **failed → exit 2** |

The unhealed posterior is ~2× too *narrow* — an artifact of noisy inner estimates, not
physics: both cures relax the widths back toward the zero-noise reference. Targeted
healing reached 1 unhealthy call from a lean base budget (retrying only the ~75 calls
that needed it) versus 12 for uniform 4× budget; effective sample size is strongly
θ-dependent, so uniform budget increases waste effort. The gate then correctly refused
to certify the one near-peak call it could not heal in two rounds. (Caveat: the healed
run used GPry seed 12 vs seed 11 elsewhere, so the *width* numbers are not
seed-controlled; the 45→1 inner-health result is.)

**UltraNest fragility (robustness item).** GPry's NORA acquisition explores the
surrogate with UltraNest, which is unseeded ("Seeded runs are not supported") and
intermittently raises `numpy.linalg.LinAlgError: Distances are not positive`
(`ultranest.mlfriends`) when the surrogate surface is momentarily degenerate — observed
both on the ESIGMA wide-bounds surface and once mid-run on the noisy demo. It is
stochastic (the identical config succeeded on other seeds). Mitigations: GPry
checkpoint/resume for long production runs; `BatchOptimizer` as an acquisition fallback
that avoids nested sampling; and the on-disk diagnostics so a crash costs no evidence.
Open interaction to fix in Phase 2: GPry checkpoint-resume reuses cached evaluations
without re-calling our wrapper, so `importance_sampling_history` does not survive a
resume — the `.jsonl` persistence is the first half of the fix.

*Context for this block: the first apples-to-apples test that the surrogate approach
actually reproduces a known answer. On a case-(1) model we can run both PE methods —
**Route A** (direct gradient sampling of the full likelihood) and **Route B** (the
GPry surrogate over 𝓛(θ_int)) — and check they agree. Agreement here is the
empirical backing for the whole fusion; the four findings that follow came out of
getting both methods to converge.*

**End-to-end cross-validation of the two PE routes (2026-07-15).** Direct gradient
sampling (Global-Local NF + MALA over the full parameter vector; "Route A") and the
GPry surrogate over the extrinsic-marginalized intrinsic likelihood ("Route B") were
run to convergence on the *same* zero-noise ESIGMA injection (0PN, `n_ode_grid=1024`,
network SNR ≈ 11) and **agree**: the (chirp_mass, eccentricity) marginals match to
Hellinger distance ≈ 0.10–0.14 with mean offsets < 0.15 σ, and both recover truth. The
result is backend-reproducible (gradient route on CPU vs GPU: Hellinger ≈ 0.10). Route
B reaches this in ~24–58 *true* waveform evaluations versus ~10⁴ gradient evaluations
for Route A — the ratio that motivates the whole surrogate approach for expensive
case-(2) models. Four findings from getting both routes to converge:

1. **Loud events stall both routes cold over broad priors.** A network-SNR-80 injection
   (a ~1000:1 needle) failed to be found by either route with wide intrinsic priors —
   the same anisotropy/multi-lobe pathology as the SNR-50 finding above, now confirmed
   to hit *gradient* sampling too (chains never locate the mode), not just GPry.
   Cheap-model-derived bounds (the Phase-2 ref-bounds step) are required for both routes,
   not a GPry-specific crutch.
2. **The T2000 GPU was 2.6× *slower* than CPU for full ESIGMA gradient PE** (6924 s GPU
   vs 2632 s CPU, matched config). This is **not** a data-movement artifact: the MCMC is
   fully device-resident — `run_chains` compiles the whole step loop into one
   `jit(vmap(chains) × lax.scan(steps))` block ([`jaxpe/kernels/base.py:126`](../jaxpe/kernels/base.py#L126)),
   and the global block likewise ([`jaxpe/sampler/global_local.py:227`](../jaxpe/sampler/global_local.py#L227));
   the only host↔device traffic is a few per-*loop* scalar `float(mean(acc))` syncs and
   checkpoints, negligible in bandwidth. The slowdown is architectural fit: the per-step
   cost is dominated by the ESIGMA diffrax ODE solve + forward-sensitivity gradient,
   which is (i) a *sequential* dependency chain of tiny kernels (latency-bound, the
   opposite of what a GPU hides), (ii) parallelized only across the few×10 chains of the
   `vmap` (far too narrow to fill the device), and (iii) mandatory **float64**, which a
   consumer Turing T2000 runs at ~1/32 of fp32. Sequential × tiny-batch × throttled-fp64
   means the GPU pays all three weaknesses and collects none of its throughput strength;
   the CPU wins on serial fp64 latency. (Contrast: the wide vmapped `best_of_prior_init`
   batch eval, [`jaxpe/sampler/global_local.py:200`](../jaxpe/sampler/global_local.py#L200),
   *is* GPU-favorable — width, not depth, is the lever.)
3. **`n_ode_grid=512` biases the log-likelihood by ≈ −1.2 (~1.5 σ) at the peak;**
   `n_ode_grid=1024` converged. See §"ODE grid" analysis: the effective resolution is
   `n_ode_grid × t_isco/t_max ≈ 0.66` (the 1.5× time-domain margin plus the frozen
   post-ISCO plateau are dead weight), and uniform-in-time sampling under-resolves the
   late chirp — a candidate for non-uniform ODE nodes / cubic-Hermite state
   reconstruction (differentiability-preserving) as a future efficiency win.
4. **Route B needs orders of magnitude fewer waveform evaluations** (finding restated
   from the eval counts above) — the correct axis on which to judge the surrogate for
   minutes-per-call models, where wall-clock is dominated by waveform generation.

*Context for this block: the cross-validation above was on ESIGMA (time-domain,
eccentric, multi-harmonic). Repeating the Route-A-vs-Route-B check on a frequency-domain
dominant-$(2,2)$-mode model (IMRPhenomD) yielded a much cheaper Route-B marginalizer and
let us measure the §D4 profiling checkpoint as a function of **signal duration** — the
axis the single Phase-1 datum could only extrapolate over.*

**PhenomD Route B via an exact closed-form marginal, and the duration-scaling profiling
campaign (2026-07-16; marginal + cross-validation done, scaling campaign running).**

*A cheaper Route B for dominant-mode FD models.* For a dominant-$(2,2)$-mode model
$h_+ = h_0(1+\cos^2\iota)/2$ and $h_\times = -i h_0\cos\iota$ share one complex factor,
and a coalescence-phase shift acts as $h_{\rm det}(\phi_c) = e^{\pm 2i\phi_c}
h_{\rm det}(0)$ *exactly*, so the $\phi_c$ integral collapses to $\ln I_0(u|Z|)$ (a
Bessel function) and only the $D_L$ integral remains (1-D quadrature) — no
spherical-harmonic-mode decomposition and no FFT, unlike the general
`ModesNetworkLikelihood`. Landed as `PhaseDistanceMarginalLikelihood`
(`jaxpe/gw/fd_marginal.py`, exported) with a **dominant-mode self-check** that measures
the residual of $h_{\rm det}(\phi_c)/h_{\rm det}(0)$ from a pure phase and warns if a
higher-mode model is plugged in — the closed form is exact only for dominant-mode models.
Validated (`tests/test_fd_marginal.py`) against an independent brute-force 2-D
$(\phi_c, D_L)$ quadrature of the *full* likelihood ($<10^{-3}$); residual $\approx 0$ for
PhenomD, above tolerance (warns) for ESIGMA $(2,2)+(3,3)$. This is now the natural Route B
for FD dominant-mode models; the mode-based marginalizer stays for genuinely
multi-harmonic ones (ESIGMA, precessing).

*Three-route cross-validation.* On a zero-noise aligned-spin PhenomD injection (network
SNR $\approx 13$), Route A (gradient, CPU **and** GPU) and Route B (closed-form marginal +
GPry) recover the full intrinsic vector $(\mathcal{M}, q, \chi_{1z}, \chi_{2z})$ and agree
at both a nonspin $(\mathcal{M}, q)$ and an aligned-spin stage
(`examples/08_fd_dominant_mode_route_comparison.py`; overlays in
`examples/output/phenomd_*_route_comparison.png`). Route B converges in $\sim 24$ (2-D) /
$\sim 320$ (4-D) true evaluations vs $\sim 10^4$ gradient steps for Route A.

*Finding #2 reversed on FD models.* On PhenomD the T2000 GPU is $\sim 1.2\times$ **faster**
than CPU for Route-A gradient PE ($1123$ s GPU vs $1340$ s CPU nonspin; $1073$ vs $1314$
aligned-spin) — the **opposite** of the ESIGMA result (finding #2: GPU $2.6\times$ slower).
That slowdown was ODE-architecture-specific: PhenomD's likelihood is a vectorized FD kernel
with no sequential diffrax ODE and no forward-sensitivity tape, so the GPU's width is usable
and fp64 throttling is outweighed. The lever is width, not depth.

*Duration-scaling profiling campaign — the §D4 checkpoint across waveform cost.*
`examples/08` now (i) reads/writes **bilby-convention** injection files (portable inputs:
component masses, `a_i`/`tilt_i` spins, `theta_jn`), (ii) auto-sizes the analysis segment
per injection (pycbc signal length; 0PN fallback; next power of two), and (iii) generates a
**matched-SNR ($\approx 15$) total-mass sweep** $80\to 10\,M_\odot$ at fixed $q=0.8$,
$\chi=+0.2/-0.1$. Lower total mass $\Rightarrow$ longer signal: the sweep spans segment
durations $4$ s ($80\,M_\odot$, $n_{\rm freq}=4097$) to $32$ s ($10\,M_\odot$,
$n_{\rm freq}=32769$) — an $8\times$ waveform-cost lever at fixed SNR. All three routes
(A-CPU, A-GPU, B) are timed across it; Route B additionally logs the wall-clock split
between **waveform generation** (timed `PhaseDistanceMarginalLikelihood` calls) and **GPry
GP-fit + acquisition** (the remainder of `engine.run()`) — precisely the split D4's port
trigger is defined on. Run at `--config fast` (timing-oriented; per-step/per-eval rates,
not convergence, carry the scaling signal). **Measured** (matched SNR 15, 4-D intrinsic
$(\mathcal{M},q,\chi_{1z},\chi_{2z})$):

| $M_{\rm tot}$ | dur. | $n_{\rm freq}$ | B evals | B total | · waveform | · GPry | A-CPU | A-GPU |
|---|---|---|---|---|---|---|---|---|
| 80 | 4 s | 4097 | 184 | 258 s | 2.6 s | 255 s | 187 s | 166 s |
| 70 | 4 s | 4097 | 376 | 651 s | 3.5 s | 648 s | 168 s | 142 s |
| 60 | 8 s | 8193 | 460 | 832 s | 5.2 s | 827 s | 244 s | 167 s |
| 50 | 8 s | 8193 | 176 | 302 s | 3.2 s | 298 s | 243 s | 161 s |
| 40 | 8 s | 8193 | 240 | 430 s | 3.6 s | 427 s | 246 s | — |
| 30 | 8 s | 8193 | 392 | 812 s | 4.9 s | 807 s | 247 s | 162 s |
| 20 | 16 s | 16385 | 504 | 1341 s | 9.1 s | 1332 s | 613 s | 210 s |

($M_{\rm tot}=10$, 32 s: the gradient graph overflowed host-LLVM / T2000 memory — a 4 GB
hardware limit, not a code issue, and two A-GPU cells hit the same OOM; the Route-B point
hit the *documented* UltraNest MLFriends fragility on its sharp posterior — both are
known-issue limits, not new failures.) Scaling figure:
[`examples/output/phenomd_duration_scaling.png`](../examples/output/phenomd_duration_scaling.png).

*D4 assessment (measured).* **Route B is $\sim 99\%$ GPry across the whole sweep:** waveform
generation is $0.5$–$1.1\%$ of wall-clock ($2.6$–$9.1$ s) while GP fit + NORA/UltraNest
acquisition is the remaining $255$–$1332$ s — GPry outweighs the waveform by $\sim
95$–$185\times$ with **no crossover**, because PhenomD is a vectorized $\sim 9$–$18$ ms/call
kernel even at $16$ s. Route B's cost tracks the *evaluation count* ($176$–$504$, set by how
sharp the intrinsic posterior is — lower masses are better-constrained), not the signal
duration. So the D4 non-waveform share is $\sim 99\%$, far past the $30\%$ port trigger, and
the indicated lever is the **acquisition** nested sampler (the **BlackJAX** backend already
exposed at the `gpry/ns_interfaces.py` seam), *not* the GP fit or a waveform port. The two
quantities this conclusion hinges on — the GP-fit-vs-acquisition split *within* that
$\sim 99\%$, and the *actual* per-call cost of a production case-(2) EOB model (the design
originally assumed $0.5$–$10$ min/call, $\sim 10^{4}\times$ PhenomD) — were still assumed
here; both were then **measured directly** (next two blocks), which confirms the port target
and *relocates* the waveform-dominated regime. Separately, the **A-route GPU advantage grows
with signal duration** ($1.1\times$ at $4$ s $\to 2.9\times$ at $16$ s: $613$ s CPU vs $210$ s
GPU at $M_{\rm tot}=20$), reinforcing finding-#2-reversed: longer FD signals are *wider*
(larger $n_{\rm freq}$), and width is the GPU's lever.

*GP-fit vs acquisition, measured (Rec 1 — closes the split half of task 1.5).* `examples/08`
now reads GPry's own per-iteration timing table (`gpry.progress.Progress`) after
`engine.run()`, so the lumped "GPry" cost is broken into GP hyperparameter-refit
(`time_fit`), acquisition nested sampling (`time_acquire`, NORA over the GP surrogate),
convergence, and MC. Measured at two eval counts (the split is **$N$-dependent**, so a single
point would mislead):

| $M_{\rm tot}$ | evals $N$ | GPry | acquisition | GP fit | MC | acq/fit |
|---|---|---|---|---|---|---|
| 80 | 136 | 196 s | $151.5$ s ($1.11$/eval, $77\%$) | $9.1$ s ($0.07$/eval, $5\%$) | $\sim35$ s ($18\%$) | $17\times$ |
| 20 | 560 | 1639 s | $1147$ s ($2.05$/eval, $70\%$) | $425$ s ($0.76$/eval, $26\%$) | $\sim66$ s ($4\%$) | $2.7\times$ |

**Acquisition is the dominant cost at both ends ($70$–$77\%$)** — the primary port target — but
the GP fit's $O(N^3)$ Cholesky grows from a negligible $5\%$ at $136$ points to a non-trivial
$26\%$ at $560$, while acquisition (a nested sampler over the GP predictive, cheaper per added
point) stays dominant. Implications for D4: **(i)** the acquisition NS is the port to do first
($70$–$77\%$, and the `vmap`/BlackJAX-friendly part — it optimizes over the GP predictive with a
*cached* factorization, so the poor-consumer-GPU-fp64 objection does **not** apply to it);
**(ii)** the GP-fit fp64-Cholesky term the "don't rewrite" argument worried about is genuinely
small only at *low* eval count — at the high-$N$ end (which is where high-dimensional case-(2)
lives) it returns as $\sim$a quarter of the loop, so a *full* speedup there eventually needs the
Cholesky ported too, and that is exactly the fp64-hostile piece the design flagged. So the port
decision is not one-shot: acquisition always; GP fit only if profiling at the target
dimensionality shows it dominating. Matches the Phase-1 demo datum (acquisition $\sim 83\%$) now
on the real FD Route B.

*Real case-(2) EOB per-call cost, BBH$\to$BNS, measured (Rec 2 — the assumption D4 rested on).*
The "interface, don't rewrite" call rested on production EOB waveforms costing $0.5$–$10$
min/call. `examples/08 --eob-timing` times real external models at the sweep's intrinsic
configuration, **with the sampling rate set from the waveform's own highest frequency content**
(the ringdown; Nyquist sized for the $(4,4)$ mode, higher-mode models capped at $(l,m)\le(4,4)$),
warmup-excluded, across a total-mass grid extended down to **BNS masses** where the long signal
and high ringdown make the ODE integration expensive. `pyseobnr` (SEOBNRv5) was installed for
this. **Measured** (median ms/call; fs in Hz set per point):

| model | $M{=}80$ | $M{=}40$ | $M{=}20$ | $M{=}10$ | $M{=}4$ | $M{=}2.8$ (BNS) | crosses GPry ($\sim1.9$ s/eval) at |
|---|---|---|---|---|---|---|---|
| TEOBResumS      | 13 | 15 | 21 | 77 | 1200 | 2170 | $M\!\approx\!3$ (BNS) |
| SEOBNRv5HM      | 36 | 40 | 52 | 100 | 1640 | 3280 | $M\!\approx\!3$–$4$ |
| SEOBNRv5PHM     | 62 | 84 | 127 | 287 | 3880 | 7200 | $M\!\approx\!4$–$5$ |
| SEOBNRv4\_opt   | 17 | 22 | 35 | 67 | 384 | 690 | none (in range) |
| SEOBNRv4        | 234 | 393 | 790 | 2270 | 22600 | 50700 | $M\!\approx\!10$ |
| SEOBNRv4HM      | 390 | 640 | 1770 | 7580 | 65000 | 218000 | $M\!\approx\!10$–$20$ |

(fs rises $2048\to 4096\to 8192\to 16384\to 32768$ Hz as $M$ drops, from the $(4,4)$-mode
Nyquist; SEOBNRv5HM's native mode set already tops at $(4,4)$. Data:
[`examples/output/phenomd_eob_call_timing.json`](../examples/output/phenomd_eob_call_timing.json).)

**The $0.5$–$10$ min/call premise is false for standard aligned-spin EOB.** Across the whole
**stellar-mass BBH** range ($M\gtrsim 10$) every production model is $13$–$800$ ms/call —
$10^{2}$–$10^{4}\times$ cheaper than assumed, i.e. *comparable to the JAX PhenomD FD kernel* —
so Route B is GPry-dominated ($25$–$130\times$ for the fast models) and the acquisition NS is
the sole bottleneck. The waveform overtakes the $\sim 1.9$ s/eval GPry overhead **only at low
mass**, and the crossover mass is strongly model-dependent: fast models (TEOBResumS,
SEOBNRv5HM) cross only at **BNS masses** ($M\!\approx\!3$); precessing/higher-mode/unoptimized
models (SEOBNRv5PHM, SEOBNRv4HM, SEOBNRv4) cross at $M\!\approx\!4$–$20$; SEOBNRv4\_opt never
crosses in range.

*D4 reframed: a per-call-cost threshold, not a case-(1)/case-(2) dichotomy.* The port trigger
is a waveform cost of $\sim 1.9$ s/call, and both cheap JAX FD models **and** standard
aligned-spin EOB across the entire stellar-mass BBH range sit below it — so a **JAX/BlackJAX
acquisition nested sampler is a broadly worthwhile partial port**, not an FD-only curiosity: it
cuts real PE wall-clock for BBH parameter estimation with *any* of these waveform families. The
waveform-dominated regime the design assumed is real but **relocated to BNS / very-low-mass
($M\lesssim 3$–$5$) and the slow higher-mode variants**, where the surrogate's value is exactly
its original one — minimizing expensive calls — and pure "interface" holds. Net: D4's
"interface, don't rewrite the GP/robustness machinery" stands, but its escape-hatch clause is
now *active for the acquisition component in the BBH regime*, and specifically names the
acquisition NS (measured $70$–$77\%$), not the GP fit (measured $5$–$26\%$, $N$-dependent).

*Eccentricity, measured (the real "minutes/call" candidate).* SEOBNRv5EHM (available via
`pyseobnr`) at the same $(4,4)$ modes / physical fs costs **$\sim 4.5$–$12\times$ its aligned
SEOBNRv5HM counterpart** — M40 $41\to 186$ ms, M20 $54\to 495$ ms, M10 $125\to 1522$ ms, M6
$413\to 3725$ ms, M4 $1.65\to 8.6$ s — from the denser eccentric ODE, and nearly
$e$-independent between $e=0.1$ and $0.3$. So eccentricity **moves the crossover up to
$M\!\approx\!8$–$10$** (from $M\!\approx\!3$ aligned), but even eccentric SEOBNRv5 is
*seconds*/call at worst ($8.6$ s at M4/$e{=}0.3$), **not** the assumed minutes: across
stellar-mass BBH $M\gtrsim 20$ it is still GPry-dominated ($495$ ms $\ll 2.6$ s/eval). The
genuine minutes/call regime needs eccentricity **compounded** with a BNS-length signal (few
$M_\odot$) and/or sub-$20$ Hz $f_{\rm low}$ — not reached here.

*Remaining caveats.* Point-particle, aligned-spin timings; at BNS masses the standard models
omit tides (an ODE-length cost *proxy*, not production BNS waveforms); SEOBNRv5PHM at aligned
config is a lower bound on precessing cost. The harness (`--eob-timing`) is one registry entry
(`_EOB_MODELS`) from timing any approximant for the follow-up.

*Visual summary.* The EOB-cost curves, the three-route PE mass-scaling, and **every recovered
posterior vs truth** are collected in the
[D4 timing report]({{ '/examples/d4_timing_report.html' | relative_url }})
(`examples/d4_timing_report.html`; figures regenerated by `examples/figures/make_d4_figures.py`).
The posterior grid also records a validation datum worth flagging: at $M_{\rm tot}=20$ (sharp
$16$ s posterior) the *converged* Route-A gradient sampler mislocates the aligned spins
($\chi_{1z}\approx-0.15$) while Route B recovers $\chi_{1z}=+0.20,\ \chi_{2z}=-0.09$ (truth
$+0.2/-0.1$) — the marginalized surrogate is the more robust route on sharp spin structure, at
the higher masses all three routes agree, and $M_{\rm tot}=10$ (32 s) is Route-B-only (the
gradient graph exceeds the T2000's memory).

### Reducing the Route-B GPry bottleneck

*Current state (measured).* Route B's wall-clock is GPry's per-eval overhead ($1.4$–$2.6$ s),
of which **acquisition is $70$–$77\%$** and the **GP fit $5$–$26\%$** ($N$-dependent); the
waveform is $\lesssim 1\%$ for every BBH model. So for cheap-waveform PE the bottleneck *is* the
acquisition nested sampler, and secondarily the $O(N^3)$ GP refit at high eval count. Options to
cut it, most-impactful first, each with its tradeoff:

- **0 — Don't surrogate a cheap likelihood; sample the marginal directly.** GPry's whole purpose
  is minimizing *expensive* true-likelihood calls. When the marginalized intrinsic likelihood is
  cheap ($13$–$800$ ms), a direct nested sampler / MCMC on it — no GP, no acquisition — pays only
  the waveform per eval and is *exact* (no surrogate error, no IS reweighting). GPry earns its
  overhead only when $t_{\rm wave} \gtrsim t_{\rm overhead}\,N_{\rm gpry}/(N_{\rm direct}-N_{\rm gpry})
  \approx 2\,{\rm s}\times 300/10^4 \approx$ **tens of ms** (in 4-D). *Tradeoff:* direct NS scales
  worse in dimension ($N_{\rm direct}\!\sim\!10^{5}$–$10^{6}$ at 10-D), so this wins for low-D BBH
  and cheap waveforms; GPry wins for high-D or expensive (eccentric/BNS) case-(2). **Cheapest
  experiment, biggest potential win — and no new GP code.** *Action:* benchmark a direct NS
  (nessai / dynesty / BlackJAX-NS) on the *same* marginal as the Route-B baseline; gate GPry on
  the measured crossover.
- **1 — JAX/BlackJAX acquisition NS (Phase 2.5).** Port the $70$–$77\%$: the acquisition
  optimizes over the GP *predictive* with a cached factorization (fp64 Cholesky *not* re-run per
  eval), so it is `vmap`/GPU-friendly and the consumer-GPU-fp64 objection does not apply.
  *Tradeoff:* a wrong acquisition silently biases the posterior — must reproduce GPry-native
  proposals within tolerance (gate G2.5), preserving NORA's batch/trust-region logic.
- **2 — Cheaper acquisition than a full NS.** NORA runs a nested sampler over the acquisition
  surface *every* iteration; the GP exposes analytic mean/std gradients
  (`return_mean_grad`/`return_std_grad`, verified seam), so a multi-start gradient (L-BFGS)
  optimum is far cheaper and `vmap`-friendly. *Tradeoff:* gradient multistart can miss modes on
  the SNR$^2$-amplified multi-lobed acquisition surfaces we have already seen — validate against
  NORA before trusting. Orthogonal to (1): it attacks the same $70$–$77\%$ algorithmically.
- **3 — Cut the eval count $N$ (multifidelity mean + tight bounds; Phase 2).** Fewer evals means
  fewer acquisition iterations *and* a smaller GP, so it attacks **both** the acquisition total
  and the $O(N^3)$ fit. The cheap-model mean gives the GP a head start; cheap-model-derived bounds
  shrink the domain (already required for high-SNR multi-lobed surfaces). *Tradeoff:* multifidelity
  gain is unproven (gate G2 $\ge 2\times$); bounds need a trustworthy cheap model.
- **4 — Batch acquisition.** A larger NORA batch (propose $B$ points/iteration) means fewer
  iterations → fewer GP refits and NS runs, amortizing the fixed per-iteration cost; GPry's NORA
  batch $\leftrightarrow$ workers already supports it. *Tradeoff:* batch points are individually
  less informative (may raise $N$ slightly); net win depends on the fixed/variable cost ratio.
  Cheapest to try.
- **5 — Incremental GP (rank-1 Cholesky update) instead of full refit.** Caps the GP-fit growth
  (the $26\%$ at $N\!\sim\!560$) by updating the factorization as points are added rather than an
  $O(N^3)$ refit each iteration; fold into the multifidelity `GaussianProcessRegressor` subclass.
  *Tradeoff:* hyperparameters drift as data grows, so periodic full refits are still needed — only
  matters once the fit dominates (high $N$ / high dimension).

*Recommendation (ordered).* **(i)** Measure Option 0 first — a direct-NS baseline on the marginal;
it may remove the bottleneck entirely for low-D cheap-waveform BBH and pins exactly when GPry is
worth using, at zero GP-engineering cost. **(ii)** Where GPry *is* warranted (expensive-waveform
or high-D), do Option 1 (Phase 2.5) together with Option 2 — port *and* cheapen the acquisition —
gated on G2.5. **(iii)** In parallel, Option 3 (multifidelity + bounds, Phase 2) to cut $N$, which
attacks both terms and is already on the roadmap; try Option 4 opportunistically (near-free).
**(iv)** Only if high-$N$/high-$D$ profiling shows the GP fit dominating, add Option 5; defer any
fp64-Cholesky-on-GPU port per D4. Throughout, **correctness $>$ speed**: Options 1/2/5 change the
surrogate internals and can *silently* bias the posterior, so each is validated against the pinned
GPry loop before it is trusted — Option 0 is the safe baseline because it carries no surrogate at
all.

### Phase 2 — Multifidelity mean (~4–6 d)

| # | task | deliverable / test |
|---|---|---|
| 2.1 | `surrogate/multifidelity.py`: mean-function `GaussianProcessRegressor` subclass (fit residuals; `predict` adds $m$; `return_mean_grad` adds $\nabla m$) | unit tests incl. gradient paths; upstream-pin compatibility test |
| 2.2 | δ-smoothness probe tool (~20-point scatter evaluation, §4 rule) | report template |
| 2.3 | Pseudo-black-box multifidelity test: target = ESIGMA @ high `n_ode_grid`/PN, mean = cheap ESIGMA config (a *controlled* fidelity pair with known truth) | measured call-reduction factor vs Phase-1 baseline at matched accuracy |

**Gate G2:** measured expensive-call reduction ≥ 2× at matched posterior accuracy on the
controlled pair (else multifidelity stays opt-in/off and we proceed single-fidelity).

### Phase 2.5 — JAX acquisition nested sampler (conditional; D4 escape hatch, now triggered)

*Rationale (measured, §9): the acquisition NS is $70$–$77\%$ of the GPry loop (dominant at both
low and high eval count) while real aligned-spin EOB is $13$–$800$ ms/call across stellar-mass
BBH, so Route B is GPry-dominated for the whole BBH regime — the acquisition, not the waveform,
is the wall-clock. This is the one partial port the §D4 checkpoint now warrants. Only the
acquisition is ported; the GP regressor, SVM/robustness, convergence and checkpointing stay
GPry (D4). The GP fit ($O(N^3)$ Cholesky) is $5\%$ at low $N$ but $\sim 26\%$ at $N\!\sim\!560$
— revisit a fit port only if high-dimensional profiling shows it dominating (task 2.5.1).*

| # | task | deliverable / test |
|---|---|---|
| 2.5.1 | Split-timing + BBH$\to$BNS EOB timing landed (`examples/08 --eob-timing`, `gpry.progress` readout; eccentric SEOBNRv5EHM measured — $\sim$4.5–12$\times$ aligned, crossover $M\!\approx\!8$–$10$); remaining: the *compounded* minutes/call corner (eccentric + BNS-length + sub-20 Hz $f_{\rm low}$, + tidal EOB) | extend the §9 EOB table; confirms which targets stay waveform-dominated |
| 2.5.2 | JAX acquisition over the GP predictive at the `gpry/ns_interfaces.py` **BlackJAX** seam: `vmap` the acquisition function on a cached GP factorization; keep GPry's NORA batch/trust-region logic | matches GPry-native acquisition proposals within tolerance on a fixed GP; wall-clock speedup measured |
| 2.5.3 | Wire behind the `SurrogateEngine` protocol as an optional backend; default stays GPry-native | opt-in flag; falls back cleanly; posterior unchanged vs native on the Phase-1 pseudo-black-box |

**Gate G2.5:** on a fixed training set the JAX acquisition reproduces GPry-native proposals
(same next-point distribution within tolerance) **and** cuts acquisition wall-clock ≥ 2× at
$N\sim$ few$\times10^2$ training points; else it stays off and we keep GPry-native (correctness
over speed — a wrong acquisition silently biases the posterior).

### Phase 3 — Production case (2) (~6–10 d + cluster time)

| # | task | deliverable / test |
|---|---|---|
| 3.1 | `external_models/teobresums.py`, `seobnrv6ehm.py` wrappers: mode extraction, `f_ref`/spin/units conventions documented and round-trip tested vs each code's own examples | convention test suite (this is where silent bugs live) |
| 3.2 | MPI deployment: NORA batch = workers; failure/timeout handling for crashed waveform calls (map to GPry's infinities classifier) | soak test with injected failures |
| 3.3 | `surrogate/reweight.py`: IS weights + ESS + hierarchical extrinsic recovery (D3) | unit tests; ESS-degradation → auto-reacquire loop |
| 3.4 | **Injection study**: SEOBNRv6EHM injection, ESIGMA mean (6D pairing) — first production-config run | ESS/N ≥ 0.2 (tune); credible-interval recovery |
| 3.5 | Precessing TEOBResumS injection (8D, single-fid) | same |
| 3.6 | **Golden-event brute-force cross-check** (once per model family, §7.4) | posterior overlay report |

**Gate G3:** injection recovery + ESS targets met on both model families; brute-force overlay
consistent.

### Deferred / explicitly out of scope now

10D eccentric+precessing production runs (do after G3); ROM-from-cache amortization (D2
optionality); gradient-enhanced kernels; **any JAX port of the GP regressor / SVM / convergence
machinery** (the §D4 checkpoint triggered a port of the *acquisition NS only* — Phase 2.5 — and
the GP fit measured $5$–$26\%$, $N$-dependent, deferred pending high-dimensional profiling);
RIFT head-to-head paper comparison.

---

## 10. Risks

*Context: the things most likely to go wrong, each paired with its mitigation. The
recurring theme is the one that makes this whole project delicate — surrogate-PE
failures are silent (a wrong posterior, not an error) — so every risk here is paired
with a *visible* check (an ESS floor, a convention test suite, a compatibility test,
the infinities classifier) that converts a silent failure into a loud one.*

- **10D precessing is the GP stress case** — mitigations: extrinsic marginalization (done by
  design), multifidelity where a sane pair exists, seeding the initial proposer from a cheap
  jaxpe posterior instead of the prior; fallback is more acquisition rounds (§1.1 says even
  10⁴ calls is affordable at these per-call costs).
- **Convention mismatches in 3.1** (`f_ref`, spin frames, mode conventions, units) are the
  most likely source of silent wrong posteriors — hence the dedicated convention test suite
  and the golden-event cross-check.
- **Upstream GPry churn** breaking the mean-function subclass — version pin + a compatibility
  test in CI; the subclass touches one stable sklearn-style class.
- **Waveform-call failures** (EOB codes can fail at extreme parameters) — must map to the SVM
  infinities classifier, not crash the loop (task 3.2).
- **Unverified claims to re-check before relying on them:** SEOBNRv6EHM spin/eccentricity
  content; TEOBResumS variant naming for eccentric/precessing modes; RIFT interpolation
  details; GPry acquisition counts at 10D.
