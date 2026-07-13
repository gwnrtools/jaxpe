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

## 1. Problem statement

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

Needs 10⁶–10⁷ *training* waveforms up front — the same infeasible budget, paid before the first
event. Rejected for case (2). (It remains attractive for case (1), orthogonal to this note.)

### 1.3 Prior art: this architecture is RIFT-shaped

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

### D1. What function does the GP learn?

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

As decided in D3. Deliverables: an IS-reweighting post-processor with ESS reporting, and the
hierarchical extrinsic-conditional sampler (jaxpe gradient kernel over 7D with cached modes).
Acceptance rule for a production run: report is incomplete without (ESS/N, number of
acquisition rounds, δ-diagnostics if multifidelity, and the convergence-criterion trace).

---

## 6. Architecture / new code layout

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
re-check at 6-10D where NORA cost grows). Notes: jaxpe requires `jax_enable_x64`
(float32 GPS times silently NaN — the driver sets it; scripts must too);
GPry's `logp_truth` is single-point (wrapped with a loop); strict-editable installs
need `pip install -e . --no-deps` re-run when a new subpackage is added.
**Remaining for G1-full:** task 1.4 (ESIGMA as pseudo-black-box — needs a mode-level
adapter since `ESIGMAInspiral` exposes only polarizations) and a full-marginal
(adaptive-IS) end-to-end run vs direct sampling; then the G2 multifidelity work.

### Phase 2 — Multifidelity mean (~4–6 d)

| # | task | deliverable / test |
|---|---|---|
| 2.1 | `surrogate/multifidelity.py`: mean-function `GaussianProcessRegressor` subclass (fit residuals; `predict` adds $m$; `return_mean_grad` adds $\nabla m$) | unit tests incl. gradient paths; upstream-pin compatibility test |
| 2.2 | δ-smoothness probe tool (~20-point scatter evaluation, §4 rule) | report template |
| 2.3 | Pseudo-black-box multifidelity test: target = ESIGMA @ high `n_ode_grid`/PN, mean = cheap ESIGMA config (a *controlled* fidelity pair with known truth) | measured call-reduction factor vs Phase-1 baseline at matched accuracy |

**Gate G2:** measured expensive-call reduction ≥ 2× at matched posterior accuracy on the
controlled pair (else multifidelity stays opt-in/off and we proceed single-fidelity).

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
optionality); gradient-enhanced kernels; any JAX port of GPry components absent a failed §D4
checkpoint; RIFT head-to-head paper comparison.

---

## 10. Risks

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
