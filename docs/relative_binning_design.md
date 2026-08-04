---
layout: default
title: Design — Relative binning (FD & TD heterodyned likelihoods)
nav_order: 2
parent: Ongoing
---

# Design Note: Relative Binning in jaxpe (Fourier- and Time-Domain)

**Status:** design drafted 2026-07; decisions D-RB1..4 accepted. **Implemented:**
RB-1 (FD, dominant mode) in
[`relative_binning_fd.py`](../jaxpe/gw/likelihood/relative_binning_fd.py); RB-3
(Toeplitz/Gohberg–Semencul infrastructure) in
[`toeplitz.py`](../jaxpe/gw/likelihood/toeplitz.py); RB-4 (TD heterodyned
likelihood, dominant mode **and higher modes**) in
[`relative_binning_td.py`](../jaxpe/gw/likelihood/relative_binning_td.py)
(`RelativeBinningTDLikelihood` and `RelativeBinningTDLikelihoodHM`). All exact
at the fiducial and validated against the exact reference to the Zackay
`beta*(1+|lnL|)` error model, with measured speedups (FD ~12x, TD ~370x / HM ~74x vs
dense, Gohberg–Semencul `C⁻¹v` ~3300x vs dense solve). Also implemented and tested:
**RB-2 FD higher modes** (`RelativeBinningFDLikelihoodHM`, per-mode/per-pair per-bin
summary data on FD modes with the diagonal covariance), the TD **detector network**
(`RelativeBinningTDNetwork`), and **t_c support** (a coalescence-time shift samples the
trial mode at `edge_times − Δt_c`; validated against the dense reference). **Not
started:** RB-5 end-to-end PE against a production waveform + sampler (needs a
production higher-mode mode source and sampler/GPry integration).
**Companion notes:** [`gpry_fusion_design.md`](gpry_fusion_design.md) (surrogate route,
mode-based marginalization — shares the `ModesData` machinery this note reuses).

---

## How to read this note

*Read by section; each restates the terms it needs. Shared vocabulary:*

- **Relative binning / heterodyning** — a likelihood-acceleration trick. Instead of
  summing the noise-weighted inner product over all `N` data points, pick a
  **fiducial** waveform `h₀` at a nearby parameter point, and use that the **ratio**
  `r = h/h₀` of any nearby trial waveform to the fiducial is a *slowly varying*
  function. Approximate `r` as piecewise-linear over a coarse set of **bins**
  (`N_bins ≈ 200–500 ≪ N`), precompute **summary data** from `(data, h₀, noise)`
  once, and every subsequent likelihood is an `O(N_bins)` sum over bins that needs
  the trial waveform only at bin edges. Speedups of 10²–10³ are typical.
- **FD relative binning** — the original scheme (Zackay, Dai & Venumadhav 2018,
  arXiv:1806.08792), in the **frequency** domain, where the noise covariance is
  **diagonal** (the Whittle likelihood jaxpe already uses). bilby's
  `RelativeBinningGravitationalWaveTransient` is the reference implementation.
- **TD relative binning** — the **time-domain** extension (Sharma, Vijaykumar &
  Kumar, arXiv:2601.11239), where the noise covariance `C` is a **non-diagonal
  symmetric Toeplitz** matrix `Cᵢⱼ = ρ(|i−j|)` (`ρ` = noise autocorrelation). This
  is the accurate likelihood for long / edge-sensitive segments (BNS, sub-second
  windows) where the FD-diagonal approximation and windowing bite. It heterodynes
  **per spherical-harmonic mode** `h_lm(t)`.
- **`NetworkLikelihood`** — jaxpe's abstract Whittle likelihood base
  ([`jaxpe/gw/likelihood/base.py`](../jaxpe/gw/likelihood/base.py)); concrete
  members `TDNetworkLikelihood`, `FDNetworkLikelihood`, `ModesNetworkLikelihood`.
- **`ModesData`** — jaxpe's container of `h_lm(t)` modes on a uniform time grid
  (`d_ref_mpc`, `t_ref`), produced by external mode models and consumed by
  `ModesNetworkLikelihood`. The TD method's fiducial `h₀_lm(t)` is exactly a
  `ModesData`.

**Provenance.** The FD scheme and bilby's formulas were read from
`bilby/gw/likelihood/relative.py`. The TD scheme — including **Appendices A
(higher-mode summary data), B (Gohberg–Semencul), and C (bin-count sufficiency)** —
was read from the full arXiv:2601.11239 PDF; equation numbers below (Eqs. 4–34,
A1–A15, B1–B2) refer to that paper. One author of the TD paper is the maintainer of
this repo — treat the paper as authoritative on any conflict.

---

## 1. Problem statement

jaxpe's inner product ([`base.py`](../jaxpe/gw/likelihood/base.py)
`NetworkLikelihood.log_likelihood`) is the full-resolution, frequency-**diagonal**
Whittle sum

$$\ln\mathcal L = (d|h) - \tfrac12(h|h),\qquad (a|b) = 4\,\mathrm{Re}\sum_{f}\frac{\tilde a(f)\,\tilde b^*(f)}{S_n(f)}\,\Delta f,$$

an `O(N_f)` sum requiring the waveform at every frequency bin. For a cheap JAX model
(`IMRPhenomD`) at 8 s this is fine; for long segments (128 s BNS, `N_f ~ 10⁵–10⁶`) or
for the accurate time-domain covariance it is the bottleneck (arXiv:2601.11239 report
~3.13 s per full TD evaluation at 128 s). We want two accelerated inner-product
engines that reproduce the exact likelihood to sampling tolerance:

1. **FD relative binning** — diagonal covariance, dominant- and higher-mode FD models.
2. **TD relative binning** — non-diagonal Toeplitz covariance, per-mode heterodyning.

Neither is a new waveform; both are new *inner-product engines* sitting behind the
same `log_likelihood(params) → scalar` contract.

---

## 2. Current state and integration points

| Piece | Where | Role for relative binning |
|---|---|---|
| `NetworkLikelihood` (ABC) | `likelihood/base.py` | new subclasses hang here; reuse `_static` eager cache for summary-data precompute, `detector_strains_fd`, `optimal_snr`, `problem()` |
| `FDNetworkLikelihood` | `likelihood/fd.py` | **exact reference** for the FD-binned lnL; provides `polarizations_fd` at bin-edge frequencies |
| `ModesData` / `ModesNetworkLikelihood` | `likelihood/modes.py` | TD method's fiducial + trial `h_lm(t)`; `spin_weighted_ylm`, per-mode FD algebra, `from_likelihood` idiom |
| `make_injection` / `likelihood_from_strain` | `gw/data.py` | construction entry points; dispatch on `waveform.is_fd`. Add a `relative_binning=…` path |
| `conditioning.py` | `gw/conditioning.py` | `rfft_freqs`, `time_shift`, `td_to_fd`, `tukey_window` |
| `psd.py` | `gw/psd.py` | PSD → (new) autocorrelation `ρ(τ)` for the Toeplitz `C` |

**Key synergy:** the TD method heterodynes per mode `h_lm(t)` with the convention
`h₊ − i h× = Σ_lm h_lm(t) ₋₂Y_lm(ι,φ)` (paper Eq. 7) — **identical** to
`ModesNetworkLikelihood`'s convention. The mode-ratio machinery, `spin_weighted_ylm`,
and the `from_likelihood(like, modes_data)` constructor are directly reusable.

---

## 3. Accepted decisions

- **D-RB1 — API shape:** relative-binning likelihoods are **sibling
  `NetworkLikelihood` subclasses** with a `from_likelihood(like, fiducial_params, …)`
  constructor (mirrors `ModesNetworkLikelihood.from_likelihood`). They are drop-in
  for the sampler / `InferenceProblem`; the host-side summary-data precompute lives
  in an override of `_static`, exactly like the existing eager mode-FFT / `half_dd`
  caching.
- **D-RB2 — TD covariance:** implement the **Gohberg–Semencul** fast inverse-Toeplitz
  matvec **in JAX** (FFT-based, `O(N log N)`), used at setup to build summary data.
- **D-RB3 — mode scope:** **higher modes from the start** — per-mode heterodyning for
  both FD and TD (not dominant-(2,2) only).
- **D-RB4 — deliverable:** this design note (`docs/relative_binning_design.md`).

---

## 4. Method 1 — Fourier-domain relative binning

**New module:** `jaxpe/gw/likelihood/relative_binning_fd.py` →
`RelativeBinningFDLikelihood(NetworkLikelihood)`.

### 4.1 Bin setup (Zackay et al. 2018, Eqs. 8–10)

Write the phase as a sum of PN power laws `Ψ(f) = Σ_i α_i f^{γ_i}` (Eq. 8). The
worst-case differential phase drift of any nearby waveform is

$$\delta\Psi_{\max}(f) = 2\pi\chi \sum_i \Big(\tfrac{f}{f_{*,i}}\Big)^{\gamma_i}\,\mathrm{sgn}(\gamma_i),\qquad f_{*,i}=\begin{cases}f_{\min}&\gamma_i<0\\ f_{\max}&\gamma_i>0\end{cases}\ \ (\text{Eq. 9}),$$

and a bin is admitted while `|δΨ_max(f_max(b)) − δΨ_max(f_min(b))| < ε` (Eq. 10). The
**exponent set is a modeling choice**: the paper motivates `γ ∈ {−5/3 (chirp mass),
−1 (sym. mass ratio, 1PN), −2/3 (spin), 5/3 (tidal), 1 (merger-time shift)}`; bilby's
implementation uses `[−5/3, −2/3, 1, 5/3, 7/3]`. Expose `γ` and the tunables `(χ, ε)`;
paper achieves target accuracy `β<0.01` with **62 bins** on GW170817 at `χ=1, ε=0.5`.
Store `bin_edges`, `bin_centers`, and the full-grid→bin index map. Because the FD
covariance is **diagonal**, bins can be chosen **per detector** and the summary data
are **per-bin** (single index `b`) — no cross-bin term, unlike TD (§5).

### 4.2 Summary data (host-side, in `_static`; Eqs. 3–6)

Evaluate the fiducial detector strain `h₀(f)` once at full resolution (reuse
`detector_strains_fd(fiducial_params)`), then per detector, per bin `b` (the paper
writes these with a dimensionless-DFT `S_n/T` normalization; use jaxpe's own continuum
`4Δf/S_n` convention — the structure is identical):

$$A_0^{(b)} = 4\Delta f\!\!\sum_{f\in b}\frac{d\,h_0^*}{S_n},\quad
A_1^{(b)} = 4\Delta f\!\!\sum_{f\in b}\frac{d\,h_0^*}{S_n}(f-f_c^b),\quad
B_0^{(b)} = 4\Delta f\!\!\sum_{f\in b}\frac{|h_0|^2}{S_n},\quad
B_1^{(b)} = 4\Delta f\!\!\sum_{f\in b}\frac{|h_0|^2}{S_n}(f-f_c^b),$$

reusing the existing `inv_psd_banded` from `_static`.

### 4.3 Hot path (jittable `log_likelihood`; Eqs. 1, 7)

Evaluate the trial waveform **only at bin edges**, ratio `r = h/h₀`, then

$$r_0^{(b)} = \tfrac12\big(r_{b+1}+r_b\big),\quad r_1^{(b)} = \frac{r_{b+1}-r_b}{\Delta f_b},$$

$$(d|h) = \sum_b \big(A_0^{(b)} r_0^{(b)*} + A_1^{(b)} r_1^{(b)*}\big),\qquad
(h|h) = \sum_b \big(B_0^{(b)}|r_0^{(b)}|^2 + 2B_1^{(b)}\,\mathrm{Re}(r_0^{(b)} r_1^{(b)*})\big),$$

`ln L = Re(d|h) − ½(h|h)` summed over detectors. Cost: `O(N_bins)` + one waveform
call at `~N_bins` frequencies.

### 4.4 Higher modes (D-RB3)

The *summed* ratio `r = (Σ_lm c_lm h_lm)/(Σ_lm c_lm h₀_lm)` is smooth only for
dominant-mode models. The original paper notes (concluding remarks) that HM "would
require a different set of bins for each mode as different modes are highly oscillatory
with respect to each other" — so each mode gets its **own** bin set and per-mode ratio
`r_lm`, with per-mode/per-bin summary data `A_n^{lm}(b), B_n^{lm}(b)` (still per-bin,
diagonal covariance). Contract with the extrinsic `c_lm(ι,φ,ψ)`. For FD HM models that
expose modes (e.g. `IMRPhenomXHM` when wrapped), use the per-mode path; dominant-(2,2)
`IMRPhenomD` uses the cheaper summed-ratio path (§4.3). (This is the FD analogue of the
TD per-mode scheme in §5.4, minus the cross-bin `B(b₁,b₂)`.)

### 4.5 Fiducial parameters & guardrail

Default fiducial = injection/trigger point. Optional refinement via
`scipy.optimize.differential_evolution` (host-side, setup-only). Add a
**smooth-ratio self-check** at construction (mirroring
`PhaseDistanceMarginalLikelihood.dominant_mode_residual`): measure `max|r − linear|`
across bins at a probe offset and warn if the linear approximation is violated
(precession, wide fiducial mismatch).

### 4.6 Validation

- `|ln L_binned − ln L_exact| ≲ 10⁻³` over prior draws vs `FDNetworkLikelihood`.
- Recover bilby's `A/B` formulas exactly on a shared injection.

---

## 5. Method 2 — Time-domain relative binning (arXiv:2601.11239)

**New modules:** `jaxpe/gw/likelihood/toeplitz.py` (covariance infra) and
`jaxpe/gw/likelihood/relative_binning_td.py` →
`RelativeBinningTDLikelihood(NetworkLikelihood)`, built from `ModesData`.

### 5.1 Likelihood and covariance

$$\ln\mathcal L(d|\theta) = -\tfrac12\sum_{i,j}(d_i - s_i)\,(C^{-1})_{ij}\,(d_j - s_j) + \text{const},\qquad C_{ij} = \rho(|i-j|),$$

with `ρ(τ)` the noise autocorrelation (`ρ = IFFT` of the two-sided PSD; new helper in
`psd.py`). `C` is symmetric positive-definite Toeplitz.

### 5.2 Gohberg–Semencul fast inverse (D-RB2, `toeplitz.py`, paper App. B)

`C⁻¹` is needed only **at setup** (to weight `d` and the fiducial modes into summary
data); it never appears in the hot path. Dense inversion is `O(N³)` time / `O(N²)`
memory (terabytes at 128 s) — infeasible. The paper's two-step scheme (App. B):

1. **Generators (once).** Solve the two Toeplitz systems (Eq. B2) `C x = e₀` and
   `yᵀC = e_{N-1}ᵀ` by Levinson–Durbin (`scipy.linalg.solve_toeplitz`), needing only
   the first row/column of `C` → `O(N)` memory, `O(N²)` time, **once per PSD**. For
   **symmetric** `C` only `x` is solved and `y_k = x_{N-k}`.
2. **Matvec (Gohberg–Semencul, Eq. B1).** `C⁻¹` is a difference of products of
   triangular Toeplitz matrices built from `x, y`:
   $$C^{-1} = \frac{1}{x_0}\big(L(x)\,U(y) - \tilde L(y)\,\tilde U(x)\big),$$
   where `L(x)`/`U(y)` are lower/upper-triangular Toeplitz (generators `x`, `y`) and
   `\tilde L,\tilde U` the strictly-triangular shifts. Each triangular-Toeplitz × vector
   is an FFT linear convolution → `C⁻¹v` in **`O(N log N)`** (the paper's `matmul_toeplitz`;
   Fig. 8 confirms `N log N` vs Levinson's `N²`).

**D-RB2 realization:** generators host-side via `scipy.solve_toeplitz` (once); the
**matvec in JAX** (`jnp.fft`-based triangular-Toeplitz products), so the `N_bins ×
N_modes` right-hand sides of §5.4 batch under `vmap` on GPU, and the whole map is
jittable/differentiable. Never materialize `C` or `C⁻¹`.

### 5.3 Adaptive time bins (paper Eqs. 28–34, App. C)

Derived from a phase-perturbation argument (Eqs. 23–29): the sampled waveform is the
fiducial plus a term ∝ `δΨ(t)`, with `Ψ(t)=Σ_k α_k t^{γ_k}` the PN phase powers. The
**maximum phase drift** is `δΨ_max(t) = 2πχ Σ_k (t/t_*)^{γ_k} sgn(γ_k)` (Eq. 33,
`t_* = |t_max|` if `γ_k>0` else `|t_min|`), and a bin `b` is admitted while
`|δΨ_max(t_f(b)) − δΨ_max(t_i(b))| < ε` (Eq. 34). This is applied to the **inspiral**;
**merger-ringdown** uses a heuristic — narrow constant-width bins near merger, widening
outward in constant-width segments (Fig. 2). Tunables `(χ, ε)`; paper uses **191–486
bins for 2–128 s**. **Bin count matters (App. C / Fig. 9):** 187 bins bias a 128 s
posterior while 443 recover it — add a bin-sufficiency check (§7, risk iv) and expose `(χ,ε)`.

### 5.4 Per-mode summary data (host-side; D-RB3; paper Eqs. 13–22, A1–A15)

Decompose the log-likelihood (Eq. 13) into `L₁(d,d)` (data only, once), `L₁(d,s)`, and
`L₁(s,s)`, with `L₁(a,b) ≡ Σ_{ij} a_i C⁻¹_{ij} b_j`. Fiducial modes `h₀_lm(t)` = one
`ModesData` (reuse the external mode-model + `ModeCache` path from
`MarginalizedIntrinsicLikelihood`). Using `h₊−ih× = Σ_lm h_lm\,{}_{-2}Y_lm` and the
reality `h^{l,-m}=(-1)^l(h^{lm})^*` (fold to `m>0`), the summary data are (Eqs. A4,
A13–A15):

$$A_0^{lm}(b)=\sum_{j\in b}(C^{-1}d)_j\,h_{0,j}^{lm},\qquad
A_1^{lm}(b)=\sum_{j\in b}(C^{-1}d)_j\,h_{0,j}^{lm}(t_j-t_c),$$

$$B_n^{l_1m_1,l_2m_2}(b_1,b_2)=\sum_{i\in b_1}\sum_{j\in b_2} \big[h_{0,i}^{l_1m_1}\ \text{or}\ h_{0,i}^{*\,l_1m_1}\big]\,C^{-1}_{ij}\,\big[h_{0,j}^{l_2m_2}\,\{1\ \text{or}\ (t_j-t_c)\}\big].$$

There are **two `A_n^{lm}` (per mode)** and **nine `B_n^{l_1m_1,l_2m_2}` (per mode
pair)**, `n=0..8` — the `(H,H)` and `(H^*,H)` combinations of Eq. A10 (App. A). `A`
needs `C⁻¹d` (one G-S solve); each `B` column needs `C⁻¹` applied to a **bin-masked
fiducial mode** `h₀_{lm}·1_{b_2}` (§5.2), i.e. **`N_bins × N_modes` G-S matvecs** at
setup — this is the whole setup cost (Table I: 8.7 s → 2×10⁴ s for 2 s → 128 s).
Detector antenna/`F₊,F×`/time-delay factors fold in exactly as in
`ModesNetworkLikelihood._phase_decomposition`.

### 5.5 Hot path (jittable)

Generate trial modes `h_lm(t)` at **bin edges only** → per-mode `r₀,r₁` (Eq. 11) →

$$L_1(d,s)=\sum_b\big[r_0^{lm}(b)A_0^{lm}(b)+r_1^{lm}(b)A_1^{lm}(b)\big]\ \ (O(N_{\rm bins})),$$
$$L_1(s,s)=\sum_{b_1,b_2}\sum_{n}\ (r\cdot r)\,B_n(b_1,b_2)\ \ (O(N_{\rm bins}^2 N_{\rm modes}^2)),$$

combined with the extrinsic `c_lm(ι,φ,ψ)` (Eqs. A3, A10). Both are pure contractions
of the precomputed tensors — no waveform, no `C⁻¹` — so at `N_bins ~ 500` the whole
evaluation is a few ms (Table I: 2.3–9.2 ms, 19–341× over full).

### 5.6 Resolved: `B` is full bin-pair; the cost is setup

The `(s,s)` term (Eqs. 20–21, A13–A15) sums over **all** bin pairs `(b₁,b₂)` with **no
truncation** — the abstract's "`O(N_bins)`" refers to waveform downsampling and the
`L₁(d,s)` term, while `L₁(s,s)` is `O(N_bins² N_modes²)` but negligible at eval time.
Design consequences: store `B` **dense** per mode-pair (`N_bins²` complex, ~2 MB per
pair at 500 bins) and precompute it once; the engineering target is the **setup**
(`N_bins × N_modes` G-S matvecs — §5.2 makes this GPU-batchable), not the hot path. A
band-limited-`B` approximation is a *possible future optimization* (inverse-Toeplitz
correlations decay), not part of the faithful implementation.

### 5.7 Validation

- `|ln L_TD-binned − ln L_dense-Toeplitz|` to setup tolerance on small `N`.
- Reproduce the paper's **P–P** test (200 BBH 8 s + 200 BNS 128 s; all p > 0.05) and
  **JS divergence** (~10⁻³) vs a direct TD sampler.
- Confirm the ~340× speedup at 128 s.

---

## 6. Module layout

```
jaxpe/gw/likelihood/
├── relative_binning_fd.py   # RelativeBinningFDLikelihood(NetworkLikelihood)
├── relative_binning_td.py   # RelativeBinningTDLikelihood(NetworkLikelihood)
├── toeplitz.py              # autocorr, Levinson, Gohberg-Semencul matvec (JAX)
└── _relative_binning.py     # shared: Zackay/phase-tolerance bin scheme, r0/r1 helpers
```
`gw/psd.py`: add `autocorrelation_from_psd(...)`. `gw/data.py`: add a
`relative_binning={"fd"|"td", fiducial_params, chi, epsilon}` path in `make_injection`
/ `likelihood_from_strain`. `gw/likelihood/__init__.py` + `gw/__init__.py`: export the
two new classes and add to `__all__`.

---

## 7. Phased plan and gates

| Phase | Scope | Gate |
|---|---|---|
| RB-1 ✅ | `relative_binning_fd.py`: Zackay bin scheme + `RelativeBinningFDLikelihood` (dominant mode) | **done** — exact at fiducial; parity to beta·(1+\|lnL\|); ~12x speedup; support-restriction handled (band beyond ringdown cutoff) |
| RB-2 ✅ (modes) | `RelativeBinningFDLikelihoodHM`: per-mode / per-mode-pair per-bin `A`/`B` on FD modes (diagonal covariance — no cross-bin, no G-S) | **done** — exact at fiducial; parity vs dense multi-mode FD likelihood (`fd_dense_loglikelihood_modes`) to beta·(1+\|lnL\|). Works on FD modes directly; a production HM FD mode source is the only missing piece for end-to-end use |
| RB-3 ✅ | `toeplitz.py`: autocorr + Levinson generators + Gohberg–Semencul matvec (JAX) | **done** — `C⁻¹v` vs dense to 1e-9..1e-11; round-trip; jit+vmap; ~3300x vs dense solve |
| RB-4 ✅ (dominant + HM) | `relative_binning_td.py`: adaptive time bins + `A0/A1` and `B0/B1/B1b/B2/B3/B3b` summary data + hot path; `RelativeBinningTDLikelihoodHM` for higher modes (per-pair cross-mode tensors, Appendix A) | **done** — exact at fiducial; parity vs dense-Toeplitz (intrinsic/extrinsic/noisy) to `beta·(1+\|lnL\|)`; single-mode HM == dominant; ~370x (dominant) / ~74x (2-mode) speedup. Network, t_c: follow-on |
| RB-5 ◐ (posterior-level) | End-to-end PE + validation | **partial** — synthetic posterior-recovery test done (`test_td_posterior_recovery_vs_dense`): the heterodyned and dense-likelihood posteriors over an intrinsic parameter agree in mean/width, both peak at truth, and their **JS divergence < 1e-3** (the paper's acceptance level). Full production PE (real HM waveform + sampler + P–P over many injections) remains |

**Risks:** (i) **Waveform-at-bin-edges** — the speedup needs modes evaluable at
arbitrary (non-uniform) times. `NRSur7dq4`/JaxNRSur support this; EOB/expensive models
must generate dense + downsample (keeps the *covariance* speedup, loses the
*waveform* one — the paper says the same). Confirm which jaxpe models expose an
arbitrary-time API. (ii) **Setup cost** (`N_bins × N_modes` G-S matvecs, hours at
128 s) is the real bottleneck — must be GPU-batched and cached; amortized over ~10⁷
evals. (iii) Gohberg–Semencul conditioning on steep BNS PSDs — `float64` + a dense
fallback for small N; validate the generators. (iv) **Bin count** too low → biased
posteriors (App. C) — ship a convergence check comparing `(χ,ε)` and doubled bins.
(v) HM ratio smoothness where a fiducial `h₀_lm` has near-zeros (ratio blows up) — the
per-mode scheme presumes `h₀_lm ≠ 0` on-support; add a guard. (vi) Tukey-window vs
exact-`C` consistency between the FD (diagonal, windowed) and TD (Toeplitz) paths.

---

## 8. References

- Zackay, Dai & Venumadhav, *Relative Binning and Fast Likelihood Evaluation for
  Gravitational Wave Parameter Estimation*, arXiv:1806.08792.
- Sharma, Vijaykumar & Kumar, *Rapid inference of gravitational-wave signals in the
  time domain using a heterodyned likelihood*, arXiv:2601.11239.
- bilby `RelativeBinningGravitationalWaveTransient`, `bilby/gw/likelihood/relative.py`.
- Gohberg & Heinig, inverse-Toeplitz representation (per paper App. B).
