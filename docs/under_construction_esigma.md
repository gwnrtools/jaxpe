---
layout: default
title: ESIGMA — ISCO & Inspiral Termination
nav_order: 100
---

# ESIGMA Inspiral Termination: `x_final`, the ISCO, and Spin

> Split out of [`under_construction.md`](under_construction.md) (the ESIGMA XLA-compilation and
> gradient log). This note collects the quantified analysis of the inspiral-termination radius
> `x_final` versus the physically-motivated ISCO. **Current modeling decision: keep esigmapy's
> fixed `r = 4M`; revisit later.**

## 1. What `x_final` actually is

**`x_final` is not the ISCO — it is a fixed `r = 4M` inspiral-termination radius.**
`ESIGMAInspiral` sets `self.x_final = 1.0 / inspiral_end_radius` with `inspiral_end_radius = 4.0`
([esigma.py:90](../jaxpe/gw/esigma.py#L90)), i.e. `x_final = 0.25`. In the PN variable
`x = (M\omega_{\rm orb})^{2/3}` (leading order `x ≈ M/r`), that is `r = 4M`. The Schwarzschild
test-particle ISCO is `r = 6M`, i.e. `x_ISCO = 1/6 ≈ 0.167` — so the code terminates the inspiral
at `x = 0.25 > 1/6`, **deeper than (inside) the ISCO**. It is a heuristic end-of-inspiral radius,
*not* an ISCO estimate, and it does not depend on mass ratio or spin.

This mirrors esigmapy's own **numba PN backend**, which sets
`f_gw_isco = 1/(TRANS^{3/2} \pi M)` with `TRANS = inspiral_end_radius = 4.0` and
`x_final = (\pi M f_gw_isco)^{2/3} = 1/4` (`pn_main.py:529,572-574`) — it *labels* this "ISCO"
but it is the same fixed `4M`. So the jaxpe adapter faithfully reproduces esigmapy's PN model, and
`test_esigma_parity` validates that agreement.

## 2. The physically-motivated ISCO (spin and mass-ratio dependent)

`r = 6M` holds only for a non-spinning test particle around a Schwarzschild BH.

- **Kerr test-particle ISCO (Bardeen–Press–Teukolsky 1972),** dimensionless spin `χ ∈ [-1,1]`
  (aligned; negative = retrograde):
  $$ r_{\rm ISCO}/M = 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2 Z_2)}, $$
  $$ Z_1 = 1 + (1-\chi^2)^{1/3}\left[(1+\chi)^{1/3} + (1-\chi)^{1/3}\right], \quad Z_2 = \sqrt{3\chi^2 + Z_1^2}, $$
  upper sign (`−`) prograde, lower (`+`) retrograde. `χ=0 → 6M`; `χ=+1 → 1M`; `χ=−1 → 9M`.

- **Finite mass-ratio + aligned spin:** esigmapy already ships this as
  `esigmapy.utils.f_ISCO_spin(m1, m2, s1z, s2z)` (used by its *surrogate* backend, not the PN one).
  It forms an **effective spin** `a_eff = a_tot + \zeta\,\eta\,(s_{1z}+s_{2z})`
  (`a_tot` the mass-weighted spin, `ζ = 0.41616`), evaluates the Kerr ISCO of the **remnant**
  (final spin `χ_f` via an `η`-dependent fit, then the BPT radius/frequency of that remnant), and
  returns the Kerr ISCO **GW frequency** in Hz. `η` enters through the final-spin/final-mass fit.
  This is a phenomenological remnant-ISCO transition frequency (Healy–Lousto–Zlochower-style
  fits), not the strict two-body ISCO; EOB radial-potential or MECO (`dE/dx = 0`) conditions are
  the more first-principles finite-`η` alternatives.

## 3. Consequence for highly spinning black holes

The fixed `x_final = 0.25` cannot track the ISCO, which in the code's own `x` variable sweeps
`x_ISCO ∈ [0.11, 0.51]` (`r ∈ [1M, 9M]`) across aligned spin. Using the Kerr formula with the
Kerr orbital frequency at ISCO, `M\omega_{\rm ISCO} = 1/(r_{\rm ISCO}^{3/2} + \chi)` and
`x_ISCO = (M\omega_{\rm ISCO})^{2/3}`:

| χ (aligned) | r_ISCO/M | x_ISCO | vs fixed x_final = 0.25 |
|---:|---:|---:|---|
| +0.99 | 1.45 | **0.510** | stops **short** (large early cut) |
| +0.90 | 2.32 | **0.370** | stops short |
| +0.70 | 3.39 | 0.275 | stops slightly short |
| +0.50 | 4.23 | 0.228 | runs slightly past |
| 0.00 | 6.00 | 0.167 | runs past |
| −0.50 | 7.56 | 0.135 | runs past |
| −0.90 | 8.72 | **0.117** | runs **far past** ISCO |
| −0.99 | 8.97 | 0.114 | runs far past |

Two distinct failure modes:

- **Prograde, χ ≳ 0.6 → premature truncation.** The true ISCO is *inside* `x = 0.25`, so the fixed
  cutoff ends the inspiral early and discards the loudest late-inspiral cycles. The termination
  frequency is off by `(x_ISCO / 0.25)^{3/2}`: at χ = +0.9 the real ISCO frequency is **1.8×
  higher** than where the code stops (≈2.9× at χ = +0.99). For PE this lost band biases the
  likelihood and pushes recovered spin/masses.
- **Retrograde and low spin, χ ≲ 0.5 → over-integration past the ISCO.** The true ISCO is *outside*
  `x = 0.25`, so the code integrates the adiabatic PN inspiral well past it — for χ = −0.9 down to
  ~3× past the ISCO frequency — into a regime with **no stable circular orbits**, where an
  inspiral-only model is unphysical (this includes the non-spinning case, since `4M` is already
  inside `6M`).

No single fixed radius can be right, since the ISCO sweeps `1M → 9M` across spin.

**PN-validity caveat that limits the upside of simply extending the cutoff.** At `x = 0.25`,
`v ≈ 0.5c` and the PN series is already marginal; for high prograde spin the ISCO sits at
`x = 0.37–0.51`, where the PN expansion (and its large spin-orbit/spin-spin terms) is genuinely
unreliable. So a spin-dependent ISCO would *properly cut* retrograde/low-spin systems at their
outer ISCO (removing the unphysical over-integration — a clear win) but would *extend* prograde
systems toward their inner ISCO only as far as PN remains trustworthy.

## 4. If we later adopt a proper ISCO in jaxpe

Replace the constant with `x_final = (m_sec · \pi · f_ISCO_spin(m1, m2, s1z, s2z))^{2/3}` (the same
`x = (M\pi f_{\rm GW})^{2/3}` mapping already used for `x_init`), porting `f_ISCO_spin` to JAX
(pure algebra — differentiable; guard `sign/abs` at zero spin).

**Interaction with the ISCO-clip gradient fix** (§17–18 of
[`under_construction.md`](under_construction.md)). Making `x_final` a function of `(η, s1z, s2z)`
is fully compatible with the `jnp.minimum(x, x_final)` cap: the cap stays differentiable and would
correctly propagate the ISCO's *parameter* dependence into the mass/spin gradients (a new,
physically-correct contribution, since `x_final` also enters `t_max`). It does **not** reintroduce
the frozen-sensitivity bug, which is about the RHS freeze and is orthogonal to the *value* of
`x_final`.

**Decision (deferred).** Two consistent choices:
1. **Keep `r = 4M`** (current) to remain a faithful, parity-validated reproduction of esigmapy's
   PN model.
2. **Adopt the `η`+spin ISCO** for better physics — but this changes where the inspiral ends,
   diverges from esigmapy's PN reference, and would require updating `test_esigma_parity` and
   re-validating against the surrogate/reference. esigmapy's *own* PN backend would then also need
   changing for consistency.

## Sources

Bardeen, Press & Teukolsky, ApJ 178, 347 (1972); Kerr ISCO calculator (L. C. Stein,
duetosymmetry.com); ISCO parameter review [arXiv:1605.04189](https://arxiv.org/abs/1605.04189);
PN/self-force ISCO conditions [arXiv:1010.2553](https://arxiv.org/abs/1010.2553);
`esigmapy/utils.py::f_ISCO_spin`.
