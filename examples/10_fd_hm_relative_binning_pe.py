r"""End-to-end higher-mode Fourier-domain relative-binning parameter estimation.

Injects a :class:`~jaxpe.gw.ToyChirpFDHM` signal (a toy multi-mode FD chirp whose (3,3)
amplitude scales with mass asymmetry, so the mass ratio is identifiable through higher
modes), builds the higher-mode relative-binning likelihood
:class:`~jaxpe.gw.likelihood.RelativeBinningFDLikelihoodHM`, and runs PE over the
intrinsic parameters (chirp mass, mass ratio) on a 2-D grid. The heterodyned posterior
is compared to the exact dense-likelihood posterior: they agree (Jensen-Shannon
divergence far below the 0.06 acceptance line) while the heterodyned likelihood evaluates
the trial waveform at only ~N_bins bin edges.

Run: ``python examples/10_fd_hm_relative_binning_pe.py``
"""

import jax

jax.config.update("jax_enable_x64", True)

import time

import jax.numpy as jnp
import numpy as np

from jaxpe.gw import ToyChirpFDHM, spin_weighted_ylm
from jaxpe.gw.likelihood import (
    RelativeBinningFDLikelihoodHM,
    fd_dense_loglikelihood_modes,
)

# ------------------------------------------------------------------- injection setup
FREQS = np.fft.rfftfreq(4096, d=1.0 / 1024.0)  # df = 0.25 Hz, up to 512 Hz
F_MIN, F_MAX = 20.0, 400.0
PSD = 1e-3 * (1.0 + (30.0 / np.clip(FREQS, FREQS[1], None)) ** 4)
TRUE = {"chirp_mass": 30.0, "mass_ratio": 0.6}
EXTRINSIC = dict(iota=0.9, phi=1.1, distance=1.0)
TARGET_SNR = 25.0

wf = ToyChirpFDHM(modes=((2, 2), (3, 3)))
KEYS = wf.modes


def mode_stack(params, fgrid):
    m = wf(params, jnp.asarray(fgrid))
    return np.stack([np.asarray(m[k]) for k in KEYS])


def main():
    # per-mode extrinsic coefficients c_lm = Y_lm(iota, phi) / D  (fixed in this run)
    coeff = np.array(
        [
            complex(
                np.asarray(spin_weighted_ylm(EXTRINSIC["iota"], EXTRINSIC["phi"], l, m))
            )
            / EXTRINSIC["distance"]
            for (l, m) in KEYS
        ]
    )
    stack_true = mode_stack(TRUE, FREQS)
    data = (coeff[:, None] * stack_true).sum(0)

    # rescale to a target network SNR
    df = float(FREQS[1] - FREQS[0])
    inv = np.where((FREQS >= F_MIN) & (FREQS <= F_MAX), 1.0 / PSD, 0.0)
    snr = np.sqrt(4.0 * df * np.sum((data.real**2 + data.imag**2) * inv))
    coeff *= TARGET_SNR / snr
    data = (coeff[:, None] * stack_true).sum(0)
    print(f"injection: {TRUE}, network SNR = {TARGET_SNR:.1f}")

    # fiducial = injection; build the heterodyned likelihood
    fmodes = {k: stack_true[i] for i, k in enumerate(KEYS)}
    like = RelativeBinningFDLikelihoodHM(
        fmodes, FREQS, data, PSD, F_MIN, F_MAX, phase_per_bin=0.5
    )
    edge_freqs = FREQS[like.edge_indices]
    n_band = int(np.sum((FREQS >= F_MIN) & (FREQS <= F_MAX)))
    print(f"relative binning: {n_band} band points -> {like.n_bins} bins")
    het_eval = jax.jit(like.log_likelihood)

    # ----------------------------------------------------------------- grid PE
    mc_grid = np.linspace(29.5, 30.5, 61)
    q_grid = np.linspace(0.45, 0.80, 61)
    het = np.empty((mc_grid.size, q_grid.size))
    dense = np.empty_like(het)
    t_het = t_dense = 0.0
    for i, mc in enumerate(mc_grid):
        for j, q in enumerate(q_grid):
            p = {"chirp_mass": mc, "mass_ratio": q}
            t0 = time.perf_counter()
            het[i, j] = float(het_eval(jnp.asarray(mode_stack(p, edge_freqs)), coeff))
            t_het += time.perf_counter() - t0
            t0 = time.perf_counter()
            dense[i, j] = fd_dense_loglikelihood_modes(
                mode_stack(p, FREQS), coeff, data, PSD, FREQS, F_MIN, F_MAX
            )
            t_dense += time.perf_counter() - t0

    def posterior(lnl):
        w = np.exp(lnl - lnl.max())
        return w / w.sum()

    ph, pd = posterior(het), posterior(dense)
    ih, jh = np.unravel_index(ph.argmax(), ph.shape)
    mix = 0.5 * (ph + pd)
    m = ph > 0
    js = 0.5 * np.sum(ph[m] * np.log(ph[m] / mix[m])) + 0.5 * np.sum(
        pd[m] * np.log(pd[m] / mix[m])
    )
    print(f"heterodyned MAP: chirp_mass={mc_grid[ih]:.3f}, mass_ratio={q_grid[jh]:.3f}")
    print(f"JS(heterodyned || dense) = {js:.2e}  (acceptance line 0.06)")
    print(
        f"grid eval time: heterodyned {t_het:.2f}s, dense {t_dense:.2f}s "
        f"(waveform generated at {like.n_bins + 1} vs {n_band} frequencies per point)"
    )

    # ----------------------------------------------------------------- plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        MC, Q = np.meshgrid(mc_grid, q_grid, indexing="ij")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contour(MC, Q, ph, levels=6, colors="C0")
        ax.contour(MC, Q, pd, levels=6, colors="C1", linestyles="--")
        ax.plot(TRUE["chirp_mass"], TRUE["mass_ratio"], "k+", ms=12, label="injection")
        ax.plot([], [], color="C0", label="heterodyned")
        ax.plot([], [], color="C1", ls="--", label="dense")
        ax.set(
            xlabel="chirp mass [Msun]",
            ylabel="mass ratio q",
            title="FD higher-mode relative-binning PE",
        )
        ax.legend()
        out = "examples/figures/fd_hm_relative_binning_pe.png"
        import os

        os.makedirs("examples/figures", exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")
    except ImportError:
        print("matplotlib not available; skipping the plot")


if __name__ == "__main__":
    main()
