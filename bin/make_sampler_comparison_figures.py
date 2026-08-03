#!/usr/bin/env python
r"""Cross-sampler figures for the mass-sweep PE suite (docs/bns_ce_pe_benchmark.md).

Reads several ``run_mass_sweep_pe.py`` output trees -- one per ``--kernel``, all run
over the SAME injection grid (same ``--seed``, so identical masses, spins, durations
and target SNRs at every index) -- and writes two families of PNG into ``docs/assets``:

``sampler_corner_injNN_M*.png`` (one per binary)
    Every kernel's posterior for that one injection, overlaid on a single corner
    plot: the four sampled parameters (chirp mass, eta, spin1z, spin2z) plus the
    three derived ones (m1, m2, chi_eff) computed per-sample. The injected truth is
    drawn once. Because the kernels return different numbers of samples, the 1-D
    marginals are drawn as densities, not counts -- otherwise a kernel that simply
    ran longer would appear to have a taller posterior.

    Axis ranges are COMMON across kernels within a figure (the union of each
    kernel's 0.5-99.5 percentile range), which is what makes the overlay legible;
    it also means a kernel whose chains diverged visibly widens the axes rather
    than silently vanishing off-frame.

``sampler_timing_comparison.png``
    Time to the convergence gate versus total mass, one line per kernel, log-x.
    Runs that did NOT pass the gate inside the budget are drawn as hollow markers
    and are the honest reading of "this kernel did not converge here", not a
    missing point -- their wall time is a budget floor, not a convergence time.

Encoding notes
--------------
Four MH-corrected kernels take categorical hue slots 1-4 of the validated default
palette (all-pairs CVD-separated; the aqua slot is below 3:1 on white, so every
series is also direct-labelled and legended). ULD is drawn in ink, dashed, and NOT
as a fifth hue: it has no Metropolis correction, so its stationary distribution is
biased by O(step_size^2) *by construction* (see ``jaxpe/kernels/uld.py``). It is
categorically not a peer estimator of the same target, and encoding it as one more
colour would say it is. Every series additionally carries a distinct line style, so
identity never rests on colour alone.

Run:
    python bin/make_sampler_comparison_figures.py --root examples/output/sampler_sweep
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# bin/ is a script directory, not an installed package, so the sibling module is
# imported by path rather than by distribution -- the derived-parameter and
# axis-scaling helpers must be the SAME code that draws the single-sampler corner
# plots, or the two figure families would silently disagree on m1/m2/chi_eff.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bns_ce_figures import _chirp_mass_offset_scale, _derived_m1_m2_chieff

INK, MUTED = "#0b0b0b", "#52514e"

# slot -> (display label, colour, line-plot dash, contour dash). Order is fixed and
# never cycled. hmc/mala/mmala/random-walk take categorical slots 1-4 (validated
# all-pairs); uld is deliberately ink, not a fifth hue -- see the module docstring.
# The two dash entries differ because matplotlib's contour ``linestyles`` accepts
# only the four NAMED styles, not the (offset, on-off) tuples Line2D takes -- so
# the 1-D marginals carry the finer-grained dash vocabulary and the contours reuse
# the four names (colour still separates every pair).
KERNEL_STYLE = {
    "hmc": ("HMC", "#2a78d6", "-", "solid"),
    "mala": ("MALA", "#eb6834", (0, (5, 1.6)), "dashed"),
    "mmala": ("MMALA", "#1baf7a", (0, (1.4, 1.4)), "dotted"),
    "random-walk": ("RandomWalk", "#4a3aa7", (0, (6, 1.5, 1.4, 1.5)), "dashdot"),
    "uld": ("ULD (unadjusted)", INK, (0, (3, 2.2)), "dashed"),
}
KERNEL_ORDER = tuple(KERNEL_STYLE)

LABELS = (
    None,  # filled in per figure: the chirp-mass offset scale is adaptive
    r"$\eta$",
    r"$\chi_{1z}$",
    r"$\chi_{2z}$",
    r"$m_1$  [$M_\odot$]",
    r"$m_2$  [$M_\odot$]",
    r"$\chi_{\rm eff}$",
)


def _read_summary(path):
    """sweep_summary.csv -> {injection index: row}."""
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {int(r["index"]): r for r in np.atleast_1d(rows)}


def discover(root):
    """{kernel: (summary_rows, sweep_dir)} for every kernel subdirectory present."""
    found = {}
    for name in KERNEL_ORDER:
        csv = Path(root) / name / "sweep_summary.csv"
        if csv.exists():
            found[name] = (_read_summary(csv), Path(root) / name)
        else:
            print(f"no sweep for kernel {name!r}: missing {csv}")
    if not found:
        raise SystemExit(f"no kernel sweeps found under {root}")
    return found


def _params(npz, mc_true_offset):
    """(n, 7) sampled+derived array for one run, in this module's fixed column order.

    Also returns the fraction of stored draws that were non-finite and had to be
    dropped. That is NOT a plotting detail to be hidden: an unadjusted integrator
    that blows up writes NaN/inf into its own chains, and the fraction is a direct
    measure of how badly. Measured here: zero for all four MH-corrected kernels on
    every injection; up to 27% for ULD (and 0% on the cheapest one, so it is
    injection-dependent, not a constant property of the kernel).
    """
    d = np.load(npz)
    s, truth, names = d["samples"], d["truth"], [str(n) for n in d["names"]]
    i = {n: k for k, n in enumerate(names)}
    m1, m2, chi_eff, m1_t, m2_t, chieff_t = _derived_m1_m2_chieff(s, truth, i)
    x = np.column_stack(
        [
            s[:, i["chirp_mass"]] - mc_true_offset,
            s[:, i["eta"]],
            s[:, i["spin1z"]],
            s[:, i["spin2z"]],
            m1,
            m2,
            chi_eff,
        ]
    )
    good = np.isfinite(x).all(axis=1)
    bad_frac = 1.0 - good.mean()
    t = np.array(
        [
            0.0,
            truth[i["eta"]],
            truth[i["spin1z"]],
            truth[i["spin2z"]],
            m1_t,
            m2_t,
            chieff_t,
        ]
    )
    return x[good], t, bad_frac, d


def _common_ranges(arrays):
    """Union of each run's 0.5-99.5 percentile box, per column.

    Percentiles rather than min/max so a handful of stranded chains cannot set the
    axes for everyone; a union rather than an intersection so a kernel that sampled
    somewhere else entirely stays VISIBLE (widening the frame) instead of being
    silently cropped out of its own comparison figure.
    """
    lo = np.min([np.percentile(a, 0.5, axis=0) for a in arrays], axis=0)
    hi = np.max([np.percentile(a, 99.5, axis=0) for a in arrays], axis=0)
    pad = 0.04 * np.where(hi > lo, hi - lo, 1.0)
    return list(zip(lo - pad, hi + pad))


def corner_overlay(runs, truth, out, *, title, info_lines, dropped=None):
    """One corner figure with every kernel's posterior overlaid.

    ``runs`` is an ordered {kernel: (n, 7) array} mapping; ``truth`` the shared
    injected point in the same 7 columns.
    """
    import corner

    arrays = list(runs.values())
    mc_scale, mc_exp = _chirp_mass_offset_scale(
        np.max([np.std(a[:, 0]) for a in arrays])
    )
    scaled = [np.column_stack([a[:, 0] * mc_scale, a[:, 1:]]) for a in arrays]
    t = np.concatenate([[truth[0] * mc_scale], truth[1:]])
    labels = list(LABELS)
    labels[0] = (
        rf"$\mathcal{{M}}_c - \mathcal{{M}}_c^{{\rm true}}$  "
        rf"[$10^{{{mc_exp}}}\,M_\odot$]"
    )

    rng = _common_ranges(scaled)
    fig = None
    for (name, _), x in zip(runs.items(), scaled):
        _, color, dash, cdash = KERNEL_STYLE[name]
        fig = corner.corner(
            x,
            fig=fig,
            labels=labels,
            # truths are drawn once, on the first pass only, so the marker is not
            # over-plotted five times at five slightly different alpha stackings
            truths=t if fig is None else None,
            truth_color="#e34948",
            color=color,
            range=rng,
            bins=55,
            smooth=0.9,
            # NO smooth1d: with it set, corner draws the 1-D marginal as a Line2D
            # rather than via ax.hist, and a Line2D rejects ``density`` -- which is
            # the one hist option this figure cannot do without (below).
            levels=(0.5, 0.9),
            label_kwargs={"fontsize": 11, "color": INK},
            # density, not counts: the kernels return different sample counts, and
            # a count histogram would read a longer run as a taller posterior
            hist_kwargs={"linewidth": 1.7, "density": True, "linestyle": dash},
            contour_kwargs={"linewidths": 1.3, "linestyles": cdash},
            plot_datapoints=False,
            fill_contours=False,
            no_fill_contours=True,
            # corner's 2-D density image is drawn per call, so five overlays stack
            # five grey layers into a haze that hides the contours doing the actual
            # work here. Contours only.
            plot_density=False,
            max_n_ticks=4,
        )

    # A kernel that wrote non-finite draws gets that stated ON the legend entry --
    # the surviving contours would otherwise look like an ordinary posterior.
    dropped = dropped or {}
    handles = [
        plt.Line2D(
            [],
            [],
            color=KERNEL_STYLE[k][1],
            linestyle=KERNEL_STYLE[k][2],
            linewidth=2.0,
            label=KERNEL_STYLE[k][0]
            + (f"  [{dropped[k]:.0%} non-finite dropped]" if dropped.get(k) else ""),
        )
        for k in runs
    ] + [plt.Line2D([], [], color="#e34948", linewidth=2.0, label="injection")]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=False,
        fontsize=11,
        labelcolor=INK,
    )
    fig.suptitle(title, fontsize=13.5, color=INK, y=1.012)
    fig.text(
        0.60, 0.80, info_lines, fontsize=10, color=MUTED, va="top", linespacing=1.55
    )
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _val(row, key):
    """Float field of a summary row, NaN for a missing row or an unparseable cell.

    Timed-out injections legitimately have empty timing cells, and a missing row
    must not be silently read as zero -- a zero bar reads as "instant", which is
    the opposite of what a timeout means.
    """
    if row is None:
        return float("nan")
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _conv(row):
    """True only if this row explicitly recorded a passed convergence gate.

    Anything else -- missing row, blank cell, timed-out injection -- is False:
    absence of evidence is not a pass.
    """
    if row is None:
        return False
    try:
        return str(row["converged"]).strip().lower() == "true"
    except (KeyError, TypeError, ValueError):
        return False


def timing_figure(found, out):
    """Cost vs total mass, one line per kernel: production only, and end to end.

    Two panels because they answer two different questions and only one of them is
    about the sampler. Setup (injection, MAP+Laplace, the relative-binning parity
    refinement) is byte-for-byte identical across kernels at a given injection --
    it is the same code on the same grid -- so including it in a *sampler*
    comparison only adds a common offset that flatters whichever kernel happens to
    sit beside the most expensive setup. The left panel is therefore the honest
    like-for-like comparison; the right panel is the honest end-to-end cost of
    actually getting a posterior, which is what a user pays.
    """
    # GROUPED BARS, not lines against mass. The cost here is set by the
    # relative-binning bin count (which the spin prior forces up on some
    # injections), NOT by total mass -- so joining the points with a line against a
    # mass axis draws a trend that does not exist, and leaves visual gaps wherever
    # an injection is missing. One group per binary, annotated with its bin count,
    # says what actually varies.
    idx = sorted(set().union(*(set(r) for r, _ in found.values())))
    idx = [
        i
        for i in idx
        if any(np.isfinite(_val(found[k][0].get(i), "total")) for k in found)
    ]
    names = list(found)
    w = 0.8 / len(names)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.6), sharex=True)
    for ax in axes:
        ax.grid(True, axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c9c4")
        ax.tick_params(colors=MUTED, labelsize=9)

    for ax, key, title in (
        (axes[0], "production", "Production only — the sampler's own cost"),
        (axes[1], "total", "End to end, including the shared per-injection setup"),
    ):
        for j, name in enumerate(names):
            rows = found[name][0]
            label, color, _, _ = KERNEL_STYLE[name]
            xs = np.arange(len(idx)) + (j - (len(names) - 1) / 2) * w
            vals = np.array([_val(rows.get(i), key) / 60.0 for i in idx])
            conv = np.array([_conv(rows.get(i)) for i in idx])
            # hatched = did not pass the gate, so the bar is a budget floor rather
            # than a cost; a 2px surface gap keeps adjacent bars readable
            ax.bar(
                xs,
                np.nan_to_num(vals),
                width=w * 0.92,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=1.6,
                hatch=["" if c else "///" for c in conv],
                zorder=3,
            )
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_ylabel("wall clock  [min]", fontsize=10.5, color=INK)

    ref = found[names[0]][0]
    axes[1].set_xticks(np.arange(len(idx)))
    axes[1].set_xticklabels(
        [
            f"{_val(ref.get(i), 'total_mass'):.1f} $M_\\odot$\n"
            f"{_val(ref.get(i), 'rb_n_bins'):.0f} bins"
            for i in idx
        ],
        fontsize=9.5,
        color=INK,
    )
    # legend sits ABOVE the axes: the bars run to the budget ceiling on the
    # hatched groups, so any in-axes placement collides with data
    axes[0].legend(
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.06),
        ncol=5,
        columnspacing=1.6,
        handlelength=1.5,
    )
    fig.suptitle(
        "Mass sweep at Cosmic Explorer: cost per sampler, on identical CPU hardware\n"
        "hatched = budget exhausted without passing the R-hat/ESS gate "
        "(that bar is a floor, not a convergence time)",
        fontsize=12.0,
        color=INK,
        x=0.015,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root",
        default="examples/output/sampler_sweep",
        help="directory holding one <kernel>/ subdirectory per sweep",
    )
    ap.add_argument("--assets", default="docs/assets")
    args = ap.parse_args()

    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    found = discover(args.root)
    print(f"kernels found: {', '.join(found)}")

    indices = sorted(set.intersection(*(set(rows) for rows, _ in found.values())))
    for idx in indices:
        runs, truth, ref_row, dropped = {}, None, None, {}
        # the injected chirp mass is the same for every kernel at this index (one
        # grid, one seed), and sweep_summary.csv already records it -- so the
        # offset every kernel is plotted against comes from the grid, not from
        # whichever run happened to be read first
        mc_ref = float(next(iter(found.values()))[0][idx]["mc"])
        for name, (rows, sweep_dir) in found.items():
            row = rows[idx]
            npz = (
                sweep_dir
                / f"inj_{idx:02d}_M{float(row['total_mass']):.1f}"
                / "samples.npz"
            )
            if not npz.exists():
                print(f"  injection {idx}, kernel {name}: no {npz}, skipping series")
                continue
            x, t, bad_frac, _ = _params(npz, mc_ref)
            if x.shape[0] < 100:
                print(
                    f"  injection {idx}, kernel {name}: only {x.shape[0]} finite "
                    "draws, skipping series"
                )
                continue
            if bad_frac > 0:
                print(
                    f"  injection {idx}, kernel {name}: dropped "
                    f"{bad_frac:.1%} non-finite draws"
                )
                dropped[name] = max(dropped.get(name, 0.0), bad_frac)
            runs[name], truth, ref_row = x, t, row
        if not runs:
            print(f"injection {idx}: no samples from any kernel, skipping figure")
            continue

        mtot = float(ref_row["total_mass"])
        gate = [k for k in runs if bool(found[k][0][idx]["converged"])]
        info = (
            f"network SNR {float(ref_row['achieved_snr']):.0f}\n"
            f"$m_1$ = {float(ref_row['m1']):.2f}, $m_2$ = {float(ref_row['m2']):.2f} "
            f"$M_\\odot$\n"
            f"$\\chi_{{1z}}$ = {float(ref_row['spin1z']):+.3f}, "
            f"$\\chi_{{2z}}$ = {float(ref_row['spin2z']):+.3f}\n"
            f"passed the gate: {', '.join(KERNEL_STYLE[k][0] for k in gate) or 'none'}"
        )
        out = assets / f"sampler_corner_inj{idx:02d}_M{mtot:.1f}.png"
        corner_overlay(
            runs,
            truth,
            out,
            title=f"$M_{{\\rm tot}}$ = {mtot:.2f} $M_\\odot$ — all samplers overlaid",
            info_lines=info,
            dropped=dropped,
        )
        print("wrote", out)

    print("wrote", timing_figure(found, assets / "sampler_timing_comparison.png"))


if __name__ == "__main__":
    raise SystemExit(main())
