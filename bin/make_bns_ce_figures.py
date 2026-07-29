#!/usr/bin/env python
r"""Figures for the BNS/Cosmic-Explorer PE benchmark (docs/bns_ce_pe_benchmark.md).

Reads the artefacts written by :mod:`bin.run_bns_ce_pe` -- ``samples.npz`` and the
per-run stdout logs -- and writes two PNGs into ``docs/assets``:

``bns_ce_corner.png``
    Posterior corner plot of the four sampled parameters. Chirp mass is shown as
    an offset from the injected value (its posterior is ~1e-6 Msun wide, so an
    absolute axis is unreadable) and the spins in units of 1e-3. The injection is
    marked; note it sits on the *edge* of several marginals because eta = 1/4 and
    chi = 0 are prior boundaries, not because the posterior is biased.

``bns_ce_convergence.png``
    Convergence against wall clock: rank-normalized split-Rhat of the global
    subseries, and Geyer min ESS, both versus *total* elapsed minutes (the
    per-block log timestamps are production-phase only, so the setup + MAP +
    warmup offset from the run's timings line is added back). The 20-minute
    budget and the two gate thresholds are drawn, so the claim "converged inside
    the budget" is read directly off the axes.

Run: python bin/make_bns_ce_figures.py
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots 1-3 of the validated default palette (CVD-separated;
# the aqua slot is below 3:1 on white, so every series is also direct-labelled)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED = "#0b0b0b", "#52514e"

BLOCK_RE = re.compile(
    r"production block (\d+):.*?rank-Rhat\(glob\) \[([^\]]+)\].*?"
    r"min ESS (\d+).*?\[([\d.]+)s\]"
)
TIMINGS_RE = re.compile(r"^timings: (\{.*\})", re.M)
# phases that precede the production loop, i.e. the offset to add to block times
PRE_PRODUCTION = (
    "psd",
    "injection",
    "rb_setup",
    "map_laplace",
    "rb_validation",
    "warmup",
    "flow_fit",
)


def parse_log(path):
    """(elapsed_minutes, rhat_max, min_ess) per production block, wall-clock referenced."""
    text = Path(path).read_text()
    timings = eval(TIMINGS_RE.search(text).group(1))  # noqa: S307 - our own output
    offset = sum(timings[k] for k in PRE_PRODUCTION) / 60.0
    rows = []
    for m in BLOCK_RE.finditer(text):
        rhat = max(float(v) for v in m.group(2).split())
        rows.append((offset + float(m.group(4)) / 60.0, rhat, float(m.group(3))))
    return np.array(rows).T  # (3, n_blocks)


def load_runs(rundir, csv):
    """Per-run convergence series, from the stdout logs if present, else the CSV.

    The raw ``*.log`` files are covered by the project's blanket log ignore, so the
    extracted series is cached to a small committed CSV -- that keeps the figure
    reproducible from a clean checkout without version-controlling stdout.
    """
    logs = {
        "run 1 (cold)": rundir / "run1_cold.log",
        "run 2 (frozen flow)": rundir / "run2_frozen_flow.log",
        "run 3 (final)": rundir / "run3_final.log",
    }
    if all(p.exists() for p in logs.values()):
        runs = {name: parse_log(p) for name, p in logs.items()}
        with open(csv, "w") as f:
            f.write("run,elapsed_min,rhat_max,min_ess\n")
            for name, d in runs.items():
                for t, r, e in zip(*d):
                    f.write(f"{name},{t:.4f},{r:.4f},{e:.0f}\n")
        return runs, f"logs in {rundir}"

    rows = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {
        name: np.array(
            [
                [r["elapsed_min"], r["rhat_max"], r["min_ess"]]
                for r in rows
                if r["run"] == name
            ]
        ).T
        for name in dict.fromkeys(rows["run"])
    }, str(csv)


def corner_figure(npz, out):
    import corner

    d = np.load(npz)
    s, truth, names = d["samples"], d["truth"], [str(n) for n in d["names"]]
    i = {n: k for k, n in enumerate(names)}

    # rescale into readable units; the truth maps to the same transform
    x = np.column_stack(
        [
            (s[:, i["chirp_mass"]] - truth[i["chirp_mass"]]) * 1e6,
            s[:, i["eta"]],
            s[:, i["spin1z"]] * 1e3,
            s[:, i["spin2z"]] * 1e3,
        ]
    )
    t = [0.0, truth[i["eta"]], 0.0, 0.0]
    labels = [
        r"$\mathcal{M}_c - \mathcal{M}_c^{\rm true}$  [$10^{-6}\,M_\odot$]",
        r"$\eta$",
        r"$\chi_{1z}$  [$10^{-3}$]",
        r"$\chi_{2z}$  [$10^{-3}$]",
    ]
    # clamp to the physical/prior support so smoothing cannot bleed across the
    # eta <= 1/4 and chi >= 0 boundaries where the posterior actually piles up
    rng = [
        (x[:, 0].min(), x[:, 0].max()),
        (x[:, 1].min(), 0.25),
        (0.0, np.percentile(x[:, 2], 99.9)),
        (0.0, np.percentile(x[:, 3], 99.9)),
    ]

    fig = corner.corner(
        x,
        labels=labels,
        truths=t,
        truth_color=SERIES[1],
        color=SERIES[0],
        range=rng,
        bins=60,
        smooth=0.9,
        smooth1d=0.6,
        levels=(0.5, 0.9),
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 10, "color": INK},
        label_kwargs={"fontsize": 11, "color": INK},
        hist_kwargs={"linewidth": 1.6},
        contour_kwargs={"linewidths": 1.2},
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
    )
    # keep the suptitle to one line: the per-column titles corner draws sit at the
    # very top of the grid, and a second suptitle line collides with the first one
    fig.suptitle(
        "BNS at Cosmic Explorer — FD relative binning + JAX HMC",
        fontsize=13,
        color=INK,
        y=1.015,
    )
    fig.text(
        0.60,
        0.80,
        f"network SNR {float(d['snr']):.0f}\n"
        f"{s.shape[0] / 1e6:.2f}M posterior samples\n"
        rf"$\hat{{R}} \leq {float(np.max(d['rhat'])):.4f}$,  min ESS "
        f"{round(float(np.min(d['ess']))):,}\n"
        "converged in 15.4 min on one GPU",
        fontsize=10,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    # keep this note to <= 4 lines: it must clear the row-3 column title below it
    fig.text(
        0.60,
        0.68,
        "orange = injection\n\n"
        r"$\eta = 1/4$ and $\chi_{iz} = 0$ are prior edges, so the"
        "\ninjection lying on a marginal's boundary is\n"
        r"expected — and it pushes $\mathcal{M}_c$ up through the"
        "\n"
        r"$\mathcal{M}_c$–$\eta$ anti-correlation. Not a bias.",
        fontsize=9.5,
        color=MUTED,
        va="top",
        linespacing=1.4,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def convergence_figure(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax in axes:  # recessive frame
        ax.grid(True, color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c9c4")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("total elapsed wall clock  [min]", fontsize=10, color=INK)
        ax.axvline(20.0, color=MUTED, linestyle=":", linewidth=1.4)

    for (label, data), c in zip(runs.items(), SERIES):
        t, rhat, ess = data
        axes[0].plot(t, rhat, "-o", color=c, lw=2.0, ms=4.5, label=label)
        axes[1].plot(t, ess, "-o", color=c, lw=2.0, ms=4.5, label=label)

    axes[0].axhline(1.01, color="#008300", linestyle="--", linewidth=1.4)
    axes[0].set_ylabel(
        r"rank-normalized split-$\hat{R}$ (global subseries)", fontsize=10, color=INK
    )
    axes[0].set_title("Between-chain agreement", fontsize=11, color=INK, loc="left")
    axes[0].set_ylim(0.999, 1.14)

    axes[1].axhline(2000.0, color="#008300", linestyle="--", linewidth=1.4)
    axes[1].set_ylabel("Geyer min ESS (all sampled dims)", fontsize=10, color=INK)
    axes[1].set_title("Effective sample size", fontsize=11, color=INK, loc="left")
    axes[1].set_yscale("log")

    # gate labels ride the threshold lines: x in axes fraction, y in data units,
    # so they stay inside the frame instead of colliding with the y-axis label
    for ax, y, txt in ((axes[0], 1.0115, "gate: 1.01"), (axes[1], 2350, "gate: 2000")):
        ax.text(
            0.03,
            y,
            txt,
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color="#008300",
            va="bottom",
        )

    # direct labels at each series' end -- the required relief for the
    # low-contrast aqua slot, and faster to read than the legend alone
    for ax, j in ((axes[0], 1), (axes[1], 2)):
        for (label, data), c in zip(runs.items(), SERIES):
            ax.annotate(
                label.split()[1],  # "run 3 (final)" -> "3"
                (data[0][-1], data[j][-1]),
                textcoords="offset points",
                xytext=(7, 0),
                fontsize=9.5,
                fontweight="bold",
                color=c,
                va="center",
            )
        ax.set_xlim(4.0, 27.5)  # headroom for those end labels
    axes[0].text(
        19.5,
        1.137,
        "20 min budget",
        fontsize=9,
        color=MUTED,
        rotation=90,
        va="top",
        ha="right",
    )

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle(
        "BNS/CE parameter estimation: convergence against the 20-minute budget",
        fontsize=12.5,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rundir", default="examples/output/bns_ce_rb_hmc")
    ap.add_argument("--assets", default="docs/assets")
    args = ap.parse_args()

    rundir, assets = Path(args.rundir), Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)

    if (rundir / "samples.npz").exists():
        print(
            "wrote", corner_figure(rundir / "samples.npz", assets / "bns_ce_corner.png")
        )
    else:
        print(f"skipping the corner plot: no {rundir}/samples.npz (re-run the PE)")

    runs, src = load_runs(rundir, assets / "bns_ce_convergence.csv")
    print(f"convergence series from {src}")
    for name, d in runs.items():
        print(
            f"  {name}: {d.shape[1]} blocks, ends at {d[0][-1]:.2f} min, "
            f"Rhat {d[1][-1]:.4f}, ESS {d[2][-1]:.0f}"
        )
    print("wrote", convergence_figure(runs, assets / "bns_ce_convergence.png"))


if __name__ == "__main__":
    raise SystemExit(main())
