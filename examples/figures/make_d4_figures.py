"""Regenerate the §D4-checkpoint figures for docs/gpry_fusion_design.md and
examples/d4_timing_report.html, from the committed measurement artifacts.

Inputs (committed):
  examples/output/phenomd_eob_call_timing.json   -- real EOB per-call timing (Rec 2)
  examples/output/phenomd_M*_{gradient,surrogate}_*.npz -- PE sweep runs
Outputs:
  examples/figures/fig_eob_timing.png
  examples/figures/fig_pe_scaling.png
  examples/figures/fig_posteriors.png       -- 1-D marginals, every run vs truth (overview)
  examples/figures/fig_corner_M<mass>.png   -- one 2-D corner per total mass, all routes

Run:  conda run -n lalsuite-dev python examples/figures/make_d4_figures.py
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "output"))  # examples/output
FIG = HERE  # examples/figures

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 140, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

# GPry per-eval overhead band (ms): 1.44 s at M80 .. 2.64 s at M20 (from the npz).
GPRY_LO, GPRY_HI = 1440.0, 2640.0

# Eccentric SEOBNRv5EHM (e=0.1), measured 2026-07-17 (pyseobnr, same (4,4) modes / fs
# as the aligned run; warmup-excluded median). Not in the harness json (eccentric needs
# extra params); recorded here for the figure with provenance.
ECC_M = np.array([40.0, 20.0, 10.0, 6.0, 4.0])
ECC_MS = np.array([185.9, 495.4, 1522.3, 3725.3, 8573.7])


def load_pe():
    pe = {}
    for f in glob.glob(os.path.join(OUT, "phenomd_M*_*.npz")):
        m = re.match(
            r"phenomd_M(\d+)_(gradient_cpu|gradient_gpu|surrogate_cpu)\.npz",
            os.path.basename(f),
        )
        if not m:
            continue
        mass, route = int(m.group(1)), m.group(2)
        d = np.load(f, allow_pickle=True)
        k = set(d.files)

        def g(key, dv=np.nan):
            return float(d[key]) if key in k else dv

        pe.setdefault(mass, {})[route] = dict(
            wall=g("wall_seconds"), evals=g("likelihood_evaluations"),
            dur=g("duration"), wave=g("waveform_seconds"), gp=g("gp_seconds"),
            ncalls=g("n_waveform_calls"),
        )
    return pe


def eob_series(models, model):
    ms, ts = [], []
    for v in models[model].values():
        if "seconds" in v:
            ms.append(v["total_mass"])
            ts.append(v["seconds"] * 1e3)
    o = np.argsort(ms)
    return np.array(ms)[o], np.array(ts)[o]


def fig_eob():
    models = json.load(open(os.path.join(OUT, "phenomd_eob_call_timing.json")))["models"]
    styles = {
        "TEOBResumS": ("#0072B2", "o", "-"), "SEOBNRv5HM": ("#009E73", "s", "-"),
        "SEOBNRv4_opt": ("#56B4E9", "^", "-"), "SEOBNRv5PHM": ("#CC79A7", "D", "-"),
        "SEOBNRv4": ("#E69F00", "v", "--"), "SEOBNRv4HM": ("#D55E00", "P", "--"),
    }
    order = ["TEOBResumS", "SEOBNRv5HM", "SEOBNRv4_opt", "SEOBNRv5PHM",
             "SEOBNRv4", "SEOBNRv4HM"]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axhspan(GPRY_LO, GPRY_HI, color="0.6", alpha=0.30, zorder=0)
    ax.text(2.75, GPRY_HI * 1.02, "GPry overhead per eval (1.4-2.6 s)", fontsize=9,
            color="0.25", va="bottom")
    for name in order:
        c, mk, ls = styles[name]
        mm, tt = eob_series(models, name)
        ax.plot(mm, tt, ls, marker=mk, color=c, label=name, markersize=5.5, lw=1.6)
    ax.plot(ECC_M, ECC_MS, ":", marker="X", color="#111111", markersize=6.5, lw=1.6,
            label="SEOBNRv5EHM (e=0.1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total mass  $M_{\\rm tot}$  [$M_\\odot$]   (<- longer signal, higher fs)")
    ax.set_ylabel("waveform generation time per call  [ms]")
    ax.set_title("Real EOB per-call cost vs total mass\n"
                 "(fs from the (4,4) ringdown; HM capped at (l,m)<=(4,4); warmup-excluded)")
    ax.set_xticks([2.8, 4, 6, 10, 20, 40, 80])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.axvspan(2.5, 5, color="#E69F00", alpha=0.06, zorder=0)
    ax.text(3.5, 6e4, "BNS /\nlow mass", fontsize=8, ha="center", color="0.4")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.set_xlim(2.6, 95)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_eob_timing.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def fig_pe():
    pe = load_pe()
    A = sorted(m for m in pe if pe[m].get("gradient_cpu", {}).get("evals", 0) > 5000)
    B = sorted(pe)
    acpu = np.array([pe[m]["gradient_cpu"]["wall"] for m in A])
    agpu = np.array([pe[m]["gradient_gpu"]["wall"] for m in A])
    bwall = np.array([pe[m]["surrogate_cpu"]["wall"] for m in B])
    bwave = np.array([pe[m]["surrogate_cpu"]["wave"] for m in B])
    bgp = np.array([pe[m]["surrogate_cpu"]["gp"] for m in B])
    bncall = np.array([pe[m]["surrogate_cpu"]["ncalls"] for m in B])
    gp_per_eval = bgp / bncall

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axs[0, 0]
    ax.plot(A, acpu, "o-", color="#0072B2", label="Route A - gradient (CPU)", markersize=6)
    ax.plot(A, agpu, "s-", color="#009E73", label="Route A - gradient (GPU)", markersize=6)
    ax.plot(B, bwall, "D-", color="#D55E00", label="Route B - GPry surrogate (CPU)",
            markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("total mass  $M_{\\rm tot}$  [$M_\\odot$]")
    ax.set_ylabel("PE wall time (exec, compile-excluded)  [s]")
    ax.set_title("(a) End-to-end PE cost per run vs mass")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    for m, y in zip(A, acpu):
        ax.annotate(f"{pe[m]['gradient_cpu']['dur']:.0f}s", (m, y),
                    textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center",
                    color="0.4")

    ax = axs[0, 1]
    speed = acpu / agpu
    ax.plot(A, speed, "o-", color="#009E73", markersize=7)
    for m, s in zip(A, speed):
        ax.annotate(f"{s:.1f}x", (m, s), textcoords="offset points", xytext=(0, 8),
                    fontsize=9, ha="center")
    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    ax.set_xlabel("total mass  $M_{\\rm tot}$  [$M_\\odot$]")
    ax.set_ylabel("Route A speedup  CPU / GPU")
    ax.set_title("(b) GPU advantage grows toward longer signals")
    ax.invert_xaxis()
    ax.set_ylim(0, 4.3)

    ax = axs[1, 0]
    ax.plot(B, bgp, "D-", color="#7d3c98", label="GPry (GP fit + acquisition)", markersize=6)
    ax.plot(B, bwave, "o-", color="#c0392b", label="waveform generation (JAX PhenomD)",
            markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("total mass  $M_{\\rm tot}$  [$M_\\odot$]")
    ax.set_ylabel("wall time  [s]")
    ax.set_title("(c) Route B: GPry overhead dominates the waveform ~100x")
    ax.invert_xaxis()
    ax.legend(frameon=False)

    ax = axs[1, 1]
    ax.plot(B, gp_per_eval, "D-", color="#7d3c98", markersize=6)
    ax.set_xlabel("total mass  $M_{\\rm tot}$  [$M_\\odot$]")
    ax.set_ylabel("GPry cost per eval  [s]", color="#7d3c98")
    ax.tick_params(axis="y", labelcolor="#7d3c98")
    ax.set_title("(d) GPry per-eval cost tracks eval count, not duration")
    ax.invert_xaxis()
    ax.set_ylim(0, 3)
    ax2 = ax.twinx()
    ax2.bar(B, bncall, width=3.0, color="0.75", alpha=0.5, zorder=0)
    ax2.set_ylabel("GPry truth evaluations", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.5")
    ax2.grid(False)

    fig.suptitle("PhenomD PE scaling with total mass - three routes "
                 "(matched SNR~15, 4-D intrinsic)", fontsize=13, y=1.00)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_pe_scaling.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def _load_run(mass, route):
    p = os.path.join(OUT, f"phenomd_M{mass}_{route}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    w = d["weights"].astype(float)
    return dict(x=d["samples"].astype(float), w=w / w.sum(),
                names=[str(n) for n in d["names"]], truth=d["truth"].astype(float),
                dur=float(d["duration"]) if "duration" in d.files else np.nan,
                evals=float(d["likelihood_evaluations"])
                if "likelihood_evaluations" in d.files else np.nan)


def fig_posteriors():
    """Every posterior from every run: masses (rows) x the 4 intrinsic parameters
    (columns), all routes overlaid vs the injected truth. Peak-normalized 1-D marginals."""
    masses = [80, 70, 60, 50, 40, 30, 20, 10]
    routes = [("gradient_cpu", "#1f6fb2", "Route A – gradient (CPU)"),
              ("gradient_gpu", "#0f8f80", "Route A – gradient (GPU)"),
              ("surrogate_cpu", "#c0392b", "Route B – GPry surrogate")]
    labels = [r"chirp mass $\mathcal{M}$", "mass ratio $q$",
              r"$\chi_{1z}$", r"$\chi_{2z}$"]
    nrow, ncol = len(masses), 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(11, 15.2))

    for r, mass in enumerate(masses):
        runs = {rt: _load_run(mass, rt) for rt, _, _ in routes}
        ref = next((v for v in runs.values() if v is not None), None)
        if ref is None:
            continue
        truth = ref["truth"]
        # is Route A a converged full run here? (8350 steps vs the 1400-step benchmark)
        a = runs.get("gradient_cpu")
        a_short = a is not None and a["evals"] < 5000
        for c in range(ncol):
            ax = axs[r, c]
            # common x-range from the routes present
            lo, hi = np.inf, -np.inf
            for rt, _, _ in routes:
                v = runs[rt]
                if v is None:
                    continue
                xs, ws = v["x"][:, c], v["w"]
                mu = np.average(xs, weights=ws)
                sd = np.sqrt(np.average((xs - mu) ** 2, weights=ws))
                lo, hi = min(lo, mu - 4 * sd), max(hi, mu + 4 * sd)
            lo, hi = min(lo, truth[c]), max(hi, truth[c])
            pad = 0.08 * (hi - lo + 1e-9)
            grid = np.linspace(lo - pad, hi + pad, 300)
            for rt, col, _ in routes:
                v = runs[rt]
                if v is None:
                    continue
                xs, ws = v["x"][:, c], v["w"]
                if xs.std() < 1e-9:
                    continue
                try:
                    dens = gaussian_kde(xs, weights=ws)(grid)
                except Exception:
                    continue
                dens = dens / dens.max()
                dashed = (rt == "gradient_cpu" or rt == "gradient_gpu") and a_short
                ax.plot(grid, dens, color=col, lw=1.5,
                        ls="--" if dashed else "-",
                        alpha=0.55 if dashed else 0.95)
                ax.fill_between(grid, dens, color=col, alpha=0.05)
            ax.axvline(truth[c], color="#222", ls=(0, (4, 2)), lw=1.1, zorder=5)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=8)
            for s in ("top", "left", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(labels[c], fontsize=11)
            if c == 0:
                tag = "  (A: 1.4k*)" if a_short else ""
                only_b = all(runs[rt] is None for rt, _, _ in routes[:2])
                tag = "  (B only)" if only_b else tag
                ax.set_ylabel(f"$M_{{\\rm tot}}={mass}$\n{ref['dur']:.0f} s{tag}",
                              rotation=0, ha="right", va="center", fontsize=9,
                              labelpad=28)
    handles = [Line2D([0], [0], color=col, lw=2, label=lab) for _, col, lab in routes]
    handles.append(Line2D([0], [0], color="#222", ls=(0, (4, 2)), lw=1.2,
                          label="injected truth"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Recovered intrinsic posteriors — every run vs truth "
                 "(peak-normalized 1-D marginals; matched SNR≈15)",
                 fontsize=12.5, y=1.008)
    fig.text(0.5, -0.004,
             "* M70/M50/M30 Route A are short 1400-step benchmarks (dashed, less "
             "converged); all other A runs are 8350-step converged; Route B is "
             "GPry-converged throughout.", ha="center", fontsize=8, color="0.4")
    fig.tight_layout(rect=[0.02, 0.0, 1, 0.985])
    p = os.path.join(FIG, "fig_posteriors.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def _wquantile(x, w, q):
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(q, c, xs)


def fig_corners():
    """One 2-D corner plot per total mass, with every run at that mass overlaid
    (Route A CPU/GPU + Route B) against the injected truth. Weighted 50%/90% credible
    contours on a common range so the routes are directly comparable."""
    import corner  # optional dep (env: lalsuite-dev)

    masses = [80, 70, 60, 50, 40, 30, 20, 10]
    routes = [("gradient_cpu", "#1f6fb2", "Route A – gradient (CPU)"),
              ("gradient_gpu", "#0f8f80", "Route A – gradient (GPU)"),
              ("surrogate_cpu", "#c0392b", "Route B – GPry surrogate")]
    labels = [r"$\mathcal{M}\,[M_\odot]$", "$q$", r"$\chi_{1z}$", r"$\chi_{2z}$"]

    for mass in masses:
        present = [(rt, col, lab, _load_run(mass, rt))
                   for rt, col, lab in routes]
        present = [(rt, col, lab, v) for rt, col, lab, v in present if v is not None]
        if not present:
            continue
        truth = present[0][3]["truth"]
        dur = present[0][3]["dur"]
        ndim = len(labels)
        rng = []
        for d in range(ndim):
            los, his = [], []
            for _, _, _, v in present:
                lo, hi = _wquantile(v["x"][:, d], v["w"], [0.005, 0.995])
                los.append(lo)
                his.append(hi)
            lo, hi = min(los + [truth[d]]), max(his + [truth[d]])
            pad = 0.10 * (hi - lo + 1e-9)
            rng.append((lo - pad, hi + pad))

        fig = None
        a_short = False
        for rt, col, _, v in present:
            if rt in ("gradient_cpu", "gradient_gpu") and v["evals"] < 5000:
                a_short = True
            fig = corner.corner(
                v["x"], weights=v["w"], labels=labels, range=rng, bins=32,
                color=col, smooth=1.0, levels=(0.5, 0.9),
                plot_datapoints=False, plot_density=False, fill_contours=True,
                contour_kwargs=dict(linewidths=1.1),
                contourf_kwargs=dict(alpha=0.28),
                hist_kwargs=dict(density=True, lw=1.5),
                truths=truth if fig is None else None, truth_color="#222",
                labelpad=0.08, fig=fig,
            )
        handles = [Line2D([0], [0], color=col, lw=2.4, label=lab)
                   for _, col, lab, _ in present]
        handles.append(Line2D([0], [0], color="#222", ls=(0, (4, 2)), lw=1.3,
                              label="injected truth"))
        note = "  (Route A: short 1.4k-step benchmark)" if a_short else ""
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=11)
        fig.suptitle(
            f"$M_{{\\rm tot}} = {mass}\\,M_\\odot$   ·   {dur:.0f} s   ·   SNR $\\approx$ 15"
            f"   ·   4-D intrinsic posterior{note}",
            fontsize=13, y=1.02)
        p = os.path.join(FIG, f"fig_corner_M{mass}.png")
        fig.savefig(p, bbox_inches="tight", dpi=130)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    fig_eob()
    fig_pe()
    fig_posteriors()
    fig_corners()
