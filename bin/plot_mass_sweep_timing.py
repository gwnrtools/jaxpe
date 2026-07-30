#!/usr/bin/env python
r"""Plot completion time vs total mass from a run_mass_sweep_pe.py summary.

Reads ``sweep_summary.csv`` (written by ``bin/run_mass_sweep_pe.py``) and plots
total wall-clock time to convergence against total mass, log-x (the sweep spans
more than an order of magnitude in mass). Non-converged or crashed injections
(``converged`` false, or a missing ``total``) are marked distinctly rather than
silently dropped, so a partial sweep is visibly partial.

Run: python bin/plot_mass_sweep_timing.py examples/output/mass_sweep_pe/sweep_summary.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary_csv")
    ap.add_argument("--out", default=None, help="default: alongside the input CSV")
    args = ap.parse_args()

    csv_path = Path(args.summary_csv)
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: float(r["total_mass"]))

    mtot = [float(r["total_mass"]) for r in rows]
    total_min = [float(r["total"]) / 60.0 if r["total"] else float("nan") for r in rows]
    converged = [r["converged"] == "True" for r in rows]
    snr = [float(r["achieved_snr"]) if r["achieved_snr"] else float("nan") for r in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = [c for c in converged]
    ax.plot(
        [m for m, c in zip(mtot, ok) if c],
        [t for t, c in zip(total_min, ok) if c],
        "o-", color="#2a78d6", ms=7, lw=1.5, zorder=3, label="converged",
    )
    bad_m = [m for m, c in zip(mtot, ok) if not c]
    bad_t = [t for t, c in zip(total_min, ok) if not c]
    if bad_m:
        ax.plot(bad_m, bad_t, "x", color="#c0392b", ms=10, mew=2, zorder=4,
                 label="NOT converged / failed")

    for m, t, s in zip(mtot, total_min, snr):
        if t == t:  # not nan
            ax.annotate(f"SNR {s:.0f}", (m, t), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="#555555")

    ax.set_xscale("log")
    ax.set_xlabel(r"Total mass $M_1+M_2$ [$M_\odot$]")
    ax.set_ylabel("Wall-clock time to convergence [min]")
    ax.set_title("BNS -> BBH mass sweep: PE completion time vs total mass")
    ymin, ymax = min(t for t in total_min if t == t), max(t for t in total_min if t == t)
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + 2.2 * pad)
    ax.grid(True, which="both", alpha=0.3)
    if bad_m:
        ax.legend(frameon=False)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else csv_path.with_name("completion_time_vs_mass.png")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
