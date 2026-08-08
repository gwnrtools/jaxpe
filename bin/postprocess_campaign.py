#!/usr/bin/env python3
"""General-purpose post-processing for a suite of jaxpe PE runs.

Not tied to any one campaign: given a results directory laid out as
``<results_dir>/<variant>/<run_id>/``, where each ``<run_id>`` is the ``--outdir``
of one ``jaxpe run-pe`` + ``jaxpe process-samples`` invocation (so it contains
``injection.json``, ``run_config.json``, ``posterior_samples.npy``,
``raw_samples.npz``), this script auto-discovers every variant and run under that
directory and produces one self-contained report:

1. **Corner plots for every run.** Reuses each run's own ``corner_thinned.png`` if
   present; regenerates it from ``posterior_samples.npy`` (via
   ``jaxpe.diagnostics.plots.corner_plot``, the same function ``jaxpe
   process-samples`` uses) for any run where it is missing -- e.g. because that
   run's own corner-plot step failed but its samples are fine. A small WebP
   thumbnail is generated per run for the gallery (efficient storage: the report
   never duplicates the full-resolution images, only small thumbnails plus
   relative links to the originals already on disk).
2. **One PP (probability-probability) plot per variant**, testing calibration of
   that variant's credible intervals against its population of injected truths --
   valid whenever the injections were drawn from the same prior used for recovery
   (``injection.parameters: "prior"`` in the run configuration). Handles both
   unweighted (HMC/MALA, uniform weight) and weighted (NS/GPry) posteriors.
3. **A static index.html** linking everything, viewable locally by opening the
   file directly in a browser (no server needed) -- ``file://.../index.html``.

Designed to be reused for any future suite of per-injection jaxpe runs organized
the same way; nothing here is specific to particular variant names, parameter
names, or run counts -- all of that is discovered from the directory layout and
each run's own ``run_config.json``.

Usage
-----
    python bin/postprocess_campaign.py --results-dir campaign/results \\
        --outdir campaign/report

    # Restrict to specific variants / skip auto-discovery:
    python bin/postprocess_campaign.py --results-dir campaign/results \\
        --outdir campaign/report --variants phenomd_hmc phenomd_gpry esigma_gpry
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_variants(results_dir: Path, requested=None):
    if requested:
        return [results_dir / v for v in requested]
    return sorted(d for d in results_dir.iterdir() if d.is_dir())


def discover_runs(variant_dir: Path):
    """Every immediate subdirectory of ``variant_dir`` that looks like a finished
    run (has an injection.json and a posterior_samples.npy), sorted numerically
    where the directory name is an integer (the run/injection index), lexically
    otherwise."""
    runs = []
    for d in sorted(variant_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "injection.json").exists() and (d / "posterior_samples.npy").exists():
            runs.append(d)
    runs.sort(key=lambda d: (0, int(d.name)) if d.name.isdigit() else (1, d.name))
    return runs


def load_run(run_dir: Path):
    injection = json.loads((run_dir / "injection.json").read_text())
    injection.pop("metadata", None)
    run_config = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        run_config = json.loads(cfg_path.read_text())
    samples = np.load(run_dir / "posterior_samples.npy")
    names = run_config.get("param_names") or [f"x_{i}" for i in range(samples.shape[-1])]

    weights = None
    if run_config.get("weighted"):
        npz_path = run_dir / "raw_samples.npz"
        if npz_path.exists():
            with np.load(npz_path) as npz:
                if "weights" in npz.files:
                    weights = np.asarray(npz["weights"], dtype=float).ravel()
    return dict(
        run_dir=run_dir,
        injection=injection,
        run_config=run_config,
        samples=samples,
        weights=weights,
        names=names,
    )


# --------------------------------------------------------------------------- #
# PP-plot statistics
# --------------------------------------------------------------------------- #


def credible_level(samples_1d: np.ndarray, weights, truth: float) -> float:
    """Fraction of posterior mass at or below ``truth`` -- the PP-plot rank
    statistic. For a calibrated sampler and injections drawn from the recovery
    prior, this is Uniform(0, 1)-distributed across a population of runs."""
    w = np.ones_like(samples_1d) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(w[samples_1d <= truth].sum())


def pp_confidence_bands(n: int, levels=(0.68, 0.95, 0.997)):
    """(level, x, lower, upper) bands from Beta-distributed order statistics --
    the standard PP-plot credible envelope (e.g. bilby's make_pp_plot)."""
    from scipy.stats import beta as beta_dist

    k = np.arange(1, n + 1)
    x = k / (n + 1)
    bands = []
    for level in levels:
        lower = beta_dist.ppf((1 - level) / 2, k, n - k + 1)
        upper = beta_dist.ppf((1 + level) / 2, k, n - k + 1)
        bands.append((level, x, lower, upper))
    return bands


def make_pp_plot(variant_name: str, percentiles_by_param: dict, out_path: Path):
    """One combined PP plot for a variant: every parameter's empirical CDF of
    credible levels overlaid, plus the Beta-order-statistic confidence bands and
    a per-parameter KS-test p-value against Uniform(0,1) in the legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kstest

    fig, ax = plt.subplots(figsize=(6, 6))
    any_n = 0
    for level, x, lower, upper in pp_confidence_bands(
        max((len(v) for v in percentiles_by_param.values()), default=1)
    ):
        ax.fill_between(x, lower, upper, color="gray", alpha=0.12, lw=0)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")

    for name, percentiles in percentiles_by_param.items():
        p = np.sort(np.asarray(percentiles))
        n = len(p)
        any_n = max(any_n, n)
        if n == 0:
            continue
        y = np.arange(1, n + 1) / (n + 1)
        ks_p = kstest(p, "uniform").pvalue if n >= 2 else float("nan")
        ax.step(
            np.concatenate([[0.0], p, [1.0]]),
            np.concatenate([[0.0], y, [1.0]]),
            where="post",
            label=f"{name} (n={n}, KS p={ks_p:.2f})",
        )

    ax.set_xlabel("Credible level of injected truth")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"PP plot: {variant_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return any_n


# --------------------------------------------------------------------------- #
# Corner plots + thumbnails
# --------------------------------------------------------------------------- #


def ensure_corner_plot(run: dict) -> Path | None:
    """Return the path to this run's corner plot, regenerating it from
    posterior_samples.npy if the per-run process-samples step didn't leave one
    (e.g. it failed on this particular run but the samples themselves are fine).
    """
    existing = run["run_dir"] / "corner_thinned.png"
    if existing.exists():
        return existing
    try:
        from jaxpe.diagnostics.plots import corner_plot
    except ImportError:
        return None
    try:
        truths = [run["injection"].get(n) for n in run["names"]]
        kwargs = {}
        if run["weights"] is not None:
            kwargs["weights"] = run["weights"]
        fig = corner_plot(run["samples"], names=run["names"], truths=truths, **kwargs)
        fig.savefig(existing, dpi=120)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return existing
    except Exception as e:  # pragma: no cover - best-effort regeneration
        print(f"  WARNING: could not regenerate corner plot for {run['run_dir']}: {e}")
        return None


def make_thumbnail(src: Path, dest: Path, size=(320, 320)) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail(size)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, "WEBP", quality=80)
        return True
    except Exception as e:  # pragma: no cover
        print(f"  WARNING: could not thumbnail {src}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


def build_report(results_dir: Path, outdir: Path, variants):
    outdir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = outdir / "thumbnails"
    pp_dir = outdir / "pp_plots"
    pp_dir.mkdir(parents=True, exist_ok=True)

    variant_sections = []
    for variant_dir in discover_variants(results_dir, variants):
        if not variant_dir.is_dir():
            print(f"WARNING: variant directory {variant_dir} not found, skipping")
            continue
        variant = variant_dir.name
        run_dirs = discover_runs(variant_dir)
        print(f"{variant}: {len(run_dirs)} completed run(s) found")

        runs, gallery_items = [], []
        percentiles_by_param: dict[str, list] = {}
        for run_dir in run_dirs:
            try:
                run = load_run(run_dir)
            except Exception as e:
                print(f"  WARNING: failed to load {run_dir}: {e}")
                continue
            runs.append(run)

            for i, name in enumerate(run["names"]):
                truth = run["injection"].get(name)
                if truth is None:
                    continue
                level = credible_level(run["samples"][:, i], run["weights"], float(truth))
                percentiles_by_param.setdefault(name, []).append(level)

            corner_path = ensure_corner_plot(run)
            thumb_rel = None
            if corner_path is not None:
                thumb_path = thumbs_dir / f"{variant}_{run_dir.name}.webp"
                if make_thumbnail(corner_path, thumb_path):
                    thumb_rel = relpath(thumb_path, outdir)
            gallery_items.append(
                dict(
                    run_id=run_dir.name,
                    thumb=thumb_rel,
                    full=relpath(corner_path, outdir) if corner_path else None,
                )
            )

        pp_path = None
        if percentiles_by_param:
            pp_file = pp_dir / f"{variant}_pp.png"
            make_pp_plot(variant, percentiles_by_param, pp_file)
            pp_path = relpath(pp_file, outdir)

        variant_sections.append(
            dict(
                name=variant,
                n_runs=len(runs),
                pp_plot=pp_path,
                gallery=gallery_items,
            )
        )

    write_index_html(outdir, results_dir, variant_sections)
    return variant_sections


def write_index_html(outdir: Path, results_dir: Path, sections):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>jaxpe PE campaign report</title>",
        "<style>",
        "body{font-family:sans-serif;margin:2em;background:#111;color:#eee}",
        "h1{margin-bottom:0}",
        ".sub{color:#999;margin-top:0}",
        "h2{border-bottom:1px solid #444;padding-bottom:.3em;margin-top:2.5em}",
        ".pp{max-width:520px;display:block;margin:1em 0}",
        ".gallery{display:flex;flex-wrap:wrap;gap:8px}",
        ".gallery a{display:block}",
        ".gallery img{width:160px;height:160px;object-fit:cover;border:1px solid #444;"
        "border-radius:4px}",
        ".gallery .label{font-size:11px;color:#aaa;text-align:center}",
        "</style></head><body>",
        "<h1>jaxpe PE campaign report</h1>",
        f"<p class='sub'>results dir: {html.escape(str(results_dir))}</p>",
    ]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec['name'])} ({sec['n_runs']} runs)</h2>")
        if sec["pp_plot"]:
            parts.append(
                f"<a href='{html.escape(sec['pp_plot'])}'>"
                f"<img class='pp' src='{html.escape(sec['pp_plot'])}'></a>"
            )
        else:
            parts.append("<p><em>No PP plot (no runs with a matching injection truth).</em></p>")
        parts.append("<div class='gallery'>")
        for item in sec["gallery"]:
            if item["full"] is None:
                continue
            img = item["thumb"] or item["full"]
            parts.append(
                f"<a href='{html.escape(item['full'])}'>"
                f"<img loading='lazy' src='{html.escape(img)}'>"
                f"<div class='label'>{html.escape(item['run_id'])}</div></a>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(parts))
    print(f"Wrote {outdir / 'index.html'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="<results_dir>/<variant>/<run_id>/ layout")
    p.add_argument("--outdir", required=True, help="Where to write the report")
    p.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Restrict to these variant subdirectory names (default: auto-discover all)",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    outdir = Path(args.outdir).resolve()
    build_report(results_dir, outdir, args.variants)


if __name__ == "__main__":
    main()
