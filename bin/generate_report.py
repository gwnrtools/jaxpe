import json
import shutil
from pathlib import Path

events = ["GW150914", "GW170729", "GW170104", "GW190412", "GW190521"]
base_dir = Path("output/production_events")
artifact_dir = Path(
    "/home/prayush/.gemini/antigravity-ide/brain/9b5855a2-5f93-4c50-81cc-aa4c5155177f"
)

markdown = """# jaxpe Parameter Estimation Suite
## Frequency-Domain IMRPhenomD Injection Recovery

This document summarizes the results of running the fully autonomous `jaxpe` sampler suite across 5 synthetic gravitational-wave injections based on parameters of prominent GWTC events.

The inference pipeline utilized a zero-noise realization with advanced LIGO Zero-detuning high-power curves, integrated with our global-local normalizing flow Metropolis-Hastings kernel.

### Recovery Summary

| Event | Network SNR | Sampling Time (s) | Samples | Chirp Mass (M⊙) | Mass Ratio (q) | Distance (Mpc) |
|---|---|---|---|---|---|---|
"""

for ev in events:
    sum_file = base_dir / ev / "summary.json"
    if not sum_file.exists():
        continue
    with open(sum_file, "r") as f:
        data = json.load(f)

    snr_net = (
        data["snr"]["H1"] ** 2 + data["snr"]["L1"] ** 2 + data["snr"]["V1"] ** 2
    ) ** 0.5
    t_samp = data["timings"]["sampling_time_s"]
    n_samp = data["n_samples"]

    rec = data["recovery"]

    # Check for thinned posterior samples
    post_file = base_dir / ev / "posterior_samples.npy"
    if post_file.exists():
        import numpy as np

        phys = np.load(post_file)
        n_samp_str = f"{phys.shape[0]:,} (thinned)"
        mc_med = np.median(phys[:, 0])
        q_med = np.median(phys[:, 1])
        dl_med = np.median(phys[:, 4])

        mc = f"{mc_med:.1f} ({rec['chirp_mass']['true']:.1f})"
        q = f"{q_med:.2f} ({rec['mass_ratio']['true']:.2f})"
        dl = f"{dl_med:.0f} ({rec['luminosity_distance']['true']:.0f})"
    else:
        n_samp_str = f"{n_samp:,} (raw)"
        mc = f"{rec['chirp_mass']['median']:.1f} ({rec['chirp_mass']['true']:.1f})"
        q = f"{rec['mass_ratio']['median']:.2f} ({rec['mass_ratio']['true']:.2f})"
        dl = f"{rec['luminosity_distance']['median']:.0f} ({rec['luminosity_distance']['true']:.0f})"

    markdown += f"| **{ev}** | {snr_net:.1f} | {t_samp:.1f} | {n_samp_str} | {mc} | {q} | {dl} |\n"

markdown += """
*(Values in parentheses denote the true injected values)*

---

### Posterior Corner Plots

"""

for ev in events:
    png_path_thinned = base_dir / ev / "corner_thinned.png"
    png_path_raw = base_dir / ev / "corner.png"

    png_path = png_path_thinned if png_path_thinned.exists() else png_path_raw
    if png_path.exists():
        # Copy to artifacts directory
        artifact_png = artifact_dir / f"{ev}_{png_path.name}"
        shutil.copy(png_path, artifact_png)

        title = f"{ev} (Thinned)" if png_path == png_path_thinned else f"{ev} (Raw)"
        markdown += f"#### {title}\n"
        markdown += f"![{title} Posterior Corner Plot]({artifact_png.absolute()})\n\n"


# Write to repo docs
out_file = Path("docs/under_construction_fd.md")
out_file.parent.mkdir(exist_ok=True)
with open(out_file, "w") as f:
    f.write(markdown)

# Write as artifact
artifact_md = artifact_dir / "under_construction_fd.md"
with open(artifact_md, "w") as f:
    f.write(markdown)

print(f"Successfully generated {out_file} and {artifact_md}")
