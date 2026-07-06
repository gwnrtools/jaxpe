"""PE on real GWOSC data: GW150914 in H1+L1 (requires network access; gwpy/gwosc).

Strain is downloaded once (gwpy caches it), PSDs are Welch-median-estimated from
off-source data preceding the event, and the same global-local sampler runs on GPU.

Caveat: ToyChirp is a pipeline-validation toy, so the recovered parameters will show
waveform systematics relative to published LVC posteriors (which used IMRPhenom/EOB
models); chirp mass and coalescence time should nevertheless land in the right
neighbourhood (~28-32 Msun detector frame, t_c within ms). Swap in your production
JAX waveform via the same WaveformModel signature for science-grade results.
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from jaxpe.diagnostics import corner_plot, split_rhat
from jaxpe.gw import ToyChirp, bbh_priors, fetch_open_strain, likelihood_from_strain
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler

OUT = Path(__file__).parent / "output"

TRIGGER = 1126259462.4  # GW150914
PSD_PAD = 512.0  # off-source seconds before the analysis segment, for Welch PSDs


def main():
    seg_start = TRIGGER + 2.0 - 8.0
    strain, psd_strain = {}, {}
    for det in ("H1", "L1"):
        print(f"fetching {det} strain (gwpy cache) ...")
        s, fs = fetch_open_strain(det, seg_start - PSD_PAD, TRIGGER + 4.0)
        i_seg = int(round(PSD_PAD * fs))
        strain[det] = s[i_seg:]
        psd_strain[det] = s[:i_seg]  # strictly off-source
    print(f"sampling rate {fs} Hz")

    like = likelihood_from_strain(
        ToyChirp(f_start=20.0),
        strain=strain,
        strain_start=seg_start,
        sampling_rate=fs,
        trigger_time=TRIGGER,
        duration=8.0,
        psd_strain=psd_strain,
        f_min=20.0,
        f_max=512.0,
    )

    prior = bbh_priors(
        chirp_mass=(15.0, 45.0),
        mass_ratio=(0.25, 1.0),
        luminosity_distance=(50.0, 2000.0),
        geocent_time=TRIGGER,
        time_width=0.1,
    )
    problem = like.problem(prior)

    cfg = GlobalLocalConfig(
        n_chains=80,
        n_training_loops=15,
        n_production_loops=8,
        n_local_steps=100,
        n_global_steps=50,
        local_thin=5,
        flow_layers=8,
        nn_width=64,
        n_epochs=6,
    )
    sampler = Sampler(MALA(step_size=0.05), problem=problem, config=cfg)

    t0 = time.time()
    res = sampler.run(jax.random.PRNGKey(4))
    print(f"sampling wall time: {time.time() - t0:.1f} s")

    phys = sampler.to_physical(res.samples)
    flat = phys.reshape(-1, problem.n_dim)
    names = list(problem.names)
    print("R-hat:", dict(zip(names, np.round(split_rhat(phys), 3))))
    print("\nposterior (median [16%, 84%]):")
    for i, n in enumerate(names):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(f"  {n:22s} {q50:12.4f} [{q16:12.4f}, {q84:12.4f}]")

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "gw150914_samples.npy", flat)
    fig = corner_plot(flat, names=names)
    fig.savefig(OUT / "gw150914_corner.png", dpi=120)
    print(f"corner -> {OUT / 'gw150914_corner.png'}")


if __name__ == "__main__":
    main()
