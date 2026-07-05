"""Cross-validate the jaxpe sampler against bilby+dynesty on the same injection.

Both samplers target the *identical* posterior: bilby's likelihood wraps our jitted
NetworkLikelihood, and bilby priors mirror jaxpe's. Any disagreement therefore
isolates sampling error rather than likelihood/waveform differences.

Run examples/03_gw_injection.py first (it saves the jaxpe posterior samples), then
this script; it prints per-parameter Jensen-Shannon divergences and writes an overlay
corner plot. JS < ~0.02 bits per parameter is the usual "same posterior" threshold
used in GW sampler reviews.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import bilby
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import jensenshannon

from jaxpe.gw import ToyChirp, make_injection

sys.path.insert(0, str(Path(__file__).parent))
inj_mod = __import__("03_gw_injection")

OUT = Path(__file__).parent / "output"
NOISE_SEED = 42


class WrappedLikelihood(bilby.Likelihood):
    def __init__(self, like, names):
        super().__init__(parameters=dict.fromkeys(names))
        self._like = like
        self._names = names
        self._fn = jax.jit(like.log_likelihood)

    def log_likelihood(self):
        params = {n: jnp.asarray(self.parameters[n]) for n in self._names}
        return float(self._fn(params))


def bilby_priors(t_c):
    p = bilby.core.prior.PriorDict()
    p["chirp_mass"] = bilby.core.prior.Uniform(25.0, 35.0, "chirp_mass")
    p["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1.0, "mass_ratio")
    p["luminosity_distance"] = bilby.core.prior.PowerLaw(2.0, 100.0, 2000.0, "luminosity_distance")
    p["inclination"] = bilby.core.prior.Sine(name="inclination")
    p["phase"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, "phase")
    p["ra"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, "ra")
    p["dec"] = bilby.core.prior.Cosine(name="dec")
    p["psi"] = bilby.core.prior.Uniform(0.0, np.pi, "psi")
    p["geocent_time"] = bilby.core.prior.Uniform(t_c - 0.1, t_c + 0.1, "geocent_time")
    return p


def js_bits(a, b, bins=60):
    """Jensen-Shannon divergence (bits) between two 1-D sample sets."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return jensenshannon(pa + 1e-12, pb + 1e-12, base=2) ** 2


def main(nlive=400):
    like = make_injection(
        ToyChirp(f_start=20.0), inj_mod.INJECTION,
        detector_names=("H1", "L1"), duration=8.0, sampling_rate=2048.0,
        f_min=20.0, noise_seed=NOISE_SEED,
    )
    names = list(bilby_priors(inj_mod.T_C).keys())

    jaxpe_file = OUT / f"gw_injection_seed{NOISE_SEED}_samples.npy"
    if not jaxpe_file.exists():
        raise SystemExit(f"run 03_gw_injection.py first ({jaxpe_file} missing)")
    jaxpe_samples = np.load(jaxpe_file)

    result = bilby.run_sampler(
        likelihood=WrappedLikelihood(like, names),
        priors=bilby_priors(inj_mod.T_C),
        sampler="dynesty",
        nlive=nlive,
        sample="rwalk",
        outdir=str(OUT / "dynesty"),
        label=f"toychirp_seed{NOISE_SEED}",
        resume=True,
        npool=1,
    )
    dyn = result.posterior[names].to_numpy()

    print(f"\ndynesty posterior samples: {len(dyn)}, jaxpe samples: {len(jaxpe_samples)}")
    print("\nper-parameter JS divergence (bits), threshold ~0.02:")
    worst = 0.0
    for i, n in enumerate(names):
        js = js_bits(jaxpe_samples[:, i], dyn[:, i])
        worst = max(worst, js)
        flag = "" if js < 0.02 else "  <-- CHECK"
        print(f"  {n:22s} {js:.4f}{flag}")
    print(f"\nworst JS: {worst:.4f} -> {'PASS' if worst < 0.02 else 'INSPECT'}")

    # overlay corner
    import corner as corner_module
    import matplotlib

    matplotlib.use("Agg")
    truths = [inj_mod.INJECTION[n] for n in names]
    fig = corner_module.corner(
        dyn, labels=names, truths=truths, color="C1", bins=40,
        hist_kwargs=dict(density=True),
    )
    corner_module.corner(
        jaxpe_samples[np.random.default_rng(0).choice(len(jaxpe_samples), min(len(dyn) * 4, len(jaxpe_samples)), replace=False)],
        fig=fig, color="C0", bins=40, hist_kwargs=dict(density=True),
    )
    fig.savefig(OUT / "validate_overlay_corner.png", dpi=120)
    print(f"overlay corner (blue=jaxpe, orange=dynesty) -> {OUT / 'validate_overlay_corner.png'}")


if __name__ == "__main__":
    main()
