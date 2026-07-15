"""Cross-validation of two parameter-estimation methods on a PhenomD injection.

This is the frequency-domain, aligned-spin counterpart of
``examples/07_route_comparison.py``. It injects one IMRPhenomD signal and recovers its
intrinsic parameters two different ways, checking that the posteriors agree. Agreement
is a strong end-to-end test: the two methods share almost no code, so a match validates
both the differentiable likelihood and the marginalized likelihood at once.

The two methods
---------------
1. Gradient-based direct sampling
   Samples the full set of free parameters directly from the differentiable
   likelihood with the normalizing-flow-enhanced Global-Local sampler (Metropolis-
   adjusted Langevin kernel). It explores the intrinsic parameters plus the two
   extrinsic parameters left free (coalescence phase and luminosity distance).

2. Surrogate marginalized inference
   Marginalizes coalescence phase and luminosity distance out of the likelihood
   analytically, leaving a likelihood over the intrinsic parameters only, and builds a
   Gaussian-process surrogate of it by active learning (GPry) from a few dozen
   evaluations. This is the method intended for expensive, non-differentiable waveform
   models where each likelihood call is precious.

Two stages of increasing dimensionality are provided:
  * ``nonspin``      -- recover (chirp_mass, mass_ratio); spins fixed at zero.
  * ``alignedspin``  -- recover (chirp_mass, mass_ratio, spin1z, spin2z).
Sky position, inclination and coalescence time are held FIXED at truth in both methods
so the comparison is like-for-like over the same intrinsic parameters.

How Route B differs from example 07 (and why)
---------------------------------------------
Example 07 marginalizes phase and distance with the general *mode-based* marginalizer
(:class:`~jaxpe.gw.marginalized.ModesNetworkLikelihood`), which decomposes the waveform
into spherical-harmonic modes -- necessary for a genuinely multi-harmonic model like
the eccentric ESIGMA. IMRPhenomD is a dominant-(2,2)-mode model, so here Route B uses
the exact closed-form marginal :class:`~jaxpe.gw.PhaseDistanceMarginalLikelihood`
instead: the phase integral collapses to a Bessel function ln I0(u|Z|) and the distance
integral to a 1-D quadrature, with no mode decomposition or FFT. That object self-checks
that the plugged-in model really is dominant-mode (it warns otherwise), and it is
model-agnostic, so any frequency-domain model registered in ``MODELS`` below can be
swapped in via ``--model``.

Hardware note (opposite to example 07)
--------------------------------------
The likelihood must run in float64. For the ODE-based ESIGMA model of example 07 the
gradient sampler is *slower* on a small consumer GPU than on CPU; for the vectorized
frequency-domain IMRPhenomD the gradient sampler is *faster* on the GPU. Run the
gradient method once per backend (via ``JAX_PLATFORMS``) to reproduce that finding --
the overlay figure keeps the CPU and GPU posteriors as separate curves so you can see
they coincide.

Results are persisted per run to ``examples/output/`` and every run present for a stage
is overlaid on one figure, mirroring the ``route_comparison_3way`` convention. So a
single ``--method both`` run on CPU gives a two-curve figure; adding a GPU gradient run
upgrades it to three curves.

Examples
--------
    # gradient (whatever JAX sees) + surrogate, both stages, then overlay:
    JAX_PLATFORMS=cpu python examples/08_phenomd_route_comparison.py
    # add the GPU gradient curve to the same figures:
    python examples/08_phenomd_route_comparison.py --method gradient
    # just rebuild the overlays from already-saved runs:
    python examples/08_phenomd_route_comparison.py --overlay-only

Requires jaxpe with the ``surrogate`` extra (GPry) for the surrogate method.
"""

import argparse
import time
from pathlib import Path

import jax

# float64 is mandatory: GPS times (~1.1e9 s) and the Whittle inner product lose all
# precision in float32, silently turning the log-likelihood into NaN.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------- the injection
LOWER_FREQUENCY_HZ = 20.0
SEGMENT_DURATION_S = 4.0
SAMPLING_RATE_HZ = 2048.0
COALESCENCE_TIME_GPS = 1126259462.4
REFERENCE_DISTANCE_MPC = (
    1000.0  # distance at which Route B evaluates its reference strain
)

# The shared aligned-spin binary. The two stages differ only in whether the spins are
# injected (and recovered) or pinned at zero.
BASE_PARAMETERS = dict(
    chirp_mass=25.0,  # solar masses
    mass_ratio=0.8,
    luminosity_distance=3700.0,  # Mpc -- network signal-to-noise ~13
    inclination=0.4,
    phase=1.5,
    geocent_time=COALESCENCE_TIME_GPS,
    ra=1.2,
    dec=0.5,
    psi=0.8,
)
STAGE_SPINS = {
    "nonspin": dict(spin1z=0.0, spin2z=0.0),
    "alignedspin": dict(spin1z=0.2, spin2z=-0.1),
}

# Prior ranges. Distance uses a distance^2 (volume) prior over these bounds; phase is
# uniform on [0, 2*pi]. The surrogate marginalizes phase and distance with these same
# priors, so the two methods are compared on identical footing.
CHIRP_MASS_PRIOR = (23.0, 27.0)
MASS_RATIO_PRIOR = (0.5, 1.0)
SPIN_PRIOR = (-0.5, 0.5)
DISTANCE_PRIOR_MPC = (1000.0, 8000.0)


# A tiny registry so any frequency-domain, dominant-(2,2)-mode model can be swapped in.
# The model must take the standard params dict and return (h+, hx) in the frequency
# domain; the closed-form Route B self-checks that it is dominant-mode.
def _build_model(name, f_ref):
    from jaxpe.gw import IMRPhenomD

    factories = {
        "phenomd": lambda: IMRPhenomD(f_ref=f_ref),
        # add here, e.g. "phenomxas": lambda: IMRPhenomXAS(f_ref=f_ref),
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; known: {tuple(factories)}")
    return factories[name]()


MODELS = ("phenomd",)


def intrinsic_names(stage):
    """The parameters both methods estimate at a given stage."""
    if stage == "alignedspin":
        return ("chirp_mass", "mass_ratio", "spin1z", "spin2z")
    return ("chirp_mass", "mass_ratio")


def injected_parameters(stage):
    return {**BASE_PARAMETERS, **STAGE_SPINS[stage]}


def build_injection(model_name, stage):
    """Inject the stage's signal into two detectors (Hanford, Livingston), no noise."""
    from jaxpe.gw import make_injection

    injection = injected_parameters(stage)
    waveform = _build_model(model_name, LOWER_FREQUENCY_HZ)
    likelihood = make_injection(
        waveform,
        injection,
        detector_names=("H1", "L1"),
        duration=SEGMENT_DURATION_S,
        sampling_rate=SAMPLING_RATE_HZ,
        f_min=LOWER_FREQUENCY_HZ,
        noise_seed=None,
    )
    per_detector = likelihood.optimal_snr(
        {k: jnp.asarray(v) for k, v in injection.items()}
    )
    network = float(np.sqrt(sum(s**2 for s in per_detector.values())))
    print(
        f"[injection:{stage}] {model_name} signal-to-noise per detector "
        f"{per_detector}, network {network:.1f}"
    )
    return likelihood, network


# ------------------------------------------------- method 1: gradient direct sampling
def run_gradient_direct_sampling(
    likelihood, stage, n_chains=48, n_production=150, seed=0
):
    """Sample the intrinsic parameters plus phase and distance directly with gradients.

    Uses the Global-Local sampler (normalizing-flow global proposals plus a local
    Metropolis-adjusted Langevin kernel). Returns the intrinsic marginal together with
    timing and cost information.
    """
    from jaxpe.core.priors import JointPrior, PowerLaw, Uniform
    from jaxpe.core.problem import InferenceProblem
    from jaxpe.kernels import MALA
    from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

    device = jax.devices()[0]
    print(
        f"\n=== Method 1: gradient-based direct sampling [{stage}] "
        f"(device: {device}) ==="
    )

    injection = injected_parameters(stage)
    sampled = {
        "chirp_mass": Uniform(low=CHIRP_MASS_PRIOR[0], high=CHIRP_MASS_PRIOR[1]),
        "mass_ratio": Uniform(low=MASS_RATIO_PRIOR[0], high=MASS_RATIO_PRIOR[1]),
        "phase": Uniform(low=0.0, high=2 * np.pi),
        "luminosity_distance": PowerLaw(
            alpha=2.0, low=DISTANCE_PRIOR_MPC[0], high=DISTANCE_PRIOR_MPC[1]
        ),
    }
    if stage == "alignedspin":
        sampled["spin1z"] = Uniform(low=SPIN_PRIOR[0], high=SPIN_PRIOR[1])
        sampled["spin2z"] = Uniform(low=SPIN_PRIOR[0], high=SPIN_PRIOR[1])
    prior = JointPrior(sampled)

    # Everything not sampled is supplied fixed at truth (sky, inclination, time, and the
    # spins in the non-spinning stage).
    fixed_keys = ["ra", "dec", "psi", "inclination", "geocent_time"]
    if stage == "nonspin":
        fixed_keys += ["spin1z", "spin2z"]
    fixed = {k: jnp.asarray(injection[k]) for k in fixed_keys}
    problem = InferenceProblem(
        prior=prior,
        log_likelihood=lambda s: likelihood.log_likelihood({**fixed, **s}),
    )

    buffer = n_chains * 20
    config = GlobalLocalConfig(
        n_chains=n_chains,
        n_prelim_loops=2,
        n_training_loops=15,
        n_production_loops=n_production,
        n_local_steps=50,
        n_global_steps=50,
        local_thin=3,
        buffer_size=15 * buffer,
        flow_layers=6,
        nn_width=48,
        n_epochs=40,
        batch_size=min(1024, 15 * buffer),
    )
    sampler = Sampler(MALA(step_size=0.03), problem=problem, config=config)

    key = jax.random.PRNGKey(seed)
    start_points = best_of_prior_init(key, problem, n_chains, n_draws=5000)

    started = time.time()
    result = sampler.run(key, x0=start_points)
    wall_seconds = time.time() - started

    samples = sampler.to_physical(result.samples).reshape(-1, problem.n_dim)
    names = list(problem.names)
    order = [names.index(n) for n in intrinsic_names(stage)]
    gradient_steps = (
        config.n_prelim_loops + config.n_training_loops + config.n_production_loops
    ) * config.n_local_steps
    print(
        f"[gradient:{stage}] {samples.shape[0]} samples, ~{gradient_steps} gradient "
        f"steps, {wall_seconds:.0f} s on {device}"
    )
    return dict(
        samples=samples[:, order],
        weights=None,
        wall_seconds=wall_seconds,
        likelihood_evaluations=gradient_steps,
        method="gradient",
        device=device.platform,
        n_samples=samples.shape[0],
    )


# --------------------------------------------- method 2: surrogate marginalized inference
def run_surrogate_marginalized_inference(likelihood, stage, seed=11):
    """Learn a surrogate of the phase-and-distance-marginalized likelihood (GPry).

    Route B here is the closed-form :class:`~jaxpe.gw.PhaseDistanceMarginalLikelihood`,
    valid because IMRPhenomD is a dominant-(2,2)-mode model with a fixed sky. Its
    constructor self-checks that factorization on the injected parameters.
    """
    from jaxpe.gw import PhaseDistanceMarginalLikelihood
    from jaxpe.surrogate import GPryEngine

    print(f"\n=== Method 2: surrogate marginalized inference [{stage}] (GPry) ===")
    injection = injected_parameters(stage)
    names = intrinsic_names(stage)

    bounds = {"chirp_mass": CHIRP_MASS_PRIOR, "mass_ratio": MASS_RATIO_PRIOR}
    if stage == "alignedspin":
        bounds["spin1z"] = SPIN_PRIOR
        bounds["spin2z"] = SPIN_PRIOR
    fixed_ext = {k: injection[k] for k in ("ra", "dec", "psi", "inclination")}
    if stage == "nonspin":
        fixed_ext["spin1z"] = 0.0
        fixed_ext["spin2z"] = 0.0

    marginal = PhaseDistanceMarginalLikelihood(
        likelihood,
        names,
        fixed_ext,
        dist_bounds=DISTANCE_PRIOR_MPC,
        dist_power=2.0,
        d_ref=REFERENCE_DISTANCE_MPC,
        check_params=injection,  # verifies the dominant-mode factorization for this model
    )
    print(
        f"[surrogate:{stage}] dominant-mode residual "
        f"{marginal.dominant_mode_residual:.2e} (0 => closed form exact)"
    )

    started = time.time()
    diagnostics = None
    for attempt in range(3):  # UltraNest's MLFriends is stochastic; retry on failure
        try:
            engine = GPryEngine(
                marginal, bounds=bounds, options={"seed": seed + attempt}, verbose=0
            )
            diagnostics = engine.run()
            break
        except Exception as error:  # noqa: BLE001
            print(f"  [attempt {attempt} failed: {type(error).__name__}: {error}]")
    if diagnostics is None:
        raise RuntimeError(f"GPry failed three times on {stage}")
    posterior = engine.sample()
    wall_seconds = time.time() - started

    print(
        f"[surrogate:{stage}] converged={diagnostics['has_converged']}, "
        f"{diagnostics['n_truth_evals']} waveform evaluations, "
        f"{wall_seconds:.0f} s, {len(posterior.x)} samples"
    )
    return dict(
        samples=np.asarray(posterior.x),
        weights=np.asarray(posterior.weights),
        wall_seconds=wall_seconds,
        likelihood_evaluations=diagnostics["n_truth_evals"],
        method="surrogate",
        device="cpu",
        n_samples=len(posterior.x),
    )


# ------------------------------------------------------------------ persistence + overlay
# Stable colour + label per (method, device), so a curve keeps its identity across runs.
_RUN_STYLE = {
    ("gradient", "cpu"): ("#1f6fb2", "gradient direct (CPU)"),
    ("gradient", "gpu"): ("#0f8f80", "gradient direct (GPU)"),
    ("surrogate", "cpu"): ("#c0392b", "surrogate marginalized"),
}


def persist_run(stage, result, max_samples=20000):
    """Save one run's intrinsic posterior to output/phenomd_<stage>_<method>_<device>.npz.

    Gradient chains are thinned to keep the file small; the surrogate posterior is
    already compact and is saved with its importance weights.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    samples = np.asarray(result["samples"])
    weights = (
        np.ones(len(samples))
        if result["weights"] is None
        else np.asarray(result["weights"])
    )
    if len(samples) > max_samples:  # thin (weighted) for a lightweight, plottable file
        rng = np.random.default_rng(0)
        pick = rng.choice(
            len(samples), max_samples, replace=False, p=weights / weights.sum()
        )
        samples, weights = samples[pick], weights[pick]
    truth = [injected_parameters(stage)[n] for n in intrinsic_names(stage)]
    path = OUTPUT_DIR / f"phenomd_{stage}_{result['method']}_{result['device']}.npz"
    np.savez(
        path,
        samples=samples,
        weights=weights,
        names=np.array(intrinsic_names(stage)),
        truth=np.array(truth),
        wall_seconds=result["wall_seconds"],
        likelihood_evaluations=result["likelihood_evaluations"],
        method=result["method"],
        device=result["device"],
    )
    print(f"  saved run to {path}")


def load_runs(stage):
    """Load every persisted run for a stage, keyed by (method, device)."""
    runs = {}
    for path in sorted(OUTPUT_DIR.glob(f"phenomd_{stage}_*.npz")):
        d = np.load(path, allow_pickle=True)
        runs[(str(d["method"]), str(d["device"]))] = dict(
            samples=d["samples"],
            weights=d["weights"],
            names=[str(n) for n in d["names"]],
            truth=d["truth"],
            wall_seconds=float(d["wall_seconds"]),
            likelihood_evaluations=int(d["likelihood_evaluations"]),
        )
    return runs


def credible_interval(values, weights):
    """Median and 16th/84th percentiles of one parameter (weighted)."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order] / weights.sum())
    return np.interp([0.16, 0.5, 0.84], cumulative, values[order])


def report(stage, runs):
    """Print each run's recovered intrinsic posterior and its cost."""
    names = intrinsic_names(stage)
    truth = [injected_parameters(stage)[n] for n in names]
    print(f"\n--- {stage}: recovered intrinsic posterior (median [16th, 84th]) ---")
    for (method, device), r in runs.items():
        label = _RUN_STYLE.get((method, device), (None, f"{method} ({device})"))[1]
        cols = []
        for i, n in enumerate(names):
            lo, mid, hi = credible_interval(r["samples"][:, i], r["weights"])
            cols.append(f"{n} {mid:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  {label:26s} " + "   ".join(cols))
    print(
        f"  {'injected truth':26s} "
        + "   ".join(f"{n} {t:.3f}" for n, t in zip(names, truth))
    )
    print(f"--- {stage}: cost per run ---")
    for (method, device), r in runs.items():
        label = _RUN_STYLE.get((method, device), (None, f"{method} ({device})"))[1]
        print(
            f"  {label:26s} {r['likelihood_evaluations']:7d} evaluations   "
            f"{r['wall_seconds']:7.0f} s   {len(r['samples']):6d} samples"
        )


def overlay(stage, network_snr=None):
    """Overlay every persisted run's intrinsic posterior for a stage on one figure."""
    import corner
    import matplotlib
    import matplotlib.lines as mlines

    matplotlib.use("Agg")

    runs = load_runs(stage)
    if not runs:
        print(f"[overlay:{stage}] no saved runs found; nothing to plot")
        return
    names = intrinsic_names(stage)
    truth = list(next(iter(runs.values()))["truth"])
    # common plotting range across all runs so contours are comparable
    ranges = [
        (
            min(r["samples"][:, i].min() for r in runs.values()),
            max(r["samples"][:, i].max() for r in runs.values()),
        )
        for i in range(len(names))
    ]

    figure, legend = None, []
    for (method, device), r in runs.items():
        color, label = _RUN_STYLE.get(
            (method, device), ("#555555", f"{method} ({device})")
        )
        weights = r["weights"] / r["weights"].sum()
        figure = corner.corner(
            r["samples"],
            weights=weights,
            fig=figure,
            labels=list(names) if figure is None else None,
            truths=truth if figure is None else None,
            truth_color="k",
            color=color,
            range=ranges,
            hist_kwargs=dict(density=True),
            plot_datapoints=False,
            smooth=1.0,
            levels=(0.393, 0.865),
            bins=30,
        )
        legend.append(mlines.Line2D([], [], color=color, label=label))
    legend.append(mlines.Line2D([], [], color="k", label="injected truth"))
    figure.legend(handles=legend, loc="upper right", frameon=False, fontsize=10)
    snr = f" (network SNR ~{network_snr:.0f})" if network_snr else ""
    figure.suptitle(f"PhenomD {stage}: PE routes on one injection{snr}", y=1.02)
    path = OUTPUT_DIR / f"phenomd_{stage}_route_comparison.png"
    figure.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[overlay:{stage}] saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODELS, default="phenomd")
    parser.add_argument(
        "--stage", choices=["nonspin", "alignedspin", "both"], default="both"
    )
    parser.add_argument(
        "--method",
        choices=["gradient", "surrogate", "both"],
        default="both",
        help="which method(s) to run this invocation",
    )
    parser.add_argument(
        "--n-chains", type=int, default=48, help="chains for the gradient sampler"
    )
    parser.add_argument(
        "--n-production",
        type=int,
        default=150,
        help="production loops for the gradient sampler",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="skip running; just rebuild overlays from saved runs",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    stages = ["nonspin", "alignedspin"] if args.stage == "both" else [args.stage]

    for stage in stages:
        network = None
        if not args.overlay_only:
            likelihood, network = build_injection(args.model, stage)
            if args.method in ("gradient", "both"):
                result = run_gradient_direct_sampling(
                    likelihood,
                    stage,
                    n_chains=args.n_chains,
                    n_production=args.n_production,
                    seed=args.seed,
                )
                persist_run(stage, result)
            if args.method in ("surrogate", "both"):
                result = run_surrogate_marginalized_inference(likelihood, stage)
                persist_run(stage, result)
        runs = load_runs(stage)
        if runs:
            report(stage, runs)
            overlay(stage, network_snr=network)


if __name__ == "__main__":
    main()
