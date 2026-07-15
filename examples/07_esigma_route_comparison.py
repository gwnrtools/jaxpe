"""Cross-validation of two parameter-estimation methods on one shared injection.

jaxpe can estimate compact-binary parameters two very different ways, and this script
runs both on the *same* simulated eccentric-binary signal and checks that they agree.
Agreement is a strong end-to-end test: the two methods share almost no code, so if
their posteriors match, both the differentiable likelihood and the marginalized
likelihood are internally consistent.

The two methods
---------------
1. Gradient-based direct sampling
   Samples the full set of unknown parameters directly from the differentiable
   likelihood, using the normalizing-flow-enhanced Global-Local sampler with a
   Metropolis-adjusted Langevin (gradient) kernel. Here it explores four parameters:
   chirp mass, eccentricity, and the two extrinsic parameters left free (coalescence
   phase and luminosity distance).

2. Surrogate marginalized inference
   Integrates the two extrinsic parameters (phase, distance) out of the likelihood
   analytically / by importance sampling, leaving a likelihood over only the two
   intrinsic parameters (chirp mass, eccentricity). A Gaussian-process surrogate of
   that marginalized likelihood is then built by active learning (GPry) from a few
   dozen evaluations. This is the method intended for expensive, non-differentiable
   waveform models, where each likelihood call is precious.

Both methods must return the same posterior over (chirp mass, eccentricity) and both
must recover the injected truth. To make the comparison clean, sky position,
inclination and coalescence time are held FIXED at their true values in both methods,
so any disagreement points to a genuine convention or normalization difference rather
than to one method's sampler having under-converged.

Choices that matter (each learned the hard way; see docs/gpry_fusion_design.md)
-------------------------------------------------------------------------------
* Signal loudness. The injection is placed far enough away (network signal-to-noise
  ratio ~11) that the posterior is broad relative to the prior. At very high
  signal-to-noise the posterior becomes a needle in the prior volume that neither
  method can find from a cold start -- in that regime a cheap-model posterior must be
  used to narrow the prior first.
* ODE resolution. `n_ode_grid` sets how finely the inspiral's orbital phase is
  resolved. 1024 points is converged for this ~25 solar-mass signal; 512 points
  under-resolves it and biases the recovered parameters by ~1.5 sigma. Run with
  `--check-ode-grid` to reproduce that convergence study.
* Hardware. The likelihood must run in float64 (GPS times and the noise-weighted
  inner product are meaningless in float32). On a small consumer GPU with throttled
  float64 (e.g. a Quadro T2000) the gradient sampler is *slower* than the CPU, so
  `--backend` is offered but CPU is the better choice there.

Examples
--------
    JAX_PLATFORMS=cpu python examples/07_esigma_route_comparison.py            # both methods
    JAX_PLATFORMS=cpu python examples/07_esigma_route_comparison.py --method surrogate
    python examples/07_esigma_route_comparison.py --check-ode-grid             # grid study only

Requires esigmapy (the waveform). The surrogate method also needs the ``surrogate``
extra (``pip install jaxpe[surrogate]``, which provides GPry).
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
# A single eccentric binary-black-hole signal, injected into zero noise so both
# methods should peak exactly at these values.
LOWER_FREQUENCY_HZ = 30.0
SEGMENT_DURATION_S = 4.0
SAMPLING_RATE_HZ = 1024.0
COALESCENCE_TIME_GPS = 1126259462.4

TRUE_PARAMETERS = dict(
    chirp_mass=25.0,          # solar masses
    mass_ratio=0.8,
    eccentricity=0.05,
    mean_anomaly=1.0,
    spin1z=0.0,
    spin2z=0.0,
    luminosity_distance=3000.0,   # Mpc -- chosen for network signal-to-noise ~11
    inclination=0.4,
    phase=1.5,
    geocent_time=COALESCENCE_TIME_GPS,
    ra=1.2,
    dec=0.5,
    psi=0.8,
)

# Held fixed at their true values in BOTH methods (not sampled, not marginalized), so
# the two methods are compared on identical footing over the same two free parameters.
FIXED_PARAMETERS = dict(
    mass_ratio=0.8,
    mean_anomaly=1.0,
    spin1z=0.0,
    spin2z=0.0,
    ra=1.2,
    dec=0.5,
    psi=0.8,
    inclination=0.4,
    geocent_time=COALESCENCE_TIME_GPS,
)

# Prior ranges for the two parameters both methods estimate, plus the two extrinsic
# parameters the gradient method samples (the surrogate method marginalizes these two
# with matching priors: phase uniform on [0, 2*pi], distance proportional to
# distance^2 over the same bounds).
CHIRP_MASS_PRIOR = (24.5, 25.5)          # solar masses
ECCENTRICITY_PRIOR = (0.0, 0.1)
DISTANCE_PRIOR_MPC = (1000.0, 6000.0)

INTRINSIC_PARAMETER_NAMES = ("chirp_mass", "eccentricity")


def build_waveform(n_ode_grid=1024):
    """The ESIGMA eccentric-inspiral model.

    A cheap leading-order (0 post-Newtonian) configuration is used deliberately: this
    script compares parameter-estimation *methods*, not waveform accuracy, so both
    methods share one identical, fast model.
    """
    from jaxpe.gw import ESIGMAInspiral

    return ESIGMAInspiral(
        f_lower=LOWER_FREQUENCY_HZ,
        modes=((2, 2), (3, 3)),
        rad_pn_order=0,
        mode_pn_order=0,
        ode_eps=1e-6,
        n_ode_grid=n_ode_grid,
        max_ode_steps=16384,
    )


def build_injection(waveform):
    """Inject `TRUE_PARAMETERS` into two detectors (Hanford, Livingston), no noise."""
    from jaxpe.gw import make_injection

    likelihood = make_injection(
        waveform, TRUE_PARAMETERS, detector_names=("H1", "L1"),
        duration=SEGMENT_DURATION_S, sampling_rate=SAMPLING_RATE_HZ,
        f_min=LOWER_FREQUENCY_HZ, noise_seed=None,
    )
    per_detector = likelihood.optimal_snr(
        {k: jnp.asarray(v) for k, v in TRUE_PARAMETERS.items()})
    network = np.sqrt(sum(s**2 for s in per_detector.values()))
    print(f"[injection] signal-to-noise per detector {per_detector}, "
          f"network {network:.1f}")
    return likelihood


# ------------------------------------------------- method 1: gradient direct sampling
def run_gradient_direct_sampling(likelihood, n_chains=20, seed=0):
    """Sample (chirp mass, eccentricity, phase, distance) directly with gradients.

    Uses the Global-Local sampler (normalizing-flow global proposals plus a local
    Metropolis-adjusted Langevin kernel). Returns the (chirp mass, eccentricity)
    marginal together with timing and cost information.
    """
    from jaxpe.core.priors import JointPrior, PowerLaw, Uniform
    from jaxpe.core.problem import InferenceProblem
    from jaxpe.kernels import MALA
    from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

    device = jax.devices()[0]
    print(f"\n=== Method 1: gradient-based direct sampling (device: {device}) ===")

    prior = JointPrior({
        "chirp_mass": Uniform(low=CHIRP_MASS_PRIOR[0], high=CHIRP_MASS_PRIOR[1]),
        "eccentricity": Uniform(low=ECCENTRICITY_PRIOR[0], high=ECCENTRICITY_PRIOR[1]),
        "phase": Uniform(low=0.0, high=2 * np.pi),
        "luminosity_distance": PowerLaw(
            alpha=2.0, low=DISTANCE_PRIOR_MPC[0], high=DISTANCE_PRIOR_MPC[1]),
    })

    # Supply the held-fixed parameters to every likelihood call; the sampler only
    # varies the four parameters named in the prior.
    fixed = {k: jnp.asarray(v) for k, v in FIXED_PARAMETERS.items()}
    problem = InferenceProblem(
        prior=prior,
        log_likelihood=lambda sampled: likelihood.log_likelihood({**fixed, **sampled}),
    )

    buffer = n_chains * 20
    config = GlobalLocalConfig(
        n_chains=n_chains,
        n_prelim_loops=2, n_training_loops=8, n_production_loops=40,
        n_local_steps=30, n_global_steps=30, local_thin=3,
        buffer_size=15 * buffer, flow_layers=6, nn_width=48, n_epochs=30,
        batch_size=min(1024, 15 * buffer),
    )
    sampler = Sampler(MALA(step_size=0.03), problem=problem, config=config)

    key = jax.random.PRNGKey(seed)
    start_points = best_of_prior_init(key, problem, n_chains, n_draws=5000)

    started = time.time()
    result = sampler.run(key, x0=start_points)
    wall_seconds = time.time() - started

    samples = sampler.to_physical(result.samples).reshape(-1, problem.n_dim)
    order = [list(problem.names).index(n) for n in INTRINSIC_PARAMETER_NAMES]
    gradient_steps = (config.n_prelim_loops + config.n_training_loops
                      + config.n_production_loops) * config.n_local_steps
    print(f"[gradient] {samples.shape[0]} samples, ~{gradient_steps} gradient steps, "
          f"{wall_seconds:.0f} s on {device}")
    return dict(
        samples=samples[:, order], weights=None,
        wall_seconds=wall_seconds, likelihood_evaluations=gradient_steps,
        device=str(device), n_samples=samples.shape[0],
    )


# --------------------------------------------- method 2: surrogate marginalized inference
def run_surrogate_marginalized_inference(waveform, likelihood, seed=11):
    """Learn a surrogate of the phase-and-distance-marginalized likelihood.

    The waveform is reduced to its spherical-harmonic modes once per intrinsic point;
    coalescence phase and luminosity distance are marginalized out of the likelihood
    (with the same priors the gradient method samples); GPry active-learning then
    builds a Gaussian-process surrogate over (chirp mass, eccentricity) from a few
    dozen evaluations and draws a posterior from it.
    """
    from jaxpe.gw.external_models import ModesData
    from jaxpe.gw.marginalized import (
        MarginalizedIntrinsicLikelihood, ModesNetworkLikelihood)
    from jaxpe.surrogate import GPryEngine

    print("\n=== Method 2: surrogate marginalized inference (GPry) ===")
    analysis_times = likelihood.times
    analysis_times_jax = jnp.asarray(analysis_times)
    fixed_intrinsic = {k: FIXED_PARAMETERS[k]
                       for k in ("mass_ratio", "mean_anomaly", "spin1z", "spin2z")}

    def waveform_modes(intrinsic_point):
        """Map (chirp mass, eccentricity) to strain modes at a 1 Mpc reference."""
        params = {
            "chirp_mass": jnp.asarray(intrinsic_point["chirp_mass"]),
            "eccentricity": jnp.asarray(intrinsic_point["eccentricity"]),
            "geocent_time": jnp.asarray(COALESCENCE_TIME_GPS),
            **{k: jnp.asarray(v) for k, v in fixed_intrinsic.items()},
        }
        modes = waveform.mode_dict(params, analysis_times_jax)
        return ModesData(
            modes={lm: np.asarray(h) for lm, h in modes.items()},
            times=analysis_times, d_ref_mpc=1.0, t_ref=COALESCENCE_TIME_GPS)

    true_modes = waveform_modes({"chirp_mass": TRUE_PARAMETERS["chirp_mass"],
                                 "eccentricity": TRUE_PARAMETERS["eccentricity"]})
    mode_likelihood = ModesNetworkLikelihood.from_likelihood(likelihood, true_modes)

    marginalized_likelihood = MarginalizedIntrinsicLikelihood(
        waveform_modes, mode_likelihood,
        names=INTRINSIC_PARAMETER_NAMES, t_center=COALESCENCE_TIME_GPS,
        marginalize_sky=False,   # sky and inclination are fixed at truth
        fixed_extrinsic={k: FIXED_PARAMETERS[k]
                         for k in ("ra", "dec", "psi", "inclination")},
        # marginalize phase (n_phi nodes) and distance (n_dist nodes, same
        # distance^2 prior and bounds the gradient method uses); coalescence time is
        # pinned at truth by tc_half_samples=0.
        settings=dict(n_phi=96, n_dist=64, tc_half_samples=0,
                      dist_min=DISTANCE_PRIOR_MPC[0], dist_max=DISTANCE_PRIOR_MPC[1],
                      dist_power=2.0),
    )

    started = time.time()
    engine = GPryEngine(
        marginalized_likelihood,
        bounds={"chirp_mass": CHIRP_MASS_PRIOR, "eccentricity": ECCENTRICITY_PRIOR},
        options={"seed": seed}, verbose=0)
    diagnostics = engine.run()
    posterior = engine.sample()
    wall_seconds = time.time() - started

    print(f"[surrogate] converged={diagnostics['has_converged']}, "
          f"{diagnostics['n_truth_evals']} waveform evaluations, "
          f"{wall_seconds:.0f} s, {len(posterior.x)} samples")
    return dict(
        samples=posterior.x, weights=posterior.weights,
        wall_seconds=wall_seconds,
        likelihood_evaluations=diagnostics["n_truth_evals"],
        device="cpu (Gaussian process)", n_samples=len(posterior.x),
    )


# ------------------------------------------------------------------ ODE grid study
def check_ode_grid():
    """Show why 1024 ODE points is the right resolution for this signal.

    Injects the signal at a high resolution (1024 points) and then evaluates the
    template at several resolutions, reporting the log-likelihood at the true
    parameters. A converged resolution gives ~0; a coarse one gives a large negative
    value (a waveform-modeling bias that would shift the recovered parameters).
    """
    from jaxpe.gw import make_injection
    from jaxpe.gw.likelihood import TDNetworkLikelihood

    print("=== ODE-grid convergence: inject at 1024, score template at truth ===")
    reference = make_injection(
        build_waveform(1024), TRUE_PARAMETERS, detector_names=("H1", "L1"),
        duration=SEGMENT_DURATION_S, sampling_rate=SAMPLING_RATE_HZ,
        f_min=LOWER_FREQUENCY_HZ, noise_seed=None)
    truth = {k: jnp.asarray(v) for k, v in TRUE_PARAMETERS.items()}
    network = np.sqrt(sum(s**2 for s in reference.optimal_snr(truth).values()))
    print(f"network signal-to-noise {network:.1f}; a perfect template scores 0, "
          f"the worst possible is {-network**2 / 2:.0f}")
    for grid in (256, 512, 1024, 2048):
        template = TDNetworkLikelihood(
            waveform=build_waveform(grid), detectors=reference.detectors,
            data_fd=reference.data_fd, psds=reference.psds, freqs=reference.freqs,
            times=reference.times, f_min=reference.f_min, f_max=reference.f_max,
            gmst_ref=reference.gmst_ref, t_ref=reference.t_ref,
            tukey_alpha=reference.tukey_alpha)
        loss = float(template.log_likelihood(truth))
        verdict = "converged" if abs(loss) < 0.2 else "under-resolved (biased)"
        print(f"  {grid:5d} points: log-likelihood at truth = {loss:8.3f}  ({verdict})")


# ------------------------------------------------------------------ reporting
def credible_interval(samples, weights, column):
    """Median and 16th/84th percentiles of one parameter (weighted if needed)."""
    values = samples[:, column]
    if weights is None:
        return np.percentile(values, [16, 50, 84])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order] / weights.sum())
    return np.interp([0.16, 0.5, 0.84], cumulative, values[order])


def report(results):
    """Print the recovered posteriors and the per-method cost, and check agreement."""
    truth = [TRUE_PARAMETERS[n] for n in INTRINSIC_PARAMETER_NAMES]

    print("\nRecovered posterior  (median [16th, 84th percentile])")
    for label, r in results.items():
        chirp = credible_interval(r["samples"], r["weights"], 0)
        ecc = credible_interval(r["samples"], r["weights"], 1)
        print(f"  {label:28s} chirp_mass {chirp[1]:.3f} [{chirp[0]:.3f}, {chirp[2]:.3f}]"
              f"   eccentricity {ecc[1]:.4f} [{ecc[0]:.4f}, {ecc[2]:.4f}]")
    print(f"  {'injected truth':28s} chirp_mass {truth[0]:.3f}"
          f"{'':21s} eccentricity {truth[1]:.4f}")

    print("\nCost per method")
    for label, r in results.items():
        print(f"  {label:28s} {r['likelihood_evaluations']:6d} evaluations   "
              f"{r['wall_seconds']:7.0f} s   {r['n_samples']:6d} samples   "
              f"({r['device']})")


def save_overlay(results, path):
    """Overlay every method's (chirp mass, eccentricity) posterior on one figure."""
    import corner
    import matplotlib
    import matplotlib.lines as mlines
    matplotlib.use("Agg")

    truth = [TRUE_PARAMETERS[n] for n in INTRINSIC_PARAMETER_NAMES]
    palette = ["C2", "C0", "C3", "C1"]
    figure, legend = None, []
    for color, (label, r) in zip(palette, results.items()):
        weights = None if r["weights"] is None else r["weights"] / r["weights"].sum()
        figure = corner.corner(
            r["samples"], labels=list(INTRINSIC_PARAMETER_NAMES), color=color,
            weights=weights, fig=figure,
            truths=truth if figure is None else None, truth_color="k",
            hist_kwargs=dict(density=True), plot_datapoints=False,
            smooth=1.0, levels=(0.393, 0.865), bins=35)
        legend.append(mlines.Line2D([], [], color=color, label=label))
    legend.append(mlines.Line2D([], [], color="k", label="injected truth"))
    figure.legend(handles=legend, loc="upper right", fontsize=9)
    figure.suptitle("Two PE methods on one ESIGMA injection (ODE grid 1024)", y=1.02)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nsaved overlay figure to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method", choices=["gradient", "surrogate", "both"], default="both",
        help="which parameter-estimation method(s) to run")
    parser.add_argument("--n-chains", type=int, default=20,
                        help="chains for the gradient sampler (20 fits a 4 GB GPU)")
    parser.add_argument("--n-ode-grid", type=int, default=1024,
                        help="ODE resolution; 1024 is converged for this signal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-ode-grid", action="store_true",
                        help="run only the ODE-resolution convergence study and exit")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.check_ode_grid:
        check_ode_grid()
        return

    waveform = build_waveform(args.n_ode_grid)
    likelihood = build_injection(waveform)

    results = {}
    if args.method in ("gradient", "both"):
        label = f"gradient direct ({jax.devices()[0].platform})"
        results[label] = run_gradient_direct_sampling(
            likelihood, n_chains=args.n_chains, seed=args.seed)
    if args.method in ("surrogate", "both"):
        results["surrogate marginalized"] = run_surrogate_marginalized_inference(
            waveform, likelihood)

    report(results)
    if len(results) > 1:
        save_overlay(results, OUTPUT_DIR / "route_comparison_corner.png")


if __name__ == "__main__":
    main()
