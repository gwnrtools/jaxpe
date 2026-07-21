"""Cross-validation and duration-scaling of two PE methods on FD dominant-mode signals.

This is the frequency-domain, aligned-spin counterpart of
``examples/07_td_higher_mode_route_comparison.py``. It injects an IMRPhenomD signal and
recovers its intrinsic parameters two different ways, checking that the posteriors
agree. Agreement is a strong end-to-end test: the two methods share almost no code, so a
match validates both the differentiable likelihood and the marginalized likelihood.

The two methods
---------------
1. Gradient-based direct sampling (Route A)
   Samples the full set of free parameters directly from the differentiable likelihood
   with the normalizing-flow-enhanced Global-Local sampler (Metropolis-adjusted Langevin
   kernel): the intrinsic parameters plus coalescence phase and luminosity distance.

2. Surrogate marginalized inference (Route B)
   Marginalizes phase and distance out of the likelihood analytically (the closed-form
   :class:`~jaxpe.gw.PhaseDistanceMarginalLikelihood`, exact because IMRPhenomD is a
   dominant-(2,2)-mode model) and builds a Gaussian-process surrogate of the resulting
   intrinsic likelihood by active learning (GPry).

Sky position, inclination and coalescence time are held FIXED at truth in both methods.

What this script can do
-----------------------
* Single injection, two "stages" (default): ``nonspin`` recovers (chirp_mass,
  mass_ratio); ``alignedspin`` also recovers (spin1z, spin2z).
* Read injections from a bilby/pycbc-compatible file (``--injection-file``) and run the
  routes on each, so inputs are portable across tools.
* Generate a matched-SNR total-mass sweep as a bilby injection file
  (``--make-mass-sweep``); lower total mass => longer signal, which is the lever for the
  duration-scaling study.
* Profile Route B: how much wall time is spent generating waveforms vs training the GP
  surrogate inside GPry. Measured across this sweep, GPry's GP fit + acquisition is
  ~99% of Route B for vectorized frequency-domain waveforms (the waveform, already JAX,
  is ms-scale even at 16 s), so a jaxified acquisition sampler is the only worthwhile
  port; the waveform dominates only for minutes-per-call models (the production target).
* Summarize duration scaling across the three routes A-CPU, A-GPU and B
  (``--scaling-plot``).

Injection file format (bilby / pycbc compatible)
------------------------------------------------
A whitespace/CSV table with a header of bilby parameter names (one row per injection),
or a JSON dict / list of dicts. Recognized columns (bilby convention, with jaxpe
fallbacks): ``mass_1, mass_2`` or ``chirp_mass, mass_ratio``; aligned spins as
``a_1, a_2, tilt_1, tilt_2`` (spin_z = a*cos(tilt)) or directly ``spin1z, spin2z``;
``theta_jn`` or ``inclination``; ``luminosity_distance, phase, ra, dec, psi,
geocent_time``; optional ``duration``. This is exactly the table ``bilby_pipe
--injection-file`` consumes.

Hardware note (opposite to example 07)
--------------------------------------
The likelihood runs in float64. For the ODE-based ESIGMA of example 07 the gradient
sampler is slower on a small consumer GPU than on CPU; for the vectorized frequency-
domain IMRPhenomD it is faster on the GPU. Run the gradient method once per backend (via
``JAX_PLATFORMS``) to keep the CPU and GPU curves separate.

Examples
--------
    # single injection, both methods, both stages (then overlay):
    JAX_PLATFORMS=cpu python examples/08_fd_dominant_mode_route_comparison.py
    # generate the matched-SNR mass sweep as a bilby injection file:
    python examples/08_fd_dominant_mode_route_comparison.py --make-mass-sweep sweep.dat
    # run all three routes over a file of injections (CPU gradient + surrogate here):
    JAX_PLATFORMS=cpu python examples/08_fd_dominant_mode_route_comparison.py \
        --injection-file sweep.dat --config fast
    # add the GPU gradient curve, then build the duration-scaling summary:
    python examples/08_fd_dominant_mode_route_comparison.py --injection-file sweep.dat \
        --method gradient --config fast
    python examples/08_fd_dominant_mode_route_comparison.py --scaling-plot

Requires jaxpe with the ``surrogate`` extra (GPry). ``--make-mass-sweep`` uses pycbc's
signal-length estimator when available (a 0-post-Newtonian fallback otherwise).
"""

import argparse
import json
import time
from dataclasses import dataclass, field, replace
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

# The shared aligned-spin binary for the single-injection (stage) mode. The two stages
# differ only in whether the spins are injected/recovered or pinned at zero.
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
# priors, so the two methods are compared on identical footing. The chirp-mass prior is
# widened for the mass sweep, whose injected chirp masses span ~4 - 35 solar masses.
MASS_RATIO_PRIOR = (0.5, 1.0)
SPIN_PRIOR = (-0.5, 0.5)
DISTANCE_PRIOR_MPC = (1000.0, 8000.0)
CHIRP_MASS_PRIOR_STAGE = (23.0, 27.0)  # narrow prior around the single stage injection
CHIRP_MASS_HALF_WIDTH = 2.0  # sweep: prior is truth +- this many solar masses

# Sampler / GPry workload presets. "full" reproduces the committed cross-validation;
# "fast" is a timing-oriented preset for the duration-scaling study, where the per-step
# and per-evaluation rates -- not full posterior convergence -- carry the signal.
CONFIGS = {
    "extreme": dict(n_chains=48, n_training=15, n_production=150, max_total=4000),
    "full": dict(n_chains=48, n_training=15, n_production=150, max_total=2000),
    "fast": dict(n_chains=32, n_training=6, n_production=20, max_total=560),
}

# Matched-SNR total-mass sweep defaults (the duration-scaling campaign).
SWEEP_TOTAL_MASSES = (80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0)
SWEEP_MASS_RATIO = 0.8
SWEEP_SPIN1Z = 0.2
SWEEP_SPIN2Z = -0.1
SWEEP_TARGET_SNR = 15.0
SWEEP_SKY = dict(
    inclination=0.4,
    phase=1.5,
    ra=1.2,
    dec=0.5,
    psi=0.8,
    geocent_time=COALESCENCE_TIME_GPS,
)


# A tiny registry so any frequency-domain, dominant-(2,2)-mode model can be swapped in.
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
_MODEL_DISPLAY = {"phenomd": "PhenomD"}


# ============================================================ injection description
@dataclass
class Injection:
    """One injection plus the analysis grid and the parameters to recover.

    ``params`` uses jaxpe names (chirp_mass, mass_ratio, spin1z, spin2z,
    luminosity_distance, inclination, phase, ra, dec, psi, geocent_time). ``recover`` is
    the ordered tuple of intrinsic parameters both methods estimate; every other
    parameter is held fixed at its injected value.
    """

    label: str
    params: dict
    recover: tuple
    duration: float
    sampling_rate: float = SAMPLING_RATE_HZ
    network_snr: float = field(default=None)

    def truth(self):
        return [self.params[n] for n in self.recover]


def chirp_mass_of(mass_1, mass_2):
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def component_masses(total_mass, mass_ratio):
    """(m1, m2) from total mass and mass_ratio = m2/m1 <= 1 (m1 >= m2)."""
    m1 = total_mass / (1.0 + mass_ratio)
    return m1, total_mass - m1


def signal_length_seconds(mass_1, mass_2, f_low):
    """Time in band from f_low to merger. Uses pycbc's estimator; 0PN fallback."""
    try:
        from pycbc.waveform import get_waveform_filter_length_in_time

        return float(
            get_waveform_filter_length_in_time(
                "IMRPhenomD", mass1=mass_1, mass2=mass_2, f_lower=f_low
            )
        )
    except Exception:
        # leading-order (0PN) chirp time; verified to ~10% of the pycbc value
        G, c, msun = 6.674e-11, 2.998e8, 1.989e30
        mc = G * chirp_mass_of(mass_1, mass_2) * msun / c**3  # seconds
        return (5.0 / 256.0) * mc ** (-5.0 / 3.0) * (np.pi * f_low) ** (-8.0 / 3.0)


def auto_duration(mass_1, mass_2, f_low, post_trigger=2.0, safety=1.25, min_dur=4.0):
    """Segment duration (power of 2 s) that contains the signal plus padding."""
    needed = safety * signal_length_seconds(mass_1, mass_2, f_low) + post_trigger + 1.0
    return float(2.0 ** np.ceil(np.log2(max(needed, min_dur))))


# ------------------------------------------------------------ injection file I/O
# Defaults for extrinsic parameters absent from an injection file.
_EXTRINSIC_DEFAULTS = dict(
    inclination=0.0,
    phase=0.0,
    ra=0.0,
    dec=0.0,
    psi=0.0,
    geocent_time=COALESCENCE_TIME_GPS,
)


def _row_to_jaxpe_params(row):
    """Translate one bilby/pycbc/jaxpe parameter row into a jaxpe params dict."""

    def g(*keys):
        return next((row[k] for k in keys if k in row and _finite(row[k])), None)

    # masses: prefer chirp_mass/mass_ratio, else component masses (bilby/pycbc)
    chirp = g("chirp_mass")
    q = g("mass_ratio", "q")
    m1, m2 = g("mass_1", "mass1"), g("mass_2", "mass2")
    if chirp is None or q is None:
        if m1 is None or m2 is None:
            raise ValueError(
                "injection row needs chirp_mass+mass_ratio or mass_1+mass_2"
            )
        m1, m2 = (m1, m2) if m1 >= m2 else (m2, m1)  # enforce m1 >= m2
        chirp, q = chirp_mass_of(m1, m2), m2 / m1

    # aligned spins: prefer explicit z-components, else a*cos(tilt) (bilby convention)
    def spin_z(idx):
        z = g(f"spin{idx}z", f"spin_{idx}z", f"chi_{idx}")
        if z is not None:
            return float(z)
        a, tilt = g(f"a_{idx}"), g(f"tilt_{idx}")
        return float(a) * np.cos(float(tilt)) if a is not None else 0.0

    params = dict(
        chirp_mass=float(chirp),
        mass_ratio=float(q),
        spin1z=spin_z(1),
        spin2z=spin_z(2),
        luminosity_distance=float(
            g("luminosity_distance", "distance") or REFERENCE_DISTANCE_MPC
        ),
        inclination=float(
            g("inclination", "theta_jn") or _EXTRINSIC_DEFAULTS["inclination"]
        ),
    )
    for key, default in _EXTRINSIC_DEFAULTS.items():
        if key == "inclination":
            continue
        val = g(key)
        params[key] = float(val) if val is not None else float(default)
    return params


def _finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_injections(
    path,
    recover=("chirp_mass", "mass_ratio", "spin1z", "spin2z"),
    f_low=LOWER_FREQUENCY_HZ,
    sampling_rate=SAMPLING_RATE_HZ,
):
    """Read a bilby/pycbc-compatible injection file into a list of Injection objects."""
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else [payload]
    else:  # whitespace/CSV table with a header of parameter names
        import pandas as pd

        sep = "," if path.suffix == ".csv" else r"\s+"
        rows = pd.read_csv(path, sep=sep, comment="#").to_dict("records")

    injections = []
    for i, row in enumerate(rows):
        params = _row_to_jaxpe_params(row)
        m1, m2 = _component_from_chirp_q(params["chirp_mass"], params["mass_ratio"])
        dur = (
            float(row["duration"])
            if ("duration" in row and _finite(row.get("duration")))
            else auto_duration(m1, m2, f_low)
        )
        label = str(row.get("label") or f"inj{i}")
        injections.append(Injection(label, params, tuple(recover), dur, sampling_rate))
    return injections


def _component_from_chirp_q(chirp, q):
    """(m1, m2) from chirp mass and mass_ratio = m2/m1.

    Inverts Mc = M_total * eta^(3/5) with eta = q/(1+q)^2, i.e.
    M_total = Mc * (1+q)^(6/5) / q^(3/5).
    """
    total = chirp * (1.0 + q) ** 1.2 / q**0.6
    return component_masses(total, q)


def write_injection_file(path, injections):
    """Write injections as a bilby-convention whitespace table (a_i, tilt_i spins)."""
    cols = [
        "label",
        "mass_1",
        "mass_2",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "theta_jn",
        "luminosity_distance",
        "phase",
        "ra",
        "dec",
        "psi",
        "geocent_time",
        "duration",
    ]
    lines = [
        "# bilby-convention injection table (spin_z = a * cos(tilt))",
        " ".join(cols),
    ]
    for inj in injections:
        p = inj.params
        m1, m2 = _component_from_chirp_q(p["chirp_mass"], p["mass_ratio"])
        a1, t1 = abs(p["spin1z"]), (0.0 if p["spin1z"] >= 0 else np.pi)
        a2, t2 = abs(p["spin2z"]), (0.0 if p["spin2z"] >= 0 else np.pi)
        vals = [
            inj.label,
            m1,
            m2,
            a1,
            a2,
            t1,
            t2,
            p["inclination"],
            p["luminosity_distance"],
            p["phase"],
            p["ra"],
            p["dec"],
            p["psi"],
            p["geocent_time"],
            inj.duration,
        ]
        lines.append(" ".join(str(v) for v in vals))
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------- stage / sweep constructors
def stage_injection(stage):
    """Build the single-injection Injection for the default 'stage' mode."""
    params = {**BASE_PARAMETERS, **STAGE_SPINS[stage]}
    recover = ("chirp_mass", "mass_ratio") + (
        ("spin1z", "spin2z") if stage == "alignedspin" else ()
    )
    return Injection(stage, params, recover, SEGMENT_DURATION_S, SAMPLING_RATE_HZ)


def make_mass_sweep(
    model_name,
    total_masses=SWEEP_TOTAL_MASSES,
    mass_ratio=SWEEP_MASS_RATIO,
    spin1z=SWEEP_SPIN1Z,
    spin2z=SWEEP_SPIN2Z,
    target_snr=SWEEP_TARGET_SNR,
    f_low=LOWER_FREQUENCY_HZ,
    sampling_rate=SAMPLING_RATE_HZ,
):
    """Build a matched-SNR total-mass sweep (fixed q and spins), tuning distance per
    injection to `target_snr`. Lower total mass => longer signal => the duration lever.
    """
    from jaxpe.gw import make_injection

    injections = []
    for total in total_masses:
        m1, m2 = component_masses(total, mass_ratio)
        chirp = chirp_mass_of(m1, m2)
        dur = auto_duration(m1, m2, f_low)
        base = dict(
            chirp_mass=chirp,
            mass_ratio=mass_ratio,
            spin1z=spin1z,
            spin2z=spin2z,
            luminosity_distance=REFERENCE_DISTANCE_MPC,
            **SWEEP_SKY,
        )
        waveform = _build_model(model_name, f_low)
        like = make_injection(
            waveform,
            base,
            detector_names=("H1", "L1"),
            duration=dur,
            sampling_rate=sampling_rate,
            f_min=f_low,
            noise_seed=None,
        )
        snr_ref = float(
            np.sqrt(
                sum(
                    s**2
                    for s in like.optimal_snr(
                        {k: jnp.asarray(v) for k, v in base.items()}
                    ).values()
                )
            )
        )
        base["luminosity_distance"] = REFERENCE_DISTANCE_MPC * snr_ref / target_snr
        params = base
        injections.append(
            Injection(
                f"M{int(total)}",
                params,
                ("chirp_mass", "mass_ratio", "spin1z", "spin2z"),
                dur,
                sampling_rate,
                network_snr=target_snr,
            )
        )
        print(
            f"[sweep] M_total={total:5.1f}  chirp={chirp:6.2f}  duration={dur:5.1f}s  "
            f"distance={params['luminosity_distance']:7.0f}Mpc  (SNR->{target_snr:.0f})"
        )
    return injections


# ============================================================ likelihood + methods
def build_injection_likelihood(model_name, inj: Injection):
    """Inject `inj` into a two-detector network (Hanford, Livingston), no noise."""
    from jaxpe.gw import make_injection

    waveform = _build_model(model_name, LOWER_FREQUENCY_HZ)
    likelihood = make_injection(
        waveform,
        inj.params,
        detector_names=("H1", "L1"),
        duration=inj.duration,
        sampling_rate=inj.sampling_rate,
        f_min=LOWER_FREQUENCY_HZ,
        noise_seed=None,
    )
    per_detector = likelihood.optimal_snr(
        {k: jnp.asarray(v) for k, v in inj.params.items()}
    )
    inj.network_snr = float(np.sqrt(sum(s**2 for s in per_detector.values())))
    n_freq = int(len(likelihood.freqs))
    print(
        f"[injection:{inj.label}] {model_name}  duration={inj.duration:.1f}s  "
        f"n_freq={n_freq}  network SNR {inj.network_snr:.1f}"
    )
    return likelihood, n_freq


def _chirp_prior(inj: Injection):
    """Chirp-mass prior: narrow fixed window for the stage demo, truth-centred for a sweep."""
    if inj.label in STAGE_SPINS:
        return CHIRP_MASS_PRIOR_STAGE
    mc = inj.params["chirp_mass"]
    half_width = 0.2 * mc
    lower_bound = max(0.2, mc - half_width)
    return (lower_bound, mc + half_width)


# ------------------------------------------------- method 1: gradient direct sampling
def run_gradient_direct_sampling(
    likelihood, inj: Injection, n_freq, config="full", seed=0
):
    """Sample the intrinsic parameters plus phase and distance directly with gradients."""
    from jaxpe.core.priors import JointPrior, PowerLaw, Uniform
    from jaxpe.core.problem import InferenceProblem
    from jaxpe.kernels import MALA
    from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

    cfg = dict(CONFIGS[config])
    if inj.params.get("chirp_mass", 20) < 15:  # M20 has mc ~ 8.7
        cfg["n_chains"] = min(cfg["n_chains"], 8)

    device = jax.devices()[0]
    print(
        f"\n=== Route A: gradient direct sampling [{inj.label}] (device: {device}) ==="
    )

    cm_prior = _chirp_prior(inj)
    sampled = {
        "chirp_mass": Uniform(low=cm_prior[0], high=cm_prior[1]),
        "mass_ratio": Uniform(low=MASS_RATIO_PRIOR[0], high=MASS_RATIO_PRIOR[1]),
        "phase": Uniform(low=0.0, high=2 * np.pi),
        "luminosity_distance": PowerLaw(
            alpha=2.0, low=DISTANCE_PRIOR_MPC[0], high=DISTANCE_PRIOR_MPC[1]
        ),
    }
    if "spin1z" in inj.recover:
        sampled["spin1z"] = Uniform(low=SPIN_PRIOR[0], high=SPIN_PRIOR[1])
        sampled["spin2z"] = Uniform(low=SPIN_PRIOR[0], high=SPIN_PRIOR[1])
    prior = JointPrior(sampled)

    fixed = {k: jnp.asarray(v) for k, v in inj.params.items() if k not in sampled}
    problem = InferenceProblem(
        prior=prior,
        log_likelihood=lambda s: likelihood.log_likelihood({**fixed, **s}),
    )

    buffer = cfg["n_chains"] * 20
    gl_config = GlobalLocalConfig(
        n_chains=cfg["n_chains"],
        n_prelim_loops=2,
        n_training_loops=cfg["n_training"],
        n_production_loops=cfg["n_production"],
        n_local_steps=50,
        n_global_steps=50,
        local_thin=3,
        buffer_size=15 * buffer,
        flow_layers=6,
        nn_width=48,
        n_epochs=40,
        batch_size=min(256, 15 * buffer),
    )
    kernel = MALA(step_size=0.03)
    sampler = Sampler(kernel, problem=problem, config=gl_config)
    key = jax.random.PRNGKey(seed)
    start_points = best_of_prior_init(
        key,
        problem,
        cfg["n_chains"],
        n_draws=5000,
        batch_size=min(256, cfg["n_chains"] * 2),
    )

    # Exclude JIT/XLA compilation from the performance measurement, identically on CPU and
    # GPU: run a minimal-loop probe (loop counts dropped to 1, everything else -- shapes,
    # the *same* kernel instance, flow architecture -- unchanged) which warms the
    # module-level jit cache (jaxpe.kernels.base._run_chains_jit, _global_block, the flow
    # trainer step; all @jit / @eqx.filter_jit). The timed run then reuses those compiled
    # executables on whatever backend this process is on, so its wall time is
    # execution-only. Compile is reported separately.
    probe_config = replace(
        gl_config,
        n_prelim_loops=1,
        n_training_loops=1,
        n_production_loops=1,
        n_epochs=1,
    )
    t_probe = time.time()
    Sampler(kernel, problem=problem, config=probe_config).run(key, x0=start_points)
    compile_seconds = time.time() - t_probe

    started = time.time()
    result = sampler.run(key, x0=start_points)
    wall_seconds = time.time() - started  # execution only (compile warmed above)

    samples = sampler.to_physical(result.samples).reshape(-1, problem.n_dim)
    names = list(problem.names)
    order = [names.index(n) for n in inj.recover]
    gradient_steps = (
        gl_config.n_prelim_loops
        + gl_config.n_training_loops
        + gl_config.n_production_loops
    ) * gl_config.n_local_steps
    print(
        f"[gradient:{inj.label}] {samples.shape[0]} samples, ~{gradient_steps} gradient "
        f"steps, {wall_seconds:.0f} s exec (+{compile_seconds:.0f} s compile) on {device}"
    )
    return dict(
        samples=samples[:, order],
        weights=None,
        wall_seconds=wall_seconds,  # execution only (compile excluded)
        compile_seconds=compile_seconds,
        likelihood_evaluations=gradient_steps,
        method="gradient",
        device=device.platform,
        n_samples=samples.shape[0],
        duration=inj.duration,
        n_freq=n_freq,
        network_snr=inj.network_snr,
    )


# --------------------------------------------- method 2: surrogate marginalized inference
def _gpry_timing_split(engine) -> dict:
    """Sum GPry's own per-iteration wall-clock timers over ``engine.run()``.

    GPry records a per-iteration timing table internally (``gpry.progress.Progress``,
    one row per active-learning iteration). We read it here to break the otherwise
    lumped "GPry" cost into the stages the §D4 port decision turns on:

    - ``acquire_seconds`` (``time_acquire``): optimizing the acquisition function --
      NORA runs a nested sampler over the GP surrogate. Embarrassingly parallel,
      vmap-friendly, uses a *cached* GP factorization (no per-eval fp64 Cholesky):
      the clean JAX/BlackJAX port target if it dominates.
    - ``fit_seconds`` (``time_fit``): refitting the GP after adding points --
      hyperparameter optimization + O(N^3) fp64 Cholesky over the training set. On a
      consumer GPU with throttled fp64 this can be *slower* than LAPACK at N<=few*10^3
      (design note D4): a port here may lose.
    - ``gpry_truth_seconds`` (``time_truth``): GPry's own timing of the true-model
      (waveform) calls -- an independent cross-check on our ``timed_loglike`` sum.
    - ``convergence_seconds`` / ``inloop_mc_seconds`` (``time_convergence`` /
      ``time_mc``): the convergence criterion and any in-loop MC sampling.

    We also read GPry's surrogate-evaluation *counters* (``evals_acquire``,
    ``evals_fit``). GPry increments these by ``len(X)`` per predictive call
    (``gpr.py``: ``self.n_eval += len(X)``), so they count surrogate **points**, not
    calls. That makes ``acquire_seconds / acquire_evals`` a wall-time *per acquisition
    point*, directly comparable to a measured cost per point of the GP posterior
    predictive -- which is what decides whether the acquisition is flops-bound (a JAX
    port buys the ratio of the two linear algebras) or overhead-bound (it does not: the
    win would have to come from jitting away the sampler's Python loop instead).

    Missing columns/rows (older GPry, non-main MPI ranks) count as zero. NaN-safe.
    """
    try:
        df = engine.runner.progress.data
    except AttributeError:
        return {}

    def col(name: str) -> float:
        return float(np.nansum(df[name].to_numpy(dtype=float))) if name in df else 0.0

    return dict(
        acquire_seconds=col("time_acquire"),
        fit_seconds=col("time_fit"),
        gpry_truth_seconds=col("time_truth"),
        convergence_seconds=col("time_convergence"),
        inloop_mc_seconds=col("time_mc"),
        # surrogate-point counters (denominators for the per-point costs above)
        acquire_evals=int(col("evals_acquire")),
        fit_evals=int(col("evals_fit")),
        n_iterations=int(len(df)),
    )


def run_surrogate_marginalized_inference(
    likelihood, inj: Injection, n_freq, config="full", seed=11, jax_acquisition=False
):
    """Learn a GPry surrogate of the closed-form phase+distance-marginalized likelihood,
    profiling waveform-generation time vs GP-training/acquisition time.

    ``jax_acquisition`` selects the experimental JAX/BlackJAX acquisition nested sampler
    (Phase 2.5) instead of GPry-native NORA. It is **off by default**: the native path is
    the validated reference, and keeping it the default is what lets the two be compared
    head-to-head (gate G2.5). See ``docs/gpry_fusion_design.md``.
    """
    from jaxpe.gw import PhaseDistanceMarginalLikelihood
    from jaxpe.surrogate import GPryEngine

    cfg = CONFIGS[config]
    print(f"\n=== Route B: surrogate marginalized inference [{inj.label}] (GPry) ===")
    names = inj.recover
    bounds = {"chirp_mass": _chirp_prior(inj), "mass_ratio": MASS_RATIO_PRIOR}
    if "spin1z" in names:
        bounds["spin1z"] = SPIN_PRIOR
        bounds["spin2z"] = SPIN_PRIOR
    fixed_ext = {k: inj.params[k] for k in ("ra", "dec", "psi", "inclination")}
    for spin in ("spin1z", "spin2z"):  # any spin not recovered is held fixed at truth
        if spin not in names:
            fixed_ext[spin] = inj.params[spin]

    marginal = PhaseDistanceMarginalLikelihood(
        likelihood,
        names,
        fixed_ext,
        dist_bounds=DISTANCE_PRIOR_MPC,
        dist_power=2.0,
        d_ref=REFERENCE_DISTANCE_MPC,
        check_params=inj.params,
    )
    print(
        f"[surrogate:{inj.label}] dominant-mode residual "
        f"{marginal.dominant_mode_residual:.2e} (0 => closed form exact)"
    )

    # Profiling wrapper: accumulate time spent in the (waveform + overlap) likelihood.
    profile = dict(waveform_seconds=0.0, n_calls=0, first_call_seconds=0.0)

    def timed_loglike(x):
        t0 = time.perf_counter()
        if x.ndim == 2 and x.shape[0] > 32:
            import jax.numpy as jnp

            res = []
            for i in range(0, x.shape[0], 32):
                res.append(marginal(x[i : i + 32]))
            value = jnp.concatenate(res)
        else:
            value = marginal(x)
        dt = time.perf_counter() - t0
        if profile["n_calls"] == 0:
            profile["first_call_seconds"] = dt  # includes one-time JIT compile
        profile["waveform_seconds"] += dt
        profile["n_calls"] += 1
        return value

    # Scale n_initial relative to a M=50 reference (mc ~ 21.7)
    m_tot = inj.params.get("mass_1", 0.0) + inj.params.get("mass_2", 0.0)
    if m_tot == 0.0:
        m_tot = inj.params.get("chirp_mass", 21.7) * 2.3

    snr = getattr(inj, "network_snr", 15.0) or 15.0
    mc_prior_width = bounds["chirp_mass"][1] - bounds["chirp_mass"][0]

    # Scale based on M=50, SNR=15, prior width=4.0 (where n_init ~ 60 was sufficient)
    scale_factor = (50.0 / m_tot) ** 2 * (snr / 15.0) * (mc_prior_width / 4.0)

    base_n_initial = cfg.get("n_training", 15) * 4
    n_init = int(base_n_initial * max(1.0, scale_factor))
    n_init = min(n_init, 2000)

    started = time.time()
    diagnostics = None
    for attempt in range(3):  # UltraNest's MLFriends is stochastic; retry on failure
        try:
            engine = GPryEngine(
                timed_loglike,
                bounds=bounds,
                options={
                    "seed": seed + attempt,
                    "options": {
                        "max_total": cfg.get("max_total", 560),
                        "n_initial": n_init,
                        "max_initial": n_init * 2,
                    },
                },
                verbose=0,
                jax_acquisition=jax_acquisition,
            )
            diagnostics = engine.run()
            break
        except Exception as error:  # noqa: BLE001
            print(f"  [attempt {attempt} failed: {type(error).__name__}: {error}]")
    if diagnostics is None:
        raise RuntimeError(f"GPry failed three times on {inj.label}")
    run_seconds = time.time() - started  # active-learning loop only
    t_sample0 = time.time()
    posterior = engine.sample()  # final MC over the converged surrogate (separate cost)
    sample_seconds = time.time() - t_sample0
    wall_seconds = time.time() - started

    # Exclude compile from the performance measure, symmetric with Route A: the only JAX
    # compilation in Route B is the marginal's overlaps jit on the FIRST waveform call
    # (first_call_seconds); GPry's GP fit + acquisition are numpy/sklearn (no compile).
    # So execution-only wall = total - first_call, and the reported waveform time is the
    # steady-state (compile-excluded) total.
    compile_seconds = profile["first_call_seconds"]
    t_wave = max(profile["waveform_seconds"] - compile_seconds, 0.0)  # compile-excluded
    exec_seconds = max(wall_seconds - compile_seconds, 0.0)
    t_gp = max(exec_seconds - t_wave, 0.0)

    # Rec 1: break the lumped GPry cost into GP-refit vs acquisition-NS (the D4 split).
    # GPry's internal timers measure the loop stages; the final MC is sample_seconds.
    split = _gpry_timing_split(engine)
    fit_s = split.get("fit_seconds", float("nan"))
    acq_s = split.get("acquire_seconds", float("nan"))
    conv_s = split.get("convergence_seconds", 0.0)
    inloop_mc_s = split.get("inloop_mc_seconds", 0.0)
    acq_evals = split.get("acquire_evals", 0)
    fit_evals = split.get("fit_evals", 0)
    n_evals = diagnostics["n_truth_evals"]
    per = (lambda s: s / n_evals) if n_evals else (lambda s: float("nan"))
    print(
        f"[surrogate:{inj.label}] converged={diagnostics['has_converged']}, "
        f"{n_evals} evals, {exec_seconds:.0f} s exec "
        f"(+{compile_seconds:.1f} s compile; waveform {t_wave:.1f}s / GPry {t_gp:.1f}s), "
        f"{len(posterior.x)} samples"
    )
    print(
        f"  GPry split: fit {fit_s:.1f}s ({per(fit_s):.2f}/eval)  "
        f"acquire {acq_s:.1f}s ({per(acq_s):.2f}/eval)  "
        f"convergence {conv_s:.1f}s  in-loop MC {inloop_mc_s:.1f}s  "
        f"final MC {sample_seconds:.1f}s"
        + (
            f"  [acq/fit={acq_s / fit_s:.1f}x]"
            if fit_s and fit_s == fit_s and fit_s > 0
            else ""
        )
    )
    # Per-acquisition-point wall time: the number that separates "the acquisition is
    # flops-bound" (comparable to the GP predictive's own cost per point) from "it is
    # overhead-bound" (orders of magnitude above it -> the sampler's Python loop, not
    # the linear algebra, is the target).
    if acq_evals:
        print(
            f"  acquisition: {acq_evals} surrogate points in {acq_s:.1f}s "
            f"=> {acq_s / acq_evals * 1e6:.1f} us/point"
            + (f"  |  fit: {fit_evals} points" if fit_evals else "")
        )
    return dict(
        samples=np.asarray(posterior.x),
        weights=np.asarray(posterior.weights),
        wall_seconds=exec_seconds,  # execution only (compile excluded)
        compile_seconds=compile_seconds,
        likelihood_evaluations=n_evals,
        method="surrogate_jax" if jax_acquisition else "surrogate",
        device=jax.devices()[0].platform if jax_acquisition else "cpu",
        n_samples=len(posterior.x),
        duration=inj.duration,
        n_freq=n_freq,
        network_snr=inj.network_snr,
        waveform_seconds=t_wave,
        gp_seconds=t_gp,
        n_waveform_calls=profile["n_calls"],
        # Rec 1: GP-fit vs acquisition-NS split (the D4 port decision axis)
        fit_seconds=fit_s,
        acquire_seconds=acq_s,
        convergence_seconds=conv_s,
        inloop_mc_seconds=inloop_mc_s,
        final_mc_seconds=sample_seconds,
        loop_seconds=max(run_seconds - compile_seconds, 0.0),  # AL loop, compile-excl
        gpry_truth_seconds=split.get("gpry_truth_seconds", float("nan")),
        acquire_evals=acq_evals,
        fit_evals=fit_evals,
        n_gpry_iterations=split.get("n_iterations", 0),
    )


# ------------------------------------------------------------------ persistence + overlay
_RUN_STYLE = {
    ("gradient", "cpu"): ("#1f6fb2", "gradient direct (CPU)"),
    ("gradient", "gpu"): ("#0f8f80", "gradient direct (GPU)"),
    ("surrogate", "cpu"): ("#c0392b", "surrogate marginalized"),
}
# fields carried through the npz beyond the samples themselves
_META_FIELDS = (
    "wall_seconds",  # execution only (compile excluded)
    "compile_seconds",
    "likelihood_evaluations",
    "method",
    "device",
    "duration",
    "n_freq",
    "network_snr",
    "waveform_seconds",
    "gp_seconds",
    "n_waveform_calls",
    # Rec 1: GPry internal split (present only for surrogate runs; the D4 port axis)
    "fit_seconds",
    "acquire_seconds",
    "convergence_seconds",
    "inloop_mc_seconds",
    "final_mc_seconds",
    "loop_seconds",
    "gpry_truth_seconds",
    # surrogate-point counters: denominators for the per-point acquisition/fit cost
    "acquire_evals",
    "fit_evals",
    "n_gpry_iterations",
)


def persist_run(model, inj: Injection, result, max_samples=20000):
    """Save one run to output/<model>_<label>_<method>_<device>.npz (keyed by model so a
    future FD model writes to its own files). Gradient chains are thinned."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    samples = np.asarray(result["samples"])
    weights = (
        np.ones(len(samples))
        if result["weights"] is None
        else np.asarray(result["weights"])
    )
    if len(samples) > max_samples:
        rng = np.random.default_rng(0)
        pick = rng.choice(
            len(samples), max_samples, replace=False, p=weights / weights.sum()
        )
        samples, weights = samples[pick], weights[pick]
    meta = {k: result[k] for k in _META_FIELDS if k in result}
    path = OUTPUT_DIR / f"{model}_{inj.label}_{result['method']}_{result['device']}.npz"
    np.savez(
        path,
        samples=samples,
        weights=weights,
        names=np.array(inj.recover),
        truth=np.array(inj.truth()),
        label=inj.label,
        **meta,
    )
    print(f"  saved run to {path}")


def load_runs(model, label):
    """Load every persisted run for a model/label, keyed by (method, device)."""
    runs = {}
    for path in sorted(OUTPUT_DIR.glob(f"{model}_{label}_*.npz")):
        d = np.load(path, allow_pickle=True)
        runs[(str(d["method"]), str(d["device"]))] = {
            "samples": d["samples"],
            "weights": d["weights"],
            "names": [str(n) for n in d["names"]],
            "truth": d["truth"],
            **{k: d[k].item() for k in _META_FIELDS if k in d.files},
        }
    return runs


def credible_interval(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order] / weights.sum())
    return np.interp([0.16, 0.5, 0.84], cumulative, values[order])


def report(label, runs):
    """Print each run's recovered intrinsic posterior and its cost."""
    if not runs:
        return
    names = next(iter(runs.values()))["names"]
    truth = list(next(iter(runs.values()))["truth"])
    print(f"\n--- {label}: recovered intrinsic posterior (median [16th, 84th]) ---")
    for (method, device), r in runs.items():
        lab = _RUN_STYLE.get((method, device), (None, f"{method} ({device})"))[1]
        cols = [
            f"{n} {credible_interval(r['samples'][:, i], r['weights'])[1]:.3f}"
            for i, n in enumerate(names)
        ]
        print(f"  {lab:26s} " + "   ".join(cols))
    print(
        f"  {'injected truth':26s} "
        + "   ".join(f"{n} {t:.3f}" for n, t in zip(names, truth))
    )
    print(f"--- {label}: cost per run ---")
    for (method, device), r in runs.items():
        lab = _RUN_STYLE.get((method, device), (None, f"{method} ({device})"))[1]
        extra = ""
        if method == "surrogate" and "waveform_seconds" in r:
            extra = (
                f"  [waveform {r['waveform_seconds']:.1f}s / "
                f"GPry {r['gp_seconds']:.1f}s]"
            )
        comp = f" (+{r.get('compile_seconds', 0):.0f}s compile)"
        print(
            f"  {lab:26s} {int(r['likelihood_evaluations']):7d} evaluations   "
            f"{r['wall_seconds']:7.0f} s exec{comp}{extra}"
        )


def overlay(model, label, network_snr=None):
    """Overlay every persisted run's intrinsic posterior for a model/label."""
    import corner
    import matplotlib
    import matplotlib.lines as mlines

    matplotlib.use("Agg")
    runs = load_runs(model, label)
    if not runs:
        print(f"[overlay:{label}] no saved runs found; nothing to plot")
        return
    names = next(iter(runs.values()))["names"]
    truth = list(next(iter(runs.values()))["truth"])
    ranges = [
        (
            min(r["samples"][:, i].min() for r in runs.values()),
            max(r["samples"][:, i].max() for r in runs.values()),
        )
        for i in range(len(names))
    ]

    figure, legend = None, []
    for (method, device), r in runs.items():
        color, lab = _RUN_STYLE.get(
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
        legend.append(mlines.Line2D([], [], color=color, label=lab))
    legend.append(mlines.Line2D([], [], color="k", label="injected truth"))
    figure.legend(handles=legend, loc="upper right", frameon=False, fontsize=10)
    snr = f" (network SNR ~{network_snr:.0f})" if network_snr else ""
    display = _MODEL_DISPLAY.get(model, model)
    figure.suptitle(f"{display} {label}: PE routes on one injection{snr}", y=1.02)
    path = OUTPUT_DIR / f"{model}_{label}_route_comparison.png"
    figure.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[overlay:{label}] saved {path}")


# ------------------------------------------------------------------ duration scaling
def scaling_summary(model, labels):
    """Collect per-route wall time vs signal duration across a set of labels (the mass
    sweep), plus the Route-B waveform/GP-training split, and plot + tabulate them."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    # gather: series[(method, device)] = list of (duration, wall, extra...)
    series = {}
    wave_split = []  # (duration, waveform_seconds, gp_seconds, n_evals)
    for label in labels:
        for (method, device), r in load_runs(model, label).items():
            series.setdefault((method, device), []).append(
                (
                    r["duration"],
                    r["wall_seconds"],
                    r.get("n_freq", np.nan),
                    int(r["likelihood_evaluations"]),
                )
            )
            if method == "surrogate" and "waveform_seconds" in r:
                wave_split.append(
                    (
                        r["duration"],
                        r["waveform_seconds"],
                        r["gp_seconds"],
                        int(r["likelihood_evaluations"]),
                    )
                )
    if not series:
        print("[scaling] no runs found for the sweep labels; nothing to plot")
        return

    print("\n=== duration scaling (wall time vs signal duration) ===")
    for (method, device), pts in sorted(series.items()):
        pts.sort()
        lab = _RUN_STYLE.get((method, device), (None, f"{method} ({device})"))[1]
        print(f"  {lab}:")
        for dur, wall, nf, ev in pts:
            print(
                f"    duration {dur:6.1f}s  n_freq {int(nf):6d}  wall {wall:8.1f}s  "
                f"{ev:5d} evals"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (method, device), pts in sorted(series.items()):
        pts.sort()
        d = np.array([p[0] for p in pts])
        w = np.array([p[1] for p in pts])
        color, lab = _RUN_STYLE.get(
            (method, device), ("#555555", f"{method} ({device})")
        )
        # scatter, not lines: several masses share a (power-of-two) duration, so a
        # connecting line would zig-zag vertically at the same x
        axes[0].plot(d, w, "o", color=color, label=lab, markersize=7, alpha=0.85)
    axes[0].set(
        xlabel="signal duration (s)",
        ylabel="wall time (s)",
        title="Total wall time vs duration",
        xscale="log",
        yscale="log",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.3, which="both")

    if wave_split:
        wave_split.sort()
        d = np.array([p[0] for p in wave_split])
        wv = np.array([p[1] for p in wave_split])
        gp = np.array([p[2] for p in wave_split])
        n = np.array([p[3] for p in wave_split])
        axes[1].plot(
            d, wv, "o", color="#c0392b", label="waveform generation", markersize=7
        )
        axes[1].plot(
            d, gp, "s", color="#7d3c98", label="GP fit + acquisition", markersize=7
        )
        axes[1].set(
            xlabel="signal duration (s)",
            ylabel="wall time (s)",
            title="Route B: waveform vs GPry overhead",
            xscale="log",
            yscale="log",
        )
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].grid(alpha=0.3, which="both")
        # honest read: report an actual sign change, else name the regime + the per-call
        # waveform cost at which the waveform would overtake the GPry overhead
        gpry_per_eval = gp / n
        if np.all(wv < gp):
            print(
                f"\n[scaling] Route B is GPry-dominated across the whole range: GP fit + "
                f"acquisition is {np.min(gp / wv):.0f}-{np.max(gp / wv):.0f}x the waveform "
                f"time (no crossover). The waveform (already JAX) is "
                f"{np.min(wv / n) * 1e3:.0f}-{np.max(wv / n) * 1e3:.0f} ms/call and would "
                f"overtake GPry only above ~{np.min(gpry_per_eval):.1f}-"
                f"{np.max(gpry_per_eval):.1f} s/call. Lever: a JAX acquisition nested "
                f"sampler (the BlackJAX seam), not the GP fit or a waveform port."
            )
        elif np.all(wv > gp):
            print(
                "\n[scaling] Route B is waveform-dominated across the whole range: the "
                "surrogate's value is minimizing (expensive) waveform calls."
            )
        else:
            sign = np.sign(wv - gp)
            idx = int(np.where(np.diff(sign) != 0)[0][0])
            print(
                f"\n[scaling] Route B waveform/GPry crossover between "
                f"{d[idx]:.0f}s and {d[idx + 1]:.0f}s."
            )
    fig.suptitle(f"{_MODEL_DISPLAY.get(model, model)}: PE duration scaling", y=1.02)
    path = OUTPUT_DIR / f"{model}_duration_scaling.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[scaling] saved {path}")


# ------------------------------------- Rec 2: real (case-2) EOB per-call cost profiling
# The §D4 port decision hinges on where an external, non-JAX EOB waveform's per-call cost
# sits relative to GPry's per-eval overhead (measured ~1.1-2.6 s here). The design note
# *assumed* case-(2) models cost 0.5-10 min/call; this measures them directly, across a
# BBH->BNS total-mass grid, warmup-excluded (same compile/first-call exclusion discipline
# as the JAX routes). Timing is done "properly": the sampling rate is set from the
# waveform's own highest frequency content (the ringdown), not a fixed 2048 Hz -- planning
# for the (4,4) mode in higher-mode models -- and lower masses (longer signals, higher
# ringdown) are where the ODE-integration cost can finally overtake the GPry overhead.
#
# engine: which generator; highest_m: highest angular mode m reaching the detector (sets
# Nyquist); cap44: restrict to (l,m)<=(4,4) so "highest content" is the well-measured
# (4,4), not a marginal (5,5) that only inflates fs.
_EOB_MODELS = {
    "TEOBResumS": dict(engine="lal", highest_m=2, cap44=False),
    "SEOBNRv4": dict(engine="lal", highest_m=2, cap44=False),
    "SEOBNRv4_opt": dict(engine="lal", highest_m=2, cap44=False),
    "SEOBNRv4HM": dict(engine="lal", highest_m=4, cap44=True),
    "SEOBNRv5HM": dict(engine="pyseobnr", highest_m=4, cap44=True),
    "SEOBNRv5PHM": dict(engine="pyseobnr", highest_m=4, cap44=True),
}
_MODE_ARRAY_44 = [(2, 2), (2, 1), (3, 3), (4, 4)]  # standard HM set capped at (4,4)
# Dominant-QNM ringdown frequency of mode (m,m) relative to (2,2): f_{mm}/f_{22}. Kerr
# values shift together with remnant spin, so the ratio is stable to ~10%.
_QNM_RATIO = {2: 1.0, 3: 1.60, 4: 2.10, 5: 2.60}
# BBH -> BNS total-mass grid (Msun) at fixed q=0.8, chi=+0.2/-0.1. Lower mass = longer
# signal (more ODE steps) and higher ringdown (higher fs) -- both raise the per-call cost.
_EOB_MASS_GRID = (80.0, 60.0, 40.0, 20.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.8)


def _ringdown_22_hz(total_mass_msun):
    """Approximate (2,2) ringdown (dominant QNM) frequency of the remnant. M_f ~= 0.95
    M_tot (~5% radiated); Re(M_f omega_220) ~= 0.53 for a ~0.7 remnant (0.374 Schwarzschild,
    rising with spin). f = Re(omega)/(2 pi) . c^3/(G M_f)."""
    m_f = 0.95 * total_mass_msun
    c3_over_2piG_msun = 32312.0  # c^3 / (2 pi G M_sun), Hz
    return 0.53 * c3_over_2piG_msun / m_f


def _physical_sampling_rate(total_mass_msun, highest_m, floor=2048.0):
    """Smallest power-of-two fs whose Nyquist clears the highest ringdown mode present --
    'commensurate with the highest frequency content', not a fixed rate. The generator
    errors (does not silently alias) if fs is too low, so this is the Nyquist minimum and
    the caller bumps up only if the estimate is marginally low."""
    f_max = _QNM_RATIO[highest_m] * _ringdown_22_hz(total_mass_msun)
    return float(max(floor, 2.0 ** np.ceil(np.log2(2.0 * f_max))))


def _eob_generate(approximant, m1, m2, s1z, s2z, incl, dist_mpc, phi, f_low, fs, cap44):
    """Generate one external TD EOB waveform; return its length. Raises on model
    domain/Nyquist errors so the caller can bump fs or record a genuine model limit."""
    dt = 1.0 / fs
    cfg = _EOB_MODELS[approximant]
    if cfg["engine"] == "pyseobnr":
        from pyseobnr.generate_waveform import GenerateWaveform  # optional dep

        params = dict(
            mass1=m1,
            mass2=m2,
            spin1x=0.0,
            spin1y=0.0,
            spin1z=s1z,
            spin2x=0.0,
            spin2y=0.0,
            spin2z=s2z,
            deltaT=dt,
            f22_start=f_low,
            f_ref=f_low,
            distance=dist_mpc,
            inclination=incl,
            phi_ref=phi,
            approximant=approximant,
        )
        if cap44:
            params["mode_array"] = _MODE_ARRAY_44
        hp, _ = GenerateWaveform(params).generate_td_polarizations()
        return hp.data.length
    import lal
    import lalsimulation as ls  # optional dep

    lal_params = None
    if cap44:  # restrict LAL higher-mode models to (l,m) <= (4,4)
        lal_params = lal.CreateDict()
        mode_array = ls.SimInspiralCreateModeArray()
        for ell, m in _MODE_ARRAY_44:
            ls.SimInspiralModeArrayActivateMode(mode_array, ell, m)
            ls.SimInspiralModeArrayActivateMode(mode_array, ell, -m)
        ls.SimInspiralWaveformParamsInsertModeArray(lal_params, mode_array)
    aid = ls.GetApproximantFromString(approximant)
    hp, _ = ls.SimInspiralChooseTDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0,
        0.0,
        s1z,
        0.0,
        0.0,
        s2z,
        dist_mpc * 1e6 * lal.PC_SI,
        incl,
        phi,
        0.0,
        0.0,
        0.0,
        dt,
        f_low,
        f_low,
        lal_params,
        aid,
    )
    return hp.data.length


def eob_call_seconds(inj: Injection, approximant, repeats=5, f_low=LOWER_FREQUENCY_HZ):
    """Median warmup-excluded seconds for one external EOB call at this injection.

    Returns ``(seconds, fs_used, n_samples)``. fs is set from the waveform's highest
    ringdown mode (``_physical_sampling_rate``); if the generator still reports the content
    exceeds Nyquist we bump fs by powers of two. Repeats are reduced for slow (low-mass)
    calls so a BNS sweep stays affordable. Raises the underlying error if no fs works.
    """
    cfg = _EOB_MODELS[approximant]
    m1, m2 = _component_from_chirp_q(inj.params["chirp_mass"], inj.params["mass_ratio"])
    total_mass = m1 + m2
    s1z, s2z = inj.params.get("spin1z", 0.0), inj.params.get("spin2z", 0.0)
    incl = inj.params.get("inclination", 0.0)
    dist = inj.params.get("luminosity_distance", REFERENCE_DISTANCE_MPC)
    phi = inj.params.get("phase", 0.0)
    cap44 = cfg["cap44"]

    fs0 = _physical_sampling_rate(total_mass, cfg["highest_m"])
    fs_options = [fs0, fs0 * 2, fs0 * 4]  # Nyquist minimum, then bump if marginally low
    last_err = None
    for fs in fs_options:
        try:
            n0 = _eob_generate(
                approximant, m1, m2, s1z, s2z, incl, dist, phi, f_low, fs, cap44
            )
        except Exception as error:  # noqa: BLE001 -- domain/Nyquist; try a higher fs
            last_err = error
            continue

        # adaptive repeats: one timed call, then fewer the slower it is (keep BNS cheap)
        def _one():
            t0 = time.perf_counter()
            _eob_generate(
                approximant, m1, m2, s1z, s2z, incl, dist, phi, f_low, fs, cap44
            )
            return time.perf_counter() - t0

        times = [_one()]
        n_more = 0 if times[0] > 8.0 else (1 if times[0] > 1.5 else repeats - 1)
        times.extend(_one() for _ in range(n_more))
        return float(np.median(times)), float(fs), int(n0)
    if last_err is not None:  # exhausted the fs ladder on a real model/domain error
        raise last_err
    raise RuntimeError(f"{approximant}: no sampling rate in {fs_options} succeeded")


def timing_injections(masses, q=0.8, s1z=0.2, s2z=-0.1):
    """Lightweight injections for a pure waveform-timing sweep (no PE / SNR tuning): the
    same intrinsic configuration as the matched-SNR sweep, extended down to BNS masses.
    """
    injections = []
    for total in masses:
        m1, m2 = component_masses(total, q)
        params = {
            "chirp_mass": chirp_mass_of(m1, m2),
            "mass_ratio": q,
            "spin1z": s1z,
            "spin2z": s2z,
            "inclination": 0.4,
            "luminosity_distance": REFERENCE_DISTANCE_MPC,
            "phase": 1.5,
        }
        label = f"M{total:g}"
        injections.append(
            Injection(
                label,
                params,
                ("chirp_mass", "mass_ratio", "spin1z", "spin2z"),
                auto_duration(m1, m2, LOWER_FREQUENCY_HZ),
                SAMPLING_RATE_HZ,
            )
        )
    return injections


def eob_timing_sweep(model, injections, approximants, repeats=5):
    """Time each external EOB model across the injections and compare its per-call cost to
    the measured GPry per-eval overhead (from saved surrogate runs, where present). Writes
    a JSON table and prints where -- if anywhere -- the waveform overtakes GPry."""
    print(f"\n=== Real EOB per-call cost vs GPry per-eval overhead ({model}) ===")
    print(
        "(warmup-excluded; fs set from the (4,4) ringdown; D4 trigger: EOB/call ~ GPry/eval)"
    )
    results = {}
    for approximant in approximants:
        if approximant not in _EOB_MODELS:
            print(f"  [skip {approximant}: unknown; known = {list(_EOB_MODELS)}]")
            continue
        row = {}
        for inj in injections:
            m1, m2 = _component_from_chirp_q(
                inj.params["chirp_mass"], inj.params["mass_ratio"]
            )
            gp_per_eval = _gpry_per_eval(model, inj.label)  # from saved Route B, or nan
            try:
                sec, fs, n = eob_call_seconds(inj, approximant, repeats=repeats)
                crossed = sec > gp_per_eval  # nan compares False -> "gpry-dominated"
                row[inj.label] = dict(
                    total_mass=m1 + m2,
                    seconds=sec,
                    fs=fs,
                    n_samples=n,
                    signal_s=n / fs,
                    gpry_per_eval=gp_per_eval,
                )
                gp_note = (
                    f"  vs GPry/eval {gp_per_eval:.2f}s "
                    f"[{'WAVEFORM>GPry' if crossed else 'gpry-dominated'}, "
                    f"{max(gp_per_eval, sec) / min(gp_per_eval, sec):.0f}x]"
                    if gp_per_eval == gp_per_eval
                    else ""
                )
                print(
                    f"  {approximant:12s} {inj.label:>5s} (M={m1 + m2:5.1f}, "
                    f"sig {n / fs:6.1f}s @fs={fs:6.0f}): {sec * 1e3:9.1f} ms/call{gp_note}"
                )
            except Exception as error:  # noqa: BLE001
                row[inj.label] = dict(
                    error=f"{type(error).__name__}: {str(error)[:70]}"
                )
                print(
                    f"  {approximant:12s} {inj.label:>5s} (M={m1 + m2:5.1f}): "
                    f"FAILED ({type(error).__name__}: {str(error)[:50]})"
                )
        results[approximant] = row
    path = OUTPUT_DIR / f"{model}_eob_call_timing.json"
    with open(path, "w") as handle:
        json.dump(
            dict(
                f_low=LOWER_FREQUENCY_HZ,
                repeats=repeats,
                cap="(l,m)<=(4,4) for HM",
                models=results,
            ),
            handle,
            indent=2,
        )
    print(f"[eob-timing] saved {path}")
    return results


def _gpry_per_eval(model, label):
    """GPry per-eval overhead (gp_seconds / n_waveform_calls) from a saved Route B run,
    or NaN if that surrogate run is not on disk."""
    path = OUTPUT_DIR / f"{model}_{label}_surrogate_cpu.npz"
    if not path.exists():
        return float("nan")
    d = np.load(path, allow_pickle=True)
    if "gp_seconds" not in d or "n_waveform_calls" not in d:
        return float("nan")
    n = float(d["n_waveform_calls"])
    return float(d["gp_seconds"]) / n if n else float("nan")


# ------------------------------------------------------------------ driver
def run_injection(model, inj, method, config, seed, jax_acquisition=False):
    """Run the requested method(s) on one injection and persist each."""
    likelihood, n_freq = build_injection_likelihood(model, inj)
    if method in ("gradient", "both"):
        persist_run(
            model,
            inj,
            run_gradient_direct_sampling(
                likelihood, inj, n_freq, config=config, seed=seed
            ),
        )
    if method in ("surrogate", "both"):
        persist_run(
            model,
            inj,
            run_surrogate_marginalized_inference(
                likelihood,
                inj,
                n_freq,
                config=config,
                seed=seed,
                jax_acquisition=jax_acquisition,
            ),
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODELS, default="phenomd")
    parser.add_argument(
        "--stage",
        choices=["nonspin", "alignedspin", "both"],
        default="both",
        help="single-injection mode (ignored with a file)",
    )
    parser.add_argument(
        "--method", choices=["gradient", "surrogate", "both"], default="both"
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default="full",
        help="'full' reproduces the cross-validation; 'fast' is for timing",
    )
    parser.add_argument(
        "--injection-file", help="bilby/pycbc injection file to analyze"
    )
    parser.add_argument(
        "--make-mass-sweep",
        metavar="PATH",
        help="write the matched-SNR mass sweep to a bilby file and exit",
    )
    parser.add_argument(
        "--scaling-plot",
        action="store_true",
        help="build the duration-scaling summary from saved sweep runs",
    )
    parser.add_argument(
        "--eob-timing",
        metavar="MODELS",
        help="comma-separated external EOB approximants (e.g. "
        "'TEOBResumS,SEOBNRv4,SEOBNRv5HM,SEOBNRv5PHM') to time per-call across a "
        "BBH->BNS mass grid (fs set from the (4,4) ringdown) and compare to the GPry "
        "per-eval overhead (Rec 2 / D4 checkpoint)",
    )
    parser.add_argument(
        "--eob-mass-grid",
        metavar="M1,M2,...",
        help="override total masses (Msun) for --eob-timing (default: BBH->BNS grid)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jax-acquisition",
        action="store_true",
        help="Route B: use the experimental JAX/BlackJAX acquisition NS (Phase 2.5) "
        "instead of GPry-native NORA. Default off (native is the validated reference).",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="skip running; rebuild overlays from saved runs",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.make_mass_sweep:
        write_injection_file(args.make_mass_sweep, make_mass_sweep(args.model))
        print(f"\nwrote mass-sweep injection file to {args.make_mass_sweep}")
        return

    if args.scaling_plot:
        scaling_summary(args.model, [f"M{int(m)}" for m in SWEEP_TOTAL_MASSES])
        return

    if args.injection_file:
        injections = load_injections(args.injection_file)
    else:
        injections = [
            stage_injection(s)
            for s in (
                ["nonspin", "alignedspin"] if args.stage == "both" else [args.stage]
            )
        ]

    if args.eob_timing:
        approximants = [a.strip() for a in args.eob_timing.split(",") if a.strip()]
        if args.injection_file:
            eob_injections = injections  # time on the same injections as the PE sweep
        else:
            masses = (
                [float(m) for m in args.eob_mass_grid.split(",")]
                if args.eob_mass_grid
                else list(_EOB_MASS_GRID)
            )
            eob_injections = timing_injections(masses)
        eob_timing_sweep(args.model, eob_injections, approximants)
        return

    for inj in injections:
        try:
            if not args.overlay_only:
                run_injection(
                    args.model,
                    inj,
                    args.method,
                    args.config,
                    args.seed,
                    jax_acquisition=args.jax_acquisition,
                )
            runs = load_runs(args.model, inj.label)
            report(inj.label, runs)
            overlay(args.model, inj.label, network_snr=inj.network_snr)
        except Exception as error:  # keep a long sweep alive if one injection fails
            print(
                f"[ERROR] injection {inj.label} failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )


if __name__ == "__main__":
    main()
