"""Command-line interface for jaxpe Universal PE Drivers."""

import argparse
import json
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxpe.gw import IMRPhenomD, IMRPhenomT, bbh_priors, make_injection
from jaxpe.gw.likelihood import PhaseDistanceMarginalLikelihood
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init, PostProcessor
from jaxpe.kernels import MALA, HMC
from jaxpe.diagnostics.plots import corner_plot

# Prior ranges the CLI samples in. Injection truths are drawn from a slightly
# narrower box so that no truth sits on a prior edge (which would rail the posterior).
PRIOR_RANGES = {
    "chirp_mass": (10.0, 50.0),
    "mass_ratio": (0.1, 1.0),
    "spin1z": (-0.9, 0.9),
    "spin2z": (-0.9, 0.9),
    "luminosity_distance": (100.0, 2000.0),
}
DRAW_RANGES = {
    "chirp_mass": (15.0, 45.0),
    "mass_ratio": (0.3, 1.0),
    "spin1z": (-0.5, 0.5),
    "spin2z": (-0.5, 0.5),
    "luminosity_distance": (300.0, 1500.0),
}
FIDUCIAL_GEOCENT_TIME = 1126259462.4

# The historical fixed reference binary: a GW150914-like BBH, face-on, optimally
# oriented, at the GW150914 epoch. Kept as a stable point of comparison across runs
# and releases; selected with --fiducial. All values sit inside PRIOR_RANGES.
FIDUCIAL_INJECTION = {
    "chirp_mass": 30.0,
    "mass_ratio": 0.8,
    "spin1z": 0.0,
    "spin2z": 0.0,
    "luminosity_distance": 700.0,
    "geocent_time": FIDUCIAL_GEOCENT_TIME,
    "phase": 0.0,
    "inclination": 0.0,
    "ra": 0.0,
    "dec": 0.0,
    "psi": 0.0,
}

# Data-conditioning settings. Recorded into run_config.json so that process-samples
# reconstructs the same problem the sampler saw instead of assuming these values.
DURATION = 4.0
SAMPLING_RATE = 1024.0
F_MIN = 30.0
F_REF = 20.0
TIME_WIDTH = 0.1


def resolve_psd(spec):
    """Map a --psd specification to a psd_fn usable by ``make_injection``.

    ``aligo`` (default) is the built-in analytic Advanced-LIGO ZDHP curve; anything
    else is treated as a path to a two-column ASCII PSD file. Unknown *names* are
    rejected rather than silently ignored -- there is no built-in Cosmic Explorer
    curve in ``jaxpe.gw.psd``, so ``--psd CE`` must be given as a file.
    """
    from jaxpe.gw import aligo_zdhp_psd, psd_from_file

    if spec is None or spec.lower() in ("aligo", "zdhp", "aligo_zdhp"):
        return aligo_zdhp_psd
    path = Path(spec)
    if path.exists():
        return lambda freqs: psd_from_file(path, freqs)
    raise ValueError(
        f"--psd {spec!r} is neither the built-in 'aligo' curve nor an existing file. "
        "jaxpe.gw.psd ships only the analytic aLIGO ZDHP curve; supply any other "
        "detector (CE, ET, ...) as a two-column ASCII file path."
    )


def draw_injection(rng):
    """Draw one injection's true parameters, isotropic in orientation and sky."""
    u = rng.uniform
    return {
        "chirp_mass": float(u(*DRAW_RANGES["chirp_mass"])),
        "mass_ratio": float(u(*DRAW_RANGES["mass_ratio"])),
        "spin1z": float(u(*DRAW_RANGES["spin1z"])),
        "spin2z": float(u(*DRAW_RANGES["spin2z"])),
        "luminosity_distance": float(u(*DRAW_RANGES["luminosity_distance"])),
        "geocent_time": FIDUCIAL_GEOCENT_TIME,
        "phase": float(u(0.0, 2 * np.pi)),
        # isotropic inclination and declination, not uniform in the angle
        "inclination": float(np.arccos(u(-1.0, 1.0))),
        "ra": float(u(0.0, 2 * np.pi)),
        "dec": float(np.arcsin(u(-1.0, 1.0))),
        "psi": float(u(0.0, np.pi)),
    }


def generate_injections(args):
    print("Generating injections...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fail fast on an unusable PSD rather than at run-pe time.
    resolve_psd(args.psd)

    # --fiducial emits one fixed binary. Emitting N copies of it is exactly the
    # "N identical files" behaviour this command used to have by accident, so refuse
    # rather than silently reproduce it.
    if args.fiducial and args.n_injections != 1:
        raise ValueError(
            f"--fiducial produces a single fixed binary, but --n-injections is "
            f"{args.n_injections}; that would write {args.n_injections} identical "
            "files. Use --fiducial with --n-injections 1, or drop --fiducial to draw "
            "a distinct set."
        )

    rng = np.random.default_rng(args.seed)
    for i in range(args.n_injections):
        injection_params = (
            dict(FIDUCIAL_INJECTION) if args.fiducial else draw_injection(rng)
        )
        # Recorded so run-pe can default to the network/noise/PSD this set was
        # generated for. Popped before the dict is handed to the waveform model.
        injection_params["metadata"] = {
            "network": args.network,
            "noise": args.noise,
            "psd": args.psd,
            "seed": None if args.fiducial else args.seed,
            "index": i,
            "fiducial": bool(args.fiducial),
        }
        ipath = outdir / f"inj_{i}.json"
        with open(ipath, "w") as f:
            json.dump(injection_params, f, indent=2)
        print(
            f"Saved {'fiducial ' if args.fiducial else ''}injection {i} to {ipath} "
            f"(Mc={injection_params['chirp_mass']:.2f}, "
            f"q={injection_params['mass_ratio']:.2f}, "
            f"D={injection_params['luminosity_distance']:.0f} Mpc)"
        )


def run_pe(args):
    print("Running PE...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.injection, "r") as f:
        injection_params = json.load(f)

    # Generation-time settings recorded by generate-injections. They are defaults
    # only: an explicit flag on this command always wins. Popped so the dict handed
    # to the waveform model contains physical parameters exclusively.
    meta = injection_params.pop("metadata", {})
    network = args.network if args.network is not None else meta.get("network", "H1,L1")
    noise = args.noise if args.noise is not None else meta.get("noise", "zero")
    psd_spec = args.psd if args.psd is not None else meta.get("psd", "aligo")
    psd_fn = resolve_psd(psd_spec)

    print(f"Loaded injection params: {injection_params}")
    print(f"Network {network}, noise {noise}, psd {psd_spec}")

    if args.domain == "fd":
        waveform = IMRPhenomD(f_ref=F_REF)
    elif args.domain == "td":
        waveform = IMRPhenomT(f_ref=F_REF)
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    prior = bbh_priors(
        chirp_mass=PRIOR_RANGES["chirp_mass"],
        mass_ratio=PRIOR_RANGES["mass_ratio"],
        aligned_spins=PRIOR_RANGES["spin1z"],
        luminosity_distance=PRIOR_RANGES["luminosity_distance"],
        geocent_time=injection_params.get("geocent_time", 0.0),
        time_width=TIME_WIDTH,
    )

    like = make_injection(
        waveform,
        injection_params,
        detector_names=network.split(","),
        duration=DURATION,
        sampling_rate=SAMPLING_RATE,
        f_min=F_MIN,
        psd_fn=psd_fn,
        noise_seed=None if noise == "zero" else 42,
    )

    # Construct likelihood and prior structures
    if args.likelihood == "marginalized_phase_distance":
        if args.domain != "fd":
            raise ValueError(
                "marginalized_phase_distance is only supported for FD domain."
            )
        if args.sampler in ["hmc", "mala"]:
            raise ValueError(
                "marginalized_phase_distance is host-side and incompatible with HMC/MALA."
            )

        names = ["chirp_mass", "mass_ratio", "spin1z", "spin2z"]
        fixed_ext = {
            "ra": injection_params.get("ra", 0.0),
            "dec": injection_params.get("dec", 0.0),
            "psi": injection_params.get("psi", 0.0),
            "inclination": injection_params.get("inclination", 0.0),
        }
        like_callable = PhaseDistanceMarginalLikelihood(
            like, names=names, fixed_ext=fixed_ext, check_params=injection_params
        )
        bounds_dict = {n: PRIOR_RANGES[n] for n in names}

        def log_prob_fn(x):
            return like_callable(x)

    else:
        problem = like.problem(prior)

        # NS/GPry samplers require explicit bounds. We construct them based on the names.
        bounds_dict = {}
        for n in problem.names:
            if n in PRIOR_RANGES:
                bounds_dict[n] = PRIOR_RANGES[n]
            elif "spin" in n:
                bounds_dict[n] = PRIOR_RANGES["spin1z"]
            elif n == "phase":
                bounds_dict[n] = (0.0, 2 * np.pi)
            elif n == "geocent_time":
                bounds_dict[n] = (
                    injection_params.get("geocent_time", 0.0) - 0.1,
                    injection_params.get("geocent_time", 0.0) + 0.1,
                )
            elif n == "inclination":
                bounds_dict[n] = (0.0, np.pi)
            elif n == "ra":
                bounds_dict[n] = (0.0, 2 * np.pi)
            elif n == "dec":
                bounds_dict[n] = (-np.pi / 2, np.pi / 2)
            elif n == "psi":
                bounds_dict[n] = (0.0, np.pi)
            else:
                bounds_dict[n] = (0.0, 1.0)  # fallback

        def log_prob_fn(x):
            # ns/gpry explore the *physical* bounds box, so this must be the
            # physical-space density: log_posterior() expects unconstrained
            # coordinates and InferenceProblem has no log_prob at all. The bounds
            # box only delimits the region, so the prior density is added here --
            # without it the target would be likelihood x uniform, a different
            # posterior from the one the MCMC path samples.
            return problem.log_likelihood_vec(x) + problem.prior.log_prob(x)

    out_samples = outdir / "raw_samples.npz"
    key = jax.random.PRNGKey(42)

    if args.sampler in ["hmc", "mala"]:
        if args.sampler == "hmc":
            kernel = HMC(step_size=0.01, n_leapfrog=10)
        else:
            kernel = MALA(step_size=0.01)

        cfg = GlobalLocalConfig(
            n_chains=args.n_chains,
            n_prelim_loops=args.n_prelim_loops,
            n_training_loops=args.n_training_loops,
            n_production_loops=args.n_production_loops,
        )
        sampler = Sampler(kernel, problem=problem, config=cfg)
        x0 = best_of_prior_init(key, problem, cfg.n_chains)
        res = sampler.run(key, x0=x0)

        # The global-local sampler works in the unconstrained space; process-samples
        # must apply the inverse bijection to recover physical parameters.
        param_names = list(problem.names)
        sample_space = "unconstrained"

        np.savez(
            out_samples,
            samples=res.samples,
            log_prob=res.log_prob,
        )
    elif args.sampler == "ns":
        from jaxpe.surrogate.jax_acquisition import JAXInterfaceBlackJAX

        # JAXInterfaceBlackJAX requires bounds as a (n_dim, 2) array
        names_list = list(bounds_dict.keys())
        bounds_arr = np.array([bounds_dict[n] for n in names_list])
        ns = JAXInterfaceBlackJAX(bounds_arr, verbosity=1)
        # Use a small nlive for testing
        ns.set_precision(nlive=10, num_repeats=5, precision_criterion=0.1)

        if args.likelihood == "marginalized_phase_distance":
            from jax.scipy.special import i0e, logsumexp

            u = jnp.asarray(like_callable._u)
            log_pi = jnp.asarray(like_callable._log_pi)
            log_dD = jnp.asarray(like_callable._log_dD)
            dd = float(like_callable.dd)
            overlaps = like_callable._overlaps

            def ns_logp(x):
                zr, zi, rho2 = overlaps(jnp.asarray(x).ravel())
                abs_z = jnp.hypot(zr, zi)
                log_i0 = jnp.log(i0e(u * abs_z)) + u * abs_z
                integrand = log_pi + log_i0 - 0.5 * u**2 * rho2 + log_dD
                return logsumexp(integrand) - 0.5 * dd

        else:
            ns_logp = log_prob_fn

        @jax.jit
        def logp_jax(x):
            return ns_logp(x)

        X_MC, y_MC, w_MC, _logZ, _logZstd = ns.run(
            logp_jax, param_names=names_list, seed=42
        )

        # Nested sampling explores the *physical* box given by bounds_arr, so these
        # draws are already in physical units -- they must NOT be pushed through the
        # unconstraining bijection again.
        param_names = list(names_list)
        sample_space = "physical"

        samples_flat = np.asarray(X_MC)
        np.savez(
            out_samples,
            samples=samples_flat[None, :, :],  # shape: (1, n_samples, n_dim)
            log_prob=np.asarray(y_MC)[None, :],
            weights=np.asarray(w_MC),
        )
    elif args.sampler == "gpry":
        from jaxpe.surrogate import GPryEngine

        def timed_loglike(x):
            import numpy as np

            if x.ndim == 2:
                res = [log_prob_fn(x[i]) for i in range(x.shape[0])]
                return np.asarray(res)
            return float(log_prob_fn(x))

        ref_bounds_arr = []
        for n in bounds_dict:
            if n in injection_params:
                val = injection_params[n]
                delta = max(0.01 * abs(val), 0.1)
                ref_bounds_arr.append([val - delta, val + delta])
            else:
                ref_bounds_arr.append(bounds_dict[n])

        engine = GPryEngine(
            timed_loglike,
            bounds=bounds_dict,
            options={
                "seed": 42,
                "ref_bounds": np.array(ref_bounds_arr),
                "options": {
                    "max_total": 500,
                    "n_initial": 5,
                    "max_initial": 200,
                },
            },
        )
        engine.run()
        res_gpry = engine.sample()

        # As for nested sampling: GPry samples the physical bounds box.
        param_names = list(res_gpry.names) or list(bounds_dict)
        sample_space = "physical"

        samples_flat = res_gpry.x
        np.savez(
            out_samples,
            samples=samples_flat[None, :, :],
            log_prob=res_gpry.logpost[None, :],
            weights=res_gpry.weights,
        )
    else:
        raise NotImplementedError(
            f"Sampler {args.sampler} not fully implemented yet in generic CLI."
        )

    print(f"Saved raw samples to {out_samples}")

    # Save injection to outdir for process_samples
    with open(outdir / "injection.json", "w") as f:
        json.dump(injection_params, f, indent=2)

    # Record everything process-samples needs to rebuild the *same* problem. Without
    # this it assumed IMRPhenomD / H1,L1,V1 / 11 parameters regardless of the run,
    # which silently mismatched td runs, non-default networks, and the 4-parameter
    # marginalized likelihood.
    run_config = {
        "sampler": args.sampler,
        "likelihood": args.likelihood,
        "domain": args.domain,
        "network": network,
        "noise": noise,
        "psd": psd_spec,
        "duration": DURATION,
        "sampling_rate": SAMPLING_RATE,
        "f_min": F_MIN,
        "f_ref": F_REF,
        "time_width": TIME_WIDTH,
        "prior_ranges": {k: list(v) for k, v in PRIOR_RANGES.items()},
        "param_names": param_names,
        "sample_space": sample_space,
        "weighted": args.sampler in ("ns", "gpry"),
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run configuration to {outdir / 'run_config.json'}")


def process_samples(args):
    print("Processing samples...")
    for file_path in args.files:
        path = Path(file_path)
        out_dir = path.parent

        inj_file = out_dir / "injection.json"
        if not inj_file.exists():
            print(
                f"Warning: {inj_file} not found. Cannot reconstruct problem easily. Skipping {path}."
            )
            continue

        with open(inj_file, "r") as f:
            injection_params = json.load(f)
        injection_params.pop("metadata", None)

        # Prefer the configuration the run actually used; fall back to the historical
        # defaults (with a warning) for output directories predating run_config.json.
        cfg_file = out_dir / "run_config.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
        else:
            print(
                f"WARNING: {cfg_file} not found; assuming an fd / H1,L1,V1 / "
                f"{DURATION} s run. If the samples came from a td or "
                "non-default-network run, the reconstructed parameter mapping may be "
                "wrong. Re-run run-pe to regenerate the configuration."
            )
            cfg = {}

        domain = cfg.get("domain", "fd")
        network = cfg.get("network", "H1,L1,V1")

        data = np.load(path)
        raw = data["samples"]
        weights = data["weights"] if "weights" in data.files else None

        # Only ns/gpry write a weights array, and only they sample the physical box,
        # so its presence identifies the space for legacy runs with no run_config.
        default_space = "physical" if weights is not None else "unconstrained"
        sample_space = cfg.get("sample_space", default_space)
        weighted = cfg.get("weighted", weights is not None)

        if sample_space == "physical":
            # ns/gpry already return physical draws over the prior box, and their
            # (1, n, d) layout carries no chain structure to autocorrelate. Thinning
            # or re-applying the bijection here would corrupt them.
            phys_samples = np.asarray(raw).reshape(-1, raw.shape[-1])
            pnames = cfg.get("param_names") or [
                f"x_{i}" for i in range(phys_samples.shape[-1])
            ]
            print(
                f"{path.name}: {phys_samples.shape[0]} weighted samples already in "
                "physical units; skipping burn-in/thinning."
            )
        else:
            waveform = (
                IMRPhenomD(f_ref=cfg.get("f_ref", F_REF))
                if domain == "fd"
                else IMRPhenomT(f_ref=cfg.get("f_ref", F_REF))
            )
            like = make_injection(
                waveform,
                injection_params,
                detector_names=tuple(network.split(",")),
                duration=cfg.get("duration", DURATION),
                sampling_rate=cfg.get("sampling_rate", SAMPLING_RATE),
                f_min=cfg.get("f_min", F_MIN),
                psd_fn=resolve_psd(cfg.get("psd")),
                noise_seed=None,
            )
            ranges = cfg.get("prior_ranges", {})
            prior = bbh_priors(
                chirp_mass=tuple(ranges.get("chirp_mass", PRIOR_RANGES["chirp_mass"])),
                mass_ratio=tuple(ranges.get("mass_ratio", PRIOR_RANGES["mass_ratio"])),
                aligned_spins=tuple(ranges.get("spin1z", PRIOR_RANGES["spin1z"])),
                luminosity_distance=tuple(
                    ranges.get(
                        "luminosity_distance", PRIOR_RANGES["luminosity_distance"]
                    )
                ),
                geocent_time=injection_params.get("geocent_time", 0.0),
                time_width=cfg.get("time_width", TIME_WIDTH),
            )
            problem = like.problem(prior)

            pp = PostProcessor(problem, raw_samples_file=path)
            phys_samples = pp.process()
            pnames = list(problem.names)

        np.save(out_dir / "posterior_samples.npy", phys_samples)
        print(f"Saved {out_dir / 'posterior_samples.npy'} {phys_samples.shape}")

        try:
            truths = [injection_params.get(n, None) for n in pnames]
            kwargs = {}
            if weighted and weights is not None:
                kwargs["weights"] = np.asarray(weights).ravel()
            fig = corner_plot(phys_samples, names=pnames, truths=truths, **kwargs)
            fig.savefig(out_dir / "corner_thinned.png", dpi=120)
            print(f"Saved {out_dir / 'corner_thinned.png'}")
        except Exception as e:
            print(f"Failed to generate corner plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="jaxpe Universal Parameter Estimation Driver"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for generate-injections
    parser_gen = subparsers.add_parser(
        "generate-injections", help="Generate injections and noise data"
    )
    parser_gen.add_argument(
        "--network",
        type=str,
        default="H1,L1",
        help="Detector network recorded with the injection; run-pe uses it unless overridden",
    )
    parser_gen.add_argument(
        "--noise",
        type=str,
        choices=["zero", "gaussian"],
        default="zero",
        help="Noise realization recorded with the injection; run-pe uses it unless overridden",
    )
    parser_gen.add_argument(
        "--psd",
        type=str,
        default="aligo",
        help="'aligo' (built-in analytic aLIGO ZDHP) or a path to a two-column ASCII PSD file",
    )
    parser_gen.add_argument(
        "--n-injections",
        type=int,
        default=1,
        help="Number of distinct injections to draw",
    )
    parser_gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the injection draws (reproducible)",
    )
    parser_gen.add_argument(
        "--fiducial",
        action="store_true",
        help="Emit the fixed GW150914-like reference binary (Mc=30, q=0.8, 700 Mpc, zero spins, face-on) instead of drawing; requires --n-injections 1",
    )
    parser_gen.add_argument(
        "--outdir", type=str, required=True, help="Output directory for injection data"
    )
    parser_gen.set_defaults(func=generate_injections)

    # Subparser for run-pe
    parser_run = subparsers.add_parser("run-pe", help="Run PE on stored injection data")
    parser_run.add_argument(
        "--injection", type=str, required=True, help="Path to injection JSON file"
    )
    parser_run.add_argument(
        "--sampler",
        type=str,
        choices=["hmc", "mala", "ns", "gpry"],
        default="hmc",
        help="Sampler to use",
    )
    parser_run.add_argument(
        "--likelihood",
        type=str,
        choices=["full", "marginalized_phase_distance"],
        default="full",
        help="Likelihood form",
    )
    parser_run.add_argument(
        "--domain",
        type=str,
        choices=["td", "fd"],
        default="fd",
        help="Integration domain (time or frequency)",
    )
    parser_run.add_argument(
        "--network",
        type=str,
        default=None,
        help="Detector network (comma-separated); defaults to the value recorded in the injection",
    )
    parser_run.add_argument(
        "--noise",
        type=str,
        choices=["zero", "gaussian"],
        default=None,
        help="Noise realization; defaults to the value recorded in the injection",
    )
    parser_run.add_argument(
        "--psd",
        type=str,
        default=None,
        help="'aligo' or a path to a two-column ASCII PSD file; defaults to the value recorded in the injection",
    )
    parser_run.add_argument(
        "--n-chains", type=int, default=100, help="Number of sampler chains to run"
    )
    parser_run.add_argument(
        "--n-prelim-loops", type=int, default=1, help="Number of preliminary loops"
    )
    parser_run.add_argument(
        "--n-training-loops", type=int, default=5, help="Number of training loops"
    )
    parser_run.add_argument(
        "--n-production-loops", type=int, default=50, help="Number of production loops"
    )
    parser_run.add_argument(
        "--outdir", type=str, required=True, help="Output directory for PE results"
    )
    parser_run.set_defaults(func=run_pe)

    # Subparser for process-samples
    parser_process = subparsers.add_parser(
        "process-samples", help="Post-process raw MCMC samples"
    )
    parser_process.add_argument(
        "files", nargs="+", help="Paths to raw_samples.npz files"
    )
    parser_process.set_defaults(func=process_samples)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
