"""Command-line interface for jaxpe Universal PE Drivers.

Every physical and numerical choice this driver makes is read from a run
configuration (see :mod:`jaxpe.config`), not hardcoded here: the data conditioning,
the prior *distributions*, the population injected truths are drawn from, kernel step
sizes, sampler budgets, and every RNG seed. ``--config FILE`` supplies one; with no
``--config`` the built-in defaults apply, and those defaults are exactly the constants
this module used to carry, so existing invocations are unaffected. Command-line flags,
where they exist, override the configuration file.

Write a starting-point file with ``jaxpe write-config my_run.json``.
"""

import argparse
import json
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxpe import config as runconfig
from jaxpe.config import ConfigError
from jaxpe.core import InferenceProblem
from jaxpe.gw import IMRPhenomD, IMRPhenomT, make_injection
from jaxpe.gw.likelihood import PhaseDistanceMarginalLikelihood
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init, PostProcessor
from jaxpe.kernels import MALA, HMC
from jaxpe.diagnostics.plots import corner_plot

# Name of the resolved configuration dropped next to a generated injection set, so
# that run-pe inherits the physics the set was generated under without the user
# having to pass --config twice.
CONFIG_FILENAME = "config.json"


def load_run_config(path, *, inherit_from=None, label="configuration"):
    """Resolve the configuration for a command and report its warnings.

    Precedence: an explicit ``--config`` beats a ``config.json`` sitting beside the
    injection, which beats the built-in defaults.
    """
    source = path
    if source is None and inherit_from is not None:
        candidate = Path(inherit_from)
        if candidate.exists():
            source = candidate
    cfg, warnings = runconfig.load_config(source)
    if source is None:
        print(f"Using built-in default {label} (no --config given)")
    else:
        print(f"Loaded {label} from {source}")
    for w in warnings:
        print(f"WARNING: {w}")
    return cfg


def resolve_psd(spec):
    """Map a --psd specification to a psd_fn usable by ``make_injection``.

    Three forms, tried in order: ``aligo`` (the built-in analytic aLIGO ZDHP fit), a
    named LALSimulation design curve (``CE``, ``ET``, ``aplus``, ...), or a path to a
    two-column ASCII file. Anything else is rejected rather than silently ignored.
    """
    from jaxpe.gw import LALSIM_PSDS, aligo_zdhp_psd, lalsim_psd, psd_from_file

    if spec is None or spec.lower() in ("aligo", "zdhp", "aligo_zdhp"):
        return aligo_zdhp_psd
    # Named curves win over a same-named file only by being checked first; the names
    # are detector labels, so a file called "CE" in the cwd would be the surprise.
    if spec in LALSIM_PSDS:
        return lambda freqs: lalsim_psd(spec, freqs)
    path = Path(spec)
    if path.exists():
        return lambda freqs: psd_from_file(path, freqs)
    raise ValueError(
        f"--psd {spec!r} is not 'aligo', a known detector curve, or an existing file. "
        f"Known curves: {', '.join(sorted(LALSIM_PSDS))}. Anything else must be a "
        "two-column ASCII file path."
    )


def generate_injections(args):
    print("Generating injections...")
    cfg = load_run_config(args.config)

    # Everything that can refuse the run happens before the output directory is
    # created, so a rejected invocation leaves nothing behind.

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

    # Only --fiducial runs use injection.fiducial, so its containment in the prior is
    # a warning at load time and fatal only here -- otherwise a narrow prior (a BNS
    # config, say) would be blocked by a reference binary it never touches.
    if args.fiducial:
        outside = runconfig.fiducial_errors(cfg)
        if outside:
            raise ConfigError(
                "--fiducial cannot be used with this configuration:\n"
                + "\n".join(f"  - {p}" for p in outside)
                + "\nEdit injection.fiducial to sit inside the prior, or drop "
                "--fiducial to draw from injection.parameters instead."
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else cfg["seeds"]["injection"]
    # Draws come from the same distribution objects the prior is built from, so an
    # injection.parameters section equal to prior draws exactly from the prior.
    keys = jax.random.split(jax.random.PRNGKey(seed), max(args.n_injections, 1))
    for i in range(args.n_injections):
        injection_params = (
            runconfig.fiducial_injection(cfg)
            if args.fiducial
            else runconfig.sample_parameters(cfg, keys[i])
        )
        # Recorded so run-pe can default to the network/noise/PSD this set was
        # generated for. Popped before the dict is handed to the waveform model.
        injection_params["metadata"] = {
            "network": args.network,
            "noise": args.noise,
            "psd": args.psd,
            "seed": None if args.fiducial else seed,
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

    # Drop the resolved configuration beside the set. run-pe picks this up as its
    # default, so the prior and conditioning that defined these truths are the ones
    # they are later analysed under, without relying on the user to remember.
    cfg_path = outdir / CONFIG_FILENAME
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved resolved configuration to {cfg_path}")


def run_pe(args):
    print("Running PE...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    injection_path = Path(args.injection)
    cfg = load_run_config(
        args.config, inherit_from=injection_path.parent / CONFIG_FILENAME
    )

    with open(injection_path, "r") as f:
        injection_params = json.load(f)

    # Generation-time settings recorded by generate-injections. They are defaults
    # only: an explicit flag on this command always wins. Popped so the dict handed
    # to the waveform model contains physical parameters exclusively.
    meta = injection_params.pop("metadata", {})
    network = args.network if args.network is not None else meta.get("network", "H1,L1")
    noise = args.noise if args.noise is not None else meta.get("noise", "zero")
    psd_spec = args.psd if args.psd is not None else meta.get("psd", "aligo")
    psd_fn = resolve_psd(psd_spec)

    data_cfg = cfg["data"]
    # Trigger-relative priors are anchored on *this* injection's time, so a set
    # generated at another epoch still analyses correctly.
    trigger = float(
        injection_params.get("geocent_time", cfg["injection"]["geocent_time"])
    )
    prior = runconfig.build_prior(cfg, trigger=trigger)
    prior_box = runconfig.prior_bounds(cfg, trigger=trigger)

    print(f"Loaded injection params: {injection_params}")
    print(f"Network {network}, noise {noise}, psd {psd_spec}")
    print(
        f"Data: {data_cfg['duration']} s @ {data_cfg['sampling_rate']} Hz, "
        f"f_min={data_cfg['f_min']} Hz, f_ref={data_cfg['f_ref']} Hz"
    )

    if args.domain == "fd":
        waveform = IMRPhenomD(f_ref=data_cfg["f_ref"])
    elif args.domain == "td":
        waveform = IMRPhenomT(f_ref=data_cfg["f_ref"])
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    like = make_injection(
        waveform,
        injection_params,
        detector_names=network.split(","),
        duration=data_cfg["duration"],
        sampling_rate=data_cfg["sampling_rate"],
        f_min=data_cfg["f_min"],
        f_max=data_cfg["f_max"],
        psd_fn=psd_fn,
        noise_seed=None if noise == "zero" else cfg["seeds"]["noise"],
        post_trigger=data_cfg["post_trigger"],
        tukey_alpha=data_cfg["tukey_alpha"],
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
        bounds_dict = {n: prior_box[n] for n in names}

        def log_prob_fn(x):
            return like_callable(x)

    else:
        problem = like.problem(prior)

        # NS/GPry require an explicit box. Every sampled name comes from the prior we
        # just built, so its bounds are the prior's own support -- no fallbacks and no
        # second, divergent definition of the search region.
        bounds_dict = {n: prior_box[n] for n in problem.names}

        def log_prob_fn(x):
            # ns/gpry explore the *physical* bounds box, so this must be the
            # physical-space density: log_posterior() expects unconstrained
            # coordinates and InferenceProblem has no log_prob at all. The bounds
            # box only delimits the region, so the prior density is added here --
            # without it the target would be likelihood x uniform, a different
            # posterior from the one the MCMC path samples.
            return problem.log_likelihood_vec(x) + problem.prior.log_prob(x)

    out_samples = outdir / "raw_samples.npz"
    key = jax.random.PRNGKey(cfg["seeds"]["sampler"])

    # Settings actually used, folded into run_config.json below so the artifact
    # reflects the run rather than the file it started from.
    effective = {}

    if args.sampler in ["hmc", "mala"]:
        kernel_cfg = cfg["kernel"][args.sampler]
        if args.sampler == "hmc":
            kernel = HMC(
                step_size=kernel_cfg["step_size"],
                n_leapfrog=kernel_cfg["n_leapfrog"],
            )
        else:
            kernel = MALA(step_size=kernel_cfg["step_size"])

        sampler_kwargs = runconfig.global_local_kwargs(cfg)
        # Flags override the configuration file where they were given.
        for name in (
            "n_chains",
            "n_prelim_loops",
            "n_training_loops",
            "n_production_loops",
        ):
            value = getattr(args, name)
            if value is not None:
                sampler_kwargs[name] = value
        effective["kernel"] = {args.sampler: dict(kernel_cfg)}
        effective["sampler"] = dict(sampler_kwargs)

        cfg_gl = GlobalLocalConfig(**sampler_kwargs)
        sampler = Sampler(kernel, problem=problem, config=cfg_gl)
        x0 = best_of_prior_init(key, problem, cfg_gl.n_chains)
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

        ns_cfg = cfg["ns"]
        # JAXInterfaceBlackJAX requires bounds as a (n_dim, 2) array
        names_list = list(bounds_dict.keys())
        bounds_arr = np.array([bounds_dict[n] for n in names_list])
        ns = JAXInterfaceBlackJAX(bounds_arr, verbosity=ns_cfg["verbosity"])
        precision = {
            k: ns_cfg[k]
            for k in (
                "nlive",
                "num_repeats",
                "precision_criterion",
                "nprior",
                "max_ncalls",
            )
            if ns_cfg[k] is not None
        }
        ns.set_precision(**precision)
        effective["ns"] = dict(ns_cfg)

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
            logp_jax, param_names=names_list, seed=cfg["seeds"]["ns"]
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

        gpry_cfg = dict(cfg["gpry"])
        # Flags override the configuration file where they were given.
        for flag, key_name in (
            ("gpry_n_initial", "n_initial"),
            ("gpry_max_total", "max_total"),
            ("gpry_max_initial", "max_initial"),
            ("gpry_acquisition", "acquisition"),
        ):
            value = getattr(args, flag)
            if value is not None:
                gpry_cfg[key_name] = value

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
                delta = max(
                    gpry_cfg["ref_bounds_rel"] * abs(val), gpry_cfg["ref_bounds_abs"]
                )
                ref_bounds_arr.append([val - delta, val + delta])
            else:
                ref_bounds_arr.append(bounds_dict[n])

        # n_initial defaults to GPry's own 3*d rather than a fixed number. The
        # previous hardcoded 5 was *below* that for every problem here (4-D
        # marginalized -> 12, 11-D full -> 33), which left the SVM infinities
        # classifier under-trained and made runs fail intermittently on the
        # geometry of the particular injection rather than on anything real.
        n_initial = (
            3 * len(bounds_dict)
            if gpry_cfg["n_initial"] is None
            else gpry_cfg["n_initial"]
        )
        gpry_opts = {
            "max_total": gpry_cfg["max_total"],
            "n_initial": n_initial,
            "max_initial": gpry_cfg["max_initial"],
        }
        effective["gpry"] = {**gpry_cfg, "n_initial": n_initial}
        engine = GPryEngine(
            timed_loglike,
            bounds=bounds_dict,
            acquisition=gpry_cfg["acquisition"],
            options={
                "seed": cfg["seeds"]["gpry"],
                "ref_bounds": np.array(ref_bounds_arr),
                "options": gpry_opts,
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
    #
    # "config" is the full resolved configuration with command-line overrides folded
    # in, so the file is a complete and re-runnable description of the run. The flat
    # keys beside it are the pre-config format, kept so that output directories and
    # scripts written against it keep working.
    resolved = runconfig.merge_config(cfg, effective)
    run_config = {
        "sampler": args.sampler,
        "likelihood": args.likelihood,
        "domain": args.domain,
        "network": network,
        "noise": noise,
        "psd": psd_spec,
        "duration": data_cfg["duration"],
        "sampling_rate": data_cfg["sampling_rate"],
        "f_min": data_cfg["f_min"],
        "f_ref": data_cfg["f_ref"],
        "time_width": runconfig.time_width(cfg),
        "prior_ranges": {k: list(v) for k, v in prior_box.items()},
        "param_names": param_names,
        "sample_space": sample_space,
        "weighted": args.sampler in ("ns", "gpry"),
        "config": resolved,
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run configuration to {outdir / 'run_config.json'}")


def _config_from_run_config(cfg):
    """Recover a full configuration from a ``run_config.json`` payload.

    New runs embed the resolved configuration under ``config``. Directories written
    before that existed carry a handful of flat keys instead; map those across so
    process-samples reconstructs the right problem for them too.
    """
    if isinstance(cfg.get("config"), dict):
        return runconfig.merge_config(runconfig.DEFAULT_CONFIG, cfg["config"])

    legacy: dict = {"data": {}, "prior": {}}
    for key in ("duration", "sampling_rate", "f_min", "f_ref"):
        if key in cfg:
            legacy["data"][key] = cfg[key]
    # Legacy prior_ranges recorded [low, high] boxes only; the distributions they
    # were built with are the historical bbh_priors ones, which are this module's
    # defaults, so merging the boxes over those defaults reproduces them.
    for name, rng in (cfg.get("prior_ranges") or {}).items():
        spec = dict(runconfig.DEFAULT_CONFIG["prior"].get(name, {"dist": "uniform"}))
        spec["low"], spec["high"] = float(rng[0]), float(rng[1])
        spec.pop("relative_to_trigger", None)
        legacy["prior"][name] = spec
    if "time_width" in cfg:
        legacy["prior"]["geocent_time"] = {
            "dist": "uniform",
            "low": -float(cfg["time_width"]),
            "high": float(cfg["time_width"]),
            "relative_to_trigger": True,
        }
    return runconfig.merge_config(runconfig.DEFAULT_CONFIG, legacy)


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
                f"WARNING: {cfg_file} not found; assuming the built-in default prior. "
                "If the samples came from a run with different prior ranges, the "
                "reconstructed physical parameters will be wrong. Re-run run-pe to "
                "regenerate the configuration."
            )
            cfg = {}

        run_cfg = _config_from_run_config(cfg)

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
            # Only the prior is needed here: PostProcessor maps unconstrained draws
            # back through the bijections and never evaluates the likelihood. Building
            # an injection to obtain it cost a full waveform generation, projection,
            # FFT and jit compile per samples file, and forced the conditioning to be
            # restated somewhere it could drift from the run that produced the samples.
            trigger = float(
                injection_params.get(
                    "geocent_time", run_cfg["injection"]["geocent_time"]
                )
            )
            problem = InferenceProblem(runconfig.build_prior(run_cfg, trigger=trigger))

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


def write_config(args):
    """Emit a fully-populated configuration file to edit."""
    path = Path(args.output)
    if path.exists() and not args.force:
        raise ValueError(f"{path} already exists; pass --force to overwrite it.")
    runconfig.dump_default_config(path)
    print(f"Wrote default run configuration to {path}")
    print("Edit it and pass it with --config to generate-injections and run-pe.")


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
        "--config",
        type=str,
        default=None,
        help="Run configuration JSON (see 'jaxpe write-config'); omit for built-in defaults",
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
        help="'aligo', a named design curve (CE, ET, aplus, ...), or a two-column ASCII PSD file",
    )
    parser_gen.add_argument(
        "--n-injections",
        type=int,
        default=1,
        help="Number of distinct injections to draw from injection.parameters",
    )
    parser_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the injection draws; overrides seeds.injection in the config",
    )
    parser_gen.add_argument(
        "--fiducial",
        action="store_true",
        help="Emit the fixed reference binary from injection.fiducial instead of drawing; requires --n-injections 1",
    )
    parser_gen.add_argument(
        "--outdir", type=str, required=True, help="Output directory for injection data"
    )
    parser_gen.set_defaults(func=generate_injections)

    # Subparser for run-pe
    parser_run = subparsers.add_parser("run-pe", help="Run PE on stored injection data")
    parser_run.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Run configuration JSON (see 'jaxpe write-config'). Defaults to a "
            "config.json sitting beside the injection, then to the built-in defaults"
        ),
    )
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
        help="PSD name or file; defaults to the value recorded in the injection",
    )
    parser_run.add_argument(
        "--n-chains",
        type=int,
        default=None,
        help="Number of sampler chains to run; overrides sampler.n_chains in the config",
    )
    parser_run.add_argument(
        "--n-prelim-loops",
        type=int,
        default=None,
        help="Number of preliminary loops; overrides sampler.n_prelim_loops in the config",
    )
    parser_run.add_argument(
        "--n-training-loops",
        type=int,
        default=None,
        help="Number of training loops; overrides sampler.n_training_loops in the config",
    )
    parser_run.add_argument(
        "--n-production-loops",
        type=int,
        default=None,
        help="Number of production loops; overrides sampler.n_production_loops in the config",
    )
    parser_run.add_argument(
        "--gpry-n-initial",
        type=int,
        default=None,
        help="gpry: truth evaluations before active learning starts (default: GPry's own 3*n_dim)",
    )
    parser_run.add_argument(
        "--gpry-max-total",
        type=int,
        default=None,
        help="gpry: maximum total truth evaluations; overrides gpry.max_total in the config",
    )
    parser_run.add_argument(
        "--gpry-max-initial",
        type=int,
        default=None,
        help="gpry: maximum draws attempted while collecting the initial finite points",
    )
    parser_run.add_argument(
        "--gpry-acquisition",
        type=str,
        default=None,
        help=(
            "gpry: acquisition engine. Default leaves GPry on its native NORA "
            "nested-sampling acquisition; 'BatchOptimizer' is multi-start L-BFGS on the "
            "analytic acquisition gradient, which uses far fewer surrogate evaluations "
            "per step but was measured slower end-to-end on this problem"
        ),
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

    # Subparser for write-config
    parser_cfg = subparsers.add_parser(
        "write-config",
        help="Write a fully-populated run configuration file to edit",
    )
    parser_cfg.add_argument(
        "output", help="Path to write the JSON configuration to (e.g. my_run.json)"
    )
    parser_cfg.add_argument(
        "--force", action="store_true", help="Overwrite an existing file"
    )
    parser_cfg.set_defaults(func=write_config)

    args = parser.parse_args()
    try:
        args.func(args)
    except ConfigError as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
