"""Command-line interface for jaxpe Universal PE Drivers."""

import argparse
import json
import time
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

def generate_injections(args):
    print("Generating injections...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Default to BNS/BBH-like injection for now
    injection_params = {
        "chirp_mass": 30.0,
        "mass_ratio": 0.8,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "luminosity_distance": 700.0,
        "geocent_time": 1126259462.4,
        "phase": 0.0,
        "inclination": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "psi": 0.0,
    }
    
    for i in range(args.n_injections):
        ipath = outdir / f"inj_{i}.json"
        with open(ipath, "w") as f:
            json.dump(injection_params, f, indent=2)
        print(f"Saved injection {i} to {ipath}")

def run_pe(args):
    print("Running PE...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    with open(args.injection, "r") as f:
        injection_params = json.load(f)
        
    print(f"Loaded injection params: {injection_params}")
    
    if args.domain == "fd":
        waveform = IMRPhenomD(f_ref=20.0)
    elif args.domain == "td":
        waveform = IMRPhenomT(f_ref=20.0)
    else:
        raise ValueError(f"Unknown domain: {args.domain}")
    
    prior = bbh_priors(
        chirp_mass=(10.0, 50.0),
        mass_ratio=(0.1, 1.0),
        aligned_spins=(-0.9, 0.9),
        luminosity_distance=(100.0, 2000.0),
        geocent_time=injection_params.get("geocent_time", 0.0),
        time_width=0.1,
    )
    
    like = make_injection(
        waveform,
        injection_params,
        detector_names=args.network.split(","),
        duration=4.0,
        sampling_rate=1024.0,
        f_min=30.0,
        noise_seed=None if args.noise == "zero" else 42,
    )
    
    # Construct likelihood and prior structures
    if args.likelihood == "marginalized_phase_distance":
        if args.domain != "fd":
            raise ValueError("marginalized_phase_distance is only supported for FD domain.")
        if args.sampler in ["hmc", "mala"]:
            raise ValueError("marginalized_phase_distance is host-side and incompatible with HMC/MALA.")
            
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
        bounds_dict = {
            "chirp_mass": (10.0, 50.0),
            "mass_ratio": (0.1, 1.0),
            "spin1z": (-0.9, 0.9),
            "spin2z": (-0.9, 0.9),
        }
        
        def log_prob_fn(x):
            return like_callable(x)
            
    else:
        problem = like.problem(prior)
        
        # NS/GPry samplers require explicit bounds. We construct them based on the names.
        bounds_dict = {}
        for n in problem.names:
            if n == "chirp_mass": bounds_dict[n] = (10.0, 50.0)
            elif n == "mass_ratio": bounds_dict[n] = (0.1, 1.0)
            elif n == "luminosity_distance": bounds_dict[n] = (100.0, 2000.0)
            elif "spin" in n: bounds_dict[n] = (-0.9, 0.9)
            elif n == "phase": bounds_dict[n] = (0.0, 2*np.pi)
            elif n == "geocent_time": bounds_dict[n] = (injection_params.get("geocent_time", 0.0) - 0.1, injection_params.get("geocent_time", 0.0) + 0.1)
            elif n == "inclination": bounds_dict[n] = (0.0, np.pi)
            elif n == "ra": bounds_dict[n] = (0.0, 2*np.pi)
            elif n == "dec": bounds_dict[n] = (-np.pi/2, np.pi/2)
            elif n == "psi": bounds_dict[n] = (0.0, np.pi)
            else: bounds_dict[n] = (0.0, 1.0) # fallback
            
        def log_prob_fn(x):
            return problem.log_prob(x)
    
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
            
        X_MC, y_MC, w_MC, _logZ, _logZstd = ns.run(logp_jax, param_names=names_list, seed=42)
        
        samples_flat = np.asarray(X_MC)
        np.savez(
            out_samples,
            samples=samples_flat[None, :, :], # shape: (1, n_samples, n_dim)
            log_prob=np.asarray(y_MC)[None, :],
            weights=np.asarray(w_MC)
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
                }
            }
        )
        engine.run()
        res_gpry = engine.sample()
        samples_flat = res_gpry.x
        np.savez(
            out_samples,
            samples=samples_flat[None, :, :],
            log_prob=res_gpry.logpost[None, :],
            weights=res_gpry.weights
        )
    else:
        raise NotImplementedError(f"Sampler {args.sampler} not fully implemented yet in generic CLI.")

    print(f"Saved raw samples to {out_samples}")
    
    # Save injection to outdir for process_samples
    with open(outdir / "injection.json", "w") as f:
        json.dump(injection_params, f, indent=2)

def process_samples(args):
    print("Processing samples...")
    for file_path in args.files:
        path = Path(file_path)
        out_dir = path.parent
        
        inj_file = out_dir / "injection.json"
        if not inj_file.exists():
            print(f"Warning: {inj_file} not found. Cannot reconstruct problem easily. Skipping {path}.")
            continue
            
        with open(inj_file, "r") as f:
            injection_params = json.load(f)
            
        waveform = IMRPhenomD(f_ref=20.0)
        like = make_injection(
            waveform,
            injection_params,
            detector_names=("H1", "L1", "V1"),
            duration=4.0,
            sampling_rate=1024.0,
            f_min=30.0,
            noise_seed=None,
        )
        prior = bbh_priors(
            chirp_mass=(10.0, 50.0),
            mass_ratio=(0.1, 1.0),
            aligned_spins=(-0.9, 0.9),
            luminosity_distance=(100.0, 2000.0),
            geocent_time=injection_params.get("geocent_time", 0.0),
            time_width=0.1,
        )
        problem = like.problem(prior)
        
        pp = PostProcessor(problem, raw_samples_file=path)
        phys_samples = pp.process()
        
        np.save(out_dir / "posterior_samples.npy", phys_samples)
        
        try:
            pnames = list(problem.names)
            truths = [injection_params.get(n, None) for n in pnames]
            fig = corner_plot(phys_samples, names=pnames, truths=truths)
            fig.savefig(out_dir / "corner_thinned.png", dpi=120)
            print(f"Saved {out_dir / 'corner_thinned.png'}")
        except Exception as e:
            print(f"Failed to generate corner plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="jaxpe Universal Parameter Estimation Driver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for generate-injections
    parser_gen = subparsers.add_parser("generate-injections", help="Generate injections and noise data")
    parser_gen.add_argument("--network", type=str, default="H1,L1", help="Detector network (comma-separated, e.g., H1,L1,V1)")
    parser_gen.add_argument("--noise", type=str, choices=["zero", "gaussian"], default="zero", help="Type of noise to generate")
    parser_gen.add_argument("--psd", type=str, default="CE", help="PSD to use (e.g., CE, LIGO, or path to file)")
    parser_gen.add_argument("--n-injections", type=int, default=1, help="Number of injections to create")
    parser_gen.add_argument("--outdir", type=str, required=True, help="Output directory for injection data")
    parser_gen.set_defaults(func=generate_injections)

    # Subparser for run-pe
    parser_run = subparsers.add_parser("run-pe", help="Run PE on stored injection data")
    parser_run.add_argument("--injection", type=str, required=True, help="Path to injection JSON file")
    parser_run.add_argument("--sampler", type=str, choices=["hmc", "mala", "ns", "gpry"], default="hmc", help="Sampler to use")
    parser_run.add_argument("--likelihood", type=str, choices=["full", "marginalized_phase_distance"], default="full", help="Likelihood form")
    parser_run.add_argument("--domain", type=str, choices=["td", "fd"], default="fd", help="Integration domain (time or frequency)")
    parser_run.add_argument("--network", type=str, default="H1,L1", help="Detector network (comma-separated, e.g., H1,L1,V1)")
    parser_run.add_argument("--noise", type=str, choices=["zero", "gaussian"], default="zero", help="Type of noise to generate")
    parser_run.add_argument("--n-chains", type=int, default=100, help="Number of sampler chains to run")
    parser_run.add_argument("--n-prelim-loops", type=int, default=1, help="Number of preliminary loops")
    parser_run.add_argument("--n-training-loops", type=int, default=5, help="Number of training loops")
    parser_run.add_argument("--n-production-loops", type=int, default=50, help="Number of production loops")
    parser_run.add_argument("--outdir", type=str, required=True, help="Output directory for PE results")
    parser_run.set_defaults(func=run_pe)

    # Subparser for process-samples
    parser_process = subparsers.add_parser("process-samples", help="Post-process raw MCMC samples")
    parser_process.add_argument("files", nargs="+", help="Paths to raw_samples.npz files")
    parser_process.set_defaults(func=process_samples)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
