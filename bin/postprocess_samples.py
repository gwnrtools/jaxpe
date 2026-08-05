import argparse
import numpy as np
from pathlib import Path
from jaxpe.core import InferenceProblem
from jaxpe.sampler import PostProcessor
from jaxpe.gw import bbh_priors
from jaxpe.diagnostics import corner_plot

# Import event specifications to reconstruct the InferenceProblem
from run_phenomd_events import EVENTS


def main():
    parser = argparse.ArgumentParser(
        description="Post-process raw MCMC samples (thinning & burn-in)."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to raw_samples.npz files (e.g., output/production_events/GW190412/raw_samples.npz)",
    )
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File {path} does not exist. Skipping.")
            continue

        # Infer the event name from the parent directory
        event_name = path.parent.name

        if event_name not in EVENTS:
            print(f"Unknown event {event_name} from path {path}. Skipping.")
            continue

        print("\n=============================================")
        print(f"Post-processing {event_name}")

        # 1. Reconstruct the prior. PostProcessor only maps unconstrained draws back
        # through the bijections -- it never evaluates the likelihood -- so there is
        # no injection to rebuild here. Doing so previously cost a waveform
        # generation and FFT per file, and restated the conditioning
        # (duration/sampling_rate/network) in a second place where it could drift
        # from run_phenomd_events.py without anything noticing.
        spec = EVENTS[event_name]
        params = spec["params"]

        prior = bbh_priors(
            chirp_mass=spec["mc_prior"],
            mass_ratio=(0.1, 1.0),
            aligned_spins=(-0.9, 0.9),
            luminosity_distance=spec["dist_prior"],
            geocent_time=params["geocent_time"],
            time_width=0.1,
        )
        problem = InferenceProblem(prior)

        # 2. Run the PostProcessor
        pp = PostProcessor(problem, raw_samples_file=path)
        phys_samples = pp.process()

        out_dir = path.parent

        # 3. Save final physical samples
        final_samples_file = out_dir / "posterior_samples.npy"
        np.save(final_samples_file, phys_samples)
        print(f"Saved {final_samples_file}")

        # 4. Generate and save the corner plot
        pnames = list(problem.names)
        truths = [params[n] for n in pnames]

        try:
            fig = corner_plot(phys_samples, names=pnames, truths=truths)
            corner_file = out_dir / "corner_thinned.png"
            fig.savefig(corner_file, dpi=120)
            print(f"Saved {corner_file}")
        except Exception as e:
            print(f"Failed to generate corner plot for {event_name}: {e}")


if __name__ == "__main__":
    main()
