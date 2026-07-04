"""Standard priors for compact-binary PE, matched to the waveform parameter names."""

from ..core.priors import Cosine, JointPrior, PowerLaw, Sine, Uniform


def bbh_priors(
    chirp_mass=(20.0, 40.0),
    mass_ratio=(0.25, 1.0),
    luminosity_distance=(100.0, 2000.0),
    geocent_time=None,
    time_width: float = 0.1,
) -> JointPrior:
    """Aligned-spin-free toy-BBH prior set matching ``ToyChirp``'s parameters.

    ``geocent_time`` is the trigger-time estimate; the prior is uniform within
    +- ``time_width`` seconds around it.
    """
    import numpy as np

    if geocent_time is None:
        raise ValueError("geocent_time (trigger-time estimate) is required")
    return JointPrior(
        {
            "chirp_mass": Uniform(low=chirp_mass[0], high=chirp_mass[1]),
            "mass_ratio": Uniform(low=mass_ratio[0], high=mass_ratio[1]),
            "luminosity_distance": PowerLaw(
                alpha=2.0, low=luminosity_distance[0], high=luminosity_distance[1]
            ),
            "inclination": Sine(),
            "phase": Uniform(low=0.0, high=2 * np.pi),
            "ra": Uniform(low=0.0, high=2 * np.pi),
            "dec": Cosine(),
            "psi": Uniform(low=0.0, high=np.pi),
            "geocent_time": Uniform(
                low=geocent_time - time_width, high=geocent_time + time_width
            ),
        }
    )
