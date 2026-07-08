"""Standard Priors for Compact-Binary Parameter Estimation.

In Bayesian inference, the prior $P(\theta)$ encodes our knowledge or beliefs about
the parameters before seeing the data. The choice of prior can significantly affect
the posterior, especially for parameters that are weakly constrained by the data.

Motivation & Math
-----------------
Several parameters have standard geometric or astrophysical priors:
- **Luminosity Distance ($d_L$)**: Assuming a uniform distribution of sources in
  Euclidean volume, the number of sources scales as $V \propto d_L^3$, so the
  probability density scales as $P(d_L) \propto d_L^2$.
- **Sky Location (RA, Dec)**: To be uniform on the celestial sphere, Right Ascension
  is uniform in $[0, 2\pi]$, and Declination follows $P(\delta) \propto \cos(\delta)$
  in $[-\pi/2, \pi/2]$.
- **Inclination ($\iota$)**: For random orientations, the inclination angle follows
  $P(\iota) \propto \sin(\iota)$ in $[0, \pi]$.

This module provides ready-to-use `JointPrior` objects that construct these
standard priors for different waveform models.
"""

from ..core.priors import Cosine, JointPrior, PowerLaw, Prior, Sine, Uniform


def bbh_priors(
    chirp_mass=(20.0, 40.0),
    mass_ratio=(0.25, 1.0),
    luminosity_distance=(100.0, 2000.0),
    geocent_time=None,
    time_width: float = 0.1,
    aligned_spins: tuple | None = None,
) -> JointPrior:
    """Aligned-spin-free toy-BBH prior set matching ``ToyChirp``'s parameters.

    ``geocent_time`` is the trigger-time estimate; the prior is uniform within
    +- ``time_width`` seconds around it.
    """
    import numpy as np

    if geocent_time is None:
        raise ValueError("geocent_time (trigger-time estimate) is required")
    priors: dict[str, Prior] = {
        "chirp_mass": Uniform(low=chirp_mass[0], high=chirp_mass[1]),
        "mass_ratio": Uniform(low=mass_ratio[0], high=mass_ratio[1]),
    }
    if aligned_spins is not None:
        priors["spin1z"] = Uniform(low=aligned_spins[0], high=aligned_spins[1])
        priors["spin2z"] = Uniform(low=aligned_spins[0], high=aligned_spins[1])
    priors.update(
        {
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
    return JointPrior(priors)


def ebbh_priors(
    chirp_mass=(20.0, 40.0),
    mass_ratio=(0.25, 1.0),
    eccentricity=(0.0, 0.4),
    luminosity_distance=(100.0, 2000.0),
    geocent_time=None,
    time_width: float = 0.1,
    aligned_spins: tuple | None = None,
) -> JointPrior:
    """Eccentric aligned-spin BBH priors matching ``ESIGMAInspiral``'s parameters.

    ``aligned_spins=(low, high)`` adds uniform spin1z/spin2z priors; None (default)
    omits them (the waveform then uses spins = 0).
    """
    import numpy as np

    if geocent_time is None:
        raise ValueError("geocent_time (trigger-time estimate) is required")
    priors: dict[str, Prior] = {
        "chirp_mass": Uniform(low=chirp_mass[0], high=chirp_mass[1]),
        "mass_ratio": Uniform(low=mass_ratio[0], high=mass_ratio[1]),
        "eccentricity": Uniform(low=eccentricity[0], high=eccentricity[1]),
        "mean_anomaly": Uniform(low=0.0, high=2 * np.pi),
    }
    if aligned_spins is not None:
        priors["spin1z"] = Uniform(low=aligned_spins[0], high=aligned_spins[1])
        priors["spin2z"] = Uniform(low=aligned_spins[0], high=aligned_spins[1])
    priors.update(
        {
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
    return JointPrior(priors)
