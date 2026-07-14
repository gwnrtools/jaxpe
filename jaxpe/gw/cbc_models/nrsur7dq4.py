r"""NRSur7dq4 waveform adapter: Time-domain surrogate model for generically spinning BBHs.

Motivation & Math
-----------------
To extract the underlying astrophysics of coalescing compact binaries with fully generic
(precessing) spins, standard semi-analytic Post-Newtonian approaches often struggle to
capture the highly nonlinear dynamics near merger.

The NRSur7dq4 model is a time-domain surrogate model trained directly on 1528 highly
accurate Numerical Relativity (NR) simulations. It covers binary black holes with mass
ratios $q \leq 4$ and generic spin magnitudes up to $|\chi_1|, |\chi_2| \leq 0.8$.
The model evaluates the waveform in a co-orbital frame where the dynamics vary slowly,
then rotates the modes into the inertial frame using interpolated quaternions:
$$ h_+ - i h_\times = \sum_{l \leq 4} \sum_{m=-l}^{l} h_{lm}(t) {}_{-2}Y_{lm}(\iota, \phi) $$

This adapter leverages the `jaxnrsur` package to integrate the NRSur7dq4 surrogate seamlessly
into our `jaxpe` framework as a `TimeDomainModel`. The surrogate's internal empirical
interpolation nodes and basis functions are evaluated using JAX primitives, enabling
end-to-end auto-differentiation for full 15-dimensional precessing parameter estimation.

Implementation details:
  1. Wraps `jaxnrsur.NRSur7dq4Model` and its data loader.
  2. Evaluates the $(h_+, h_\times)$ time-domain polarizations on a user-defined time grid.
  3. Translates physical dictionary parameters (chirp mass, mass ratio, spins) into the
     internal 7D surrogate parameterization (total mass, geometric spins, etc).
"""

import jax
import jax.numpy as jnp
from .base import TimeDomainModel


class NRSur7dq4(TimeDomainModel):
    """
    JAX-differentiable NRSur7dq4 waveform model utilizing JaxNRSur.

    Parameters
    ----------
    data_path : str
        Path to the NRSur7dq4.h5 data file.
    """

    def __init__(self, data_path: str):
        import jaxnrsur
        from jaxnrsur.NRSur7dq4 import NRSur7dq4DataLoader, NRSur7dq4Model

        self.data_path = data_path
        # Load the surrogate data
        data = NRSur7dq4DataLoader(self.data_path)
        model = NRSur7dq4Model(data)

        self.jaxnrsur_model = jaxnrsur.JaxNRSur(model=model)

    def __call__(self, params: dict, grid: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Evaluate the NRSur7dq4 waveform in the time domain.

        Parameters
        ----------
        params : dict
            Contains 'chirp_mass', 'mass_ratio', 'spin1x', 'spin1y', 'spin1z',
            'spin2x', 'spin2y', 'spin2z', 'luminosity_distance', 'inclination', 'phase'.
        grid : jax.Array
            Time array in seconds.
        """
        mc = params.get("chirp_mass")
        q = params.get("mass_ratio")
        eta = q / (1.0 + q) ** 2
        mtot = mc / (eta ** (3.0 / 5.0))

        dist_mpc = params.get("luminosity_distance", 100.0)
        theta = params.get("inclination", 0.0)
        phi = params.get("phase", 0.0)

        s1x = params.get("spin1x", 0.0)
        s1y = params.get("spin1y", 0.0)
        s1z = params.get("spin1z", 0.0)
        s2x = params.get("spin2x", 0.0)
        s2y = params.get("spin2y", 0.0)
        s2z = params.get("spin2z", 0.0)

        jaxnrsur_params = jnp.array(
            [mtot, dist_mpc, theta, phi, q, s1x, s1y, s1z, s2x, s2y, s2z]
        )

        hp, hc = self.jaxnrsur_model.get_waveform_td(grid, jaxnrsur_params)
        return hp, hc
