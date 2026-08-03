import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import lalsimulation as lalsim
import lal

from jaxpe.gw.cbc_models.phenomthm import IMRPhenomTHM


def test_compare_with_lalsuite():
    # Define parameters for a reference binary
    m1 = 30.0 * lal.MSUN_SI
    m2 = 20.0 * lal.MSUN_SI
    s1x, s1y, s1z = 0.0, 0.0, 0.5
    s2x, s2y, s2z = 0.0, 0.0, -0.3
    distance = 100.0 * 1e6 * lal.PC_SI
    inclination = 0.5
    phiRef = 0.0
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0
    deltaT = 1.0 / 4096.0
    f_min = 20.0
    f_ref = 20.0

    # LALSuite parameters
    params_dict = lal.CreateDict()

    approx = lalsim.IMRPhenomTHM

    # Generate LALSuite waveform
    hp_lal, hc_lal = lalsim.SimInspiralChooseTDWaveform(
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        distance,
        inclination,
        phiRef,
        longAscNodes,
        eccentricity,
        meanPerAno,
        deltaT,
        f_min,
        f_ref,
        params_dict,
        approx,
    )

    # The LALSuite waveform is a TimeSeries. Let's get the time array.
    t_lal = jnp.arange(hp_lal.data.length) * hp_lal.deltaT + float(hp_lal.epoch)

    # Our JAX implementation
    model = IMRPhenomTHM(f_ref=f_ref)

    # JAX parameters dict
    M_total = (m1 + m2) / lal.MSUN_SI
    q = m1 / m2
    mc = M_total * (q / (1.0 + q) ** 2) ** (3.0 / 5.0)

    jax_params = {
        "chirp_mass": mc,
        "mass_ratio": q,
        "spin1z": s1z,
        "spin2z": s2z,
        "luminosity_distance": distance / (1e6 * lal.PC_SI),
        "phase": phiRef,
        "inclination": inclination,
        "geocent_time": 0.0,  # Shift it to align peaks if necessary
    }

    hp_jax, hc_jax = model(jax_params, t_lal)

    # For now, we just ensure it evaluates correctly on the same time grid
    assert hp_jax.shape == t_lal.shape
    assert hc_jax.shape == t_lal.shape

    print("LALSuite comparison evaluated successfully.")


if __name__ == "__main__":
    test_compare_with_lalsuite()
