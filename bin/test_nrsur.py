import jax.numpy as jnp
from jaxpe.gw.cbc_models.nrsur7dq4 import NRSur7dq4

# Test downloading / passing data path
try:
    import urllib.request
    import os

    if not os.path.exists("NRSur7dq4.h5"):
        print("Downloading NRSur7dq4.h5...")
        urllib.request.urlretrieve(
            "https://zenodo.org/record/3831272/files/NRSur7dq4.h5", "NRSur7dq4.h5"
        )

    model = NRSur7dq4("NRSur7dq4.h5")
    params = {
        "chirp_mass": 30.0,
        "mass_ratio": 0.5,
        "luminosity_distance": 400.0,
        "spin1z": 0.5,
    }
    times = jnp.linspace(-2.0, 0.05, 4096)
    hp, hc = model(params, times)
    print("hp shape:", hp.shape)
    print("hc shape:", hc.shape)
    print("Success!")
except Exception as e:
    print("Error:", e)
