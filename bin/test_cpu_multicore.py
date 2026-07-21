import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

print(f"JAX devices: {jax.devices()}")
