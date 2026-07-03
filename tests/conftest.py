import jax

# Correctness tests run in float64; the float32 fast path is exercised separately.
jax.config.update("jax_enable_x64", True)
