import jax
import jax.numpy as jnp


def _heavy_math(x, y):
    return x * y


heavy_math = jax.custom_vjp(lambda th, tms: _heavy_math(th, tms))


def fwd(th, tms):
    return _heavy_math(th, tms), tms


def bwd(res, g):
    # res is tms
    return (g * res, None)


heavy_math.defvjp(fwd, bwd)


@jax.jit
def test(x, y):
    # inside a jit, y is a tracer
    return heavy_math(x, y)


x = jnp.array(2.0)
y = jnp.array(3.0)
print(jax.grad(lambda x: test(x, y))(x))
