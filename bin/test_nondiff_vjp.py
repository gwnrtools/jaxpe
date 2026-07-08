import jax
import jax.numpy as jnp


def f(x, y):
    return x * y


f_vjp = jax.custom_vjp(f, nondiff_argnums=(1,))


def f_fwd(x, y):
    return f(x, y), y


def f_bwd(y, res, g):
    return (g * res,)


f_vjp.defvjp(f_fwd, f_bwd)

x = jnp.array(2.0)
y = jnp.array(3.0)
print(jax.grad(lambda x: f_vjp(x, y))(x))
