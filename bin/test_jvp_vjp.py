import jax


@jax.custom_vjp
def f(x):
    return x * x


def f_fwd(x):
    return f(x), 2 * x


def f_bwd(res, g):
    return (g * res,)


f.defvjp(f_fwd, f_bwd)

try:
    jax.jacfwd(f)(2.0)
except Exception as e:
    print(repr(e))
