import jax
import jax.numpy as jnp


def f(x):
    return jnp.exp(1j * x)


x = jnp.array(0.5)
y, vjp_fn = jax.vjp(f, x)
y_bar = 2.0 + 3.0j
(x_bar,) = vjp_fn(y_bar)

jac = jax.jacfwd(f)(x)
print("x_bar from JAX vjp:", x_bar)
print("Re(y_bar * jac):", jnp.real(y_bar * jac))
print("Re(y_bar * conj(jac)):", jnp.real(y_bar * jnp.conj(jac)))
print("Re(conj(y_bar) * jac):", jnp.real(jnp.conj(y_bar) * jac))


# Let's also check complex arrays
def g(x):
    return jnp.array([jnp.cos(x) + 1j * jnp.sin(x), x + 2j * x])


y2, vjp_fn2 = jax.vjp(g, x)
y_bar2 = jnp.array([2.0 + 3.0j, 4.0 + 5.0j])
(x_bar2,) = vjp_fn2(y_bar2)
jac2 = jax.jacfwd(g)(x)

print("x_bar2 from JAX:", x_bar2)
print("Sum Re(conj(y_bar) * jac):", jnp.sum(jnp.real(jnp.conj(y_bar2) * jac2)))
print("Sum Re(y_bar * conj(jac)):", jnp.sum(jnp.real(y_bar2 * jnp.conj(jac2))))
