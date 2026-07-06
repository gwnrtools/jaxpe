"""Hamiltonian Monte Carlo (HMC) with a fixed-length leapfrog trajectory.

Hamiltonian Monte Carlo solves the "random walk" problem of standard MCMC by
using gradient information to simulate Hamiltonian dynamics. By treating the negative
log-posterior as a potential energy well, we can assign a random momentum to our chain
and let it "roll" along the contours of the probability distribution.

Motivation & Math
-----------------
Let $x$ be the position (our parameters $\theta$) and $p$ be an auxiliary momentum variable.
We define a Hamiltonian $H(x, p)$:
$$ H(x, p) = U(x) + K(p) $$
where $U(x) = -\log \pi(x)$ is the potential energy (negative log-posterior) and
$K(p) = \frac{1}{2} p^T M^{-1} p$ is the kinetic energy, with $M$ being the mass matrix.

Hamilton's equations describe the time evolution of this system:
$$ \frac{dx}{dt} = \frac{\partial H}{\partial p} = M^{-1} p $$
$$ \frac{dp}{dt} = -\frac{\partial H}{\partial x} = -\nabla U(x) = \nabla \log \pi(x) $$

Since $H(x,p)$ is conserved along these trajectories, a perfect simulation would always
be accepted in the Metropolis-Hastings step. In practice, we discretize time using the
leapfrog integrator.

Implementation Details
----------------------
Fixed trajectory length keeps every chain's step identical in shape, so the kernel
vmaps cleanly in JAX (unlike NUTS-style dynamic trees). A per-dimension ``scale`` $d$
plays the role of $\sqrt{M^{-1}}$ (square root of the inverse mass diagonal):
momenta are drawn as $p \sim \mathcal{N}(0, \text{diag}(1/d^2))$ and the kinetic
energy is $K(p) = \frac{1}{2} ||d * p||^2$.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class HMC(Kernel):
    """
    Hamiltonian Monte Carlo Kernel.

    Proposes new states by numerically integrating Hamilton's equations using the
    leapfrog method.

    Parameters
    ----------
    step_size : float
        The step size ($\epsilon$) for the leapfrog integrator.
    n_leapfrog : int, default=10
        The number of leapfrog steps per proposal. The total integration time is
        $\epsilon \times \text{n\_leapfrog}$.
    scale : jax.Array | None, default=None
        The diagonal of the inverse mass matrix $\sqrt{M^{-1}}$. If None, defaults to
        the identity matrix ($d=1$).
    """

    needs_gradient: ClassVar[bool] = True
    step_size: jax.Array
    n_leapfrog: int = 10
    scale: jax.Array | None = None  # (n_dim,) ~ sqrt(inverse mass) diagonal

    def __init__(self, step_size: float, n_leapfrog: int = 10, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.n_leapfrog = n_leapfrog
        self.scale = None if scale is None else jnp.asarray(scale)

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_mom, key_acc = jax.random.split(key)
        d = 1.0 if self.scale is None else self.scale
        eps = self.step_size
        grad_fn = jax.value_and_grad(logp_fn)

        p0 = jax.random.normal(key_mom, state.x.shape, state.x.dtype) / d

        def leapfrog(carry, _):
            x, p, grad = carry
            p = p + 0.5 * eps * grad
            x = x + eps * d**2 * p
            _, grad = grad_fn(x)
            p = p + 0.5 * eps * grad
            return (x, p, grad), None

        (x_new, p_new, grad_new), _ = jax.lax.scan(
            leapfrog, (state.x, p0, state.grad), None, length=self.n_leapfrog
        )
        logp_new = logp_fn(x_new)

        kinetic0 = 0.5 * jnp.sum((d * p0) ** 2)
        kinetic1 = 0.5 * jnp.sum((d * p_new) ** 2)
        proposal = KernelState(x=x_new, log_prob=logp_new, grad=grad_new)
        log_ratio = logp_new - state.log_prob + kinetic0 - kinetic1
        return mh_accept(key_acc, state, proposal, log_ratio)
