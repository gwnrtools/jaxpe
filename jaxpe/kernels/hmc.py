r"""Hamiltonian Monte Carlo (HMC) with a fixed-length leapfrog trajectory.

Hamiltonian Monte Carlo (HMC) reformulates the statistical sampling problem as a
classical mechanics simulation. To efficiently explore the high-dimensional, highly
correlated posterior manifold of a gravitational-wave event, HMC suppresses random-walk
behavior by exploiting the exact gradient of the posterior, driving coherent excursions
across the parameter space.

Motivation & Math
-----------------
Let $\mathbf{q} \in \mathbb{R}^n$ denote our target parameters (the position in
configuration space) and $\mathbf{p} \in \mathbb{R}^n$ denote an auxiliary momentum vector.
We construct the phase space $(\mathbf{q}, \mathbf{p})$ and define a Hamiltonian:
$$ H(\mathbf{q}, \mathbf{p}) = U(\mathbf{q}) + K(\mathbf{p}) $$
where $U(\mathbf{q}) = -\log \pi(\mathbf{q}|d)$ is the potential energy (the negative
log-posterior of the GW data) and $K(\mathbf{p}) = \frac{1}{2} \mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}$
is the kinetic energy for a chosen mass matrix $\mathbf{M}$.

The system evolves along contours of constant $H$ according to Hamilton's equations:
$$ \frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}} = \mathbf{M}^{-1} \mathbf{p} $$
$$ \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}} = \nabla_{\mathbf{q}} \log \pi(\mathbf{q}|d) $$

By Liouville's theorem, this flow is volume-preserving. Furthermore, the symplectic
structure of Hamilton's equations ensures that the transformation is reversible.
In practice, we discretize time using a symplectic integrator—the leapfrog algorithm.
Because the numerical integration introduces $\mathcal{O}(\epsilon^2)$ energy errors,
we append a Metropolis-Hastings acceptance step based on the energy discrepancy
$\Delta H = H(\mathbf{q}_{\text{new}}, \mathbf{p}_{\text{new}}) - H(\mathbf{q}, \mathbf{p})$.

Implementation Details
----------------------
Fixed trajectory length keeps every chain's step identical in shape, so the kernel
vmaps cleanly in JAX (unlike NUTS-style dynamic trees). A per-dimension ``scale`` $d$
plays the role of $\sqrt{M^{-1}}$ (square root of the inverse mass diagonal):
momenta are drawn as $p \sim \mathcal{N}(0, \text{diag}(1/d^2))$ and the kinetic
energy is $K(p) = \frac{1}{2} ||d * p||^2$.

For strongly correlated targets, ``scale`` may instead be a **lower-triangular
Cholesky factor** $L$ with $M^{-1} = L L^T$ (e.g. ``chol(ensemble_cov(samples))``):
momenta are drawn as $p = L^{-T} v$, $v \sim \mathcal{N}(0, I)$ (so $p \sim
\mathcal{N}(0, M)$), the position update uses $M^{-1} p = L (L^T p)$, and the kinetic
energy is $K(p) = \frac{1}{2} ||L^T p||^2$. The diagonal case is recovered exactly
for $L = \text{diag}(d)$.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .base import Kernel, KernelState, LogProbFn, mh_accept


class HMC(Kernel):
    r"""
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
        Square root of the inverse mass matrix $\sqrt{M^{-1}}$: either its diagonal
        (shape ``(n_dim,)``), or a dense lower-triangular Cholesky factor $L$ with
        $M^{-1} = L L^T$ (shape ``(n_dim, n_dim)``). If None, the identity is used.
    """

    needs_gradient: ClassVar[bool] = True
    step_size: jax.Array
    n_leapfrog: int = 10
    scale: jax.Array | None = None  # (n_dim,) diag or (n_dim, n_dim) lower-tri chol

    def __init__(self, step_size: float, n_leapfrog: int = 10, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.n_leapfrog = n_leapfrog
        self.scale = None if scale is None else jnp.asarray(scale)

    def _mass_ops(self):
        """(momentum draw from N(0, M), inverse-mass matvec, sqrt(M^-1)^T matvec)."""
        d = self.scale
        if d is None:
            return (lambda v: v), (lambda p: p), (lambda p: p)
        if d.ndim == 1:
            return (lambda v: v / d), (lambda p: d**2 * p), (lambda p: d * p)
        return (
            lambda v: solve_triangular(d.T, v, lower=False),
            lambda p: d @ (d.T @ p),
            lambda p: d.T @ p,
        )

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_mom, key_acc = jax.random.split(key)
        draw, m_inv, half_t = self._mass_ops()
        eps = self.step_size
        grad_fn = jax.value_and_grad(logp_fn)

        p0 = draw(jax.random.normal(key_mom, state.x.shape, state.x.dtype))

        def leapfrog(carry, _):
            x, p, grad = carry
            p = p + 0.5 * eps * grad
            x = x + eps * m_inv(p)
            _, grad = grad_fn(x)
            p = p + 0.5 * eps * grad
            return (x, p, grad), None

        (x_new, p_new, grad_new), _ = jax.lax.scan(
            leapfrog, (state.x, p0, state.grad), None, length=self.n_leapfrog
        )
        logp_new = logp_fn(x_new)

        kinetic0 = 0.5 * jnp.sum(half_t(p0) ** 2)
        kinetic1 = 0.5 * jnp.sum(half_t(p_new) ** 2)
        proposal = KernelState(x=x_new, log_prob=logp_new, grad=grad_new)
        log_ratio = logp_new - state.log_prob + kinetic0 - kinetic1
        return mh_accept(key_acc, state, proposal, log_ratio)
