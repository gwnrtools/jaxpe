"""Bijections between an unconstrained coordinate y in R and a physical coordinate x.

The sampling engine always works in unconstrained space; these transforms (attached to
priors) map back to the physical support and supply the log-Jacobian that converts the
physical-space posterior density into an unconstrained-space density:

    log p_y(y) = log p_x(forward(y)) + log_det(y),   log_det(y) = log |d forward / dy|.

All methods are elementwise and differentiable, so they can be vmapped over parameters
and chains and sit inside a jitted log-posterior.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class Bijection(eqx.Module):
    """Elementwise map from unconstrained y in R to physical x."""

    def forward(self, y):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError

    def log_det(self, y):
        """log |d forward / dy| at unconstrained y."""
        raise NotImplementedError


class Identity(Bijection):
    """The identity map f(y) = y.

    This bijection leaves the coordinate unchanged and has a zero log-Jacobian.
    """

    def forward(self, y):
        return y

    def inverse(self, x):
        return x

    def log_det(self, y):
        return jnp.zeros_like(y)


class Affine(Bijection):
    """An affine transformation mapping y to x = loc + scale * y.

    Used to shift and scale the unconstrained coordinate.

    Parameters
    ----------
    loc : float
        The shift or location parameter.
    scale : float
        The scale multiplier. Must be strictly positive (> 0).
    """

    loc: float
    scale: float

    def forward(self, y):
        return self.loc + self.scale * y

    def inverse(self, x):
        return (x - self.loc) / self.scale

    def log_det(self, y):
        return jnp.full_like(y, jnp.log(self.scale))


class Interval(Bijection):
    """A bounded transformation mapping the real line R onto the open interval (low, high).

    The mapping is defined by x = low + (high - low) * sigmoid(y), where sigmoid
    is the logistic function.

    Parameters
    ----------
    low : float
        The lower bound of the physical interval.
    high : float
        The upper bound of the physical interval.
    """

    low: float
    high: float

    def forward(self, y):
        return self.low + (self.high - self.low) * jax.nn.sigmoid(y)

    def inverse(self, x):
        u = (x - self.low) / (self.high - self.low)
        return jnp.log(u) - jnp.log1p(-u)

    def log_det(self, y):
        # d/dy sigmoid = sigmoid(y) sigmoid(-y); use log_sigmoid for stability
        return (
            jnp.log(self.high - self.low)
            + jax.nn.log_sigmoid(y)
            + jax.nn.log_sigmoid(-y)
        )
