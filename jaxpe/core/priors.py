"""One-dimensional priors and their joint container.

Each prior knows its normalized log-density on the physical support, how to draw
samples, and which bijection maps unconstrained R onto its support. ``JointPrior``
assembles named 1-D priors into the flat parameter vector convention used by the
sampling engine: physical vectors x and unconstrained vectors y of shape (n_dim,),
ordered as ``names``.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .transforms import Affine, Bijection, Identity, Interval


class Prior(eqx.Module):
    """A normalized 1-D prior over a physical parameter."""

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, key, shape=()):
        raise NotImplementedError

    @property
    def bijection(self) -> Bijection:
        raise NotImplementedError


class Uniform(Prior):
    low: float
    high: float

    def log_prob(self, x):
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, -jnp.log(self.high - self.low), -jnp.inf)

    def sample(self, key, shape=()):
        return jax.random.uniform(key, shape, minval=self.low, maxval=self.high)

    @property
    def bijection(self):
        return Interval(self.low, self.high)


class LogUniform(Prior):
    """p(x) proportional to 1/x on [low, high], low > 0."""

    low: float
    high: float

    def log_prob(self, x):
        norm = jnp.log(jnp.log(self.high / self.low))
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, -jnp.log(x) - norm, -jnp.inf)

    def sample(self, key, shape=()):
        u = jax.random.uniform(key, shape)
        return self.low * (self.high / self.low) ** u

    @property
    def bijection(self):
        return Interval(self.low, self.high)


class PowerLaw(Prior):
    """p(x) proportional to x**alpha on [low, high]; alpha must not be -1 (use LogUniform)."""

    alpha: float
    low: float
    high: float

    def _norm(self):
        a1 = self.alpha + 1.0
        return (self.high**a1 - self.low**a1) / a1

    def log_prob(self, x):
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, self.alpha * jnp.log(x) - jnp.log(self._norm()), -jnp.inf)

    def sample(self, key, shape=()):
        u = jax.random.uniform(key, shape)
        a1 = self.alpha + 1.0
        return (self.low**a1 + u * (self.high**a1 - self.low**a1)) ** (1.0 / a1)

    @property
    def bijection(self):
        return Interval(self.low, self.high)


class Sine(Prior):
    """p(x) proportional to sin(x) on [low, high] within [0, pi] (e.g. inclination)."""

    low: float = 0.0
    high: float = jnp.pi

    def _norm(self):
        return jnp.cos(self.low) - jnp.cos(self.high)

    def log_prob(self, x):
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, jnp.log(jnp.sin(x)) - jnp.log(self._norm()), -jnp.inf)

    def sample(self, key, shape=()):
        u = jax.random.uniform(key, shape)
        return jnp.arccos(jnp.cos(self.low) - u * self._norm())

    @property
    def bijection(self):
        return Interval(self.low, self.high)


class Cosine(Prior):
    """p(x) proportional to cos(x) on [low, high] within [-pi/2, pi/2] (e.g. declination)."""

    low: float = -jnp.pi / 2
    high: float = jnp.pi / 2

    def _norm(self):
        return jnp.sin(self.high) - jnp.sin(self.low)

    def log_prob(self, x):
        inside = (x >= self.low) & (x <= self.high)
        return jnp.where(inside, jnp.log(jnp.cos(x)) - jnp.log(self._norm()), -jnp.inf)

    def sample(self, key, shape=()):
        u = jax.random.uniform(key, shape)
        return jnp.arcsin(jnp.sin(self.low) + u * self._norm())

    @property
    def bijection(self):
        return Interval(self.low, self.high)


class Gaussian(Prior):
    mu: float = 0.0
    sigma: float = 1.0

    def log_prob(self, x):
        z = (x - self.mu) / self.sigma
        return -0.5 * z**2 - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi)

    def sample(self, key, shape=()):
        return self.mu + self.sigma * jax.random.normal(key, shape)

    @property
    def bijection(self):
        return Affine(self.mu, self.sigma)


class Fixed(Prior):
    """A parameter pinned to a constant (delta prior); still occupies a slot for simplicity."""

    value: float

    def log_prob(self, x):
        return jnp.zeros_like(x)

    def sample(self, key, shape=()):
        return jnp.full(shape, self.value)

    @property
    def bijection(self):
        return Identity()


class JointPrior(eqx.Module):
    """Ordered collection of named 1-D priors defining the flat vector convention.

    Physical vectors x and unconstrained vectors y have shape (n_dim,) with components
    ordered as ``names``. Densities are elementwise-independent by construction.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    priors: tuple[Prior, ...]

    def __init__(self, priors: dict[str, Prior]):
        self.names = tuple(priors.keys())
        self.priors = tuple(priors.values())

    @property
    def n_dim(self) -> int:
        return len(self.names)

    def as_dict(self, x) -> dict:
        """Split a (..., n_dim) physical vector into named components."""
        return {name: x[..., i] for i, name in enumerate(self.names)}

    def from_dict(self, d: dict):
        return jnp.stack([d[name] for name in self.names], axis=-1)

    def sample(self, key, n: int):
        """Draw (n, n_dim) physical samples."""
        keys = jax.random.split(key, self.n_dim)
        cols = [p.sample(k, (n,)) for p, k in zip(self.priors, keys)]
        return jnp.stack(cols, axis=-1)

    def log_prob(self, x):
        """Physical-space log-density of a (n_dim,) vector."""
        terms = [p.log_prob(x[i]) for i, p in enumerate(self.priors)]
        return jnp.sum(jnp.stack(terms))

    def to_physical(self, y):
        cols = [p.bijection.forward(y[i]) for i, p in enumerate(self.priors)]
        return jnp.stack(cols)

    def to_unconstrained(self, x):
        cols = [p.bijection.inverse(x[i]) for i, p in enumerate(self.priors)]
        return jnp.stack(cols)

    def log_det(self, y):
        """Sum of log |dx/dy| over components, at unconstrained y of shape (n_dim,)."""
        terms = [p.bijection.log_det(y[i]) for i, p in enumerate(self.priors)]
        return jnp.sum(jnp.stack(terms))

    def log_prob_unconstrained(self, y):
        """Prior density expressed in unconstrained coordinates (includes Jacobian)."""
        return self.log_prob(self.to_physical(y)) + self.log_det(y)


PriorDict = dict[str, Prior]
LogLikelihood = Callable[[dict], jax.Array]
