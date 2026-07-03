"""jaxpe: normalizing-flow-enhanced gradient MCMC in JAX, with GW PE as the flagship application."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jaxpe")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from . import core, diagnostics, flows, kernels, sampler

__all__ = ["core", "kernels", "flows", "sampler", "diagnostics", "__version__"]
