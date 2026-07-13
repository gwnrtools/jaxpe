"""Active-learning surrogate PE for expensive non-JAX waveform models.

See ``docs/gpry_fusion_design.md``. Requires the ``surrogate`` extra (GPry);
imported lazily so the rest of jaxpe works without it.
"""

from .engine import SurrogateEngine, SurrogateSamples


def __getattr__(name):
    if name == "GPryEngine":  # deferred: needs the optional gpry dependency
        from .gpry_backend import GPryEngine

        return GPryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SurrogateEngine", "SurrogateSamples", "GPryEngine"]
