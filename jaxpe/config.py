r"""Declarative run configuration for the jaxpe CLI.

Every physical and numerical choice a run makes -- segment duration, sampling rate,
band limits, the prior *distributions*, the distributions injected truths are drawn
from, kernel step sizes, sampler budgets, and all RNG seeds -- lives in a JSON
configuration file rather than in :mod:`jaxpe.cli`. A production run is therefore
defined by an artifact that can be version-controlled, diffed, attached to a paper,
and replayed, instead of by a source-code edit.

Layout
------
The file is a nested JSON object with these sections, all optional::

    {
      "data":      { duration, sampling_rate, f_min, f_max, f_ref,
                     post_trigger, tukey_alpha },
      "prior":     { <parameter>: <distribution>, ... },
      "injection": { geocent_time, parameters: {<parameter>: <distribution>},
                     fiducial: {<parameter>: <value>} },
      "seeds":     { injection, noise, sampler, ns, gpry },
      "kernel":    { hmc: {step_size, n_leapfrog}, mala: {step_size} },
      "sampler":   { any field of GlobalLocalConfig },
      "ns":        { nlive, num_repeats, precision_criterion, nprior,
                     max_ncalls, verbosity },
      "gpry":      { n_initial, max_total, max_initial, acquisition,
                     ref_bounds_rel, ref_bounds_abs },
      "esigma":    { modes, rad_pn_order, mode_pn_order, ode_eps, n_ode_grid,
                     max_ode_steps, taper_on_seconds, taper_off_seconds }
    }

Anything omitted falls back to :data:`DEFAULT_CONFIG`. Merging is per-key and
recursive, so a file containing only ``{"data": {"duration": 128.0}}`` changes the
duration and nothing else.

Distributions
-------------
``prior`` and ``injection.parameters`` speak the same small language, so the box the
sampler explores and the population injected truths come from are specified the same
way and are built by the same code -- which is what a PP-plot campaign requires, since
there the injection distribution *must* be the prior. Each entry is either the
shorthand ``[low, high]`` (meaning uniform) or an object with a ``dist`` key:

===================================  ==============================================
``{"dist": "uniform", ...}``         ``low``, ``high``
``{"dist": "loguniform", ...}``      ``low``, ``high`` (both > 0); $p(x) \propto 1/x$
``{"dist": "powerlaw", ...}``        ``alpha``, ``low``, ``high``; $p(x) \propto x^\alpha$
``{"dist": "sine", ...}``            ``low``, ``high`` (default $0, \pi$); $p \propto \sin x$
``{"dist": "cosine", ...}``          ``low``, ``high`` (default $\pm\pi/2$); $p \propto \cos x$
``{"dist": "gaussian", ...}``        ``mu``, ``sigma``
``{"dist": "fixed", ...}``           ``value`` -- pins the parameter, keeping its slot
===================================  ==============================================

Any entry may also carry ``"relative_to_trigger": true``, which offsets ``low``,
``high``, ``mu`` or ``value`` by ``injection.geocent_time``. That is how the default
``geocent_time`` prior expresses "the trigger time plus or minus 0.1 s" without
hardcoding an epoch.

``injection.parameters`` may also be the single string ``"prior"``, meaning "draw
from the prior itself". That is the condition a PP-plot campaign tests, and stating
it this way keeps the two from drifting apart in a hand-edited file.

Isotropy is expressed, not assumed: an isotropically oriented population is
``inclination: {"dist": "sine"}`` with ``dec: {"dist": "cosine"}``, and a
Euclidean-volumetric one is ``luminosity_distance: {"dist": "powerlaw", "alpha": 2}``.
Changing those entries changes the population, with no code edit.

Conveniences and guard rails
----------------------------
Any key whose name begins with ``_`` is treated as a comment and ignored (JSON has no
comment syntax). Every other unrecognised key is a hard error rather than being
silently dropped -- a typo like ``"sample_rate"`` must not quietly leave the run at
1024 Hz.

:func:`validate_config` enforces the invariants that used to be prose comments in
``cli.py`` and could therefore rot:

* the support of every injection distribution lies inside the support of the
  corresponding prior, so no injected truth falls outside the prior and becomes
  unrecoverable;
* ``f_min`` is below the Nyquist frequency, and ``f_max`` (if set) is above ``f_min``
  and at or below Nyquist;
* mass ratios lie in $(0, 1]$, dimensionless spins in $[-1, 1]$, sine/cosine supports
  lie within $[0, \pi]$ and $[-\pi/2, \pi/2]$, and every range is correctly ordered;
* prior and injection sections cover exactly the parameters the driver needs.

Violations raise :class:`ConfigError` listing *all* problems at once. Choices that are
legal but rarely intended -- a reference frequency outside the analysed band, a
nested-sampling ``nlive`` low enough to be a smoke test rather than a measurement, an
injection distribution that is not the prior -- are returned as warnings instead.

The fiducial binary is the one containment check that is *not* fatal at load time:
only ``--fiducial`` runs read it, so erroring would block an otherwise fine narrow
prior (a BNS configuration, say) on account of a 30 solar-mass demo binary it never
touches. :func:`fiducial_errors` reports the problem precisely so the caller can make
it fatal at the point of use, which is what the CLI does.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG",
    "PARAMETERS",
    "PRIOR_ALIAS",
    "load_config",
    "merge_config",
    "validate_config",
    "fiducial_errors",
    "build_prior",
    "build_distribution",
    "sample_parameters",
    "spec_bounds",
    "fiducial_injection",
    "global_local_kwargs",
    "time_width",
    "dump_default_config",
]

TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0

# Value of injection.parameters meaning "draw from the prior itself".
PRIOR_ALIAS = "prior"


class ConfigError(ValueError):
    """Raised when a configuration is unusable. Reports every problem at once."""


# The parameter vector the CLI's waveform models and likelihoods require, in the
# order the components appear in every physical vector. Order matters: it fixes the
# column order of raw_samples.npz and posterior_samples.npy. It matches the ordering
# jaxpe.gw.bbh_priors has always produced, so outputs stay comparable across the
# introduction of this module.
PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "spin1z",
    "spin2z",
    "luminosity_distance",
    "inclination",
    "phase",
    "ra",
    "dec",
    "psi",
    "geocent_time",
)


# ---------------------------------------------------------------------------
# Defaults
#
# The physics here reproduces exactly what jaxpe.cli hardcoded before this module
# existed, so an invocation with no --config is unchanged. It is deliberately
# *smoke-test* scale -- see examples/configs/ for production settings.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "data": {
        # Conditioning of the simulated strain. duration x sampling_rate fixes the
        # frequency resolution df = 1/duration and the Nyquist frequency.
        "duration": 4.0,
        "sampling_rate": 1024.0,
        "f_min": 30.0,
        "f_max": None,  # None -> Nyquist
        "f_ref": 20.0,  # frequency at which spins and phase are defined
        "post_trigger": 2.0,  # seconds of data after the trigger time
        "tukey_alpha": 0.1,  # window roll-off fraction
    },
    # The distributions the sampler explores. Both spin entries are read, but
    # bbh_priors-style aligned-spin models share one range across components, so
    # differing spin1z/spin2z priors draw a warning rather than being silently merged.
    "prior": {
        "chirp_mass": {"dist": "uniform", "low": 10.0, "high": 50.0},
        "mass_ratio": {"dist": "uniform", "low": 0.1, "high": 1.0},
        "spin1z": {"dist": "uniform", "low": -0.9, "high": 0.9},
        "spin2z": {"dist": "uniform", "low": -0.9, "high": 0.9},
        # p(d) ~ d^2: sources uniform in Euclidean volume.
        "luminosity_distance": {
            "dist": "powerlaw",
            "alpha": 2.0,
            "low": 100.0,
            "high": 2000.0,
        },
        "inclination": {"dist": "sine", "low": 0.0, "high": math.pi},
        "phase": {"dist": "uniform", "low": 0.0, "high": TWO_PI},
        "ra": {"dist": "uniform", "low": 0.0, "high": TWO_PI},
        "dec": {"dist": "cosine", "low": -HALF_PI, "high": HALF_PI},
        "psi": {"dist": "uniform", "low": 0.0, "high": math.pi},
        # Trigger +- 0.1 s, without hardcoding the epoch here.
        "geocent_time": {
            "dist": "uniform",
            "low": -0.1,
            "high": 0.1,
            "relative_to_trigger": True,
        },
    },
    "injection": {
        # Trigger time shared by drawn and fiducial injections, and the origin for
        # every "relative_to_trigger" distribution.
        "geocent_time": 1126259462.4,
        # The population injected truths are drawn from. Mass, spin and distance
        # draws use a strictly narrower box than the prior so that no truth sits on a
        # prior edge and rails the posterior; orientation and sky are isotropic and
        # therefore span the full prior support. Set these equal to "prior" for a
        # PP-plot campaign, where the injection distribution must be the prior.
        "parameters": {
            "chirp_mass": {"dist": "uniform", "low": 15.0, "high": 45.0},
            "mass_ratio": {"dist": "uniform", "low": 0.3, "high": 1.0},
            "spin1z": {"dist": "uniform", "low": -0.5, "high": 0.5},
            "spin2z": {"dist": "uniform", "low": -0.5, "high": 0.5},
            "luminosity_distance": {"dist": "uniform", "low": 300.0, "high": 1500.0},
            "inclination": {"dist": "sine", "low": 0.0, "high": math.pi},
            "phase": {"dist": "uniform", "low": 0.0, "high": TWO_PI},
            "ra": {"dist": "uniform", "low": 0.0, "high": TWO_PI},
            "dec": {"dist": "cosine", "low": -HALF_PI, "high": HALF_PI},
            "psi": {"dist": "uniform", "low": 0.0, "high": math.pi},
            "geocent_time": {
                "dist": "fixed",
                "value": 0.0,
                "relative_to_trigger": True,
            },
        },
        # The fixed reference binary selected by --fiducial: a GW150914-like BBH,
        # face-on, optimally oriented. A stable point of comparison across runs and
        # releases. geocent_time defaults to injection.geocent_time.
        "fiducial": {
            "chirp_mass": 30.0,
            "mass_ratio": 0.8,
            "spin1z": 0.0,
            "spin2z": 0.0,
            "luminosity_distance": 700.0,
            "inclination": 0.0,
            "phase": 0.0,
            "ra": 0.0,
            "dec": 0.0,
            "psi": 0.0,
        },
    },
    "seeds": {
        # Kept separate rather than derived from one master seed so that a campaign
        # can be repeated with a new noise realisation while holding the injection
        # set and the sampler start fixed.
        "injection": 42,
        "noise": 42,
        "sampler": 42,
        "ns": 42,
        "gpry": 42,
    },
    "kernel": {
        "hmc": {"step_size": 0.01, "n_leapfrog": 10},
        "mala": {"step_size": 0.01},
    },
    "sampler": {
        # Any field of GlobalLocalConfig is accepted here; these four are the ones the
        # CLI has always overridden away from that dataclass's own defaults.
        "n_chains": 100,
        "n_prelim_loops": 1,
        "n_training_loops": 5,
        "n_production_loops": 50,
    },
    "ns": {
        # nlive=10 is a smoke-test value: it exits quickly and exercises the code
        # path, but the evidence and posterior it returns are not measurements.
        # validate_config warns below NLIVE_ADVISORY_MIN.
        "nlive": 10,
        "num_repeats": 5,
        "precision_criterion": 0.1,
        "nprior": None,
        "max_ncalls": None,
        "verbosity": 1,
    },
    "gpry": {
        "n_initial": None,  # None -> GPry's own 3 * n_dim
        "max_total": 500,
        "max_initial": 200,
        "acquisition": None,  # None -> GPry's native NORA
        # Reference bounds handed to GPry are a box around the injected truth of
        # half-width max(ref_bounds_rel * |truth|, ref_bounds_abs).
        "ref_bounds_rel": 0.01,
        "ref_bounds_abs": 0.1,
    },
    "esigma": {
        # Forwarded directly to jaxpe.gw.ESIGMAInspiral (f_lower comes from
        # data.f_min instead, so the two cannot drift apart).
        "modes": [[2, 2], [3, 3]],
        "rad_pn_order": 8,
        "mode_pn_order": 8,
        "ode_eps": 1e-7,
        "n_ode_grid": 2048,
        "max_ode_steps": 32768,
        "taper_on_seconds": 0.05,
        "taper_off_seconds": 0.02,
    },
}

# Below this, nested sampling exercises the code path rather than measuring a
# posterior. Not an error: the CLI's own integration tests run at nlive=10.
NLIVE_ADVISORY_MIN = 100

# Half-width of the Gaussian interval used where a *finite* box is structurally
# required (the ns/gpry bounds), since a Gaussian prior has unbounded support.
GAUSSIAN_BOUND_SIGMAS = 5.0


# ---------------------------------------------------------------------------
# The distribution mini-language
# ---------------------------------------------------------------------------

# name -> (required keys, optional keys with their defaults)
_DISTRIBUTIONS: dict = {
    "uniform": (("low", "high"), {}),
    "loguniform": (("low", "high"), {}),
    "powerlaw": (("alpha", "low", "high"), {}),
    "sine": ((), {"low": 0.0, "high": math.pi}),
    "cosine": ((), {"low": -HALF_PI, "high": HALF_PI}),
    "gaussian": (("mu", "sigma"), {}),
    "fixed": (("value",), {}),
}

# Keys every distribution accepts in addition to its own.
_UNIVERSAL_KEYS = frozenset({"dist", "relative_to_trigger"})

# Which of a spec's numeric keys are shifted by the trigger time.
_TRIGGER_SHIFTED = ("low", "high", "mu", "value")


def normalize_spec(spec, label: str, errors: list):
    """Coerce one distribution entry to a canonical dict, validating its keys.

    ``[low, high]`` is shorthand for a uniform distribution. Returns ``None`` and
    appends to ``errors`` if the entry is malformed.
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            errors.append(
                f"{label}: the [low, high] shorthand needs exactly two numbers, "
                f"got {list(spec)!r}"
            )
            return None
        spec = {"dist": "uniform", "low": spec[0], "high": spec[1]}
    if not isinstance(spec, dict):
        errors.append(
            f'{label} must be a [low, high] list or a {{"dist": ...}} object, '
            f"got {spec!r}"
        )
        return None

    name = spec.get("dist")
    if name is None:
        errors.append(
            f'{label} is missing the "dist" key; expected one of '
            f"{', '.join(sorted(_DISTRIBUTIONS))}"
        )
        return None
    if name not in _DISTRIBUTIONS:
        errors.append(
            f"{label}: unknown distribution {name!r}; expected one of "
            f"{', '.join(sorted(_DISTRIBUTIONS))}"
        )
        return None

    required, optional = _DISTRIBUTIONS[name]
    allowed = _UNIVERSAL_KEYS | set(required) | set(optional)
    for key in spec:
        if key not in allowed:
            errors.append(
                f"{label}: {name!r} does not take {key!r}; accepts "
                f"{', '.join(sorted(allowed - {'dist'}))}"
            )
    out = {"dist": name, "relative_to_trigger": bool(spec.get("relative_to_trigger"))}
    ok = True
    for key in required:
        if key not in spec:
            errors.append(f"{label}: {name!r} requires {key!r}")
            ok = False
            continue
        out[key] = spec[key]
    for key, default in optional.items():
        out[key] = spec.get(key, default)

    for key in tuple(out):
        if key in ("dist", "relative_to_trigger"):
            continue
        try:
            out[key] = float(out[key])
        except (TypeError, ValueError):
            errors.append(f"{label}: {key} must be a number, got {out[key]!r}")
            ok = False
            continue
        if not math.isfinite(out[key]):
            errors.append(f"{label}: {key} must be finite, got {out[key]}")
            ok = False
    if not ok:
        return None

    if "low" in out and "high" in out and out["low"] >= out["high"]:
        errors.append(
            f"{label} is empty or inverted: low={out['low']} must be < high={out['high']}"
        )
        return None
    if name == "loguniform" and out["low"] <= 0.0:
        errors.append(f"{label}: loguniform requires low > 0, got {out['low']}")
        return None
    if name == "powerlaw":
        if out["low"] <= 0.0:
            errors.append(f"{label}: powerlaw requires low > 0, got {out['low']}")
            return None
        if out["alpha"] == -1.0:
            errors.append(
                f"{label}: powerlaw alpha=-1 is not normalizable by this "
                'parameterisation; use {"dist": "loguniform"}'
            )
            return None
    if name == "gaussian" and out["sigma"] <= 0.0:
        errors.append(f"{label}: gaussian requires sigma > 0, got {out['sigma']}")
        return None
    if name == "sine" and not (0.0 <= out["low"] and out["high"] <= math.pi):
        errors.append(
            f"{label}: sine support must lie within [0, pi], got "
            f"[{out['low']}, {out['high']}]"
        )
        return None
    if name == "cosine" and not (-HALF_PI <= out["low"] and out["high"] <= HALF_PI):
        errors.append(
            f"{label}: cosine support must lie within [-pi/2, pi/2], got "
            f"[{out['low']}, {out['high']}]"
        )
        return None
    return out


def resolve_spec(spec: dict, trigger: float = 0.0) -> dict:
    """Apply ``relative_to_trigger`` so the spec's numbers are absolute."""
    out = dict(spec)
    if out.pop("relative_to_trigger", False):
        for key in _TRIGGER_SHIFTED:
            if key in out:
                out[key] = out[key] + trigger
    return out


def spec_support(spec: dict, trigger: float = 0.0):
    """Exact support of a resolved spec as ``(low, high)``.

    Gaussian support is unbounded and reported as infinite; use :func:`spec_bounds`
    where a finite box is structurally required.
    """
    s = resolve_spec(spec, trigger)
    name = s["dist"]
    if name == "fixed":
        return (s["value"], s["value"])
    if name == "gaussian":
        return (-math.inf, math.inf)
    return (s["low"], s["high"])


def spec_bounds(spec: dict, trigger: float = 0.0):
    """A finite ``(low, high)`` box for a spec, for samplers that require one.

    Nested sampling and GPry explore a box rather than a density, so an unbounded
    Gaussian is truncated at +- :data:`GAUSSIAN_BOUND_SIGMAS` and a Fixed parameter
    is widened to a degenerate-but-nonzero interval.
    """
    s = resolve_spec(spec, trigger)
    name = s["dist"]
    if name == "gaussian":
        half = GAUSSIAN_BOUND_SIGMAS * s["sigma"]
        return (s["mu"] - half, s["mu"] + half)
    if name == "fixed":
        v = s["value"]
        pad = abs(v) * 1e-9 + 1e-9
        return (v - pad, v + pad)
    return (s["low"], s["high"])


def build_distribution(spec: dict, trigger: float = 0.0):
    """Instantiate the :mod:`jaxpe.core.priors` object described by ``spec``."""
    from jaxpe.core.priors import (
        Cosine,
        Fixed,
        Gaussian,
        LogUniform,
        PowerLaw,
        Sine,
        Uniform,
    )

    s = resolve_spec(spec, trigger)
    name = s["dist"]
    if name == "uniform":
        return Uniform(low=s["low"], high=s["high"])
    if name == "loguniform":
        return LogUniform(low=s["low"], high=s["high"])
    if name == "powerlaw":
        return PowerLaw(alpha=s["alpha"], low=s["low"], high=s["high"])
    if name == "sine":
        return Sine(low=s["low"], high=s["high"])
    if name == "cosine":
        return Cosine(low=s["low"], high=s["high"])
    if name == "gaussian":
        return Gaussian(mu=s["mu"], sigma=s["sigma"])
    if name == "fixed":
        return Fixed(value=s["value"])
    raise ConfigError(f"unknown distribution {name!r}")  # pragma: no cover


def build_prior(cfg: dict, trigger: float | None = None):
    """Build the :class:`~jaxpe.core.priors.JointPrior` described by ``cfg['prior']``.

    Parameter order follows :data:`PARAMETERS`, which fixes the column order of every
    saved sample array.
    """
    from jaxpe.core.priors import JointPrior

    if trigger is None:
        trigger = float(cfg["injection"]["geocent_time"])
    return JointPrior(
        {
            name: build_distribution(cfg["prior"][name], trigger)
            for name in _ordered_names(cfg["prior"])
        }
    )


def sample_parameters(cfg: dict, key, trigger: float | None = None) -> dict:
    """Draw one injection's true parameters from ``cfg['injection']['parameters']``.

    Uses the same distribution objects as the prior, so an injection campaign whose
    ``parameters`` section equals its ``prior`` section is drawing from exactly the
    prior -- the condition a PP plot tests against.
    """
    import jax

    if trigger is None:
        trigger = float(cfg["injection"]["geocent_time"])
    specs = cfg["injection"]["parameters"]
    names = _ordered_names(specs)
    keys = jax.random.split(key, len(names))
    out = {}
    for name, k in zip(names, keys):
        dist = build_distribution(specs[name], trigger)
        out[name] = float(dist.sample(k, ()))
    return out


def _ordered_names(section: dict) -> list:
    """Names of ``section`` in PARAMETERS order, with any extras appended."""
    known = [n for n in PARAMETERS if n in section]
    extra = [n for n in section if n not in PARAMETERS]
    return known + extra


# ---------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------


def _strip_comments(node):
    """Drop ``_``-prefixed keys recursively. JSON has no comment syntax."""
    if isinstance(node, dict):
        return {
            k: _strip_comments(v) for k, v in node.items() if not str(k).startswith("_")
        }
    return node


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Dict values merge key-by-key, so a partial file changes only what it names --
    except that a distribution entry replaces wholesale, since merging
    ``{"dist": "powerlaw", "alpha": 2}`` onto ``{"dist": "uniform"}`` would otherwise
    leave a uniform carrying a stray alpha.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            if "dist" in value or "dist" in out[key]:
                out[key] = copy.deepcopy(value)
            else:
                out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _global_local_fields() -> frozenset:
    """Field names of GlobalLocalConfig, so the allowed key set cannot drift."""
    try:
        from jaxpe.sampler import GlobalLocalConfig

        return frozenset(GlobalLocalConfig.__dataclass_fields__)
    except Exception:  # pragma: no cover - only if the sampler cannot be imported
        return frozenset(DEFAULT_CONFIG["sampler"])


def _check_unknown_keys(cfg: dict, errors: list) -> None:
    """Reject unrecognised keys. A silently ignored typo is a wrong run."""
    for section, value in cfg.items():
        if section not in DEFAULT_CONFIG:
            errors.append(
                f"unknown section {section!r}; valid sections are "
                f"{', '.join(sorted(DEFAULT_CONFIG))}"
            )
            continue
        if not isinstance(value, dict):
            errors.append(
                f"section {section!r} must be an object, got {type(value).__name__}"
            )
            continue
        # prior and injection.parameters are open: their keys are parameter names,
        # checked against PARAMETERS by _check_parameter_coverage instead.
        if section == "prior":
            continue
        if section == "sampler":
            allowed = _global_local_fields()
            hint = "a field of GlobalLocalConfig"
        elif section == "injection":
            allowed = frozenset(DEFAULT_CONFIG["injection"])
            hint = f"one of {', '.join(sorted(allowed))}"
        else:
            allowed = frozenset(DEFAULT_CONFIG[section])
            hint = f"one of {', '.join(sorted(allowed))}"
        for key in value:
            if key not in allowed:
                errors.append(f"unknown key {section}.{key!r}; expected {hint}")

        if section == "kernel":
            for kern in ("hmc", "mala"):
                for key in value.get(kern, {}):
                    if key not in DEFAULT_CONFIG["kernel"][kern]:
                        errors.append(
                            f"unknown key kernel.{kern}.{key!r}; expected one of "
                            f"{', '.join(sorted(DEFAULT_CONFIG['kernel'][kern]))}"
                        )


def _check_parameter_coverage(section: dict, label: str, errors: list, warnings: list):
    missing = [n for n in PARAMETERS if n not in section]
    if missing:
        errors.append(
            f"{label} is missing {', '.join(missing)}. Every parameter the driver "
            "samples needs an entry; to hold one constant use "
            '{"dist": "fixed", "value": ...}, which keeps its slot in the vector.'
        )
    extra = [n for n in section if n not in PARAMETERS]
    if extra:
        warnings.append(
            f"{label} defines {', '.join(extra)}, which the CLI's waveform models do "
            "not consume; they will occupy slots in the parameter vector and be "
            "ignored by the likelihood."
        )


def _positive(value, label, errors, *, allow_zero=False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number, got {value!r}")
        return None
    if not math.isfinite(v):
        errors.append(f"{label} must be finite, got {v}")
        return None
    if v < 0 or (v == 0 and not allow_zero):
        errors.append(f"{label} must be {'>= 0' if allow_zero else '> 0'}, got {v}")
        return None
    return v


def validate_config(cfg: dict) -> list:
    """Validate a fully-merged configuration.

    Returns a list of human-readable warning strings for legal-but-suspicious
    choices. Raises :class:`ConfigError` -- listing every problem found, not just the
    first -- for choices that are wrong.
    """
    errors: list = []
    warnings: list = []

    _check_unknown_keys(cfg, errors)
    if errors:
        # Downstream checks index into sections that may not exist; report the
        # structural problems first rather than raising a confusing KeyError.
        raise ConfigError(_format(errors))

    _validate_data(cfg["data"], errors, warnings)
    prior_specs = _validate_distributions(cfg, errors, warnings)
    _validate_seeds_and_kernels(cfg, errors)
    _validate_budgets(cfg, errors, warnings)
    _validate_esigma(cfg["esigma"], errors)

    if errors:
        raise ConfigError(_format(errors))
    _advise_on_physics(cfg, prior_specs, warnings)
    return warnings


def _validate_data(data: dict, errors: list, warnings: list) -> None:
    duration = _positive(data["duration"], "data.duration", errors)
    rate = _positive(data["sampling_rate"], "data.sampling_rate", errors)
    f_min = _positive(data["f_min"], "data.f_min", errors)
    f_ref = _positive(data["f_ref"], "data.f_ref", errors)
    post_trigger = _positive(
        data["post_trigger"], "data.post_trigger", errors, allow_zero=True
    )

    nyquist = rate / 2.0 if rate else None
    if f_min is not None and nyquist is not None and f_min >= nyquist:
        errors.append(
            f"data.f_min={f_min} Hz is at or above the Nyquist frequency {nyquist} Hz "
            "(sampling_rate/2); there is no band to analyse"
        )
    f_max = data["f_max"]
    if f_max is not None:
        f_max = _positive(f_max, "data.f_max", errors)
        if f_max is not None and f_min is not None and f_max <= f_min:
            errors.append(f"data.f_max={f_max} Hz must be above data.f_min={f_min} Hz")
        if f_max is not None and nyquist is not None and f_max > nyquist:
            errors.append(
                f"data.f_max={f_max} Hz exceeds the Nyquist frequency {nyquist} Hz"
            )
    if post_trigger is not None and duration is not None and post_trigger >= duration:
        errors.append(
            f"data.post_trigger={post_trigger} s must be shorter than "
            f"data.duration={duration} s"
        )
    alpha = data["tukey_alpha"]
    try:
        if not 0.0 <= float(alpha) <= 1.0:
            errors.append(f"data.tukey_alpha must lie in [0, 1], got {alpha}")
    except (TypeError, ValueError):
        errors.append(f"data.tukey_alpha must be a number, got {alpha!r}")

    if f_ref is not None and f_min is not None and f_ref < f_min:
        warnings.append(
            f"data.f_ref={f_ref} Hz lies below data.f_min={f_min} Hz, so the reference "
            "frequency at which spins and phase are defined sits outside the analysed "
            "band. This is legal but usually unintended."
        )


def _validate_distributions(cfg: dict, errors: list, warnings: list) -> dict:
    """Normalize every distribution entry in place; return the prior specs."""
    trigger = cfg["injection"]["geocent_time"]
    try:
        trigger = float(trigger)
    except (TypeError, ValueError):
        errors.append(f"injection.geocent_time must be a number, got {trigger!r}")
        trigger = 0.0

    _check_parameter_coverage(cfg["prior"], "prior", errors, warnings)

    prior_specs = {}
    for name, spec in cfg["prior"].items():
        norm = normalize_spec(spec, f"prior.{name}", errors)
        if norm is not None:
            prior_specs[name] = norm
            cfg["prior"][name] = norm

    # injection.parameters may be the literal string "prior", meaning "draw from the
    # prior itself" -- the condition a PP-plot campaign tests. Resolving the alias
    # here means the two sections cannot drift apart in a hand-edited file.
    if cfg["injection"]["parameters"] == PRIOR_ALIAS:
        cfg["injection"]["parameters"] = copy.deepcopy(prior_specs)
    elif not isinstance(cfg["injection"]["parameters"], dict):
        errors.append(
            f"injection.parameters must be an object of distributions or the string "
            f"{PRIOR_ALIAS!r} (draw from the prior), got "
            f"{cfg['injection']['parameters']!r}"
        )
        raise ConfigError(_format(errors))

    _check_parameter_coverage(
        cfg["injection"]["parameters"], "injection.parameters", errors, warnings
    )

    inj_specs = {}
    for name, spec in cfg["injection"]["parameters"].items():
        norm = normalize_spec(spec, f"injection.parameters.{name}", errors)
        if norm is not None:
            inj_specs[name] = norm
            cfg["injection"]["parameters"][name] = norm

    _check_physical_ranges(prior_specs, "prior", trigger, errors)
    _check_physical_ranges(inj_specs, "injection.parameters", trigger, errors)

    # The invariant that used to be a comment: an injected truth must not fall
    # outside the prior, or the posterior cannot recover it.
    for name, ispec in inj_specs.items():
        pspec = prior_specs.get(name)
        if pspec is None:
            continue
        ilo, ihi = spec_support(ispec, trigger)
        plo, phi = spec_support(pspec, trigger)
        if math.isinf(plo) or math.isinf(phi):
            continue  # a Gaussian prior has full support; nothing can escape it
        if math.isinf(ilo) or math.isinf(ihi):
            warnings.append(
                f"injection.parameters.{name} has unbounded support but "
                f"prior.{name} is bounded to [{plo}, {phi}]; draws outside that "
                "range will be unrecoverable."
            )
            continue
        if ilo < plo or ihi > phi:
            errors.append(
                f"injection.parameters.{name} has support [{ilo}, {ihi}], which is "
                f"not contained in prior.{name}=[{plo}, {phi}]; injected truths "
                "would fall outside the prior and the posterior could not recover them"
            )

    # The fiducial binary should sit inside the prior too, but only --fiducial runs
    # ever use it. Erroring here would block, say, a BNS prior of [1, 2] Msun purely
    # because the demo reference binary is a 30 Msun BBH, so this is a warning at
    # load time; fiducial_errors() makes it fatal at the point of use.
    for name in cfg["injection"]["fiducial"]:
        if name not in PARAMETERS:
            errors.append(
                f"unknown key injection.fiducial.{name!r}; expected one of "
                f"{', '.join(PARAMETERS)}"
            )
    outside = fiducial_errors(cfg, prior_specs, trigger)
    if outside:
        warnings.append(
            "injection.fiducial lies outside the prior ("
            + "; ".join(outside)
            + "). Harmless unless you pass --fiducial, which will refuse to run."
        )
    return prior_specs


def fiducial_errors(cfg: dict, prior_specs: dict | None = None, trigger=None) -> list:
    """Ways the fiducial binary falls outside the prior. Empty when it is usable."""
    if trigger is None:
        trigger = float(cfg["injection"]["geocent_time"])
    if prior_specs is None:
        prior_specs = cfg["prior"]
    problems = []
    for name, value in cfg["injection"]["fiducial"].items():
        pspec = prior_specs.get(name)
        if pspec is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            problems.append(f"injection.fiducial.{name} is not a number ({value!r})")
            continue
        lo, hi = spec_support(pspec, trigger)
        if not lo <= v <= hi:
            problems.append(
                f"injection.fiducial.{name}={v} lies outside prior.{name}=[{lo}, {hi}]"
            )
    return problems


def _check_physical_ranges(specs: dict, label: str, trigger: float, errors: list):
    """Enforce the meaning of the parameters, not just the shape of the numbers."""
    for name, spec in specs.items():
        lo, hi = spec_support(spec, trigger)
        if math.isinf(lo) or math.isinf(hi):
            continue
        if name == "mass_ratio" and (lo <= 0.0 or hi > 1.0):
            errors.append(
                f"{label}.mass_ratio must lie in (0, 1] with q = m2/m1 <= 1, "
                f"got [{lo}, {hi}]"
            )
        elif name in ("spin1z", "spin2z") and (lo < -1.0 or hi > 1.0):
            errors.append(
                f"{label}.{name} must lie within [-1, 1] for a dimensionless aligned "
                f"spin, got [{lo}, {hi}]"
            )
        elif name in ("chirp_mass", "luminosity_distance") and lo <= 0.0:
            errors.append(f"{label}.{name} must be strictly positive, got low={lo}")


def _validate_seeds_and_kernels(cfg: dict, errors: list) -> None:
    for name, value in cfg["seeds"].items():
        if value is not None and not isinstance(value, int):
            errors.append(f"seeds.{name} must be an integer or null, got {value!r}")
    _positive(cfg["kernel"]["hmc"]["step_size"], "kernel.hmc.step_size", errors)
    _positive(cfg["kernel"]["mala"]["step_size"], "kernel.mala.step_size", errors)
    n_leap = cfg["kernel"]["hmc"]["n_leapfrog"]
    if not isinstance(n_leap, int) or n_leap < 1:
        errors.append(f"kernel.hmc.n_leapfrog must be an integer >= 1, got {n_leap!r}")


def _validate_budgets(cfg: dict, errors: list, warnings: list) -> None:
    smp = cfg["sampler"]
    if "n_chains" in smp and (
        not isinstance(smp["n_chains"], int) or smp["n_chains"] < 1
    ):
        errors.append(
            f"sampler.n_chains must be an integer >= 1, got {smp['n_chains']!r}"
        )
    for key in ("n_prelim_loops", "n_training_loops", "n_production_loops"):
        if key in smp and (not isinstance(smp[key], int) or smp[key] < 0):
            errors.append(f"sampler.{key} must be an integer >= 0, got {smp[key]!r}")
    if smp.get("n_production_loops", 1) == 0:
        errors.append(
            "sampler.n_production_loops must be >= 1; a run with no production loops "
            "produces no samples"
        )

    ns = cfg["ns"]
    nlive = ns["nlive"]
    # GPry's set_precision accepts the "Nd" notation (N times the dimension), so a
    # string is legal here and cannot be range-checked until the dimension is known.
    if isinstance(nlive, int):
        if nlive < 1:
            errors.append(f"ns.nlive must be >= 1, got {nlive}")
        elif nlive < NLIVE_ADVISORY_MIN:
            warnings.append(
                f"ns.nlive={nlive} is a smoke-test value: nested sampling at this "
                "resolution exercises the code path but its evidence and posterior "
                f"are not measurements. Production runs use >= {NLIVE_ADVISORY_MIN} "
                "(see examples/configs/production_bbh.json)."
            )
    elif not isinstance(nlive, str):
        errors.append(f"ns.nlive must be an integer or a 'Nd' string, got {nlive!r}")
    _positive(ns["precision_criterion"], "ns.precision_criterion", errors)

    gp = cfg["gpry"]
    for key in ("n_initial", "max_total", "max_initial"):
        val = gp[key]
        if val is not None and (not isinstance(val, int) or val < 1):
            errors.append(f"gpry.{key} must be a positive integer or null, got {val!r}")
    if (
        isinstance(gp["n_initial"], int)
        and isinstance(gp["max_total"], int)
        and gp["n_initial"] > gp["max_total"]
    ):
        errors.append(
            f"gpry.n_initial={gp['n_initial']} exceeds gpry.max_total="
            f"{gp['max_total']}; active learning would never start"
        )
    _positive(gp["ref_bounds_rel"], "gpry.ref_bounds_rel", errors, allow_zero=True)
    _positive(gp["ref_bounds_abs"], "gpry.ref_bounds_abs", errors, allow_zero=True)


def _validate_esigma(esigma: dict, errors: list) -> None:
    for key in ("rad_pn_order", "mode_pn_order", "n_ode_grid", "max_ode_steps"):
        val = esigma[key]
        if not isinstance(val, int) or val < 1:
            errors.append(f"esigma.{key} must be an integer >= 1, got {val!r}")
    _positive(esigma["ode_eps"], "esigma.ode_eps", errors)
    _positive(esigma["taper_on_seconds"], "esigma.taper_on_seconds", errors, allow_zero=True)
    _positive(esigma["taper_off_seconds"], "esigma.taper_off_seconds", errors, allow_zero=True)
    modes = esigma["modes"]
    if not isinstance(modes, list) or not modes:
        errors.append(f"esigma.modes must be a non-empty list of [l, m] pairs, got {modes!r}")
    else:
        for pair in modes:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(v, int) for v in pair)
            ):
                errors.append(f"esigma.modes entries must be [l, m] integer pairs, got {pair!r}")


def _advise_on_physics(cfg: dict, prior_specs: dict, warnings: list) -> None:
    """Legal choices that are usually mistakes. Never fatal."""
    s1, s2 = prior_specs.get("spin1z"), prior_specs.get("spin2z")
    if s1 is not None and s2 is not None and s1 != s2:
        warnings.append(
            "prior.spin1z and prior.spin2z differ. The aligned-spin prior helper "
            "applies a single range to both components, so prior.spin1z is used for "
            "both; set them equal to make that explicit."
        )
    inj = cfg["injection"]["parameters"]
    differing = [
        n
        for n in PARAMETERS
        if n in inj and n in prior_specs and inj[n] != prior_specs[n]
    ]
    if differing:
        warnings.append(
            "injection.parameters differs from prior for: "
            + ", ".join(differing)
            + ". That is the right choice for a demonstration run (it keeps truths "
            "away from prior edges), but a PP-plot campaign requires the injection "
            "distribution to *be* the prior."
        )


def _format(errors: list) -> str:
    n = len(errors)
    head = f"{n} problem{'s' if n != 1 else ''} in the run configuration:"
    return "\n".join([head] + [f"  - {e}" for e in errors])


def load_config(path=None, *, validate: bool = True) -> tuple:
    """Load ``path`` and merge it over :data:`DEFAULT_CONFIG`.

    Parameters
    ----------
    path : str or Path or None
        JSON file to read. ``None`` returns the defaults unchanged.
    validate : bool
        Run :func:`validate_config` on the merged result.

    Returns
    -------
    (config, warnings) : (dict, list of str)
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"configuration file not found: {p}")
        try:
            with open(p, "r") as f:
                user = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{p} is not valid JSON: {e}") from e
        if not isinstance(user, dict):
            raise ConfigError(
                f"{p} must contain a JSON object, got {type(user).__name__}"
            )
        cfg = merge_config(cfg, _strip_comments(user))
    warnings = validate_config(cfg) if validate else []
    return cfg, warnings


def dump_default_config(path, *, annotate: bool = True) -> Path:
    """Write :data:`DEFAULT_CONFIG` to ``path`` as a starting point for editing."""
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(DEFAULT_CONFIG)
    if annotate:
        payload = {
            "_about": (
                "jaxpe run configuration. Keys beginning with '_' are comments and are "
                "ignored. Delete any section you do not want to override; omitted keys "
                "fall back to these same defaults. These values are smoke-test scale -- "
                "see examples/configs/production_bbh.json for production settings."
            ),
            "_distributions": (
                "Entries under 'prior' and 'injection.parameters' are either "
                "[low, high] (uniform) or {'dist': ...}: uniform/loguniform (low, "
                "high), powerlaw (alpha, low, high), sine/cosine (low, high), "
                "gaussian (mu, sigma), fixed (value). Add 'relative_to_trigger': true "
                "to offset by injection.geocent_time."
            ),
            **payload,
        }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return p


# ---------------------------------------------------------------------------
# Accessors -- keep the JSON-to-object mapping in one place
# ---------------------------------------------------------------------------


def prior_bounds(cfg: dict, trigger: float | None = None) -> dict:
    """Finite ``{name: (low, high)}`` box for samplers that explore a box."""
    if trigger is None:
        trigger = float(cfg["injection"]["geocent_time"])
    return {
        name: spec_bounds(cfg["prior"][name], trigger)
        for name in _ordered_names(cfg["prior"])
    }


def time_width(cfg: dict) -> float:
    """Half-width of the ``geocent_time`` prior, for reporting and legacy metadata."""
    lo, hi = spec_support(cfg["prior"]["geocent_time"], 0.0)
    if math.isinf(lo) or math.isinf(hi):
        return math.inf
    return 0.5 * (hi - lo)


def fiducial_injection(cfg: dict) -> dict:
    """The fixed reference binary, with ``geocent_time`` stamped in.

    ``geocent_time`` is held once, in ``injection.geocent_time``, so drawn and
    fiducial injections share a trigger by construction. An explicit
    ``injection.fiducial.geocent_time`` still wins.
    """
    params = {}
    for name in _ordered_names(cfg["injection"]["fiducial"]):
        params[name] = float(cfg["injection"]["fiducial"][name])
    params.setdefault("geocent_time", float(cfg["injection"]["geocent_time"]))
    return params


def global_local_kwargs(cfg: dict) -> dict:
    """The ``sampler`` section, filtered to fields GlobalLocalConfig accepts."""
    allowed = _global_local_fields()
    return {k: v for k, v in cfg["sampler"].items() if k in allowed}
