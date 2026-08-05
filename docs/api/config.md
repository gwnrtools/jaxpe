---
title: config
parent: API Reference
layout: default
nav_order: 9
---

# `jaxpe.config`
{: .no_toc }

1. TOC
{:toc}

---

## What this module is for

`jaxpe.config` holds the declarative description of a parameter-estimation run: the data
conditioning, the prior distributions, the population injected truths are drawn from, the
kernel and sampler settings, and every RNG seed. The
[CLI]({{ site.baseurl }}/docs/cli_tutorial.html) is its main consumer, but the module is
usable on its own — it is the fastest way to build a `JointPrior` from a specification you
can serialise.

The design goal is **auditability**. A run defined by a JSON file is one you can commit,
diff against another run, attach to a paper, and replay months later. A run defined by
edited source constants is none of those things. Everything the module does follows from
that: unrecognised keys are errors rather than being ignored, physical invariants are
checked rather than assumed, and `run-pe` writes the fully-resolved configuration into its
output directory.

## The distribution algebra

A single mini-language describes both the prior and the injected population, so the two are
built by identical code. Each entry is either the shorthand `[low, high]`, meaning uniform,
or an object with a `dist` key:

| `dist` | parameters | density on its support |
|---|---|---|
| `uniform` | `low`, `high` | $$p(x) = (b-a)^{-1}$$ |
| `loguniform` | `low`, `high`, both $$>0$$ | $$p(x) \propto x^{-1}$$ |
| `powerlaw` | `alpha`, `low` $$>0$$, `high` | $$p(x) \propto x^{\alpha}$$ |
| `sine` | `low`, `high` in $$[0, \pi]$$ | $$p(x) \propto \sin x$$ |
| `cosine` | `low`, `high` in $$[-\pi/2, \pi/2]$$ | $$p(x) \propto \cos x$$ |
| `gaussian` | `mu`, `sigma` $$>0$$ | $$\mathcal{N}(\mu, \sigma^{2})$$ |
| `fixed` | `value` | $$\delta(x - v)$$ |

Each maps onto the corresponding class in
[`jaxpe.core.priors`]({{ site.baseurl }}/docs/api/core.html), so a configured prior is an
ordinary `JointPrior`: it knows its normalised log-density, how to sample itself, and the
bijection onto the unconstrained space the global-local sampler works in.

Two consequences are worth stating explicitly.

**`powerlaw` with $$\alpha = -1$$ is refused.** The normalisation used is

$$Z = \int_a^b x^{\alpha}\,\mathrm{d}x = \frac{b^{\alpha+1} - a^{\alpha+1}}{\alpha+1},$$

which is singular at $$\alpha = -1$$. That case is exactly `loguniform`, and the error
message says so rather than letting a division by zero produce a silent `inf`.

**`fixed` keeps its slot in the parameter vector.** Its bijection is the identity and its
log-density is zero, so pinning a parameter does not renumber the others or change the
column order of saved samples. This is how you hold spins at zero without the arrays
underneath shifting shape.

### Trigger-relative distributions

Any entry may carry `"relative_to_trigger": true`, which offsets whichever of `low`,
`high`, `mu`, `value` are present by `injection.geocent_time`. The coalescence-time prior
uses it so that a configuration file is portable across epochs:

```json
"geocent_time": { "dist": "uniform", "low": -0.1, "high": 0.1, "relative_to_trigger": true }
```

At analysis time the offset is applied against *that injection's* `geocent_time`, not the
configuration's, so a set generated at one epoch still analyses correctly.

### Support versus bounds

Two accessors answer two different questions, and conflating them is a real bug:

- `spec_support(spec, trigger)` — the **exact** support. A Gaussian returns
  $$(-\infty, \infty)$$; a `fixed` returns a single point. This is what containment checks
  must use.
- `spec_bounds(spec, trigger)` — a **finite** box, for the samplers that structurally
  require one. Nested sampling and GPry explore a box rather than a density, so a Gaussian
  is truncated at $$\pm 5\sigma$$ (`GAUSSIAN_BOUND_SIGMAS`) and a `fixed` parameter is
  widened to a degenerate but nonzero interval.

## Validation

`validate_config` separates two kinds of problem. **Errors** raise `ConfigError` and list
every problem found, not just the first — fixing a configuration one message per attempt is
needless friction. **Warnings** are returned as strings for the caller to print: legal
choices that are usually mistakes.

Errors cover structure (unknown sections and keys, malformed distributions) and physics:

- $$f_{\rm min} <$$ Nyquist, and $$f_{\rm min} < f_{\rm max} \le$$ Nyquist when set;
- $$q = m_2/m_1 \in (0, 1]$$; dimensionless spins in $$[-1, 1]$$; positive masses and
  distances;
- `sine` and `cosine` supports inside $$[0,\pi]$$ and $$[-\pi/2, \pi/2]$$;
- `post_trigger` $$<$$ `duration`; `tukey_alpha` in $$[0,1]$$;
- **every injection distribution contained in the corresponding prior**.

That last check is the scientifically load-bearing one. An injected truth outside the prior
cannot be recovered, so the posterior rails against a boundary and the run is wasted. It
was previously a prose comment asserting the boxes were nested, which nothing enforced.

Warnings cover a reference frequency outside the analysed band, a nested-sampling `nlive`
below `NLIVE_ADVISORY_MIN` (a smoke-test resolution whose evidence is not a measurement),
an injection population that differs from the prior, and mismatched `spin1z`/`spin2z`
priors.

### The fiducial binary is checked where it is used

`injection.fiducial` is the one containment check deliberately *not* fatal at load time.
Only `--fiducial` runs read it, so erroring would block a perfectly good narrow prior — a
BNS configuration over $$\mathcal{M} \in (1, 2)\,M_\odot$$, say — on account of a 30
$$M_\odot$$ demo binary it never touches. Instead `validate_config` warns, and
`fiducial_errors(cfg)` returns the specific problems so a caller can make them fatal at the
point of use:

```console
$ jaxpe generate-injections --config bns.json --fiducial --outdir inj
ERROR: --fiducial cannot be used with this configuration:
  - injection.fiducial.chirp_mass=30.0 lies outside prior.chirp_mass=[1.0, 2.0]
Edit injection.fiducial to sit inside the prior, or drop --fiducial to draw from
injection.parameters instead.
```

### Typos are errors

```python
>>> load_config("run.json")   # {"data": {"sample_rate": 4096.0}}
ConfigError: 1 problem in the run configuration:
  - unknown key data.'sample_rate'; expected one of duration, f_max, f_min, f_ref,
    post_trigger, sampling_rate, tukey_alpha
```

A configuration key that is silently dropped leaves the run at a value the user believes
they changed, and nothing downstream records the discrepancy. The `sampler` section's
allowed keys are read from `GlobalLocalConfig.__dataclass_fields__` rather than duplicated,
so they cannot drift from the dataclass.

Keys beginning with `_` are stripped as comments at every level, since JSON has no comment
syntax.

## Merging

`load_config` deep-merges the file over `DEFAULT_CONFIG`, so a partial file changes only
what it names. One deliberate exception: a **distribution entry replaces wholesale** rather
than merging key-by-key. Merging `{"dist": "powerlaw", "alpha": 2}` onto a stored
`{"dist": "uniform", "low": ..., "high": ...}` would otherwise leave a uniform carrying a
stray `alpha`, which the key check would then reject for reasons the user never wrote.

A corollary worth knowing: **omitting a parameter does not remove it.** The prior is always
complete, because anything absent is inherited from the defaults. To hold a parameter
constant, pin it with `{"dist": "fixed", "value": ...}`.

## Using it from Python

```python
import jax
from jaxpe.config import load_config, build_prior, sample_parameters

cfg, warnings = load_config("examples/configs/production_bbh.json")
for w in warnings:
    print("WARNING:", w)

prior = build_prior(cfg)                       # a jaxpe.core.priors.JointPrior
truths = sample_parameters(cfg, jax.random.PRNGKey(0))   # one injection, as a dict
```

| function | returns |
|---|---|
| `load_config(path=None, validate=True)` | `(config, warnings)`; `None` gives the defaults |
| `validate_config(cfg)` | warnings; raises `ConfigError` on any error |
| `merge_config(base, override)` | recursive merge, distribution entries replaced |
| `build_prior(cfg, trigger=None)` | `JointPrior` over `PARAMETERS`, in that order |
| `build_distribution(spec, trigger=0.0)` | a single `Prior` object |
| `sample_parameters(cfg, key, trigger=None)` | one injection's truths as a `dict` |
| `prior_bounds(cfg, trigger=None)` | finite `{name: (low, high)}` for box samplers |
| `spec_support` / `spec_bounds` | exact support / finite box for one spec |
| `fiducial_injection(cfg)` | the fixed reference binary, `geocent_time` stamped in |
| `global_local_kwargs(cfg)` | the `sampler` section, filtered to dataclass fields |
| `time_width(cfg)` | half-width of the `geocent_time` prior |
| `dump_default_config(path)` | writes an annotated starting-point file |

`PARAMETERS` fixes the column order of every saved sample array. It matches the ordering
`jaxpe.gw.bbh_priors` has always produced, and the test suite asserts that the default
configuration reproduces that prior's log-density exactly — so results predating this
module remain directly comparable.

## Drawing injections from the prior

Setting `injection.parameters` to the string `"prior"` binds the injected population to the
prior itself:

```json
"injection": { "parameters": "prior" }
```

This is the condition a probability-probability (PP) test requires. If injections come from
any narrower distribution — as the defaults do, to keep truths off prior edges — the
resulting PP curve is biased and tests nothing. Writing the alias rather than duplicating
eleven entries also means the two cannot drift apart when the prior is later edited.
`examples/configs/pp_campaign.json` is built this way.
