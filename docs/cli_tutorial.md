---
title: Universal CLI Tutorial
layout: default
nav_order: 6
---

# The `jaxpe` command-line interface
{: .no_toc }

1. TOC
{:toc}

---

## What this CLI is, and what it is not

`jaxpe` ships a CLI that takes you from a synthetic injection to a corner plot without
writing Python:

```
jaxpe generate-injections   →   jaxpe run-pe   →   jaxpe process-samples
```

plus `jaxpe write-config`, which emits a fully-populated configuration file to edit.

Run with no arguments beyond the essentials, it is a fast end-to-end smoke test of an
installation. Point it at a **run configuration file** and it is a production PE driver:
every physical and numerical choice — segment duration, sampling rate, $$f_{\rm min}$$,
the prior *distributions*, the population injected truths are drawn from, kernel step
sizes, sampler budgets and every RNG seed — is read from that file rather than compiled
into `jaxpe/cli.py`. See [The run configuration](#the-run-configuration).

That split matters for reproducibility. A run is defined by an artifact you can commit,
diff, attach to a paper and replay, not by a source edit that leaves no trace. `run-pe`
writes the fully-resolved configuration it actually used into `run_config.json`, so a
result directory always records the physics that produced it.

The drivers in [`bin/`]({{ site.baseurl }}/docs/scripts_and_examples.html) remain the
reference for the benchmarks on this site and for features the CLI does not expose —
relative binning, convergence gating, `ESIGMA`/`NRSur` waveforms. `bin/run_bns_ce_pe.py`
is the one every benchmark here runs.

> **Everything below was executed against the current `jaxpe/cli.py`.** Flag lists,
> defaults, output shapes and error messages are transcribed from real runs, not from the
> source by eye.

## Invoking it

The package installs a console script:

```bash
jaxpe --help
```

If that is not on your `PATH` — common with `pip install -e .` when the environment's
scripts directory has not been rehashed — the module form is always available and exactly
equivalent:

```bash
python -m jaxpe.cli --help
```

Every example below works with either. To force CPU (useful on a small or busy GPU):

```bash
JAX_PLATFORMS=cpu python -m jaxpe.cli run-pe ...
```

---

## A complete worked example

Three commands, copy-pasteable, starting from nothing:

```bash
# 1. write an injection description
python -m jaxpe.cli generate-injections --n-injections 1 --outdir demo/inj

# 2. sample it  (MALA, frequency domain, H1+L1, zero noise)
JAX_PLATFORMS=cpu python -m jaxpe.cli run-pe \
    --injection demo/inj/inj_0.json \
    --sampler mala --domain fd --likelihood full \
    --network H1,L1 --noise zero \
    --n-chains 32 --n-prelim-loops 1 --n-training-loops 3 --n-production-loops 10 \
    --outdir demo/pe

# 3. thin to independent samples, map to physical units, plot
python -m jaxpe.cli process-samples demo/pe/raw_samples.npz
```

That exact sequence runs to completion on CPU. Step 1 draws a binary and step 2 echoes the
network/noise/PSD it resolved from the injection:

```console
Saved injection 0 to demo/inj/inj_0.json (Mc=38.22, q=0.61, D=413 Mpc)
Network H1,L1, noise zero, psd aligo
```

Step 2 produces a `samples` array of shape `(300, 32, 11)` — 300 retained steps × 32 chains
× 11 parameters — and step 3 reports:

```console
Max autocorrelation time (tau): 42
Discarding 126 steps as burn-in and thinning by 42.
Extracted 160 independent samples (from 9600 raw).
Saved demo/pe/posterior_samples.npy (160, 11)
Saved demo/pe/corner_thinned.png
```

After step 2, `demo/pe/` holds `raw_samples.npz`, `injection.json` and `run_config.json`.
After step 3 it also holds `posterior_samples.npy` and `corner_thinned.png`.

Note the ratio: 9600 raw draws collapse to 160 independent ones, because the measured
autocorrelation time is ~42 steps. That is the number to reason about when deciding how
long to run — raw sample count is not sample size.

Keep the output directory intact between steps 2 and 3 — `process-samples` finds the
injection by looking for `injection.json` **next to** the `.npz` file, and skips the file
with a warning if it is missing.

---

## The run configuration

Everything physical the CLI does is read from a JSON configuration. Start from a
fully-populated one:

```bash
python -m jaxpe.cli write-config my_run.json
```

Edit it, then pass it to both commands that consume physics:

```bash
python -m jaxpe.cli generate-injections --config my_run.json --n-injections 100 --outdir inj
python -m jaxpe.cli run-pe --config my_run.json --injection inj/inj_0.json --outdir pe/0
```

You rarely need `--config` twice. `generate-injections` writes the resolved configuration
to `config.json` beside the injection set, and `run-pe` picks that up automatically, so
the prior a set was generated under is the prior it is analysed under:

```console
Loaded configuration from inj/config.json
```

**Precedence**, highest first: a command-line flag → `--config FILE` → `config.json` next
to the injection → the built-in defaults. Omitted keys fall back to the defaults, so a
file containing only

```json
{ "data": { "duration": 128.0, "sampling_rate": 4096.0, "f_min": 20.0 } }
```

changes the conditioning and leaves every prior, seed and budget alone. Keys beginning
with `_` are comments and are ignored — JSON has none of its own.

### Sections

| section | controls |
|---|---|
| `data` | `duration`, `sampling_rate`, `f_min`, `f_max`, `f_ref`, `post_trigger`, `tukey_alpha` |
| `prior` | the distribution sampled for each of the 11 parameters |
| `injection` | `geocent_time` (the trigger), `parameters` (the injected population), `fiducial` (the fixed reference binary) |
| `seeds` | `injection`, `noise`, `sampler`, `ns`, `gpry` — separately, so you can vary one and hold the rest |
| `kernel` | `hmc.step_size`, `hmc.n_leapfrog`, `mala.step_size` |
| `sampler` | any field of `GlobalLocalConfig` — chains, loops, flow architecture, adaptation |
| `ns` | `nlive`, `num_repeats`, `precision_criterion`, `nprior`, `max_ncalls`, `verbosity` |
| `gpry` | `n_initial`, `max_total`, `max_initial`, `acquisition`, reference-bounds width |

### Distributions

`prior` and `injection.parameters` speak the same language, so the box you sample and the
population you inject are specified the same way and built by the same code. Each entry is
either the shorthand `[low, high]` (meaning uniform) or an object with a `dist` key:

| `dist` | parameters | density |
|---|---|---|
| `uniform` | `low`, `high` | constant |
| `loguniform` | `low`, `high` (both $$> 0$$) | $$p(x) \propto 1/x$$ |
| `powerlaw` | `alpha`, `low`, `high` | $$p(x) \propto x^{\alpha}$$ |
| `sine` | `low`, `high` (default $$0, \pi$$) | $$p(x) \propto \sin x$$ |
| `cosine` | `low`, `high` (default $$\pm\pi/2$$) | $$p(x) \propto \cos x$$ |
| `gaussian` | `mu`, `sigma` | $$\mathcal{N}(\mu, \sigma^2)$$ |
| `fixed` | `value` | pins the parameter, keeping its slot in the vector |

Any entry may carry `"relative_to_trigger": true`, which offsets `low`, `high`, `mu` or
`value` by `injection.geocent_time`. That is how the coalescence-time prior says "the
trigger $$\pm 0.1$$ s" without hardcoding an epoch:

```json
"geocent_time": { "dist": "uniform", "low": -0.1, "high": 0.1, "relative_to_trigger": true }
```

Isotropy is expressed here rather than assumed in the code. An isotropically oriented,
Euclidean-volumetric population is

```json
"inclination":         { "dist": "sine" },
"dec":                 { "dist": "cosine" },
"luminosity_distance": { "dist": "powerlaw", "alpha": 2.0, "low": 100.0, "high": 2000.0 }
```

and changing any of those three changes the population, with no code edit. To hold a
parameter constant, pin it — omitting it does **not** remove it, because the file is merged
over the defaults:

```json
"spin1z": { "dist": "fixed", "value": 0.0 }
```

### Injections and the prior

By default `injection.parameters` is **narrower** than `prior` in the intrinsic parameters
and distance, so no injected truth lands on a prior edge and rails the posterior. A
PP-plot campaign needs the opposite — there the injected population must *be* the prior, or
the test is meaningless. Say so directly:

```json
"injection": { "parameters": "prior" }
```

The string `"prior"` binds the two together, so they cannot drift apart when the prior is
later edited. When they differ, the loader says which parameters differ and why it matters:

```console
WARNING: injection.parameters differs from prior for: chirp_mass, mass_ratio, spin1z,
spin2z, luminosity_distance, geocent_time. That is the right choice for a demonstration
run (it keeps truths away from prior edges), but a PP-plot campaign requires the injection
distribution to *be* the prior.
```

### The configuration is checked, not merely parsed

A silently ignored typo is a wrong run, so unrecognised keys are rejected and every problem
is reported at once rather than one per attempt:

```console
$ python -m jaxpe.cli run-pe --config broken.json ...
ERROR: 3 problems in the run configuration:
  - unknown key data.'sample_rate'; expected one of duration, f_max, f_min, f_ref,
    post_trigger, sampling_rate, tukey_alpha
  - data.tukey_alpha must lie in [0, 1], got 5.0
  - prior.mass_ratio is empty or inverted: low=2.0 must be < high=0.5
```

Beyond shape, the checks enforce the physics:

- $$f_{\rm min}$$ below Nyquist, and $$f_{\rm min} < f_{\rm max} \le$$ Nyquist when
  `f_max` is set;
- $$q = m_2/m_1 \in (0, 1]$$, dimensionless spins in $$[-1, 1]$$, positive masses and
  distances;
- `sine` support inside $$[0, \pi]$$ and `cosine` inside $$[-\pi/2, \pi/2]$$;
- `post_trigger` shorter than `duration`, `tukey_alpha` in $$[0, 1]$$;
- **every injection distribution contained in the corresponding prior**.

That last one used to be a comment in `cli.py` asserting the boxes were nested. It is now
enforced:

```console
ERROR: 1 problem in the run configuration:
  - injection.parameters.chirp_mass has support [5.0, 60.0], which is not contained in
    prior.chirp_mass=[10.0, 50.0]; injected truths would fall outside the prior and the
    posterior could not recover them
```

`injection.fiducial` is the one exception: it only matters to `--fiducial` runs, so a
prior that excludes it is a warning rather than an error. Narrowing to a BNS prior does
**not** force you to move the demo reference binary — but asking for it is refused:

```console
$ jaxpe generate-injections --config bns.json --fiducial --outdir inj
ERROR: --fiducial cannot be used with this configuration:
  - injection.fiducial.chirp_mass=30.0 lies outside prior.chirp_mass=[1.0, 2.0]
Edit injection.fiducial to sit inside the prior, or drop --fiducial to draw from
injection.parameters instead.
```

### Shipped examples

| file | purpose |
|---|---|
| [`examples/configs/production_bbh.json`](https://github.com/prayush/jaxpe/blob/master/examples/configs/production_bbh.json) | 8 s at 4096 Hz from 20 Hz, stellar-mass BBH prior, `nlive=1000`, 256 chains |
| [`examples/configs/pp_campaign.json`](https://github.com/prayush/jaxpe/blob/master/examples/configs/pp_campaign.json) | the same, with `"parameters": "prior"` for a PP test |

Both are validated by the test suite, so they cannot rot into invalid files. Their settings
are a defensible starting point, **not a benchmarked optimum**: `duration × sampling_rate`
sets the number of frequency bins and therefore the per-likelihood cost, and
`n_chains × loops` sets how many evaluations you pay for. Re-tune both against your own
waveform, network and hardware.

---

## `generate-injections`

```bash
python -m jaxpe.cli generate-injections --outdir DIR [--config FILE] [--n-injections N]
                                        [--seed S] [--fiducial] [--network H1,L1]
                                        [--noise {zero,gaussian}] [--psd aligo|NAME|PATH]
```

Draws `N` **distinct** injections and writes them as `inj_0.json … inj_{N-1}.json` into
`--outdir` (created if needed), plus the resolved `config.json`. `--outdir` is required;
`--n-injections` defaults to 1.

| flag | default | effect |
|---|---|---|
| `--config` | built-in defaults | run configuration; sets the population drawn from |
| `--n-injections` | 1 | how many injections to draw |
| `--seed` | `seeds.injection` (42) | RNG seed — the same seed reproduces the same set exactly |
| `--fiducial` | off | emit `injection.fiducial` instead of drawing |
| `--network` | `H1,L1` | recorded in the file; `run-pe` uses it unless overridden |
| `--noise` | `zero` | recorded in the file; `run-pe` uses it unless overridden |
| `--psd` | `aligo` | recorded in the file; a name or a two-column ASCII PSD path |

Truths are drawn from `injection.parameters` in the configuration. With the defaults that
is uniform in chirp mass $$(15, 45)\,M_\odot$$, mass ratio $$(0.3, 1.0)$$, aligned spins
$$(-0.5, 0.5)$$ and distance $$(300, 1500)$$ Mpc, with isotropic orientation and sky
position ($$p \propto \sin\iota$$, $$p \propto \cos\delta$$) and coalescence time fixed at
the trigger. Those boxes sit **strictly inside** the recovery priors, so no injected truth
lands on a prior edge — an invariant the configuration loader now enforces rather than
merely documents.

The draws use the same distribution objects the prior is built from, so setting
`"parameters": "prior"` draws from exactly the prior. See
[Injections and the prior](#injections-and-the-prior).

```console
$ python -m jaxpe.cli generate-injections --n-injections 3 --outdir demo/inj
Saved injection 0 to demo/inj/inj_0.json (Mc=38.22, q=0.61, D=413 Mpc)
Saved injection 1 to demo/inj/inj_1.json (Mc=26.12, q=0.95, D=832 Mpc)
Saved injection 2 to demo/inj/inj_2.json (Mc=37.74, q=0.55, D=1234 Mpc)
```

Each file holds the physical parameters plus a `metadata` block recording the network,
noise and PSD it was generated for:

```json
"metadata": {"network": "H1,L1", "noise": "zero", "psd": "aligo", "seed": 42, "index": 0}
```

`run-pe` reads that block and uses it as its defaults, so the network you generate for is
the network you analyse with unless you say otherwise. The block is stripped before the
parameters reach the waveform model.

### The fiducial reference binary

For regression checks and cross-run comparisons you often want the *same* binary every
time rather than a draw. `--fiducial` emits the historical reference system — a
GW150914-like BBH, face-on and optimally oriented, at the GW150914 epoch:

```console
$ python -m jaxpe.cli generate-injections --fiducial --outdir demo/ref
Saved fiducial injection 0 to demo/ref/inj_0.json (Mc=30.00, q=0.80, D=700 Mpc)
```

```json
{
  "chirp_mass": 30.0, "mass_ratio": 0.8,
  "spin1z": 0.0, "spin2z": 0.0,
  "luminosity_distance": 700.0,
  "geocent_time": 1126259462.4,
  "phase": 0.0, "inclination": 0.0,
  "ra": 0.0, "dec": 0.0, "psi": 0.0
}
```

The file records `"fiducial": true` and `"seed": null` in its metadata, so a directory of
results is self-describing about which binary produced it.

Because there is exactly one fiducial binary, combining it with `--n-injections > 1` is
refused rather than silently writing N identical files:

```console
ValueError: --fiducial produces a single fixed binary, but --n-injections is 3; that
would write 3 identical files. Use --fiducial with --n-injections 1, or drop --fiducial
to draw a distinct set.
```

`--psd` takes three forms: `aligo` (the built-in analytic ZDHP fit), a **named detector
curve** resolved through LALSimulation, or a path to a two-column ASCII file.

| name | curve |
|---|---|
| `aligo` | analytic aLIGO Zero-Detuning High-Power fit (no LALSimulation needed) |
| `CE`, `CE-wideband`, `CE-pessimistic` | Cosmic Explorer P1600143 variants |
| `ET` | Einstein Telescope P1600143 |
| `aplus`, `aligo-design`, `advirgo-O4` | A+, aLIGO design, AdVirgo O4 |

```bash
python -m jaxpe.cli generate-injections --fiducial --psd CE --outdir demo/inj
```

Names are checked before the filesystem, so a detector label always means the detector.
Anything unrecognised is rejected rather than silently ignored, and the message lists what
is available:

```console
ValueError: --psd 'Nonsense' is not 'aligo', a known detector curve, or an existing file.
Known curves: CE, CE-pessimistic, CE-wideband, ET, advirgo-O4, aligo-design, aplus.
Anything else must be a two-column ASCII file path.
```

**To analyse a specific source rather than a draw, edit the JSON.** `run-pe` reads it back
verbatim:

```bash
python -m jaxpe.cli generate-injections --n-injections 1 --outdir demo/inj
python - <<'PY'
import json, pathlib
p = pathlib.Path("demo/inj/inj_0.json")
d = json.loads(p.read_text())
d.update(chirp_mass=20.0, mass_ratio=0.6, spin1z=0.3, inclination=0.9)
p.write_text(json.dumps(d, indent=2))
PY
```

Keep edited values **inside the prior**, or the injected truth will sit outside it and the
posterior will rail against a boundary. Editing the JSON is the right tool for a one-off
source; to change the *population* every draw comes from, edit `injection.parameters` in
the configuration instead, where the containment is checked for you.

---

## `run-pe`

```bash
python -m jaxpe.cli run-pe --injection FILE --outdir DIR [--config FILE]
                           [--sampler {hmc,mala,ns,gpry}]
                           [--likelihood {full,marginalized_phase_distance}]
                           [--domain {td,fd}] [--network ...] [--noise {zero,gaussian}]
                           [--psd aligo|NAME|PATH] [--n-chains N] [--n-prelim-loops N]
                           [--n-training-loops N] [--n-production-loops N]
```

Flags override the configuration file; the configuration overrides the built-in defaults.
Where a flag is left off, the value in the "default" column below is the configuration key
it falls back to.

| flag | default | effect |
|---|---|---|
| `--injection` | *required* | path to the injection JSON |
| `--outdir` | *required* | output directory, created if absent |
| `--config` | sibling `config.json`, else built-in | run configuration |
| `--sampler` | `hmc` | `hmc`, `mala`, `ns` (BlackJAX nested sampling), `gpry` (GP surrogate) |
| `--likelihood` | `full` | `full`, or `marginalized_phase_distance` (analytic $$\phi_c$$ + $$D_L$$ marginal) |
| `--domain` | `fd` | `fd` → `IMRPhenomD`, `td` → `IMRPhenomT` (both at `data.f_ref`) |
| `--network` | *from injection* | comma-separated detectors; overrides the recorded value |
| `--noise` | *from injection* | `zero` or `gaussian` (seeded by `seeds.noise`) |
| `--psd` | *from injection* | a name or a path; overrides the recorded value |
| `--n-chains` | `sampler.n_chains` (100) | `hmc`/`mala` only |
| `--n-prelim-loops` | `sampler.n_prelim_loops` (1) | discarded warmup loops; `hmc`/`mala` only |
| `--n-training-loops` | `sampler.n_training_loops` (5) | local steps → flow fit → global block |
| `--n-production-loops` | `sampler.n_production_loops` (50) | flow frozen; `hmc`/`mala` only |
| `--gpry-acquisition` | `gpry.acquisition` (GPry's NORA) | `BatchOptimizer` swaps nested sampling for multi-start L-BFGS |
| `--gpry-n-initial` | `gpry.n_initial` (GPry's $$3d$$) | truth evaluations before active learning starts |
| `--gpry-max-total` | `gpry.max_total` (500) | total truth-evaluation budget |
| `--gpry-max-initial` | `gpry.max_initial` (200) | draws attempted while collecting the initial finite points |

Settings with no flag — step sizes, flow architecture, `nlive`, seeds, the data
conditioning, the priors — are reachable only through `--config`. The four chain/loop flags
exist because they are the ones you sweep interactively; everything else belongs in a file
you keep.

`--network`, `--noise` and `--psd` default to whatever `generate-injections` recorded in
the injection's `metadata` block, falling back to `H1,L1` / `zero` / `aligo` for files that
have none. The run echoes what it resolved:

```console
Loaded injection params: {'chirp_mass': 38.2186..., ...}
Network H1,L1,V1, noise gaussian, psd aligo
```

The four chain/loop flags are silently ignored by `ns` and `gpry`, which have their own
internal termination criteria.

### Choosing a sampler

| you want | use | why |
|---|---|---|
| the default, gradient-based MCMC | `--sampler hmc` | $$L = 10$$ leapfrog steps per proposal; travels furthest per gradient, best at prior boundaries |
| a cheaper single-gradient step | `--sampler mala` | one gradient per proposal; less to tune, mixes more slowly |
| an evidence estimate, no gradients | `--sampler ns` | BlackJAX nested sampling; returns per-sample `weights` and computes $$\ln Z$$ |
| an expensive/non-differentiable likelihood | `--sampler gpry` | GP surrogate + active learning; see [`jaxpe.surrogate`]({{ site.baseurl }}/docs/api/surrogate.html) |

> **`gpry` runs are not reproducible, and the seed does not make them so.** GPry warns
> `Seeded runs are not supported for UltraNest` — the `seed` the CLI passes is honoured by
> the GP and the initial draw, but *not* by the nested sampler underneath NORA acquisition
> and the final surrogate sampling. Measured on this page's fiducial injection at the
> default budget, four identical invocations gave **two clean runs (24–26 s) and two
> failures inside UltraNest** (one in `mlfriends`, one "not enough live points to compute
> variance"), with one run taking 4.5 minutes. Budget generously, expect to re-run, and do
> not treat a single `gpry` timing or outcome as a measurement.
>
> `--gpry-n-initial` defaults to GPry's own $$3d$$ (12 for the 4-parameter marginalized
> likelihood, 33 for the full 11-parameter one). Setting it below that under-trains the
> SVM infinities classifier and produces failures that look like properties of the source
> but are properties of the budget.

### Two combinations are rejected

`marginalized_phase_distance` is a **host-side** likelihood: it evaluates in NumPy and is
not traceable, so it cannot be paired with a gradient sampler, and it is derived for
dominant-mode frequency-domain models only. Both constraints are enforced with real
errors:

```console
$ ... --sampler hmc --domain fd --likelihood marginalized_phase_distance
ValueError: marginalized_phase_distance is host-side and incompatible with HMC/MALA.

$ ... --sampler ns --domain td --likelihood marginalized_phase_distance
ValueError: marginalized_phase_distance is only supported for FD domain.
```

So the analytic marginal pairs with `ns` or `gpry`, in `fd`:

```bash
python -m jaxpe.cli run-pe \
    --injection demo/inj/inj_0.json \
    --sampler ns --domain fd --likelihood marginalized_phase_distance \
    --network H1,L1 --outdir demo/pe_ns
```

Marginalizing $$\phi_c$$ and $$D_L$$ analytically drops the sampled space from 11
parameters to 4 — `chirp_mass`, `mass_ratio`, `spin1z`, `spin2z` — which is why this path
is dramatically cheaper. The extrinsic angles are frozen at their injected values rather
than marginalized.

### More examples

```bash
# time-domain HMC (IMRPhenomT), on CPU
JAX_PLATFORMS=cpu python -m jaxpe.cli run-pe \
    --injection demo/inj/inj_0.json --sampler hmc --domain td --likelihood full \
    --n-chains 16 --n-prelim-loops 1 --n-training-loops 3 --n-production-loops 10 \
    --outdir demo/pe_td

# three-detector network, in simulated Gaussian noise
python -m jaxpe.cli run-pe \
    --injection demo/inj/inj_0.json --sampler mala --domain fd \
    --network H1,L1,V1 --noise gaussian --outdir demo/pe_noisy

# GP surrogate over the analytic marginal
python -m jaxpe.cli run-pe \
    --injection demo/inj/inj_0.json --sampler gpry --domain fd \
    --likelihood marginalized_phase_distance --outdir demo/pe_gpry
```

### What it writes

Three files: `raw_samples.npz`, a copy of the injection as `injection.json`, and
`run_config.json`.

| sampler | `samples` | `log_prob` | `weights` | space |
|---|---|---|---|---|
| `hmc`, `mala` | `(n_samples, n_chains, n_dim)` | `(n_samples, n_chains)` | — | unconstrained |
| `ns`, `gpry` | `(1, n_samples, n_dim)` | `(1, n_samples)` | `(n_samples,)` | **physical** |

The leading `1` for `ns`/`gpry` is a dummy chain axis so downstream code can treat both
layouts uniformly. Two differences matter:

- **The two families live in different spaces.** The global-local sampler works in the
  unconstrained space and its draws must be pushed back through the prior bijection;
  nested sampling and GPry explore the physical prior box directly and must **not** be.
- **`ns`/`gpry` samples are weighted** — ignoring `weights` when computing summary
  statistics from those runs gives the wrong answer.

`run_config.json` records exactly this (`sample_space`, `weighted`, `param_names`) along
with the domain, network and PSD, so `process-samples` can rebuild the same problem
instead of guessing.

It also embeds the **fully-resolved run configuration** under a `config` key, with
command-line overrides folded in — so it describes the run that happened, not the file it
started from. If you pass `--n-chains 2`, `config.sampler.n_chains` reads 2. That makes a
result directory self-describing and re-runnable:

```bash
python -c "import json; print(json.dumps(json.load(open('pe/run_config.json'))['config'], indent=2))" > rerun.json
python -m jaxpe.cli run-pe --config rerun.json --injection pe/injection.json --outdir pe_again
```

---

## `process-samples`

```bash
python -m jaxpe.cli process-samples FILE [FILE ...]
```

Takes one or more `raw_samples.npz` paths and nothing else — there are no options beyond
`-h`. It reads the sibling `run_config.json` to rebuild the problem the sampler actually
saw, then branches on how the samples were produced:

- **Chain-based runs (`hmc`, `mala`).** Estimates the integrated autocorrelation time
  $$\tau$$, **discards burn-in and thins by $$\tau$$ automatically** (both derived from the
  chains, not configured), and maps the survivors from the unconstrained space back to
  physical parameters.
- **Weighted runs (`ns`, `gpry`).** The draws are already physical and carry no chain
  structure to autocorrelate, so burn-in, thinning and the bijection are all skipped and
  the `weights` array is carried into the corner plot.

It writes, next to each input:

- `posterior_samples.npy` — physical samples
- `corner_thinned.png` — corner plot with the injected truth marked

Batch use, since it accepts many files:

```bash
python -m jaxpe.cli process-samples demo/*/raw_samples.npz
```

For output directories produced before `run_config.json` existed, it falls back to the
historical assumptions (`fd`, `H1,L1,V1`, 4 s at 1024 Hz) and says so:

```console
WARNING: demo/pe/run_config.json not found; assuming an fd / H1,L1,V1 / 4.0 s run.
```

Re-run `run-pe` to regenerate the configuration if you see that on a `td` or
non-default-network run.

---

## `write-config`

```bash
python -m jaxpe.cli write-config OUTPUT.json [--force]
```

Writes a complete configuration — every section, every key, at its default value — with a
`_about` and `_distributions` header explaining the format. It refuses to overwrite an
existing file unless you pass `--force`, since that file is likely the definition of a run
you care about.

```console
$ python -m jaxpe.cli write-config my_run.json
Wrote default run configuration to my_run.json
Edit it and pass it with --config to generate-injections and run-pe.

$ python -m jaxpe.cli write-config my_run.json
ValueError: my_run.json already exists; pass --force to overwrite it.
```

There is no need to keep the whole file: delete every section you are not changing and the
rest falls back to the defaults.

---

## The defaults are smoke-test scale

With no `--config`, the CLI uses the values below. They are chosen so the whole pipeline
runs in seconds, **not so that any of it is accurate**, and the loader says so on every
run:

```console
WARNING: ns.nlive=10 is a smoke-test value: nested sampling at this resolution exercises
the code path but its evidence and posterior are not measurements. Production runs use
>= 100 (see examples/configs/production_bbh.json).
```

| quantity | default | config key |
|---|---|---|
| segment duration | 4.0 s | `data.duration` |
| sampling rate | 1024 Hz | `data.sampling_rate` |
| $$f_{\rm min}$$ / $$f_{\rm ref}$$ | 30 Hz / 20 Hz | `data.f_min`, `data.f_ref` |
| chirp-mass prior | uniform $$(10, 50)\ M_\odot$$ | `prior.chirp_mass` |
| mass-ratio prior | uniform $$(0.1, 1.0)$$ | `prior.mass_ratio` |
| aligned-spin priors | uniform $$(-0.9, 0.9)$$ | `prior.spin1z`, `prior.spin2z` |
| distance prior | $$p(d) \propto d^2$$ on $$(100, 2000)$$ Mpc | `prior.luminosity_distance` |
| coalescence-time prior | injected $$t_c \pm 0.1$$ s | `prior.geocent_time` |
| HMC | `step_size=0.01`, `n_leapfrog=10` | `kernel.hmc` |
| MALA | `step_size=0.01` | `kernel.mala` |
| random seeds | 42 throughout | `seeds.*` |
| nested sampling | `nlive=10`, `num_repeats=5`, `precision_criterion=0.1` | `ns` |
| GPry budget | `max_total=500`, `n_initial=` GPry's $$3d$$ | `gpry` |

Three deserve emphasis:

- **`nlive=10` is a smoke test.** Nested sampling at ten live points exercises the code
  path; its evidence and posterior are not measurements. Production runs use $$\ge 100$$.
- **`f_ref = 20` Hz sits below `f_min = 30` Hz**, so the frequency at which spins and phase
  are defined lies outside the analysed band. That is legal but rarely intended, and the
  loader warns about it. The shipped production configs set `f_ref = f_min = 20` Hz.
- **The step sizes are starting points, not tuned values.** $$\varepsilon = 0.01$$ is a
  plausible scale for a few-tens-of-$$M_\odot$$ BBH under these priors and can be badly
  wrong for another source. `GlobalLocalConfig.adapt_step_size` is on by default, so they
  are adapted during the training loops — check the acceptance rate before trusting a run.

---

## Troubleshooting

**`cli.py: error: unrecognized arguments: --burn-in ...`**
`process-samples` takes only file paths. Earlier drafts of this page advertised
`--burn-in`, `--thin` and `--plot`; those flags do not exist. Burn-in removal, thinning and
plotting all happen automatically, driven by the measured autocorrelation time.

**`WARNING: tau was non-finite for N of M parameters`**
The chains were too short for a reliable autocorrelation estimate on some parameters.
Post-processing now proceeds using the finite entries, but the resulting sample count is
an upper bound, not an effective sample size — re-run with more chains and more production
loops before believing the width of any marginal. (Before this was fixed, the same
situation raised `ValueError: cannot convert float NaN to integer` and produced nothing.)

**`Warning: .../injection.json not found ... Skipping`**
`process-samples` needs `injection.json` in the same directory as the `.npz`. `run-pe`
writes it there; keep the directory intact, or copy the injection back in.

**`ModuleNotFoundError: No module named 'equinox'`**
The environment lacks the core dependencies — see
[Installation]({{ site.baseurl }}/docs/installation.html).

**Out of memory on the GPU.** Fall back with `JAX_PLATFORMS=cpu`, or cap the fraction JAX
reserves with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`. Time-domain runs are the memory-hungry
ones; the [TD benchmark]({{ site.baseurl }}/docs/ongoing/td_phenomt_pe_benchmark.html) documents a
VRAM ceiling hit on a 4 GB card.
