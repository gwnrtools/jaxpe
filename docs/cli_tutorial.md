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

`jaxpe` ships a three-command CLI that takes you from a synthetic injection to a corner
plot without writing Python:

```
jaxpe generate-injections   →   jaxpe run-pe   →   jaxpe process-samples
```

It is best understood as a **fixed-configuration demo driver**. It exercises the whole
stack end to end and is the fastest way to confirm an installation works or to compare
samplers on a common problem. But almost every physical choice — segment duration,
sampling rate, $$f_{\rm min}$$, the prior ranges, the sampler step sizes, the random seed —
is **hardcoded in `jaxpe/cli.py`** and cannot be set from the command line. See
[What the CLI hardcodes](#what-the-cli-hardcodes) for the full list.

For real analyses — a specific source, a tuned sampler, relative binning, a convergence
gate — use the drivers in [`bin/`]({{ site.baseurl }}/docs/scripts_and_examples.html)
instead. `bin/run_bns_ce_pe.py` is the reference: it exposes masses, spins, PSD, duration,
budget and convergence criteria as real arguments, and is what every benchmark on this
site actually runs.

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

## `generate-injections`

```bash
python -m jaxpe.cli generate-injections --outdir DIR [--n-injections N] [--seed S]
                                        [--fiducial] [--network H1,L1]
                                        [--noise {zero,gaussian}] [--psd aligo|PATH]
```

Draws `N` **distinct** injections and writes them as `inj_0.json … inj_{N-1}.json` into
`--outdir` (created if needed). `--outdir` is required; `--n-injections` defaults to 1 and
`--seed` to 42.

| flag | default | effect |
|---|---|---|
| `--n-injections` | 1 | how many injections to draw |
| `--seed` | 42 | RNG seed — the same seed reproduces the same set exactly |
| `--fiducial` | off | emit the fixed reference binary instead of drawing |
| `--network` | `H1,L1` | recorded in the file; `run-pe` uses it unless overridden |
| `--noise` | `zero` | recorded in the file; `run-pe` uses it unless overridden |
| `--psd` | `aligo` | recorded in the file; `aligo` is the built-in analytic aLIGO ZDHP curve, anything else must be a path to a two-column ASCII PSD |

Parameters are drawn uniformly in chirp mass $$(15, 45)\,M_\odot$$, mass ratio
$$(0.3, 1.0)$$, aligned spins $$(-0.5, 0.5)$$ and distance $$(300, 1500)$$ Mpc, with
isotropic orientation and sky position. Those boxes sit **strictly inside** the recovery
priors, so no injected truth ever lands on a prior edge. Coalescence time is fixed at the
GW150914 GPS epoch.

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

**There is no built-in Cosmic Explorer or Einstein Telescope curve.** `jaxpe.gw.psd` ships
only the analytic aLIGO ZDHP model, so any other detector must be supplied as a file.
Passing an unknown name is rejected rather than silently ignored:

```console
$ python -m jaxpe.cli generate-injections --psd CE --outdir demo/inj
ValueError: --psd 'CE' is neither the built-in 'aligo' curve nor an existing file.
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

Keep edited values **inside the prior ranges** below, or the injected truth will sit
outside the prior and the posterior will rail against a boundary.

---

## `run-pe`

```bash
python -m jaxpe.cli run-pe --injection FILE --outdir DIR
                           [--sampler {hmc,mala,ns,gpry}]
                           [--likelihood {full,marginalized_phase_distance}]
                           [--domain {td,fd}] [--network ...] [--noise {zero,gaussian}]
                           [--psd aligo|PATH] [--n-chains 100] [--n-prelim-loops 1]
                           [--n-training-loops 5] [--n-production-loops 50]
```

| flag | default | effect |
|---|---|---|
| `--injection` | *required* | path to the injection JSON |
| `--outdir` | *required* | output directory, created if absent |
| `--sampler` | `hmc` | `hmc`, `mala`, `ns` (BlackJAX nested sampling), `gpry` (GP surrogate) |
| `--likelihood` | `full` | `full`, or `marginalized_phase_distance` (analytic $$\phi_c$$ + $$D_L$$ marginal) |
| `--domain` | `fd` | `fd` → `IMRPhenomD`, `td` → `IMRPhenomT` (both at $$f_{\rm ref} = 20$$ Hz) |
| `--network` | *from injection* | comma-separated detectors; overrides the recorded value |
| `--noise` | *from injection* | `zero` or `gaussian` (fixed seed 42); overrides the recorded value |
| `--psd` | *from injection* | `aligo` or a path; overrides the recorded value |
| `--n-chains` | 100 | `hmc`/`mala` only |
| `--n-prelim-loops` | 1 | discarded warmup loops; `hmc`/`mala` only |
| `--n-training-loops` | 5 | local steps → flow fit → global block; `hmc`/`mala` only |
| `--n-production-loops` | 50 | flow frozen; `hmc`/`mala` only |

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
with the domain, network, PSD, conditioning settings and prior ranges the run used, so
`process-samples` can rebuild the same problem instead of guessing.

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

## What the CLI hardcodes

Not reachable from any flag. To change any of these, use a `bin/` driver or the Python API.

| quantity | value |
|---|---|
| segment duration | 4.0 s |
| sampling rate | 1024 Hz |
| $$f_{\rm min}$$ / $$f_{\rm ref}$$ | 30 Hz / 20 Hz |
| chirp-mass prior | $$(10, 50)\ M_\odot$$ |
| mass-ratio prior | $$(0.1, 1.0)$$ |
| aligned-spin priors | $$(-0.9, 0.9)$$ |
| distance prior | $$(100, 2000)$$ Mpc |
| coalescence-time prior | injected $$t_c \pm 0.1$$ s |
| HMC | `step_size=0.01`, `n_leapfrog=10` |
| MALA | `step_size=0.01` |
| random seed | `PRNGKey(42)`; noise seed 42 when `--noise gaussian` |
| nested sampling | `nlive=10`, `num_repeats=5`, `precision_criterion=0.1` |
| GPry budget | `max_total=500`, `n_initial=5` |

Two of these deserve emphasis. The **step sizes are fixed, not adapted to your source** —
$$\varepsilon = 0.01$$ is a plausible scale for a few-tens-of-$$M_\odot$$ BBH under these
priors and can be badly wrong for another source; check the acceptance rate before trusting
a run. And the **nested-sampling `nlive=10` is a smoke-test setting**, far too small for a
publishable posterior; it is chosen so the path runs quickly, not so it is accurate.

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
