---
title: Scripts and Examples
layout: default
nav_order: 5
---

# Scripts and Examples

The `jaxpe` repository includes a variety of tutorials, example workflows, and robust drivers to help you go from learning the basics to running production-scale parameter estimation.

---

## 1. Tutorials & Fundamentals (`examples/`)

If you are new to MCMC or the JAX ecosystem, these tutorials use simple synthetic likelihoods to demonstrate the sampler interface:
- **`00_pedagogical_tutorial.py`**: A beginner-friendly walkthrough of setting up a Rosenbrock (banana) probability distribution and sampling from it.
- **`01_gaussian.py`**: A minimal example sampling a correlated Gaussian density using the low-level `logp_fn` interface.
- **`02_multimodal_toys.py`**: Demonstrates how flow-driven global proposals overcome isolation in multimodal distributions (Gaussian mixture, dual moons).

---

## 2. Gravitational-Wave PE Examples (`examples/`)

End-to-end demonstrations of Bayesian inference on Gravitational Wave signals:
- **`03_gw_injection.py`**: Parameter estimation on a synthetic Toy BBH injection, perfect for testing full pipeline runs locally.
- **`04_gw150914.py`**: Runs PE on real, open GWOSC data for GW150914.
- **`05_esigma_injection.py`**: Demonstration using the `ESIGMA` waveform model.
- **`06_phenomd_injection.py`**: Demonstration using the standard `IMRPhenomD` aligned-spin BBH waveform model.

---

## 3. Validation & Advanced Topics (`examples/`)

- **`07_td_higher_mode_route_comparison.py`** / **`08_fd_dominant_mode_route_comparison.py`**: Cross-validates JAX gradient sampling against nested sampling via GPry on identical injections.
- **`09_validate_injection_vs_dynesty.py`**: Direct posterior comparison against the `dynesty` sampler.
- **`10_fd_hm_relative_binning_pe.py`**: Examples of running the frequency-domain relative binning likelihood.

---

## 4. The Universal CLI (`jaxpe`)

A CLI that runs the whole stack without writing Python, available as `jaxpe <command>` or
equivalently `python -m jaxpe.cli <command>`. Every physics setting — duration, sampling
rate, $$f_{\rm min}$$, the prior *distributions*, the injected population, step sizes,
budgets and seeds — is read from a **run configuration JSON**, so a run is defined by an
artifact you can commit and replay rather than by a source edit:

```bash
jaxpe write-config my_run.json     # emit a fully-populated file to edit
jaxpe generate-injections --config my_run.json --n-injections 100 --outdir inj/
jaxpe run-pe --config my_run.json --injection inj/inj_0.json --outdir pe/0
```

With no `--config` it falls back to smoke-test-scale defaults, which is the fastest way to
confirm an installation works. `examples/configs/` ships production and PP-campaign
configurations. The `bin/` drivers below remain the reference for relative binning,
convergence gating and the `ESIGMA`/`NRSur` waveforms, which the CLI does not expose.

**[➡️ Full CLI tutorial: the configuration format, every flag, worked examples](cli_tutorial.md)**

### `jaxpe generate-injections`
Draws `N` distinct BBH injections as JSON, seeded by `--seed` for reproducibility, with
truths drawn strictly inside the recovery priors. The `--network`, `--noise` and `--psd`
settings are recorded in each file and become `run-pe`'s defaults. Use `--fiducial` for the
fixed GW150914-like reference binary instead of a draw, or edit the JSON for any other
specific source.
```bash
jaxpe generate-injections --n-injections 3 --network H1,L1 --outdir my_injection_data/
jaxpe generate-injections --fiducial --outdir reference_injection/
```

### `jaxpe run-pe`
Runs parameter estimation on a saved injection, selecting the sampler (`hmc`, `mala`, `ns`,
`gpry`), the likelihood construction, and the integration domain. `--network` and `--noise`
are honoured here.
```bash
jaxpe run-pe --injection my_injection_data/inj_0.json --sampler hmc --domain fd \
    --network H1,L1 --outdir pe_results/
```

### `jaxpe process-samples`
Post-processes raw `.npz` chains: estimates the autocorrelation time, thins to independent
draws, maps them from the unconstrained space back to physical parameters, and writes
`posterior_samples.npy` plus a corner plot. Takes file paths only — thinning is automatic,
and there are no `--burn-in`/`--thin`/`--plot` options.
```bash
jaxpe process-samples pe_results/raw_samples.npz
```

---

## 5. Profiling & Diagnostics (`bin/`)

- **`benchmark_diffrax_compile.py`**: Measures XLA compile-graph sizes versus right-hand-side complexity for JAX ODE solvers.
- **`profile_sampler_scaling.py`**: Detailed breakdown of fixed-vs-marginal costs for the HMC sampling loop.
- **`make_bns_ce_figures.py`** & **`make_sampler_comparison_figures.py`**: Specialized plotting scripts for generating the figures found in our benchmarks.
