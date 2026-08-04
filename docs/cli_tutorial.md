---
title: Universal CLI Tutorial
layout: default
nav_order: 6
---

# Universal CLI Tutorial

The `jaxpe` framework ships with a powerful, unified Command-Line Interface (CLI). This CLI allows you to generate injections, run parameter estimation (PE) using various domains, samplers, and likelihood constructions, and process the resulting posterior samples—all without writing any custom Python scripts.

You can access the CLI using the `jaxpe` command or via `python -m jaxpe.cli`.

---

## 1. Generating Injections

Before running PE, you need an injection (a simulated gravitational-wave signal). The `generate-injections` command creates one or more injections with specific noise characteristics and stores them as JSON files.

**Example: Generate a zero-noise injection for Cosmic Explorer**
```bash
jaxpe generate-injections \
    --network H1,L1,V1 \
    --noise zero \
    --psd CE \
    --n-injections 1 \
    --outdir my_injections/
```

**Example: Generate 5 injections in simulated Gaussian noise for LIGO/Virgo**
```bash
jaxpe generate-injections \
    --network H1,L1,V1 \
    --noise gaussian \
    --psd LIGO \
    --n-injections 5 \
    --outdir my_injections/
```
*This will create `inj_0.json`, `inj_1.json`, etc., in the `my_injections/` directory.*

---

## 2. Running Parameter Estimation (`run-pe`)

The `run-pe` command is the core of the universal CLI. It maps your configuration into the correct JAX backend, constructs the likelihood and priors, and dispatches to the requested sampling algorithm.

### Use Case A: Time-Domain HMC (Standard MCMC)
Use the fully differentiable `full` likelihood in the Time Domain (`td`) with the `hmc` (Hamiltonian Monte Carlo) sampler.
```bash
JAX_PLATFORMS=cpu jaxpe run-pe \
    --injection my_injections/inj_0.json \
    --sampler hmc \
    --domain td \
    --likelihood full \
    --n-chains 4 \
    --n-prelim-loops 2 \
    --n-training-loops 5 \
    --n-production-loops 10 \
    --outdir my_pe_results/
```
*(Note: Omit `JAX_PLATFORMS=cpu` if you want to run on GPU/TPU).*

### Use Case B: Frequency-Domain Nested Sampling
Use the `marginalized_phase_distance` analytic likelihood in the Frequency Domain (`fd`) and run Direct Nested Sampling (`ns`) via BlackJAX. This method builds a fully jittable likelihood function that evaluates natively on the accelerator.
```bash
jaxpe run-pe \
    --injection my_injections/inj_0.json \
    --sampler ns \
    --domain fd \
    --likelihood marginalized_phase_distance \
    --outdir my_pe_results/
```

### Use Case C: Surrogate Active Learning (GPry)
For extremely expensive likelihoods, use the `gpry` sampler. GPry trains a Gaussian Process surrogate over the parameter space and performs nested sampling on the surrogate, requesting true likelihood evaluations only where uncertainty is highest.
```bash
jaxpe run-pe \
    --injection my_injections/inj_0.json \
    --sampler gpry \
    --domain fd \
    --likelihood marginalized_phase_distance \
    --outdir my_pe_results/
```

### Use Case D: MALA (Metropolis-Adjusted Langevin Algorithm)
An alternative to HMC, MALA uses only single-step gradient information.
```bash
jaxpe run-pe \
    --injection my_injections/inj_0.json \
    --sampler mala \
    --domain fd \
    --likelihood full \
    --n-chains 4 \
    --outdir my_pe_results/
```

---

## 3. Post-Processing Samples

Once PE finishes, you will find a `raw_samples.npz` file in your output directory. The `process-samples` command computes derived physical parameters (like chirp mass and effective spin), applies burn-in and thinning, and generates corner plots.

**Example: Process samples, apply a 20% burn-in, thin by a factor of 10, and plot**
```bash
jaxpe process-samples \
    my_pe_results/raw_samples.npz \
    --burn-in 0.2 \
    --thin 10 \
    --plot
```

This will save `processed_samples.npy` and `corner_plot.png` (if `--plot` is passed) in the same directory as the raw samples.

---

## Summary of `run-pe` Options

- `--sampler`: `hmc`, `mala`, `ns`, `gpry`.
- `--domain`: `td` (Time Domain, e.g., `IMRPhenomT`), `fd` (Frequency Domain, e.g., `IMRPhenomD`).
- `--likelihood`: `full` (standard differentiable likelihood), `marginalized_phase_distance` (analytically marginalizes over phase and distance; requires `fd`). 
- `--n-chains`, `--n-training-loops`, `--n-production-loops`: Configure MCMC integration lengths (only applicable to `hmc`/`mala`).
