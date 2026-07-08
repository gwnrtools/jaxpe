# Parameter Estimation for Prominent BBH Events using IMRPhenomD

**Status: Currently Running (Background Task)**

This document tracks the results of simulating and recovering 5 prominent Binary Black Hole (BBH) events using the `jaxpe` framework with the `IMRPhenomD` frequency-domain model. The sampler configuration uses a robust, publication-ready configuration: `n_chains=100`, `n_epochs=100`, `n_production=1000`.

## Event Summary

| Event Name | Characteristic | Est. $M_c$ ($M_\odot$) | Est. $q$ | $d_L$ (Mpc) | Target SNR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GW150914** | First detection, heavy, equal-mass | ~28.1 | 0.81 | 410.0 | *Pending...* |
| **GW170729** | Massive early event | ~35.7 | 0.68 | 2840.0 | *Pending...* |
| **GW170104** | Intermediate-mass, typical | ~21.1 | 0.62 | 880.0 | *Pending...* |
| **GW190412** | Highly asymmetric mass ratio | ~13.3 | 0.28 | 740.0 | *Pending...* |
| **GW190521** | Intermediate-Mass Black Hole (IMBH) | ~64.4 | 0.78 | 5300.0 | *Pending...* |

## Execution Profiling

*The following profiling data will be populated as each event completes its MCMC phase.*

| Event | `best_of_prior_init` | `Sampler.run` (incl. JIT) | Total Production Samples |
| :--- | :--- | :--- | :--- |
| GW150914 | - | - | - |
| GW170729 | - | - | - |
| GW170104 | - | - | - |
| GW190412 | - | - | - |
| GW190521 | - | - | - |

## Results & Corner Plots

*(Corner plots and 1D marginal credible intervals will be embedded below as the background task finishes generating the `output/production_events` artifacts).*
