---
title: diagnostics
parent: API Reference
layout: default
---

# `jaxpe.diagnostics`
{: .no_toc }

1. TOC
{:toc}

The `diagnostics` module provides tools to analyze and visualize the posterior samples drawn by the sampler.

## Convergence Diagnostics

### Split R-hat ($\hat{R}$)
Evaluates the convergence of the MCMC chains. Values close to 1.0 indicates that chains have likely converged to the target distribution.

### Effective Sample Size (ESS)
Estimates the number of independent samples drawn from the posterior.

## Visualizations

### Corner Plots
Generate 1D and 2D marginal distributions of the posterior.

### Trace Plots
Visualize the evolution of the chains over time, checking for proper mixing and burn-in.
