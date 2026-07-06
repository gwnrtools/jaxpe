---
title: diagnostics
parent: jaxpe
layout: default
---

# Sec. V: Diagnostics and Utilities (`jaxpe.diagnostics`)
{: .no_toc }

1. TOC
{:toc}

In this section, we provide the diagnostic tools necessary to analyze and validate the posterior samples. 

## Convergence Diagnostics

### Split R-hat

Evaluates the convergence of the MCMC chains by comparing the variance between chains to the variance within chains. Values close to 1.0 indicate that chains have likely converged to the target distribution.

### Effective Sample Size (ESS)

Estimates the number of independent samples drawn from the posterior, correcting for autocorrelation along the Markov chain.

## Visualizations

The module provides tools to generate 1D and 2D marginal distributions of the posterior (Corner Plots) and visualize the evolution of the chains over time (Trace Plots).
