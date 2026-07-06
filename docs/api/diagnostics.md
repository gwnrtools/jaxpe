---
title: diagnostics
parent: jaxpe
layout: default
---

# Sec. V: Diagnostics and Utilities (`jaxpe.diagnostics`)
{: .no_toc }

1. TOC
{:toc}

In this section, we provide the rigorous diagnostic tools necessary to validate the stationarity and convergence of the posterior Markov chains.

## Convergence Diagnostics

### Split R-hat

The Split R-hat ($$\hat{R}$$) statistic [1, 2] evaluates the convergence of the MCMC chains by comparing the variance between the chains to the variance within the chains. For a parameter $$\theta$$ sampled across $$M$$ chains each of length $$N$$, we first split each chain in half to form $$2M$$ chains of length $$N/2$$. 

We compute the between-chain variance $$B$$ and the within-chain variance $$W$$:

$$
B = \frac{N/2}{2M - 1} \sum_{m=1}^{2M} (\bar{\theta}_{m\cdot} - \bar{\theta}_{\cdot\cdot})^2
$$

$$
W = \frac{1}{2M} \sum_{m=1}^{2M} s_m^2
$$

where $$\bar{\theta}_{m\cdot}$$ is the mean of chain $$m$$, $$\bar{\theta}_{\cdot\cdot}$$ is the grand mean, and $$s_m^2$$ is the empirical variance of chain $$m$$. The marginal posterior variance is estimated as a weighted average:

$$
\widehat{\text{Var}}^{+}(\theta) = \frac{N/2 - 1}{N/2} W + \frac{1}{N/2} B
$$

The potential scale reduction factor is then:

$$
\hat{R} = \sqrt{\frac{\widehat{\text{Var}}^{+}(\theta)}{W}}
$$

As the chains converge to the true stationary distribution, the between-chain variance approaches the within-chain variance, and $$\hat{R} \to 1$$. Values greater than 1.05 indicate lack of convergence.

### Effective Sample Size (ESS)

Markov chains generate correlated samples, reducing the actual amount of independent information gathered. The Effective Sample Size (ESS) quantifies this by integrating the autocorrelation function $$\rho_t$$ across lags $$t$$:

$$
N_{\text{eff}} = \frac{MN}{1 + 2 \sum_{t=1}^{\infty} \hat{\rho}_t}
$$

where $$\hat{\rho}_t$$ is the estimated autocorrelation at lag $$t$$ [2]. Higher values of ESS indicate better chain mixing and smaller Monte Carlo standard errors.

## Visualizations

### Corner Plots

The `diagnostics` module leverages [corner.py](https://corner.readthedocs.io) to generate 1D and 2D marginal distributions. These plots are critical for identifying parameter degeneracies, correlations (e.g., mass-ratio vs. spin), and isolated multimodal peaks in the GW posterior.

### Trace Plots

Visualize the raw evolution of the chains over the iteration index. This exposes transient non-stationary behavior, allowing researchers to set optimal burn-in (warm-up) thresholds.

### REFERENCES

[1] A. Gelman and D. B. Rubin, "Inference from Iterative Simulation Using Multiple Sequences," Stat. Sci. **7**, 457 (1992).  
[2] A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, and P. C. Bürkner, "Rank-Normalization, Folding, and Localization: An Improved $\widehat{R}$ for Assessing Convergence of MCMC," Bayesian Anal. **16**, 667 (2021).
