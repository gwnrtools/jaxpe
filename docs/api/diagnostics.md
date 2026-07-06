---
title: diagnostics
parent: jaxpe
layout: default
---

# Sec. V: Diagnostics and Utilities (`jaxpe.diagnostics`)
{: .no_toc }

1. TOC
{:toc}

How do we know when to stop? If we run our samplers indefinitely, we waste precious computational resources. But if we stop too early, we might mistake a transient, wandering chain for the true stationary posterior. In this section, we provide the rigorous diagnostic tools necessary to validate the convergence of our Markov chains.

## Convergence Diagnostics

### Split R-hat

Imagine asking ten different chefs in ten different kitchens to prepare the exact same recipe. If the recipe (our MCMC algorithm) and the ingredients (our posterior) are well-defined, eventually every kitchen should produce an indistinguishable soup. If some kitchens are producing salty broth while others produce sweet stew, the process has not yet converged. 

This is the essence of the Split R-hat ($$\hat{R}$$) statistic [1, 2]. It evaluates convergence by comparing the variance *between* the chains to the variance *within* the chains. For a parameter $$\theta$$ sampled across $$M$$ chains each of length $$N$$, we first split each chain in half to form $$2M$$ chains of length $$N/2$$ (to detect non-stationarity within a single chain). 

We compute the between-chain variance $$B$$:

$$
B = \frac{N/2}{2M - 1} \sum_{m=1}^{2M} (\bar{\theta}_{m\cdot} - \bar{\theta}_{\cdot\cdot})^2
$$

where $$\bar{\theta}_{m\cdot}$$ is the mean of chain $$m$$, and $$\bar{\theta}_{\cdot\cdot}$$ is the grand mean. Next, we compute the average within-chain variance $$W$$:

$$
W = \frac{1}{2M} \sum_{m=1}^{2M} s_m^2
$$

where $$s_m^2$$ is the empirical variance of chain $$m$$. The marginal posterior variance is estimated as a weighted average of these two quantities:

$$
\widehat{\text{Var}}^{+}(\theta) = \frac{N/2 - 1}{N/2} W + \frac{1}{N/2} B
$$

The potential scale reduction factor is then simply the ratio:

$$
\hat{R} = \sqrt{\frac{\widehat{\text{Var}}^{+}(\theta)}{W}}
$$

As the chains converge to the true stationary distribution, the between-chain variance vanishes relative to the within-chain variance, and $$\hat{R} \to 1$$. Values greater than 1.05 strongly indicate that our chefs are not yet cooking the same soup.

### Effective Sample Size (ESS)

Drawing 10,000 samples from a Markov chain is not the same as drawing 10,000 independent samples from the prior. Because each step depends on the previous one, the chain exhibits autocorrelation, reducing the actual amount of independent information gathered. 

The Effective Sample Size (ESS) elegantly quantifies this loss of efficiency. We define the true autocorrelation function at lag $$t$$ as $$\rho_t = \text{Cov}(\theta_0, \theta_t) / \text{Var}(\theta)$$. The ESS is formulated by summing this autocorrelation across all lags:

$$
N_{\text{eff}} = \frac{MN}{1 + 2 \sum_{t=1}^{\infty} \hat{\rho}_t}
$$

where $$\hat{\rho}_t$$ is the empirically estimated autocorrelation [2]. A higher ESS guarantees better chain mixing and smaller Monte Carlo standard errors, ensuring our parameter estimates are statistically robust.

## Visualizations

### Corner Plots

The `diagnostics` module leverages [corner.py](https://corner.readthedocs.io) to generate 1D and 2D marginal distributions. These corner plots are the bread and butter of gravitational-wave astronomy, instantly revealing complex physical degeneracies (like the infamous mass-ratio vs. effective-spin correlation) that raw numbers cannot.

### Trace Plots

Visualize the raw evolution of the chains over the iteration index. Trace plots expose transient, non-stationary behavior, allowing you to confidently trim the initial "burn-in" (warm-up) phase before the chains have found the true posterior mass.

### REFERENCES

[1] A. Gelman and D. B. Rubin, "Inference from Iterative Simulation Using Multiple Sequences," Stat. Sci. **7**, 457 (1992).  
[2] A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, and P. C. Bürkner, "Rank-Normalization, Folding, and Localization: An Improved $\widehat{R}$ for Assessing Convergence of MCMC," Bayesian Anal. **16**, 667 (2021).
