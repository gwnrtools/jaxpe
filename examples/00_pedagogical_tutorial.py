"""
A Pedagogical Introduction to JAXPE: Parameter Estimation Step-by-Step

Welcome to `jaxpe`! This script is designed for early graduate students and beginners
to understand the fundamentals of Parameter Estimation (PE) using `jaxpe`.
We will bypass the complexity of Gravitational Wave (GW) data analysis for now and
focus entirely on how to set up a probability distribution, configure an MCMC sampler,
and run parameter estimation.

In this tutorial, we will sample from a "Rosenbrock" density (often called the
"banana" function). It is a classic test function for optimization and sampling algorithms
due to its curved, non-linear correlations, which make it tricky for simple samplers
but manageable for advanced ones like Hamiltonian Monte Carlo (HMC) or MALA.

Let's dive in!
"""

import jax

# We usually want to enforce double-precision (64-bit) floating point in scientific computing.
# JAX defaults to 32-bit for neural network speed, but for PE we need the accuracy.
jax.config.update("jax_enable_x64", True)

# Import the sampler, kernel, and diagnostic tools from jaxpe
from jaxpe.sampler import GlobalLocalConfig, Sampler
from jaxpe.kernels import MALA
from jaxpe.diagnostics import effective_sample_size, split_rhat

# ==============================================================================
# Step 1: Define the Target Log-Probability Density
# ==============================================================================
# In Bayesian Parameter Estimation, we want to draw samples from a posterior
# probability distribution: P(theta | data).
# By Bayes' theorem: P(theta | data) \propto P(data | theta) * P(theta)
# Thus, we need a function that computes the natural logarithm of the posterior
# density (up to an additive constant).
#
# For this tutorial, we define our target density as a 2D Rosenbrock-like distribution.
# We will define a log-likelihood and assume a uniform prior over the whole space.

a = 1.0
b = 100.0


def log_likelihood(theta):
    """
    Computes the log-likelihood of the 2D Rosenbrock distribution.

    Parameters
    ----------
    theta : jnp.ndarray
        A 1D array containing the parameters we want to sample. For our 2D
        distribution, theta has a length of 2 (let's call them x and y).

    Returns
    -------
    logL : float
        The log-probability density at the given parameter values.
    """
    # Unpack parameters
    x = theta[0]
    y = theta[1]

    # The standard Rosenbrock function is f(x, y) = (a - x)^2 + b * (y - x^2)^2
    # We define our probability density as p(x,y) \propto exp(-f(x,y) / 20.0)
    # The division by 20.0 is just to "widen" the distribution a bit for easier sampling.
    val = (a - x) ** 2 + b * (y - x**2) ** 2

    # We return the log of the probability, so it's simply -val / 20.0
    return -val / 20.0


# In a fully Bayesian setting, we might also have a log-prior.
# If the prior is uniform across all real numbers, log_prior is a constant
# and can be ignored (or return 0).
def log_prior(theta):
    return 0.0


def log_posterior(theta):
    """The total log-posterior probability."""
    return log_likelihood(theta) + log_prior(theta)


# ==============================================================================
# Step 2: Setup the Sampler
# ==============================================================================
def main():
    print("Welcome to JAXPE Pedagogical Tutorial!")
    print("Setting up the 2D Rosenbrock sampling problem...")

    # We configure how the sampler will behave using `GlobalLocalConfig`.
    # JAXPE uses an advanced sampling strategy that combines:
    #   1. "Local" MCMC steps (like MALA, HMC) to explore the local neighborhood.
    #   2. "Global" Normalizing Flow proposals to jump between distant regions or modes.
    #
    # Let's define the configuration:
    #   - n_chains: Number of parallel walkers (we use many parallel chains in JAX for speed).
    #   - n_training_loops: How many phases we spend adapting the Normalizing Flow.
    #   - n_production_loops: How many phases we spend actually collecting valid samples.
    #   - n_local_steps: Number of MCMC steps taken between flow updates.
    cfg = GlobalLocalConfig(
        n_chains=128, n_training_loops=5, n_production_loops=5, n_local_steps=200
    )

    # We choose an MCMC Kernel. We will use MALA (Metropolis-Adjusted Langevin Algorithm).
    # MALA uses the gradient of the log-probability to "surf" the probability landscape.
    # The step_size controls how large of a jump MALA proposes.
    kernel = MALA(step_size=0.1)

    # Initialize the Sampler with our kernel, our log-posterior function,
    # the dimensionality of the problem (2), and the configuration.
    sampler = Sampler(kernel, logp_fn=log_posterior, n_dim=2, config=cfg)

    # ==============================================================================
    # Step 3: Initialization and Sampling
    # ==============================================================================
    # JAX relies on explicit pseudo-random number generator (PRNG) keys.
    # We start by creating a seed key.
    key = jax.random.PRNGKey(42)

    # We must provide initial starting points for all our chains.
    # We initialize them around (x=0, y=0) with a small Gaussian scatter.
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (cfg.n_chains, 2))

    print("Running sampler... (this will compile the JAX code first)")
    # The `sampler.run()` function handles everything:
    # MCMC sampling, adapting the Normalizing Flow, and collecting samples.
    # JAX uses Just-In-Time (JIT) compilation, so the very first run might take a
    # moment to compile before executing incredibly fast.
    results = sampler.run(key, x0=x0)

    # ==============================================================================
    # Step 4: Diagnostics and Analysis
    # ==============================================================================
    print("\n--- Sampling Complete! ---")

    # The results object contains our samples.
    # They have shape (n_chains, n_production_loops * n_local_steps, n_dim)
    # We can flatten all chains together to get a big array of samples.
    flat_samples = results.flat()

    print(f"Total valid samples collected: {flat_samples.shape[0]}")

    # Let's compute some basic statistics of the marginal distributions.
    mean_x, mean_y = flat_samples.mean(axis=0)
    std_x, std_y = flat_samples.std(axis=0)
    print(f"Mean parameter values: x = {mean_x:.3f}, y = {mean_y:.3f}")
    print(f"Standard deviations:   x = {std_x:.3f}, y = {std_y:.3f}")

    # MCMC diagnostics are crucial to ensure we actually sampled the correct distribution.
    #
    # 1. Gelman-Rubin split R-hat statistic (should be < 1.05 for convergence)
    rhat = split_rhat(results.samples)
    print(f"Split R-hat (should be < ~1.05): {rhat}")

    # 2. Effective Sample Size (ESS). MCMC samples are correlated. ESS estimates
    #    how many *independent* samples our correlated chains are worth.
    ess = effective_sample_size(results.samples)
    print(f"Effective Sample Size: {ess}")

    # 3. Acceptance probabilities. We want this to be in a healthy range (e.g., 20% - 80%)
    print(
        f"Local (MALA) Acceptance rates per loop: {[f'{a:.2f}' for a in results.local_acceptance]}"
    )
    print(
        f"Global (Flow) Acceptance rates per loop: {[f'{a:.2f}' for a in results.global_acceptance]}"
    )

    print(
        "\nCongratulations! You have successfully run your first Parameter Estimation with JAXPE."
    )
    # (In a real scenario, you would now use `jaxpe.diagnostics.plots.corner_plot` to plot them!)


if __name__ == "__main__":
    main()
