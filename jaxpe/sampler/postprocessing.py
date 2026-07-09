import numpy as np
import jax
import emcee


class PostProcessor:
    """
    Post-processes raw MCMC samples by calculating the integrated autocorrelation time
    and extracting independent, uncorrelated samples. Maps unconstrained samples back
    to physical parameter space.
    """

    def __init__(self, problem, raw_samples=None, raw_samples_file=None):
        """
        Initialize the PostProcessor.

        Parameters
        ----------
        problem : InferenceProblem
            The inference problem containing the prior mapping logic.
        raw_samples : np.ndarray, optional
            The raw unconstrained MCMC samples of shape (n_steps, n_chains, n_dim).
        raw_samples_file : str or Path, optional
            Path to the raw_samples.npz file containing the 'samples' array.
        """
        self.problem = problem

        if raw_samples is not None:
            self.samples = np.asarray(raw_samples)
        elif raw_samples_file is not None:
            data = np.load(raw_samples_file)
            self.samples = data["samples"]
        else:
            raise ValueError("Must provide either raw_samples or raw_samples_file")

        if self.samples.ndim != 3:
            raise ValueError(
                f"Expected samples of shape (n_steps, n_chains, n_dim), got {self.samples.shape}"
            )

        self.n_steps, self.n_chains, self.n_dim = self.samples.shape

    def compute_autocorr(self, tol=10):
        """
        Compute the integrated autocorrelation time (tau) for each parameter.
        If chains are too short to reliably estimate tau (length < tol * tau),
        this issues a warning and falls back to the best available estimate.

        Returns
        -------
        tau : np.ndarray
            Array of length n_dim with the estimated tau for each parameter.
        """
        try:
            tau = emcee.autocorr.integrated_time(self.samples, tol=tol)
        except emcee.autocorr.AutocorrError as e:
            print(f"WARNING: Chains may not be fully converged. {e}")
            tau = e.tau

        return tau

    def extract_independent_samples(self, tau=None, burnin_multiplier=3, min_burnin=0):
        """
        Discard burn-in and thin the chains by the maximum autocorrelation time to
        yield independent samples.

        Returns
        -------
        flat_samples : np.ndarray
            Unconstrained, thinned samples of shape (N_independent, n_dim).
        """
        if tau is None:
            tau = self.compute_autocorr()

        max_tau = int(np.max(tau))
        if np.isnan(max_tau) or max_tau <= 0:
            print("WARNING: Max tau is NaN or <= 0. Defaulting to 100 for safety.")
            max_tau = 100

        burnin = max(int(burnin_multiplier * max_tau), min_burnin)
        thin = max_tau

        print(f"Max autocorrelation time (tau): {max_tau}")
        print(f"Discarding {burnin} steps as burn-in and thinning by {thin}.")

        if burnin >= self.n_steps:
            print(
                "CRITICAL WARNING: Burn-in exceeds total chain length. The sampler did not converge."
            )
            print("Falling back to discarding the first 50% of the chain.")
            burnin = self.n_steps // 2

        thinned = self.samples[burnin::thin]
        flat_samples = thinned.reshape(-1, self.n_dim)

        print(
            f"Extracted {flat_samples.shape[0]} independent samples (from {self.n_steps * self.n_chains} raw)."
        )
        return flat_samples

    def to_physical(self, flat_samples, batch_size=100_000):
        """
        Map unconstrained samples to the physical parameter space.
        Uses batched execution to prevent JAX from exhausting GPU memory.
        """
        phys_list = []
        _map_fn = jax.jit(jax.vmap(self.problem.prior.to_physical))

        for i in range(0, flat_samples.shape[0], batch_size):
            phys_list.append(np.asarray(_map_fn(flat_samples[i : i + batch_size])))

        phys = np.concatenate(phys_list, axis=0)
        return phys

    def process(self):
        """
        Run the full post-processing pipeline: compute tau, extract independent samples,
        and map to physical parameters.

        Returns
        -------
        phys_samples : np.ndarray
            The final physical, independent posterior samples.
        """
        tau = self.compute_autocorr()
        flat_unconstrained = self.extract_independent_samples(tau=tau)
        phys_samples = self.to_physical(flat_unconstrained)
        return phys_samples
