"""GPry backend for the jaxpe surrogate-engine seam.

Wraps ``gpry.Runner`` (pinned optional dependency; ``pip install jaxpe[surrogate]``)
behind :class:`~jaxpe.surrogate.engine.SurrogateEngine`. GPry supplies the GP
regressor, the acquisition engine (NORA/BatchOptimizer), the SVM infinities
classifier, convergence criteria, MPI-parallel truth evaluation and checkpointing --
none of which we reimplement (design note, D4).

The likelihood passed in is an opaque host-side callable ``loglike(x) -> float`` over
the *intrinsic* parameter vector -- typically
``ModesNetworkLikelihood.log_marginal_likelihood_full`` composed with an external
waveform model, or a jaxpe case-(1) model treated as a pseudo-black-box for
validation. It must never be jitted or traced.
"""

import numpy as np

from .engine import SurrogateSamples


class GPryEngine:
    """Active-learning surrogate of an expensive log-likelihood, via GPry.

    Parameters
    ----------
    loglike
        ``(d,) -> float`` log-likelihood over the (intrinsic) parameters. Called
        hundreds-to-thousands of times; host-side Python only.
    bounds
        ``{name: (low, high)}`` (ordered) or a (d, 2) array of prior bounds; the
        prior is uniform within them (transform parameters upstream if not).
    checkpoint, load_checkpoint
        Passed to ``gpry.Runner``; ``load_checkpoint`` is required by GPry whenever
        ``checkpoint`` is set ("resume" or "overwrite").
    options
        Extra ``gpry.Runner`` keyword arguments (surrogate/acquisition/convergence
        specifications) passed through verbatim.
    """

    def __init__(
        self,
        loglike,
        bounds,
        checkpoint: str | None = None,
        load_checkpoint: str = "resume",
        verbose: int = 1,
        options: dict | None = None,
    ):
        from gpry import Runner  # deferred: optional dependency

        if isinstance(bounds, dict):
            self.names = tuple(bounds)
            bounds_arr = np.asarray([list(bounds[n]) for n in self.names], dtype=float)
            params = list(self.names)
        else:
            bounds_arr = np.asarray(bounds, dtype=float)
            self.names = tuple(f"x_{i + 1}" for i in range(len(bounds_arr)))
            params = None
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError(f"bounds must be (d, 2); got {bounds_arr.shape}")

        kwargs = dict(options or {})
        if checkpoint is not None:
            kwargs.update(checkpoint=checkpoint, load_checkpoint=load_checkpoint)
        self.runner = Runner(
            loglike, bounds=bounds_arr, params=params, verbose=verbose, **kwargs
        )

    def run(self) -> dict:
        """Run acquisition-training-convergence to completion; return diagnostics."""
        self.runner.run()
        return self.diagnostics()

    def surrogate_logp(self, x) -> np.ndarray:
        """Surrogate log-posterior at (n, d) points (uniform prior: loglike + const)."""
        return np.asarray(self.runner.logp(np.atleast_2d(np.asarray(x, dtype=float))))

    def true_logp(self, x) -> np.ndarray:
        """True log-posterior at (n, d) points -- expensive; used for IS reweighting.

        GPry's ``logp_truth`` is single-point; each point is one full expensive
        likelihood call, so the Python loop is not the bottleneck.
        """
        pts = np.atleast_2d(np.asarray(x, dtype=float))
        return np.asarray([float(self.runner.logp_truth(p)) for p in pts])

    def sample(self, sampler=None, add_options=None) -> SurrogateSamples:
        """MC-sample the surrogate posterior (GPry's nested-sampler interfaces)."""
        self.runner.generate_mc_sample(sampler=sampler, add_options=add_options)
        s = self.runner.last_mc_samples()
        x = np.asarray(s["X"])
        w = np.ones(len(x)) if s["w"] is None else np.asarray(s["w"])
        return SurrogateSamples(
            x=x, weights=w, logpost=np.asarray(s["logpost"]), names=self.names
        )

    def diagnostics(self) -> dict:
        return dict(
            n_truth_evals=int(self.runner.surrogate.n_total),
            has_run=bool(self.runner.has_run),
            has_converged=bool(self.runner.has_converged),
        )
