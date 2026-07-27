"""The GPry-facing scalar likelihood over intrinsic parameters.

:class:`MarginalizedIntrinsicLikelihood` composes an external mode model
(``theta_int dict -> ModesData``; one expensive host-side call per point) with a
shared :class:`~jaxpe.gw.likelihood.modes.ModesNetworkLikelihood` data context,
reusing that context's jit-compiled extrinsic-marginal evaluator across every
intrinsic point. It is the ``theta_int -> lnL`` callable consumed by the surrogate
(design note section 3), an :class:`~jaxpe.gw.likelihood.base.IntrinsicLikelihood`.
"""

import jax.numpy as jnp
import numpy as np

from .base import IntrinsicLikelihood
from .modes import ModesNetworkLikelihood


class LowEffectiveSampleSizeError(RuntimeError):
    """An inner extrinsic marginal stayed below the effective-sample-size floor.

    Raised only in strict mode (``on_low_effective_sample_size="raise"``) after all
    escalating extra rounds were exhausted. Carries the offending intrinsic point
    so a checkpointed pipeline can resume with a larger budget.
    """

    def __init__(self, theta, effective_sample_size, floor, extra_rounds):
        self.theta = theta
        self.effective_sample_size = effective_sample_size
        self.floor = floor
        self.extra_rounds = extra_rounds
        super().__init__(
            f"effective sample size {effective_sample_size:.1f} < floor {floor:.1f} "
            f"at theta={theta} after {extra_rounds} escalating extra rounds"
        )


class MarginalizedIntrinsicLikelihood(IntrinsicLikelihood):
    """The GPry-facing scalar likelihood: theta_int -> extrinsic-marginalized lnL.

    Composes an external mode model (``theta_int dict -> ModesData``; one expensive,
    host-side call per point) with a shared :class:`ModesNetworkLikelihood` data
    context whose jit-compiled marginal evaluator is reused across every intrinsic
    point (``marginal_eval_fn``; the modes enter as traced arguments). Optionally
    caches every generated ModesData to disk (``ModeCache``) -- the cache feeds the
    IS-reweighting/extrinsic-recovery stage and doubles as ROM training data
    (design note, D2/D3).

    Parameters
    ----------
    mode_model
        ``theta_int dict -> ModesData``, all at the template's d_ref/t_ref/grid.
        NEVER traced; may take minutes per call for production models.
    like
        The data context (detectors, PSDs, injected/observed data, grids), with a
        template ModesData fixing the mode set and conventions.
    names
        Intrinsic parameter names; defines the vector order of ``__call__``.
    t_center
        Center of the coalescence-time prior window (``geocent_time``).
    marginalize_sky
        True: full (phi_c, t_c, D_L, ra, dec, psi, iota) marginal via adaptive IS
        (production). False: (phi_c, t_c, D_L) only, at the fixed extrinsic angles
        in ``fixed_extrinsic`` -- cheaper; used by validation tests.
    settings
        Keyword options forwarded to the marginal-likelihood methods.

    Attributes
    ----------
    importance_sampling_history
        In full-marginal mode, one record per ``__call__`` with the
        importance-sampling diagnostics of that evaluation: ``theta``, ``logz``,
        ``effective_sample_size``, ``extra_rounds_used``, ``failed``, ``n_eval``,
        ``lnl_max``, ``logz_rounds``. A converged-looking GPry run cannot certify
        the *inner* extrinsic marginals -- inspect the minimum effective sample
        size over this history (``importance_sampling_summary()``); a call with a
        low effective sample size means that theta's L(theta_int) is biased low
        (by ~ 1/(2 x effective sample size) in the log) and locally noisy.
    """

    def __init__(
        self,
        mode_model,
        like: ModesNetworkLikelihood,
        names,
        t_center: float,
        marginalize_sky: bool = True,
        fixed_extrinsic: dict | None = None,
        cache=None,
        settings: dict | None = None,
        effective_sample_size_floor: float = 0.0,
        max_extra_importance_sampling_rounds: int = 1,
        on_low_effective_sample_size: str = "accept",
    ):
        self.mode_model = mode_model
        self.like = like
        self.names = tuple(names)
        self.t_center = float(t_center)
        self.marginalize_sky = marginalize_sky
        self.cache = cache
        self.settings = dict(settings or {})
        # kwargs owned by __call__ / the healing mechanism, not user settings
        for owned in (
            "return_diagnostics",
            "effective_sample_size_target",
            "max_extra_rounds",
        ):
            self.settings.pop(owned, None)
        # self-healing: if a call's inner-marginal effective sample size is below
        # the floor after the base rounds, up to max_extra_importance_sampling_rounds
        # escalating rounds are added, with every batch recycled into the estimate
        # (BalanceHeuristicAccumulator) -- measured: low-effective-sample-size calls
        # occur *in the posterior peak region*, where their
        # ~1/sqrt(effective sample size) log-likelihood scatter directly perturbs
        # the Gaussian-process fit
        self.effective_sample_size_floor = float(effective_sample_size_floor)
        self.max_extra_importance_sampling_rounds = int(
            max_extra_importance_sampling_rounds
        )
        if on_low_effective_sample_size not in ("accept", "raise"):
            raise ValueError(
                "on_low_effective_sample_size must be 'accept' or 'raise', got "
                f"{on_low_effective_sample_size!r}"
            )
        # "accept": after the retries, take the last estimate regardless (record it);
        # "raise": raise LowEffectiveSampleSizeError instead -- pairs with GPry
        # checkpointing, since it aborts the acquisition loop mid-run
        self.on_low_effective_sample_size = on_low_effective_sample_size
        self.importance_sampling_history: list[dict] = []
        if not marginalize_sky:
            if fixed_extrinsic is None:
                raise ValueError("fixed_extrinsic required when marginalize_sky=False")
            self._node = np.asarray(
                [[fixed_extrinsic[k] for k in ("ra", "dec", "psi", "inclination")]]
            )

    def _modes_ab(self, theta: dict):
        md = self.cache.load(theta) if self.cache is not None else None
        if md is None:
            md = self.mode_model(theta)
            if self.cache is not None:
                self.cache.save(theta, md)
        return self.like.modes_fd_arrays(md)

    def __call__(self, x) -> float:
        theta = dict(zip(self.names, np.asarray(x, dtype=float).ravel()))
        a, b = self._modes_ab(theta)
        if self.marginalize_sky:
            # the marginalization itself escalates while below the quality floor,
            # recycling every batch into the estimate; no discard-and-restart
            log_z, diag = self.like.log_marginal_likelihood_full(
                {"geocent_time": self.t_center},
                modes_ab=(a, b),
                return_diagnostics=True,
                effective_sample_size_target=(
                    self.effective_sample_size_floor
                    if self.effective_sample_size_floor > 0
                    else None
                ),
                max_extra_rounds=self.max_extra_importance_sampling_rounds,
                **self.settings,
            )
            failed = diag["effective_sample_size"] < self.effective_sample_size_floor
            self.importance_sampling_history.append(
                dict(theta=theta, logz=float(log_z), failed=bool(failed), **diag)
            )
            if failed and self.on_low_effective_sample_size == "raise":
                raise LowEffectiveSampleSizeError(
                    theta,
                    diag["effective_sample_size"],
                    self.effective_sample_size_floor,
                    diag["extra_rounds_used"],
                )
            return float(log_z)
        ext_batch = self.settings.get("ext_batch", 1)
        inner = {k: v for k, v in self.settings.items() if k != "ext_batch"}
        eval_nodes = self.like.marginal_eval_fn(ext_batch=ext_batch, **inner)
        return float(eval_nodes(a, b, jnp.asarray(self._node), self.t_center)[0])

    def importance_sampling_summary(
        self,
        effective_sample_size_floor: float = 100.0,
        peak_efolds: float | None = None,
    ) -> dict:
        """Aggregate the per-call importance-sampling diagnostics of a full-marginal run.

        Returns min/median effective sample size over all L(theta_int) evaluations
        and the list of thetas whose inner marginal fell below
        ``effective_sample_size_floor`` -- those values are biased low by
        ~1/(2 x effective sample size) and noisy, and can silently distort the
        Gaussian-process fit even when GPry itself reports convergence.

        With ``peak_efolds`` set, additionally reports
        ``n_below_floor_near_peak``: unhealthy calls whose log-marginal lies within
        that many e-folds of the best call. Measured on the demo problems, low
        effective sample sizes in the *tails* are harmless (exponentially small
        posterior weight) while those *near the peak* directly perturb the
        surrogate -- this count is the reliability-gate quantity.
        """
        if not self.importance_sampling_history:
            return dict(n_calls=0)
        history = self.importance_sampling_history
        sizes = np.array([h["effective_sample_size"] for h in history])
        low = [
            h
            for h in history
            if h["effective_sample_size"] < effective_sample_size_floor
        ]
        out = dict(
            n_calls=len(sizes),
            effective_sample_size_min=float(sizes.min()),
            effective_sample_size_median=float(np.median(sizes)),
            n_below_floor=len(low),
            thetas_below_floor=[h["theta"] for h in low],
        )
        if peak_efolds is not None:
            logz_max = max(h["logz"] for h in history)
            near = [h for h in low if logz_max - h["logz"] < peak_efolds]
            out["n_below_floor_near_peak"] = len(near)
            out["thetas_below_floor_near_peak"] = [h["theta"] for h in near]
        return out
