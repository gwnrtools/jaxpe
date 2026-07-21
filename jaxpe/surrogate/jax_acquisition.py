import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence
from gpry.gp_acquisition import NORA
from gpry.ns_interfaces import InterfaceBlackJAX


# --------------------------------------------------------------------------- kernels
# All GP-predictive numerics are written as *pure* functions of (query point, training
# arrays, kernel hyperparameters). Nothing is closed over as a constant, so the same
# compiled artifact is reused across GPry iterations even as the training set grows and
# the hyperparameters are refit -- the fix for the per-iteration XLA recompile that made
# the closure-based version pathological on GPU (a training set closed over as a baked
# constant forces a fresh compile every acquisition step). See ``docs/gpry_fusion_design``.


def _kernel_fn(kernel_name, nu):
    """Return a pure kernel ``k(x1, x2, length_scale, constant)`` matching GPry's.

    ``x1`` is a single (1, d) point and ``x2`` a (N, d) training block; the return is
    (N,). Only the anisotropic RBF and Matern (nu in {1/2, 3/2, 5/2}) families GPry uses
    are supported; anything else raises rather than silently mis-modelling the kernel.
    """
    if kernel_name == "RBF":

        def k(x1, x2, length_scale, constant):
            dist_sq = jnp.sum(((x1 - x2) / length_scale) ** 2, axis=-1)
            return constant * jnp.exp(-0.5 * dist_sq)

        return k

    if kernel_name == "Matern":
        if nu not in (0.5, 1.5, 2.5):
            raise NotImplementedError(
                f"Matern nu={nu} not supported in JAX acquisition (only 1/2, 3/2, 5/2)."
            )

        def k(x1, x2, length_scale, constant):
            dist = jnp.sqrt(jnp.sum(((x1 - x2) / length_scale) ** 2, axis=-1) + 1e-12)
            if nu == 1.5:
                kk = (1.0 + jnp.sqrt(3.0) * dist) * jnp.exp(-jnp.sqrt(3.0) * dist)
            elif nu == 2.5:
                kk = (1.0 + jnp.sqrt(5.0) * dist + 5.0 / 3.0 * dist**2) * jnp.exp(
                    -jnp.sqrt(5.0) * dist
                )
            else:  # nu == 0.5 (exponential / Ornstein-Uhlenbeck)
                kk = jnp.exp(-dist)
            return constant * kk

        return k

    raise NotImplementedError(f"Kernel {kernel_name} not supported in JAX acquisition.")


def _predictive_mean_fn(kernel_name, nu):
    """Return a pure GP posterior *mean* ``predict(x, X_train, alpha, W, b, length_scale,
    constant, y_mean, y_std)`` -- all GP state passed as arguments (not closed over).

    The mean is ``sum_i k(x, X_i) alpha_i`` in the y-preprocessed space, de-standardised
    by ``y_std, y_mean``; ``x`` is mapped into the X-preprocessed space by the affine
    ``x W + b``. Because the mean is linear in ``alpha``, zero-padding ``alpha`` (and the
    matching ``X_train`` rows) contributes exactly zero -- so a fixed training capacity
    can be used without changing the result (see ``extract_predictive_params``).
    """
    k = _kernel_fn(kernel_name, nu)

    def predict(x, X_train, alpha, W, b, length_scale, constant, y_mean, y_std):
        x_scaled = jnp.dot(x, W) + b
        K_vec = k(x_scaled[None, :], X_train, length_scale, constant)
        return jnp.dot(K_vec, alpha) * y_std + y_mean

    return predict


def extract_predictive_params(surrogate, pad_to=None):
    """Pull the fitted GP-predictive state out of a GPry SurrogateModel as JAX arrays.

    Returns ``(kernel_name, nu, params)`` where ``params`` is a dict of the argument
    arrays ``_predictive_mean_fn`` expects. If ``pad_to`` is given, the training set is
    padded to that many rows with zero ``alpha`` (exact: the padded rows drop out of the
    mean), giving a *fixed* shape so a jitted predictive is not retraced as N grows.
    """
    X_train = np.asarray(surrogate.gpr.X_train_, dtype=float)
    alpha = np.asarray(surrogate.gpr.alpha_, dtype=float).ravel()
    n, d = X_train.shape[0], surrogate.d
    if pad_to is not None:
        if pad_to < n:
            raise ValueError(f"pad_to={pad_to} < current training size {n}")
        X_padded = np.zeros((pad_to, d))
        X_padded[:n] = X_train
        alpha_padded = np.zeros(pad_to)
        alpha_padded[:n] = alpha
        X_train, alpha = X_padded, alpha_padded

    prep = surrogate.preprocessing_X
    b = prep.transform(np.zeros((1, d)))[0]
    W = prep.transform(np.eye(d)) - b

    prep_y = surrogate.preprocessing_y
    if hasattr(prep_y, "mean_") and hasattr(prep_y, "std_"):
        y_mean, y_std = float(np.asarray(prep_y.mean_)), float(np.asarray(prep_y.std_))
    else:
        y_mean, y_std = 0.0, 1.0

    kernel = surrogate.gpr.kernel_
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        constant, base_kernel = kernel.k1.constant_value, kernel.k2
    else:
        constant, base_kernel = 1.0, kernel
    kernel_name = base_kernel.__class__.__name__
    nu = getattr(base_kernel, "nu", None)

    params = dict(
        X_train=jnp.asarray(X_train),
        alpha=jnp.asarray(alpha),
        W=jnp.asarray(W),
        b=jnp.asarray(b),
        length_scale=jnp.asarray(np.atleast_1d(base_kernel.length_scale), dtype=float),
        constant=jnp.asarray(float(constant)),
        y_mean=jnp.asarray(y_mean),
        y_std=jnp.asarray(y_std),
    )
    return kernel_name, nu, params


def build_jax_predictive_mean(surrogate, pad_to=None):
    """Build a JAX predictive-mean closure from a fitted GPry surrogate.

    Backward-compatible thin wrapper over ``_predictive_mean_fn`` +
    ``extract_predictive_params``; ``pad_to`` pads the training set to a fixed capacity
    (exact). The returned closure takes a single (d,) query point and returns the scalar
    posterior mean.
    """
    kernel_name, nu, p = extract_predictive_params(surrogate, pad_to=pad_to)
    predict = _predictive_mean_fn(kernel_name, nu)

    def jax_predictive_mean(x_test):
        return predict(
            x_test,
            p["X_train"],
            p["alpha"],
            p["W"],
            p["b"],
            p["length_scale"],
            p["constant"],
            p["y_mean"],
            p["y_std"],
        )

    return jax_predictive_mean


def _bucket_size(n, bucket=64):
    """Round a training-set size up to the next multiple of ``bucket``.

    Padding the GP training arrays to a bucketed capacity keeps the jitted predictive at
    a *fixed* input shape across many GPry iterations, so it is compiled once per bucket
    rather than once per iteration (the N grows every iteration -> retrace pathology).
    """
    return int(np.ceil(max(int(n), 1) / bucket) * bucket)


class JAXInterfaceBlackJAX(InterfaceBlackJAX):
    """BlackJAX NSS interface with a JAX-native GP predictive.

    Two things distinguish it from the stock ``InterfaceBlackJAX``:

    * the acquisition log-likelihood is a real JAX function (no ``pure_callback``
      point-by-point escape to numpy), and
    * (``run_predictive``) the GP-predictive state is passed to a *cached* jitted NS
      step as **arguments** -- padded to a fixed capacity -- so the compiled artifact is
      reused across GPry iterations instead of being retraced every iteration as the
      training set grows and hyperparameters are refit. That per-iteration recompile was
      the dominant cost of the closure-based path (pathological on GPU).

    The closure-based ``run`` (below) is retained as a generic fallback for an arbitrary
    ``logp_func``; the pipeline uses ``run_predictive``.
    """

    def _resolve_param_names(self, param_names):
        from gpry.tools import generic_params_names

        if param_names is None:
            return list(generic_params_names(self.dim))
        if isinstance(param_names[0], (list, tuple)):
            return [p[0] for p in param_names]
        if not isinstance(param_names, Sequence):
            raise ValueError("'param_names' must be a list of parameter names.")
        return list(param_names)

    def _ns_loop_and_finalise(self, step_callable, state, param_names, rng_key):
        """Run the NSS outer loop with ``step_callable(step_key, state) -> (state, info)``
        and return ``(X_mc, y_mc, w_mc)``. Shared by ``run`` and ``run_predictive``."""

        ns_utils = self.globals["ns_utils"]
        nlive = self.precision_settings["nlive"]
        max_steps = self.precision_settings["max_steps"]
        num_delete = self.precision_settings["num_delete"]
        precision_criterion = self.precision_settings["precision_criterion"]

        dead_list = []
        dead_logls = []
        for i in range(max_steps):
            rng_key, step_key = jax.random.split(rng_key)
            state, info = step_callable(step_key, state)
            dead_list.append(info)
            dead_logls.extend(np.sort(np.asarray(info.particles.loglikelihood).ravel()))

            if precision_criterion is not None and i > 0:
                check_every = max(1, nlive // max(1, num_delete))
                if (i + 1) % check_every == 0:
                    should_stop, frac_remain = self._stop_by_remaining_evidence(
                        dead_logls=dead_logls,
                        live_logl=np.asarray(state.particles.loglikelihood),
                        nlive=nlive,
                        precision_criterion=precision_criterion,
                    )
                    if should_stop:
                        if self.verbose > 3:
                            print(
                                f"JAX BlackJAX NS converged at step {i + 1}, "
                                f"remaining evidence fraction = {frac_remain:.4g}"
                            )
                        break

        dead_info = ns_utils.finalise(state, dead_list, update_info=False)
        rng_key, weight_key = jax.random.split(rng_key)
        logw = ns_utils.log_weights(weight_key, dead_info).mean(axis=-1)
        particles = dead_info.particles

        X_mc = np.column_stack(
            [np.asarray(particles.position[name]) for name in param_names]
        )
        y_mc = np.asarray(particles.loglikelihood)
        w_mc = np.exp(logw - np.max(logw))
        w_mc = w_mc / np.sum(w_mc)
        self.X_MC, self.y_MC, self.w_MC = X_mc, y_mc, w_mc
        return X_mc, y_mc, w_mc

    def run(self, logp_func, param_names=None, out_dir=None, seed=None):
        """Generic fallback: run NSS on an arbitrary JAX ``logp_func`` (closure-based,
        recompiles per call as the closure changes -- see ``run_predictive``)."""
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        nss_api = self.globals["nss_api"]
        ns_utils = self.globals["ns_utils"]
        nlive = self.precision_settings["nlive"]
        num_inner_steps = self.precision_settings["num_inner_steps"]
        num_delete = self.precision_settings["num_delete"]

        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng_key = jax.random.PRNGKey(seed)
        param_names = self._resolve_param_names(param_names)
        bounds_dict = {
            name: (float(self.bounds[i, 0]), float(self.bounds[i, 1]))
            for i, name in enumerate(param_names)
        }
        rng_key, init_key = jax.random.split(rng_key)
        particles, logprior_fn = ns_utils.uniform_prior(init_key, nlive, bounds_dict)

        def loglikelihood_fn(params):
            x = jnp.array([params[name] for name in param_names])
            return logp_func(x)

        algorithm = nss_api(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=num_inner_steps,
            num_delete=num_delete,
        )
        step_fn = jax.jit(algorithm.step)
        rng_key, init_key = jax.random.split(rng_key)
        state = algorithm.init(particles, rng_key=init_key)
        X_mc, y_mc, w_mc = self._ns_loop_and_finalise(
            step_fn, state, param_names, rng_key
        )
        return X_mc, y_mc, w_mc, None, None

    def _get_cached_kernels(
        self, kernel_name, nu, param_names, logprior_fn, mean_func, cache_key
    ):
        """Build (once) and cache the jitted ``init`` and ``step`` for a given predictive
        signature. Both take the GP-predictive arrays as *arguments*, so the compiled
        artifact is reused across iterations (and different training-set values)."""

        cache = self._compiled_runtime_cache
        if cache_key in cache:
            return cache[cache_key]

        nss_api = self.globals["nss_api"]
        num_inner_steps = self.precision_settings["num_inner_steps"]
        num_delete = self.precision_settings["num_delete"]
        predict = _predictive_mean_fn(kernel_name, nu)
        pnames = tuple(param_names)

        def _algo(args):
            def loglikelihood_fn(params):
                x = jnp.array([params[name] for name in pnames])
                mean = predict(x, *args)
                if mean_func is not None:
                    mean = mean + mean_func(x[None, :])[0]
                return mean

            return nss_api(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                num_inner_steps=num_inner_steps,
                num_delete=num_delete,
            )

        @jax.jit
        def init_fn(particles, rng_key, *args):
            return _algo(args).init(particles, rng_key=rng_key)

        @jax.jit
        def step_fn(rng_key, state, *args):
            return _algo(args).step(rng_key, state)

        cache[cache_key] = (init_fn, step_fn)
        return init_fn, step_fn

    def run_predictive(
        self, kernel_name, nu, params, param_names=None, mean_func=None, seed=None
    ):
        """Run NSS over the GP posterior mean, reusing a cached jitted step.

        ``params`` are the padded predictive arrays from ``extract_predictive_params``
        (in the fixed positional order the pure predictive expects). The jitted init/step
        are keyed on the *shapes*/structure, not the values, so a run at the same bucket
        capacity and precision reuses the compiled artifact.
        """

        jax.config.update("jax_enable_x64", True)
        ns_utils = self.globals["ns_utils"]
        nlive = self.precision_settings["nlive"]
        num_inner_steps = self.precision_settings["num_inner_steps"]
        num_delete = self.precision_settings["num_delete"]

        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng_key = jax.random.PRNGKey(seed)
        param_names = self._resolve_param_names(param_names)
        bounds_dict = {
            name: (float(self.bounds[i, 0]), float(self.bounds[i, 1]))
            for i, name in enumerate(param_names)
        }
        rng_key, init_key = jax.random.split(rng_key)
        particles, logprior_fn = ns_utils.uniform_prior(init_key, nlive, bounds_dict)

        args = (
            params["X_train"],
            params["alpha"],
            params["W"],
            params["b"],
            params["length_scale"],
            params["constant"],
            params["y_mean"],
            params["y_std"],
        )
        # Cache key: everything that changes the compiled program. The training-set
        # capacity (padded) enters via X_train.shape; values do not.
        cache_key = (
            kernel_name,
            nu,
            tuple(param_names),
            int(params["X_train"].shape[0]),
            nlive,
            num_inner_steps,
            num_delete,
            None if mean_func is None else id(mean_func),
            tuple(map(tuple, np.asarray(self.bounds).tolist())),
        )
        init_fn, step_fn = self._get_cached_kernels(
            kernel_name, nu, param_names, logprior_fn, mean_func, cache_key
        )

        rng_key, init_key = jax.random.split(rng_key)
        state = init_fn(particles, init_key, *args)

        def step_callable(step_key, st):
            return step_fn(step_key, st, *args)

        X_mc, y_mc, w_mc = self._ns_loop_and_finalise(
            step_callable, state, param_names, rng_key
        )
        return X_mc, y_mc, w_mc, None, None


class JAXNORA(NORA):
    """JAX-accelerated NORA acquisition using JAX-compiled BlackJAX nested sampling."""

    def _init_nested_sampler(self, sampler=None):
        this_sampler = sampler or self.sampler
        if this_sampler == "blackjax":
            self.sampler_interface = JAXInterfaceBlackJAX(self.bounds_, self.verbose)
            self.sampler = "blackjax"
        else:
            super()._init_nested_sampler(this_sampler)

    def _do_mc_sample_blackjax(self, surrogate, bounds=None, rng=None):
        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        self.sampler_interface.set_precision(**self.update_NS_precision(surrogate))
        seed = rng.integers(2**31 - 1) if rng is not None else None

        # Pad the training set to a bucketed capacity so the jitted NS step keeps a fixed
        # input shape across GPry iterations (compile reused per bucket, not per
        # iteration); alpha-zero padding leaves the posterior mean exact.
        n_train = surrogate.gpr.X_train_.shape[0]
        kernel_name, nu, params = extract_predictive_params(
            surrogate, pad_to=_bucket_size(n_train)
        )
        mean_func = getattr(surrogate, "mean_func", None)

        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run_predictive(
            kernel_name, nu, params, mean_func=mean_func, seed=seed
        )
        self.sampler_interface.delete_output()
        return X_mc, None, None, w_mc, logZ, logZstd
