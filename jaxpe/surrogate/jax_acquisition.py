import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence
from gpry.gp_acquisition import NORA
from gpry.ns_interfaces import InterfaceBlackJAX


def build_jax_predictive_mean(surrogate):
    """Builds a JAX-jittable predictive mean function from a fitted GPry surrogate."""
    import numpy as np

    X_train = jnp.array(surrogate.gpr.X_train_)
    alpha = jnp.array(surrogate.gpr.alpha_)

    d = surrogate.d
    prep = surrogate.preprocessing_X
    b_np = prep.transform(np.zeros((1, d)))[0]
    W_np = prep.transform(np.eye(d)) - b_np

    W = jnp.array(W_np)
    b = jnp.array(b_np)

    prep_y = surrogate.preprocessing_y
    if hasattr(prep_y, "mean_") and hasattr(prep_y, "std_"):
        y_mean_jax = jnp.array(prep_y.mean_)
        y_std_jax = jnp.array(prep_y.std_)
    else:
        y_mean_jax = jnp.array(0.0)
        y_std_jax = jnp.array(1.0)

    kernel = surrogate.gpr.kernel_
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        constant_value = kernel.k1.constant_value
        base_kernel = kernel.k2
    else:
        constant_value = 1.0
        base_kernel = kernel

    length_scale = jnp.array(base_kernel.length_scale)

    kernel_name = base_kernel.__class__.__name__
    if kernel_name == "Matern":
        nu = base_kernel.nu

        def k_func(x1, x2):
            dist = jnp.sqrt(jnp.sum(((x1 - x2) / length_scale) ** 2, axis=-1) + 1e-12)
            if nu == 1.5:
                K = (1.0 + jnp.sqrt(3.0) * dist) * jnp.exp(-jnp.sqrt(3.0) * dist)
            elif nu == 2.5:
                K = (1.0 + jnp.sqrt(5.0) * dist + 5.0 / 3.0 * dist**2) * jnp.exp(
                    -jnp.sqrt(5.0) * dist
                )
            else:
                K = jnp.exp(-dist)
            return constant_value * K

    elif kernel_name == "RBF":

        def k_func(x1, x2):
            dist_sq = jnp.sum(((x1 - x2) / length_scale) ** 2, axis=-1)
            return constant_value * jnp.exp(-0.5 * dist_sq)

    else:
        raise NotImplementedError(
            f"Kernel {kernel_name} not supported in JAX acquisition."
        )

    def jax_predictive_mean(x_test):
        x_scaled = jnp.dot(x_test, W) + b
        K_vec = k_func(x_scaled[None, :], X_train)
        y_mean = jnp.dot(K_vec, alpha)
        return y_mean * y_std_jax + y_mean_jax

    return jax_predictive_mean


class JAXInterfaceBlackJAX(InterfaceBlackJAX):
    """Overrides InterfaceBlackJAX to bypass jax.pure_callback for pure JAX logp_func."""

    def run(self, logp_func, param_names=None, out_dir=None, seed=None):
        import jax
        import jax.numpy as jnp
        from gpry.tools import generic_params_names

        jax.config.update("jax_enable_x64", True)
        nss_api = self.globals["nss_api"]
        ns_utils = self.globals["ns_utils"]
        nlive = self.precision_settings["nlive"]
        max_steps = self.precision_settings["max_steps"]
        num_inner_steps = self.precision_settings["num_inner_steps"]
        precision_criterion = self.precision_settings["precision_criterion"]
        num_delete = self.precision_settings["num_delete"]

        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng_key = jax.random.PRNGKey(seed)

        if param_names is None:
            param_names = generic_params_names(self.dim)
        elif isinstance(param_names[0], (list, tuple)):
            param_names = [p[0] for p in param_names]
        elif not isinstance(param_names, Sequence):
            raise ValueError("'param_names' must be a list of parameter names.")

        bounds_dict = {
            name: (float(self.bounds[i, 0]), float(self.bounds[i, 1]))
            for i, name in enumerate(param_names)
        }

        rng_key, init_key = jax.random.split(rng_key)
        particles, logprior_fn = ns_utils.uniform_prior(init_key, nlive, bounds_dict)

        # JAX-native log likelihood directly calling logp_func
        def loglikelihood_fn(params):
            x = jnp.array([params[name] for name in param_names])
            return logp_func(x)

        algorithm = nss_api(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=num_inner_steps,
            num_delete=num_delete,
        )
        init_fn = algorithm.init
        # JIT the BlackJAX step!
        step_fn = jax.jit(algorithm.step)

        rng_key, init_key = jax.random.split(rng_key)
        state = init_fn(particles, rng_key=init_key)

        dead_list = []
        dead_logls = []
        for i in range(max_steps):
            rng_key, step_key = jax.random.split(rng_key)
            state, info = step_fn(step_key, state)
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
        self.X_MC = X_mc
        self.y_MC = y_mc
        self.w_MC = w_mc
        return self.X_MC, self.y_MC, self.w_MC, None, None


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
        import warnings

        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        self.sampler_interface.set_precision(**self.update_NS_precision(surrogate))
        seed = rng.integers(2**31 - 1) if rng is not None else None

        jax_logp = build_jax_predictive_mean(surrogate)

        has_mean_func = hasattr(surrogate, "mean_func")
        if has_mean_func:
            mean_func = surrogate.mean_func

            def final_logp(x):
                return jax_logp(x) + mean_func(x[None, :])[0]

        else:
            final_logp = jax_logp

        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run(
            final_logp,
            out_dir=self._get_output_folder(),
            seed=seed,
        )
        self.sampler_interface.delete_output()
        y_mc = None
        return X_mc, y_mc, None, w_mc, logZ, logZstd
