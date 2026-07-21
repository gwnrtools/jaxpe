import numpy as np
import jax
from gpry.gpr import GaussianProcessRegressor


class MultifidelityGaussianProcessRegressor(GaussianProcessRegressor):
    """Gaussian Process Regressor with a custom prior mean function.

    This is used for multifidelity surrogate modeling, e.g. using a cheap
    waveform model's likelihood as the prior mean for the expensive model's GP surrogate.
    """

    def __init__(self, mean_func, *args, **kwargs):
        """
        Parameters
        ----------
        mean_func : callable
            A JAX-differentiable function mapping an array of shape (N, d) to (N,)
            representing the prior mean values.
        *args, **kwargs
            Passed to gpry.gpr.GaussianProcessRegressor.
        """
        super().__init__(*args, **kwargs)
        self.mean_func = jax.jit(mean_func)

        # GPry's predict only supports return_mean_grad=True when X has shape (1, d).
        # We prepare a gradient function for a single (d,) point.
        def single_point_func(x_single):
            return mean_func(x_single[None, :])[0]

        self.mean_func_grad = jax.jit(jax.grad(single_point_func))

    def fit(self, X, y, **kwargs):
        """Fit the GP to the residuals (y - mean_func(X)).

        Extra keyword arguments are forwarded verbatim to
        ``gpry.gpr.GaussianProcessRegressor.fit`` (``noise_level``,
        ``fit_hyperparameters``, ``validate``). Note GPry's fit does **not** accept a
        sklearn-style ``y_std``/``alpha``; per-point noise is passed as ``noise_level``.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        mean_vals = np.asarray(self.mean_func(X))
        y_res = y - mean_vals
        return super().fit(X, y_res, **kwargs)

    def predict(
        self,
        X,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
    ):
        """Predict using the GP posterior on the residuals, added back to the mean func."""
        X = np.asarray(X)
        out = super().predict(
            X,
            return_std=return_std,
            return_mean_grad=return_mean_grad,
            return_std_grad=return_std_grad,
            validate=validate,
        )

        # Unpack the output tuple depending on the boolean flags
        if return_std and return_mean_grad and return_std_grad:
            y_mean, y_std, y_mean_grad, y_std_grad = out
        elif return_std and return_mean_grad:
            y_mean, y_std, y_mean_grad = out
        elif return_std:
            y_mean, y_std = out
        elif return_mean_grad:
            y_mean, y_mean_grad = out
        else:
            y_mean = out

        # Add the prior mean function to the GP mean
        mean_vals = np.asarray(self.mean_func(X))
        y_mean = y_mean + mean_vals

        # Add the prior mean function gradient to the GP mean gradient
        if return_mean_grad:
            # GPry returns y_mean_grad with shape (d,)
            grad_vals = np.asarray(self.mean_func_grad(X[0]))
            y_mean_grad = y_mean_grad + grad_vals

        # Pack the output back into the correct tuple format
        if return_std and return_mean_grad and return_std_grad:
            return y_mean, y_std, y_mean_grad, y_std_grad
        elif return_std and return_mean_grad:
            return y_mean, y_std, y_mean_grad
        elif return_std:
            return y_mean, y_std
        elif return_mean_grad:
            return y_mean, y_mean_grad
        else:
            return y_mean
