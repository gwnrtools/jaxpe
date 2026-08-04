---
title: surrogate
parent: API Reference
layout: default
nav_order: 8
---

# Sec. VIII: Active Learning and Surrogate Inference (`jaxpe.surrogate`)
{: .no_toc }

1. TOC
{:toc}

---

## The regime this module exists for

Every other sampler in `jaxpe` rests on one assumption: the log-posterior is a JAX
function, so $$\nabla_\theta \ln \pi$$ costs about as much as $$\ln \pi$$ itself by
reverse-mode automatic differentiation. That assumption buys HMC, MALA and the
normalizing-flow proposals of [`jaxpe.sampler`]({{ site.baseurl }}/docs/api/sampler.html).

It fails completely for effective-one-body (EOB) waveform families — `SEOBNRv5`,
`TEOBResumS` — whose generators are compiled C or Cython libraries wrapped in Python.
They are not traceable, not differentiable, and not vectorizable across chains. A single
call costs $$13$$–$$800$$ ms across stellar-mass binary black holes, against the
microseconds of a vectorized `IMRPhenomD` evaluation. Under those conditions the entire
cost model of gradient MCMC inverts, and the correct strategy is no longer *take more
steps* but *take fewer, better-chosen likelihood evaluations*.

`jaxpe.surrogate` implements that strategy: build a probabilistic **surrogate** of the
expensive log-likelihood from a few hundred carefully chosen evaluations, sample the
surrogate cheaply, and correct the result back to the exact posterior by importance
reweighting. The design rationale, the profiling that motivates it, and the measured
cost breakdown live in
[the GPry-fusion design note]({{ site.baseurl }}/docs/ongoing/gpry_fusion_design.html); this page
documents the interface and the mathematics.

---

## Gaussian-process regression of a log-likelihood

Place a Gaussian-process prior directly on the log-likelihood as a function of the
intrinsic parameters $$\theta \in \mathbb{R}^d$$:

$$
\ln \mathcal{L}(\theta) \;\sim\; \mathcal{GP}\big(m(\theta),\, k(\theta, \theta')\big),
$$

with mean function $$m$$ and covariance kernel $$k$$. Regressing the *log*-likelihood
rather than the likelihood is deliberate: $$\mathcal{L}$$ spans tens of orders of
magnitude across a prior volume and is essentially zero almost everywhere, which no
stationary kernel models well, whereas $$\ln \mathcal{L}$$ is a smooth, bounded-curvature
surface over the region that matters.

Given $$N$$ truth evaluations $$\mathcal{D} = \{(\theta_i, y_i)\}$$ with
$$y_i = \ln\mathcal{L}(\theta_i)$$, the posterior predictive at a query point
$$\theta_*$$ is the standard conditional Gaussian,

$$
\mu(\theta_*) = m(\theta_*) + \mathbf{k}_*^{\mathsf T} \big(K + \sigma_n^2 I\big)^{-1}
\big(\mathbf{y} - m(\Theta)\big),
\qquad
\sigma^2(\theta_*) = k(\theta_*, \theta_*) - \mathbf{k}_*^{\mathsf T}
\big(K + \sigma_n^2 I\big)^{-1} \mathbf{k}_*,
$$

where $$K_{ij} = k(\theta_i, \theta_j)$$ and $$(\mathbf{k}_*)_i = k(\theta_*, \theta_i)$$.
Writing $$\alpha = (K + \sigma_n^2 I)^{-1}(\mathbf{y} - m(\Theta))$$ — computed once per
refit by Cholesky factorization at $$O(N^3)$$ — the predictive mean collapses to a single
weighted kernel sum,

$$
\mu(\theta_*) = m(\theta_*) + \sum_{i=1}^{N} \alpha_i \, k(\theta_*, \theta_i),
$$

which costs $$O(N)$$ per query. This identity is what makes the JAX port below both fast
and *exactly* paddable.

The kernel families supported are exactly those GPry uses, and
`jaxpe.surrogate.jax_acquisition` raises rather than silently mis-modelling anything
else: the anisotropic radial basis function

$$
k_{\rm RBF}(\theta, \theta') = C \exp\!\left(-\tfrac{1}{2} \sum_{j=1}^{d}
\frac{(\theta_j - \theta'_j)^2}{\ell_j^2}\right),
$$

and the Matérn family at $$\nu \in \{1/2,\, 3/2,\, 5/2\}$$, whose sample paths are
$$\lceil \nu \rceil - 1$$ times differentiable. Anisotropic length scales $$\ell_j$$ are
essential here: the chirp-mass direction of a compact-binary posterior is orders of
magnitude narrower than the spin directions, and an isotropic kernel would be forced to
adopt the smallest scale everywhere, destroying the sample efficiency the surrogate
exists to provide.

### Active learning: where to spend the next evaluation

The surrogate is only worth building if the $$N$$ evaluations are chosen well. Rather
than sampling the prior, active learning selects each new $$\theta_{N+1}$$ by maximizing
an **acquisition function** that trades exploitation (regions of high predicted
likelihood) against exploration (regions of high predictive variance). GPry's default is
of the form

$$
a(\theta) \;=\; \mu(\theta) + \zeta\, \sigma(\theta),
$$

evaluated under the surrogate rather than the truth, so optimizing it is free relative to
a real waveform call. Two optimizers are exposed:

- **NORA** — nested sampling over the acquisition surface. Robust and global, but it
  costs many thousands of *surrogate* evaluations per acquisition step.
- **`BatchOptimizer`** — multi-start L-BFGS against the analytic acquisition gradient.
  Far fewer surrogate evaluations, at the cost of relying on local ascent from a finite
  set of starts.

The choice matters more than it might appear. Measured across the design note's sweep,
**acquisition, not waveform generation, is $$70$$–$$77\%$$ of the loop** — the surrogate
machinery, not the expensive physics, dominates the wall clock once the physics is only
called a few hundred times.

An SVM **infinities classifier** runs alongside the regressor, learning the boundary of
the region where the likelihood is finite so that acquisition never proposes points in
the excluded volume. jaxpe does not reimplement it; it is GPry's.

---

## Correctness: importance sampling against the truth

A surrogate posterior is not the posterior. `jaxpe.surrogate` never presents it as one.
Let $$q(\theta)$$ be the surrogate posterior actually sampled — recorded per-sample in
`SurrogateSamples.logpost` — and let $$\pi(\theta) \propto \mathcal{L}(\theta)\, p(\theta)$$
be the exact target. Self-normalized importance sampling assigns

$$
w_i \;\propto\; \frac{\pi(\theta_i)}{q(\theta_i)},
\qquad
\mathbb{E}_\pi[f] \;\approx\; \frac{\sum_i w_i f(\theta_i)}{\sum_i w_i},
$$

which is consistent for *any* proposal $$q$$ with support covering $$\pi$$, however
inaccurate the surrogate. Surrogate error therefore degrades the *efficiency* of the
estimator, not its target — the same guarantee that lets the normalizing flow in
[`jaxpe.sampler`]({{ site.baseurl }}/docs/api/sampler.html) be arbitrarily bad without
biasing the chain.

The price is paid in effective sample size. With normalized weights
$$\bar w_i = w_i / \sum_j w_j$$, Kish's estimate

$$
N_{\rm eff} \;=\; \frac{1}{\sum_i \bar w_i^{\,2}}
\;=\; \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^{\,2}}
$$

collapses toward $$1$$ when the surrogate misplaces posterior mass, because a single
sample then carries nearly all the weight. **$$N_{\rm eff}$$ is the diagnostic that
decides whether a surrogate run is usable**, and it is the reason `SurrogateSamples`
carries `logpost` as a first-class field rather than discarding the proposal density
after sampling. Reweighting costs one true likelihood call per retained sample, so it is
budgeted like any other truth evaluation.

---

## The interface

The seam between `jaxpe` and any active-learning backend is deliberately narrow — four
operations, no speculative generality:

```python
class SurrogateEngine(Protocol):
    def run(self) -> dict: ...                                  # drive active learning
    def surrogate_logp(self, x: np.ndarray) -> np.ndarray: ...  # for IS reweighting
    def sample(self) -> SurrogateSamples: ...                   # draw from the surrogate
    def diagnostics(self) -> dict: ...                          # eval count, convergence
```

Everything crossing this seam is **host-side NumPy by construction**. The expensive
likelihood is an opaque Python callable that must never enter a JAX trace — that is not a
style preference but the defining property of the regime, and violating it (by jitting a
wrapper around an EOB generator, say) fails at trace time rather than degrading quietly.

---

## Making acquisition compile once, not every iteration

`jaxpe.surrogate.jax_acquisition` is the one component jaxpe does implement rather than
delegate, and it exists to fix a specific pathology.

Active learning grows its training set by one point per iteration. A naive JAX port of
the GP predictive closes over $$\Theta$$ and $$\alpha$$ as traced constants, so **every
iteration presents XLA with a new shape and triggers a full recompile** — and because the
hyperparameters are refit as well, even a shape-stable version would invalidate the cache.
On GPU the compile dominated the acquisition it was meant to accelerate.

The fix has two parts, both visible in the module's structure:

**1. Purity.** Every predictive numeric is written as a pure function of
`(query point, training arrays, kernel hyperparameters)`. Nothing is closed over. The
same compiled artifact is therefore valid as the training set grows and the
hyperparameters change, because those enter as *arguments*.

**2. Exact padding.** Arguments still have shapes, so $$N$$ growing by one would still
retrace. `extract_predictive_params(surrogate, pad_to=...)` pads the training block to a
fixed capacity, rounded up in buckets (`_bucket_size`, default $$64$$), filling the extra
rows with $$\alpha_i = 0$$. By the kernel-sum identity above,

$$
\mu(\theta_*) = m(\theta_*) + \sum_{i=1}^{N_{\rm pad}} \alpha_i \, k(\theta_*, \theta_i),
\qquad \alpha_i = 0 \;\; \text{for} \;\; i > N,
$$

so **the padded rows contribute exactly zero** — this is an algebraic identity, not a
numerical approximation, and the padded predictive is bit-comparable to the unpadded one.
A run therefore compiles once per bucket boundary rather than once per iteration.

`JAXInterfaceBlackJAX` then feeds this predictive to BlackJAX nested sampling as a real
JAX function, eliminating the `pure_callback` escape to NumPy that would otherwise
serialize the acquisition sampler point-by-point. `JAXNORA` subclasses GPry's `NORA` to
select that interface while inheriting the rest of GPry's acquisition logic unchanged.

> **Status.** The JAX acquisition path is experimental and is selected by
> `jax_acquisition=True`, which is mutually exclusive with an explicit `acquisition=`
> specification. The default path leaves GPry on its native NORA.

---

## Multifidelity: regress the discrepancy, not the likelihood

When a cheap, differentiable model of the same physics exists — `IMRPhenomD` standing in
for an EOB waveform — the GP should not be asked to learn $$\ln\mathcal{L}$$ from scratch.
`MultifidelityGaussianProcessRegressor` instead uses the cheap model's log-likelihood as
the GP **prior mean**, so the process models only the discrepancy between fidelities:

$$
\ln\mathcal{L}_{\rm exp}(\theta) \;=\; \underbrace{m(\theta)}_{\text{cheap, JAX}}
\;+\; g(\theta), \qquad g \sim \mathcal{GP}\big(0,\, k(\theta,\theta')\big).
$$

`fit` subtracts the mean function and regresses the residuals $$y - m(\Theta)$$; `predict`
adds it back. The advantage is statistical, not computational: two waveform families that
model the same binary agree to a phase difference that is small and slowly varying, so
$$g$$ has far lower amplitude and curvature than $$\ln\mathcal{L}$$ itself and is learnable
from correspondingly fewer expensive evaluations. The GP's job shrinks from *the physics*
to *the correction*.

Because $$m$$ is a JAX function, $$\nabla_\theta m$$ is available by automatic
differentiation, and the class precomputes it (`jax.grad`) so gradient-based acquisition
can differentiate the full predictive mean — the cheap-model term analytically, the GP
term through the kernel.

> **Numerical caution.** The prior mean is only a help where the two fidelities actually
> agree. Where the cheap model is qualitatively wrong — near merger for a strongly
> precessing or highly eccentric system — the residual inherits that structure, and a
> stationary kernel will model it with a short length scale that erases the sample-
> efficiency gain. The multifidelity path is a bet on model agreement and should be
> validated per waveform pair rather than assumed.

---

## API Reference

### `SurrogateSamples`
**`jaxpe.surrogate.SurrogateSamples`**

`NamedTuple` of Monte-Carlo samples drawn from a surrogate posterior.

| field | shape | meaning |
|---|---|---|
| `x` | `(n, d)` | sample positions, columns ordered as `names` |
| `weights` | `(n,)` | non-negative sample weights (all ones for unweighted chains) |
| `logpost` | `(n,)` | surrogate log-posterior at the samples — the proposal density $$q$$ that importance reweighting divides by |
| `names` | `tuple` | parameter names, one per column of `x` |

### `SurrogateEngine`
**`jaxpe.surrogate.SurrogateEngine`**

A `runtime_checkable` `Protocol` declaring the four operations the pipeline requires:
`run()`, `surrogate_logp(x)`, `sample()`, `diagnostics()`. Any object providing them is a
valid backend; there is no base class to inherit.

### `GPryEngine`
**`jaxpe.surrogate.GPryEngine`**

Active-learning surrogate of an expensive log-likelihood, wrapping `gpry.Runner`. Imported
lazily via a module-level `__getattr__`, so `jaxpe` works without the optional dependency
(`pip install jaxpe[surrogate]`).

```python
GPryEngine(
    loglike,                      # (d,) -> float, host-side Python only
    bounds,                       # {name: (low, high)} ordered, or (d, 2) array
    checkpoint=None,              # path; GPry requires load_checkpoint when set
    load_checkpoint="resume",     # "resume" or "overwrite"
    verbose=1,
    options=None,                 # extra gpry.Runner kwargs, passed verbatim
    jax_acquisition=False,        # experimental JAX/BlackJAX NORA
    acquisition=None,             # native GPry acquisition, e.g. "BatchOptimizer"
)
```

The prior is uniform within `bounds`; transform parameters upstream if a non-uniform prior
is wanted. Passing a `dict` names the parameters, an array leaves them as `x_1 … x_d`.
`jax_acquisition` and `acquisition` are mutually exclusive.

Beyond the protocol it also exposes **`true_logp(x)`**, which calls the *real* likelihood
rather than the surrogate — the numerator of the importance weights.
`sample(sampler=None, add_options=None)` forwards to GPry's MC sampler.

### `MultifidelityGaussianProcessRegressor`
**`jaxpe.surrogate.multifidelity.MultifidelityGaussianProcessRegressor`**

`gpry.gpr.GaussianProcessRegressor` subclass taking a JAX-differentiable
`mean_func: (N, d) -> (N,)` as its prior mean; `fit` regresses residuals and `predict`
restores the mean. Extra `fit` keywords forward verbatim to GPry — note GPry takes
per-point noise as `noise_level`, **not** a scikit-learn-style `y_std`/`alpha`.

### JAX acquisition helpers
**`jaxpe.surrogate.jax_acquisition`**

| object | role |
|---|---|
| `extract_predictive_params(surrogate, pad_to=None)` | pull fitted GP state out as JAX arrays; returns `(kernel_name, nu, params)`, optionally zero-padded to fixed capacity |
| `build_jax_predictive_mean(surrogate, pad_to=None)` | convenience closure `(d,) -> scalar` posterior mean |
| `JAXInterfaceBlackJAX` | BlackJAX nested-sampling interface with a JAX-native GP predictive |
| `JAXNORA` | `gpry.gp_acquisition.NORA` subclass selecting the above interface |

---

### REFERENCES

1. Rasmussen, C. E. & Williams, C. K. I., *Gaussian Processes for Machine Learning*,
   MIT Press (2006).
2. El Gammal, J., Schöneberg, N., Torrado, J. & Fidler, C., *Fast and robust Bayesian
   inference using Gaussian processes with GPry*, JCAP **10** (2023) 021.
3. Kish, L., *Survey Sampling*, Wiley (1965) — the effective-sample-size estimator.
4. Owen, A. B., *Monte Carlo theory, methods and examples* (2013), Ch. 9 — self-normalized
   importance sampling and its variance.
5. Kennedy, M. C. & O'Hagan, A., *Predicting the output from a complex computer code when
   fast approximations are available*, Biometrika **87** (2000) 1 — the multifidelity
   prior-mean construction.
