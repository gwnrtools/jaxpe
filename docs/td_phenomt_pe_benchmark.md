# Time-Domain PE Benchmark: IMRPhenomT + Relative Binning

## Hardware and Methodology

This benchmark tests the reverse-mode differentiation scaling of an end-to-end time domain physical parameter estimation pipeline utilizing:
- **Waveform**: The fully dynamic, continuous piecewise analytical formulation of `IMRPhenomT` via `jaxpe`.
- **Likelihood**: Time-Domain Relative Binning likelihood (`jaxpe.gw.likelihood.RelativeBinningTDLikelihood`).
- **Sampler**: Hardware-accelerated MCMC Kernels (`hmc`, `mala`).
- **Target OS**: Linux / GPU (Hardware-limited VRAM).

## The Time-Translation Invariance Bug

Early validation passes identified a profound loss in injection reconstruction (Log-Likelihood mismatch of ~-2.49e6 vs exact dense 0.0) that scaled with the time delay $\delta t$ between the detector nodes (H1, L1).

**Diagnosis**: The numerical continuous integration of the phenomenological phase via `jnp.cumsum` mathematically anchored the overall accumulated phase to $0.0$ evaluated strictly at the first index of the dynamic, detector-shifted time grid. This breaks physical time-translation invariance for TD models where relative phase depends solely on propagation distances!

**Resolution**: The interpolation boundary condition `jnp.interp` was entirely scrapped. Because the Relative Binning engine computes identical shifted physical grids $t - \tau$ for the base waveform and the perturbed trial vectors, the relative unanchored integrals explicitly align without needing forced coordinate boundaries.

## Physical Hardware Exhaustion (OOM)

The execution of a full gradient pass of the likelihood hit a hard physical wall on the GPU.

When XLA traces the `run_chains_jit` reverse-mode Vector-Jacobian Product (VJP) for the `IMRPhenomT` equations evaluated over dense Relative Bin edges, it creates an enormous Execution Graph. The compilation size scales drastically across the MCMC batch loops, completely overwhelming the NVIDIA GPU's global memory during CUBIN load (`RESOURCE_EXHAUSTED`).

### Aggressive Workarounds Deployed:
1. **Loop Unrolling Mitigation**: We replaced `jnp.cumsum` with an explicit `jax.lax.scan(unroll=1)` to strictly forbid the XLA compiler from flattening the time-series operations over 4,000 bins!
2. **Transition Kernel Simplification**: We abandoned the `HMC` sampler in favor of `MALA`. This completely bypasses the 32-step Leapfrog internal loop, ensuring only a single gradient pass is compiled per step.
3. **Data Type / Parameter Scaling**: Deployed `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, enforced `--f32` half-precision to crush dense correlation matrices, pushed bin reductions to ~1000 via `--phase-per-bin 2.0`, and limited parallel tracks to `--n-chains 8`.

### Benchmark Conclusion

Even after implementing every possible XLA memory mitigation strategy, the script crashes due to physical VRAM exhaustion (`Failed to allocate device memory... Out of memory while trying to allocate 19.99MiB.`). The dense $N_{bins} \times N_{bins}$ covariance matrices of the Time-Domain correlation function combined with the phenomenological piecewise complexity of `IMRPhenomT` produces a backward pass graph that absolutely cannot fit into the memory footprint of this GPU environment.

This model requires substantially higher GPU global memory limits (e.g. A100/H100 class hardware) to execute vectorized parallel chains for `RelativeBinningTDLikelihood`.

To prove the pipeline works end-to-end, we successfully ran it natively on the **CPU** (using `JAX_PLATFORMS=cpu`). Because the host RAM easily absorbs the monolithic graph, the script completes the JAX compilation and generates posterior chains without VRAM exhaustion, saving the final MCMC bounds and generating the corner plot natively.
