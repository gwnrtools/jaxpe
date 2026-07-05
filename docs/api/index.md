---
title: API Reference
layout: default
nav_order: 4
has_children: true
---

# API Reference

This section provides the API documentation for the various components of `jaxpe`.

- [core]({{ site.baseurl }}/docs/api/core/): Priors, unconstraining transforms, and the `InferenceProblem` interface.
- [gw]({{ site.baseurl }}/docs/api/gw/): Detectors, PSDs, data handling, FD likelihood, and GW priors.
- [kernels]({{ site.baseurl }}/docs/api/kernels/): Local MCMC kernels + step-size/mass adaptation.
- [sampler]({{ site.baseurl }}/docs/api/sampler/): The global-local orchestration loop.
- [flows]({{ site.baseurl }}/docs/api/flows/): Normalizing-flow wrapper and trainer.
- [diagnostics]({{ site.baseurl }}/docs/api/diagnostics/): R-hat, ESS, corner/trace plots.
