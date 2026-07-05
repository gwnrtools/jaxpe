---
title: flows
parent: API Reference
layout: default
---

# `jaxpe.flows`
{: .no_toc }

1. TOC
{:toc}

The `flows` module wraps normalizing flows, predominantly using [flowjax](https://github.com/danielward27/flowjax), and provides training loops to learn global proposal distributions.

## Architecture

The standard choice in `jaxpe` is a rational-quadratic-spline coupling flow, which can effectively learn multi-modal distributions like the time-delay ring or phase degeneracies.

## Training

The flow is trained on accumulated chain samples during the warm-up phase. The trained flow is then used to drive Metropolis-Hastings *global* proposals.
