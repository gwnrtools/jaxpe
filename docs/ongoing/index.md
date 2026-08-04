---
title: Ongoing
layout: default
nav_order: 90
has_children: true
---

# Ongoing work

These pages are **not** library documentation. They are working notes kept under version
control alongside the code: design rationale for features being built, implementation
status against those designs, benchmark logs, and records of experiments that did not
pan out.

They are published rather than kept private for one reason — most of what they record is
*negative* or *retracted* results, and those are the expensive findings to reproduce. A
measurement that turned out to be an artifact of the configuration it was taken in is
worth more written down than deleted.

Read them for the reasoning and the measurements, not for a stable description of what
the library does. For that, see
[Getting Started]({{ site.baseurl }}/docs/getting-started.html) and the
[API Reference]({{ site.baseurl }}/docs/api/).

**These pages go stale.** Where an ongoing note and the API Reference disagree about
present behaviour, the API Reference is authoritative.

## Designs and status

- [Design — Fusing GPry into jaxpe]({{ site.baseurl }}/docs/ongoing/gpry_fusion_design.html):
  active-learning surrogate PE for expensive EOB waveforms; the design behind
  [`jaxpe.surrogate`]({{ site.baseurl }}/docs/api/surrogate.html).
- [Design — Relative binning]({{ site.baseurl }}/docs/ongoing/relative_binning_design.html):
  heterodyned likelihoods in the frequency and time domains.
- [Status — Relative binning implementation]({{ site.baseurl }}/docs/ongoing/relative_binning_status.html):
  what of that design is built and validated.

## Benchmarks

- [BNS PE with FD relative binning + HMC]({{ site.baseurl }}/docs/ongoing/bns_ce_pe_benchmark.html):
  end-to-end Cosmic Explorer benchmark, a five-kernel comparison, and a mass sweep.
- [TD PE with IMRPhenomT + relative binning]({{ site.baseurl }}/docs/ongoing/td_phenomt_pe_benchmark.html):
  time-domain pipeline, and the VRAM ceiling it currently hits.

## Experiments

- [Under Construction (Experiments)]({{ site.baseurl }}/docs/ongoing/under_construction.html)
- [ESIGMA — ISCO & inspiral termination]({{ site.baseurl }}/docs/ongoing/under_construction_esigma.html)
- [FD IMRPhenomD injection recovery]({{ site.baseurl }}/docs/ongoing/under_construction_fd.html)
