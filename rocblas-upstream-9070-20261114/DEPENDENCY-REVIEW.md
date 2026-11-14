# RX 9070 XT extension: dependency and safety review

2026-11-14. Reuse the already reviewed, clean official rocBLAS source at commit
`4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89` and existing ROCm Core SDK 10.1.0.
No additional source or dependency acquisition is required. The same-day
[9060 XT review](../rocblas-upstream-20261114/DEPENDENCY-REVIEW.md) applies to the
same runtime, compiler and RX 9000 driver family, including its limitations and
the recorded unresolved CVE-2026-0457. No CVE-free claim is made.

The added gfx1201 native-BLAS code pack is already present in the retained
runtime package. SHA-256:
`4928f4e39c7a9d25feb1fb617f41dd6599611b87d9ccb34ef4a1b2fc79ab92ad`.
It is copied privately; no production file is changed. Build-time downloads,
Tensile, hipBLASLt and upstream clients stay disabled. A separate build directory
targets gfx1201 with the same source, compiler and optimization settings. Only
the harmless device-export diagnostic is additionally suppressed in this build.

The operator confirmed a minus-30-percent power limit on the 9070 XT. Keep that
setting unchanged. This matches the reported relative limit on the 9060 XT;
it does not imply identical watts or establish performance at stock power.

The isolated guard is a hash-traced copy of the existing guard with its selected
device changed to the 9070 XT. Thermal limits remain 80 C core / 95 C hotspot,
with half-second sampling, a three-second stale-data abort and owned-process-only
cleanup. The 9070-specific sampled power stop is 300 W; this is a conservative
experimental abort threshold, not a hardware power setting or a guarantee.
AMD lists 304 W typical reference board power in its
[9070 XT specifications](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-9000-series/amd-radeon-rx-9070-xt.html).

Six idle telemetry samples (utilization at most 20%) precede each memory probe;
active unrelated work causes an abort rather than process termination. Keep
the fresh HIP memory probe, 2 GiB entry requirement, 640 MiB explicit-allocation
cap and 1 GiB immediate allocation headroom. No 9060 compute, peer communication,
driver update, global installation, harness access, production change, tuning
write or iCUE access is part of this extension.
