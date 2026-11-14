# Retained versus current upstream rocBLAS

The question is whether the earlier custom FP32 matrix-vector kernel still has
an advantage over current upstream rocBLAS on an RX 9060 XT under native Windows.
This experiment does not measure tokens per second or run a language model.

## Controlled comparison

- Official source: ROCm/rocm-libraries develop commit
  `4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89`, pinned on 2026-11-14.
- Retained package: rocBLAS `5.8.0.3d8f334b`; new source declares `5.9.0`.
- Both arms use one identical benchmark executable and unchanged pinned HIP
  runtime, driver, synthetic inputs, dimensions, and GPU selection.
- Upstream is built with existing ROCm Core SDK 10.1.0 tools, Release/O3,
  gfx1200 only. Optional Tensile and hipBLASLt GEMM backends are disabled;
  native SGEMV implementation and dispatch are not modified.
- The retained binary's original build flags/toolchain are not independently
  reconstructed. This is a retained-package versus local-upstream-build
  comparison, not a proof that every difference is caused solely by a commit.
- The benchmark's only C++ change makes optional backend module-location checks
  conditional on the module being loaded. Both HIP and rocBLAS remain required
  adjacent modules. Numerical kernels, checks and timings are unchanged.

## Measurements and checks

One smoke run per library must pass before full measurements. Then four full
native-BLAS runs in retained/upstream/upstream/retained order, run sequentially.
Each run covers rows x reduction length: 4096x256, 65536x128, 65536x256,
65536x512 and 248144x256, in both storage layouts.

Follow-up recorded after the planned four runs: the 4096x256 row-major retained
session medians varied from 27.285 to 61.4725 us; the common custom control also
varied from 9.0375 to 18.215 us. One further retained/upstream pair was therefore
run with unchanged inputs/settings. All six full runs are included, with three
session medians per library. No faster or slower session was discarded. The
small case remains explicitly qualified for variability.

Row-major data uses rocblas_sgemv with transpose; column-major uses non-transpose.
The custom kernel is tested in each layout as a common control. Alpha=1,
beta=0, unit vector strides, FP32 and host pointer mode. Seeded bounded values
are exactly representable binary fractions. Two input matrices alternate.

Every output is compared against the custom implementation; selected first/last
rows are also checked against an independent double-precision CPU calculation.
The prior tolerance remains abs(error) <= 0.0002 + 0.00002*abs(reference).
Check finite outputs, 32-value canaries on both ends, and post-timing outputs.
Each full BLAS run checks 2,695,648 numerical values and 3,840 canary values.
Smoke also exercises Q8, odd row counts and an 8 MiB device copy.

Timing is the same-stream GPU wall-clock span enclosing eight calls in a HIP
graph, divided by eight. Two warm replays precede six alternating-order samples
per variant. Record enclosing host timing and calibration. Small warm data can
fit in cache; this is not a cold-memory bandwidth benchmark. No best-sample
selection or inference-speed extrapolation. Each process must finish and its
owned helpers close before the next begins; validate nonoverlapping windows.

## Safety and provenance

9060 XT only; no peer API or 9070 compute. Existing user-reported minus-30-percent
power setting is unchanged, not programmatically re-attested. Retain the prior
640 MiB owned-allocation cap, 1 GiB immediate headroom check, fresh allocation-free
HIP probe, and independently polled ADLX watchdog. No iCUE or general hardware
monitor. Record missing ADLX VRAM as unavailable, not zero.

All artifacts, receipts and raw logs are isolated. Source, compiler, executable,
and runtime hashes are recorded; protected production dependency hashes are
checked read-only before/after. The harness is not accessed. No system install,
driver/runtime update, tuning change, launcher edit, or public upload is part of
this experiment.
