# Current upstream rocBLAS versus a narrow GEMV kernel on Windows RDNA4

Date: 2026-11-14. GPU: Radeon RX 9060 XT, gfx1200, native Windows.

The performance gap survives comparison with current upstream rocBLAS. On four
larger row-major FP32 matrix-vector cases, the experimental wave32 kernel is
**1.56x to 4.80x faster** than the locally built upstream library. This upstream
build takes **14.8% to 24.7% longer** than the retained library on those cases.
Both libraries still beat the custom kernel on every column-major case.

These are timings for individual synthetic operations, not tokens per second,
model accuracy results, or a system-wide ROCm speedup.

## Main results

Times are microseconds per operation, lower is better. Each value is the median
of three session medians; each session has six balanced-order timing samples.
The custom column uses measurements from the upstream arm. All inputs, benchmark
code, compiler settings for the benchmark, HIP runtime and GPU selection are
the same across the two library arms.

| Row-major matrix: output rows x reduction | Retained rocBLAS | Current upstream | Custom kernel | Custom speed advantage over upstream |
|---|---:|---:|---:|---:|
| 65,536 x 128 | 430.86 us | 523.01 us | 109.05 us | 4.80x |
| 65,536 x 256 | 454.52 us | 532.54 us | 217.43 us | 2.45x |
| 65,536 x 512 | 581.63 us | 667.84 us | 428.79 us | 1.56x |
| 248,144 x 256 | 1,592.16 us | 1,986.00 us | 813.64 us | 2.44x |

For example, the largest operation takes about 1.99 ms through upstream rocBLAS
and 0.81 ms through the custom kernel. This is a measured operation-level lead.
Its effect on a complete inference workload has not been measured here.

The opposite storage layout matters. For the 248,144 x 256 column-major case,
upstream rocBLAS takes 811.17 us and the custom kernel takes 970.50 us. Across
all five column-major shapes, rocBLAS remains faster than this custom kernel.
The four larger column-major native-library comparisons are approximately flat
to 2.4% faster for upstream; the smallest column-major case is about 13.9% slower.

### Variability retained in the evidence

The 4,096 x 256 row-major retained session medians were 27.285, 61.4725 and
61.3975 us. Its custom control also varied: 9.0375, 18.215 and 18.145 us.
Upstream medians were 79.7625, 79.3900 and 79.4975 us. The faster initial retained
session is preserved. Its cause was not established, and the small case is not
used for the stable 14.8%-24.7% headline comparison.

The custom column-major 65,536 x 128 timing also varies substantially, while the
native-library times for that case are much steadier. Full data and every
session median are available in [RESULTS.json](RESULTS.json) and the complete
layout table in [MEASUREMENTS.md](MEASUREMENTS.md). No session was discarded.

## What was actually built and changed

| Component | Comparison setup |
|---|---|
| Retained rocBLAS | 5.8.0.3d8f334b; existing private runtime copy |
| Upstream rocBLAS | 5.9.0.4e81d94; official source pinned at `4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89` |
| Upstream library build | Release/O3, gfx1200 only, unmodified upstream source |
| Optional GEMM backends | Tensile and hipBLASLt disabled in the upstream build; unused by SGEMV |
| HIP compiler/headers | Existing ROCm Core SDK 10.1.0; no SDK update |
| Loaded HIP runtime | Same pinned DLL in both arms; reported version 71726405 |
| Benchmark | One identical executable in both arms; existing numerical and timing code |
| Power setting | Existing user-reported minus-30-percent limit; unchanged, not reconfigured or independently re-attested |
| Isolation | Separate source, build and runtime folders; no system installation |

The benchmark's sole C++ adjustment makes module-location checks for optional
GEMM-backend DLLs conditional on whether those modules are loaded. HIP and
rocBLAS remain required adjacent DLLs. The numerical kernels, synthetic values,
reference checks, tolerances, allocation bounds and timing method are unchanged.
The upstream DLL imports the existing HIP runtime and Windows runtime libraries.
The acquired upstream worktree remains clean after the build and runs.

This is a scoped native-BLAS build, not a complete production replacement.
The retained package's original compiler and build flags were not reconstructed.
Consequently, the slower upstream result applies to **this local upstream build**;
it does not establish that the source changes alone caused a release regression.

SHA-256 provenance:

- Benchmark executable, both arms: `788914bfdd0c78b505708bed2a3782437ef860fec09d6bd15cbe1c280d5f9eb3`
- Retained rocBLAS DLL: `0843520978a4d9e8bc0ed26bd1e2bae7fadb446259c9c312e3f2cbc7b5452e61`
- Upstream rocBLAS DLL: `5ba05a7ea975ba3bf89f2ee45cc5175e0da09ac2ff40fee7da19019c0a0aa55a`
- Benchmark source: `732e187af0d86f9b692b98dfc6384fb90bafba90780fb7bef0eecb753a2a2fdd`

## Correctness and measurement controls

Both smoke suites and all six full native-BLAS suites completed successfully.
The full comparison checked **16,173,888 numerical values** and **23,040 buffer
canaries**. Including smoke checks, totals are **16,225,996 numerical comparisons**,
**8,096,384 Q8 bit comparisons**, and **37,376 canary checks**, with zero mismatches.
The two 8 MiB device-copy smoke checks also passed.

FP32 checks compare every output between implementations, independently check
selected first/last rows against a double-precision CPU calculation, reject
nonfinite outputs, check buffer boundaries, and repeat comparisons after timing.
Tolerance is unchanged: absolute error <= 0.0002 + 0.00002 times reference magnitude.
Inputs are deterministic bounded binary fractions. This coverage does not claim
all numerical corner cases or the complete rocBLAS API. The Q8 smoke checks are
for the unchanged synthetic Q8 kernels, not a new Q8 rocBLAS implementation.

Timing uses same-stream GPU wall-clock markers around eight calls in a HIP graph,
after two warm replays. Host timing and clock calibration are recorded alongside
all GPU samples. Calibration's host/device increment ratio ranged from 0.968 to
1.096. Small warm matrices can reside in cache. See [PROTOCOL.md](PROTOCOL.md) for
the predetermined comparison and the documented follow-up pair triggered by
small-case variability.

## Stability, dependency review and preservation

Only the 9060 XT performed compute. All eight guarded runs completed, their
time windows were verified not to overlap, and owned child processes exited
normally. No watchdog abort or HIP error occurred. Guard windows were 5.21-7.01
seconds, so these results are not an endurance or dual-GPU stability test.

Sampled maxima were 28 C core, 37 C hotspot and 77 W. Half-second telemetry can
miss short bursts; these are observed samples, not guaranteed physical peaks.
The HIP preflight reported 16,158 MiB free before each run. Maximum explicit
benchmark allocation was 486.551 MiB; runtime-internal allocations are additional.
ADLX's unavailable VRAM reading is retained as null rather than treated as zero.

The [dependency and advisory review](DEPENDENCY-REVIEW.md) preceded source
acquisition. Only pinned official source was acquired; existing tools and runtime
dependencies were reused. Automatic build-time dependency fetching was disabled.
The review records the existing unresolved driver advisory and makes no CVE-free
claim. Configuration initially needed an explicit device-library path; the final
build succeeded without source patches. Device-side export-attribute warnings
and a Windows linker warning about an ignored ELF-specific option were recorded.

The protected harness was not accessed. Production launchers/builds, model
files, Vulkan, driver installation, GPU tuning and iCUE were not modified.
Reviewed dependency hashes remained unchanged. Raw logs, private paths and build
receipts stay local; this report and its derived result data contain no usernames,
machine names, credentials, serial numbers, prompts, or model responses.

## What this supports next

This is evidence for a specific rocBLAS optimization opportunity: the narrow
reduction in transposed FP32 GEMV on this Windows/gfx1200 setup. Current source
still assigns a 256-thread block to an output in the relevant branch, whereas
the experimental control uses a wave per output and four outputs per block.
The newer upstream grid-stride change addresses oversized launches rather than
this work assignment. These are source observations, not an independent runtime
kernel-name trace; links and limitations are in [SOURCE-NOTES.md](SOURCE-NOTES.md).

The next useful development step is an isolated native rocBLAS dispatch/kernel
variant, followed by broader alpha/beta, stride, dimension and numerical tests.
The current specialized control is not ready to replace the general library API.
Any inference benefit would then need a separate representative end-to-end test.
