# Windows RDNA4 rocBLAS narrow GEMV: RX 9060 XT and RX 9070 XT

Date: 2026-11-14. Separate single-card tests, both at operator-confirmed minus-30-percent power limits. No dual-GPU inference or stock-power comparison is measured.

The specialized kernel outperformed the current upstream build on all five tested row-major/transposed shapes on both cards. The four larger shapes below show median speed ratios of 1.56-4.80x on the 9060 XT and 1.54-7.96x on the 9070 XT. Upstream rocBLAS won every column-major control. This identifies a specific dispatch/work-assignment opportunity; it does not establish a general replacement for rocBLAS.

## Direct comparison

FP32 row-major matrix-vector multiplication through the transposed rocblas_sgemv API. Times are GPU microseconds per operation, lower is better. Three session medians per library, six balanced-order samples per session. Custom timings below come from the upstream arm.

| GPU | Output rows x reduction | Retained rocBLAS us | Upstream rocBLAS us | Custom kernel us | Upstream / custom | Upstream time change vs retained |
|---|---|---:|---:|---:|---:|---:|
| RX 9060 XT | 65,536 x 128 | 430.86 | 523.01 | 109.05 | 4.80x | +21.4% |
| RX 9060 XT | 65,536 x 256 | 454.52 | 532.54 | 217.43 | 2.45x | +17.2% |
| RX 9060 XT | 65,536 x 512 | 581.63 | 667.84 | 428.79 | 1.56x | +14.8% |
| RX 9060 XT | 248,144 x 256 | 1592.16 | 1986.00 | 813.64 | 2.44x | +24.7% |
| RX 9070 XT | 65,536 x 128 | 200.78 | 230.38 | 28.93 | 7.96x | +14.7% |
| RX 9070 XT | 65,536 x 256 | 214.82 | 300.74 | 110.89 | 2.71x | +40.0% |
| RX 9070 XT | 65,536 x 512 | 297.26 | 334.28 | 216.37 | 1.54x | +12.5% |
| RX 9070 XT | 248,144 x 256 | 806.06 | 993.75 | 408.88 | 2.43x | +23.3% |

An asterisk marks more than 15% spread between at least one set of session medians; interpret that row with its individual sessions. All samples and small cases are retained. These ratios describe specific operations, not inference throughput.

## Layout controls and variability

- RX 9060 XT: upstream rocBLAS beats the custom implementation in 5 of 5 column-major cases.
- Variable case on RX 9060 XT: 4096 x 256, row_major_via_transpose. retained rocblas_sgemv: [27.285, 61.472, 61.398] us; retained custom_wave32_four_rows: [9.037, 18.215, 18.145] us; upstream rocblas_sgemv: [79.762, 79.390, 79.498] us; upstream custom_wave32_four_rows: [20.780, 18.675, 18.612] us.
- Variable case on RX 9060 XT: 65536 x 128, column_major_nontransposed. retained rocblas_sgemv: [110.797, 113.095, 112.510] us; retained custom_wave32_four_rows: [197.418, 373.010, 373.247] us; upstream rocblas_sgemv: [109.403, 111.655, 109.855] us; upstream custom_wave32_four_rows: [204.550, 377.195, 204.620] us.
- RX 9070 XT: upstream rocBLAS beats the custom implementation in 5 of 5 column-major cases.
- Variable case on RX 9070 XT: 4096 x 256, row_major_via_transpose. retained rocblas_sgemv: [14.018, 12.992, 31.432] us; retained custom_wave32_four_rows: [5.338, 4.965, 10.250] us; upstream rocblas_sgemv: [17.900, 17.890, 40.323] us; upstream custom_wave32_four_rows: [5.350, 5.340, 10.215] us.

The 15% flag is a descriptive reporting threshold, not a statistical significance test. On the 9070 XT, upstream 65,536 x 128 session medians span 226.60-256.82 us, and its custom control spans 28.75-30.52 us. Upstream 65,536 x 256 spans 265.56-301.98 us; its custom control spans 110.73-110.91 us. The table preserves their medians, but the retained/upstream percentage differences are not precise universal regression estimates.

The smaller row-major case is shown in the full per-card results rather than the main larger-matrix table. No run was discarded. Hardware caches, power state and background display work can affect short cases; no specific explanation for the observed variability has been established.

## Correctness and scope

All 16 guarded runs completed: one smoke suite per library plus six full BLAS suites per GPU. Combined checks: 16,192,768 Q8 bit comparisons, 32,451,992 numerical comparisons, and 74,752 canary values. Zero mismatches. Q8 checks are separate smoke coverage of unchanged synthetic Q8 kernels; the rocBLAS comparison itself is FP32.

Every FP32 output is compared between implementations; selected first/last rows are checked against an independent double-precision CPU calculation. Canaries, finite outputs and post-timing values are checked. The tolerance remains abs(error) <= 0.0002 + 0.00002*abs(reference). Inputs are deterministic bounded binary fractions, not exhaustive numerical corner cases.

Each timing is a same-stream GPU wall-clock interval enclosing eight calls inside a HIP graph, after two warm replays. Host timings, calibration and all trials are retained. These are warm synthetic calculations. No model weights, prompts, responses or production harness are used.

## Build and runtime

Official upstream source pinned at `4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89`; rocBLAS 5.9.0.4e81d94 versus retained 5.8.0.3d8f334b. Both upstream targets use the existing ROCm Core SDK 10.1.0 compiler and Release/O3, with optional unused Tensile/hipBLASLt GEMM backends disabled. Source is unmodified. The gfx1201 build additionally suppresses the device-export annotation diagnostic; arithmetic and optimization settings are unchanged.

Within each GPU comparison, retained and upstream use the same benchmark executable and pinned HIP runtime (reported version 71726405). Across GPUs, the benchmark is recompiled for gfx1200 or gfx1201 and asserts the corresponding device name. Source equality after normalizing only those identifiers is verified. The retained package original compiler flags are unknown; any retained/upstream difference is a local-build comparison, not proof of a source-only release regression.

The minus-30-percent setting is user-confirmed and left unchanged. Equal percentage settings do not imply equal watts. Removing that cap might improve power-limited operations, but no stock-power gain is established here.

## Bounded stability

- RX 9060 XT: sampled maxima 28 C core, 37 C hotspot and 77 W.
- RX 9070 XT: sampled maxima 40 C core, 45 C hotspot and 94 W.

Half-second telemetry can miss short bursts. These are short functional benchmarks, not endurance tests or dual-GPU stability validation. Owned-process cleanup and nonoverlapping run windows are checked. ADLX unavailable VRAM values are recorded as null; fresh HIP free-memory probes gate allocations. No peer access or second-card compute is requested.

Dependency/CVE review preceded acquisition in the 9060 experiment; the 9070 extension reused the same reviewed source and toolchain without downloads. Existing unresolved advisory caveats remain documented. No CVE-free claim is made. Harness, production launchers/builds, model files, Vulkan, drivers, GPU tuning and iCUE were not modified.

## Evidence

- [9070 full measurements](MEASUREMENTS.md) and [all 9070 samples](RESULTS.json).
- [9060 report](../rocblas-upstream-20261114/REPORT.md) and [all 9060 samples](../rocblas-upstream-20261114/RESULTS.json).
- [Combined sanitized numeric summary](TWO-CARD-RESULTS.json).
- [9070 protocol](PROTOCOL.md) and [dependency review](DEPENDENCY-REVIEW.md).

The focused development opportunity is narrow-reduction transposed FP32 GEMV dispatch/work assignment on the tested Windows RDNA4 configurations. A production library change needs broader alpha/beta, stride, dimension and numerical coverage. A GPU architecture-wide or inference-wide speedup has not been demonstrated.
