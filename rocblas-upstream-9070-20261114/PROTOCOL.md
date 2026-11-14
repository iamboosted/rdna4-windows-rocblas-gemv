# RX 9070 XT: retained versus current upstream rocBLAS

Run the same model-free benchmark as the 9060 XT comparison, separately on the
9070 XT. The operator confirmed a minus-30-percent power limit; leave it as-is.
This is single-card inference-library math testing, not dual-GPU inference.

Reuse official rocBLAS source commit
`4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89` and the existing SDK/compiler/runtime.
Build upstream rocBLAS for gfx1201 in a new directory. Retained rocBLAS remains
5.8.0.3d8f334b. Optional unused GEMM backends stay disabled in the upstream build.
No dependency acquisition, global install, driver/tuning change or production edit.

Within this GPU's comparison, both arms use one identical benchmark executable.
Relative to the 9060 executable, only its device identity/architecture assertions
and offload target change. Preparation verifies that reversing those identity
substitutions reproduces the original benchmark source exactly. Arithmetic,
inputs, dimensions, reference tolerances, canaries and timers are unchanged.

Pass one smoke suite per library first. Then run six full BLAS suites in order:
retained-01, upstream-01, upstream-02, retained-02, retained-03, upstream-03.
Wait for normal exit and guard cleanup before launching the next process.

Every full suite tests 4096x256, 65536x128, 65536x256, 65536x512 and 248144x256
in row-major/transposed and column-major/nontransposed storage/API combinations.
FP32, alpha=1, beta=0, contiguous vectors and deterministic binary-fraction inputs.
Two matrices alternate; every output is compared between implementations, selected
rows are checked against an independent double-precision CPU reference, and
canaries plus post-timing results are checked. The tolerance is
abs(error) <= 0.0002 + 0.00002*abs(reference), unchanged from the earlier run.
Each full suite expects 2,695,648 numerical comparisons and 3,840 canary checks.

Same-stream GPU wall-clock markers surround eight operations per HIP graph.
Two warm replays precede six balanced-order samples per variant. Record host
timing, calibration and every sample. Report three session medians per library;
retain and flag variable cases. Cache effects and power/clock state are possible
in these short workloads; do not convert an operation speedup into inference tok/s.

The guard targets only the 9070 XT, with 80 C/95 C/300 W sampled abort thresholds,
0.5-second polling and a three-second stale-sample stop. A six-sample idle check
precedes the fresh allocation-free HIP memory probe. Require at least 2 GiB free
at entry, cap explicit allocations at 640 MiB, and preserve 1 GiB immediate
allocation headroom. Stop only owned processes. An active unrelated workload is
not terminated. The 9060 XT, harness, production stack and iCUE remain untouched.

The user asked whether removing the minus-30-percent cap would improve speed.
That remains a separate unmeasured question; this protocol does not change the
cap. Equal percentage limits on different cards do not imply equal wattage.
