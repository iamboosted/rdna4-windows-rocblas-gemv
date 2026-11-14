# Windows RDNA4 rocBLAS performance reports

Report date: 2026-11-14.

- [Two-card report, including RX 9070 XT](rocblas-upstream-9070-20261114/REPORT.md).
- [Detailed RX 9060 XT comparison](rocblas-upstream-20261114/REPORT.md).
- [Portable Windows reproducer: source, build/run steps and validation](reproducer/README.md).

Separate single-GPU tests on native Windows, both at operator-confirmed -30%
power limits. The specialized FP32 kernel outperforms the pinned current upstream
rocBLAS build on the tested row-major/transposed shapes. rocBLAS wins every
column-major control. These reports measure synthetic operations, not inference
tokens per second. All samples, variability and correctness limits are included.

## Publication scope

The original twelve report/data files remain byte-identical to the initial
publication. The reproducer adds the exact tested source, configurable Windows
build/run commands, selected dependency provenance, licensing notices and
packaging-validation results. All 27 files are explicitly allowlisted and
checksummed. Private logs, machine build receipts, binaries, runtime payloads,
environment dumps and the machine-specific ADLX guard are excluded.

Run `python -B verify.py --self-test` for the publication checks and
`python -B reproducer/test_reproducer.py` for CPU-only failure-path checks and
recalculation of the reported medians. See the reproducer's publication-audit
scope for the limits of these checks, including Git history and generic CI
account names. No system driver/runtime installation or model workload is needed
to verify the supplied evidence.

## AMD destination

The technical reporting destination is
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/issues), with
both cards covered by one focused performance issue linking these reports and
the portable reproducer.
The [old rocBLAS repository](https://github.com/ROCm/rocBLAS) is retired.
AMD explicitly welcomes performance notes in its
[rocBLAS contribution guidance](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/contribute.html).

Suggested title:

> [rocBLAS][Windows][gfx1200/gfx1201] Narrow-reduction transposed SGEMV performance opportunity

Maintainer question: would a shape-specific wave32 dispatch for contiguous FP32
transposed SGEMV be worth developing and validating more broadly?

Lead with the operation, shapes, measured timings, correctness and reproduction
steps. Preserve the column-major controls and build-comparison limitations.
The repository's generic bug template asks for full environment/log files;
only reviewed, sanitized evidence should be supplied for this report.
