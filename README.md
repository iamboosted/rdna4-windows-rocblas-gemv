# Windows RDNA4 rocBLAS performance reports

Report date: 2026-11-14.

- [Two-card report, including RX 9070 XT](rocblas-upstream-9070-20261114/REPORT.md).
- [Detailed RX 9060 XT comparison](rocblas-upstream-20261114/REPORT.md).

Separate single-GPU tests on native Windows, both at operator-confirmed -30%
power limits. The specialized FP32 kernel outperforms the pinned current upstream
rocBLAS build on the tested row-major/transposed shapes. rocBLAS wins every
column-major control. These reports measure synthetic operations, not inference
tokens per second. All samples, variability and correctness limits are included.

## Publication scope

This folder contains twelve reviewed Markdown/JSON evidence files, this index,
and SHA256SUMS. The evidence files are byte-identical to their verified local
report manifests. It excludes private logs, build receipts, binaries, runtime
files, machine-specific runners, environment dumps and Git history. It contains
no private account names, hostnames, absolute machine paths or credentials.
Report links resolve within this folder or point to public upstream sources.

The report bundle is ready to publish as documentation and evidence. It is not
yet a standalone reproduction package: the tested benchmark source, portable
build/run commands, licensing/notice review and exact public toolchain recipe
still need packaging and validation before calling an AMD submission reproducible.
No further GPU test or external publication was performed while staging it.

## AMD destination

Open one focused performance issue in
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/issues), covering
both cards, with links to these reports and a portable reproducer.
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
No issue, pull request or other message has been sent to AMD.
