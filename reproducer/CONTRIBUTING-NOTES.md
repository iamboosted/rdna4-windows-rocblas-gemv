# Upstream contribution guidance reviewed

The requested [ROCm libraries CONTRIBUTING.md at the tested commit](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/CONTRIBUTING.md)
was read before packaging and submitting the performance issue. Its issue route
is ROCm/rocm-libraries. The project-specific
[rocBLAS guidance](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/projects/rocblas/.github/CONTRIBUTING.rst)
welcomes performance notes; the
[issue template](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/projects/rocblas/.github/ISSUE_TEMPLATE/bug_report.md)
asks for an exact version/commit, reproduction steps, environment, expected
behavior and supporting evidence.

This submission supplies those items as a focused performance report with a
standalone reproducer. It asks maintainers whether a shape-specific dispatch
is worth pursuing. It does not propose an unvalidated production-library patch,
claim an established source-only regression, or claim a system-wide speedup.
The two GPU datasets and the column-major controls remain available separately.

The generic template's Linux environment-dump commands do not describe this
native-Windows setup. Only selected hardware/software versions, source hashes
and sanitized numerical records are shared. Original machine logs and account
details remain private.

The root guide's branch naming, develop-branch workflow, pre-commit hooks,
auto-review routing and CI expectations apply to an eventual code PR. The
rocBLAS guidance also describes clang-format, device-memory management and
broader library coding conventions. No library PR is claimed here, and no
pre-commit package, hook or dependency installer was run for this report.
The benchmark sources retain their tested bytes; preserving their hashes is
intentional and is not a claim that their formatting is ready for upstream.

An eventual library contribution would need the project's formatting and tests,
broader alpha/beta, stride, shape, datatype and numerical coverage, and maintainer
agreement on the dispatch. A model-level inference improvement would require
separate end-to-end measurements.
