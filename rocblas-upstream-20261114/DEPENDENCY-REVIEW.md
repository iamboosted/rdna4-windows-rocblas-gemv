# Upstream rocBLAS comparison: pre-acquisition review

Review date: 2026-11-14. This review precedes acquisition and execution.

Acquire only official ROCm/rocm-libraries source at pinned develop commit
`4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89`. The top-level rocBLAS CMake files,
build options, dependency acquisition helper, and Windows toolchain were reviewed
read-only first. Source declares rocBLAS 5.9.0. The comparison is against the
previous private retained rocBLAS 5.8.0.3d8f334b, not against a newly installed SDK.

Existing tools: ROCm Core SDK 10.1.0 compiler/headers/import library,
ROCmCMakeBuildTools 0.14.0, CMake 4.4.3, Git 2.56.0.windows.3,
Python 3.15.2 standard library, and Visual Studio 18.11.2 with bundled Ninja.
No package manager, global installation, driver update, or new Python package.

Native BLAS library build: gfx1200 only, Release shared library, Tensile OFF,
hipBLASLt OFF, clients/tests/samples OFF. The external existing benchmark supplies
the same correctness checks. These optional GEMM backends are not used by SGEMV.
This is a scoped native-BLAS build, not a complete production replacement.
Use installed CMake helpers and disable FetchContent downloads. Do not run the
upstream install/dependency scripts or its vcpkg toolchain. All output is private.

The [ROCm libraries security page](https://github.com/ROCm/rocm-libraries/security)
currently reports no published advisories. A targeted rocBLAS/ROCm-CMake advisory
search found no specific applicable advisory; this is not a CVE-free guarantee.
The [AMD graphics bulletin](https://www.amd.com/en/resources/product-security/bulletin/amd-sb-6028.html)
lists RX 9000 Windows mitigations through Adrenalin 26.7.1 and an unresolved
CVE-2026-0457 entry. The operator-confirmed retained driver is 26.9.2; no new live
driver inventory or whole-driver clearance is inferred. The November shader
bulletin AMD-SB-4026 lists older integrated graphics rather than this target.
RCCL and Device Metrics Exporter are neither acquired nor used. No network
service, arbitrary model/parser input, or peer communication is exercised.

Source retrieval uses a new sparse checkout with hooks and automatic submodules
disabled, pinned commit verification, and no source execution during acquisition.
The unused large Tensile logic corpus is excluded. All generated manifests and
build logs remain local; sanitized results will omit private paths and raw logs.

GPU execution will use only the uniquely identified RX 9060 XT/gfx1200, the
existing minus-30-percent power setting, the same guarded synthetic benchmark,
same correctness tolerances, independent CPU checks and buffer canaries. Preserve
the 640 MiB owned-allocation cap, fresh memory admission check, thermal/power
watchdog, and owned-process-only cleanup. Do not access the protected harness,
production builds, launchers, model files, Vulkan, GPU tuning, or iCUE.
