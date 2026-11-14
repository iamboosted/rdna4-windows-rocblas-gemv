# Portable Windows reproducer

This reproduces the synthetic operations behind the
[two-card report](../rocblas-upstream-9070-20261114/REPORT.md) and
[9060 XT report](../rocblas-upstream-20261114/REPORT.md). It needs no model,
inference harness or launcher. The two benchmark source files are byte-identical
to the tested sources; they differ only in GPU identity and compilation target.

"Portable" here means configurable dependency/output paths on another Windows
machine with the supported hardware. This is not a Linux port or a universal GPU
benchmark. The original guard checks remain: a unique RX 9060 XT/gfx1200 or RX
9070 XT/gfx1201, wave32, 16,304 MiB reported capacity, and a 100 MHz wall-clock
counter. Other configurations fail explicitly instead of silently changing the
experiment. See [provenance.json](provenance.json) for exact versions and hashes.

## Requirements

- Native Windows, one of the two tested 16 GiB cards, and an already installed
  compatible AMD driver. Original measurements used operator-reported Adrenalin
  26.9.2 and a -30% limit on both cards, tested separately. The scripts never
  change drivers, clocks, voltage, fan control or power limits.
- An x64 Visual Studio Developer PowerShell with MSVC and the Windows SDK set
  up (`INCLUDE` and `LIB` present). Tested Visual Studio version: 18.11.2.
- Existing Python 3.10 or later (tested 3.15.2), CMake (tested 4.4.3), Ninja, Git,
  and an existing ROCm SDK laid out with `lib/llvm/bin/clang++.exe`, `include`,
  `lib/amdhip64.lib`, `lib/rocblas.lib`, and device bitcode.
- Existing ROCm CMake Build Tools (tested 0.14.0) for rebuilding rocBLAS.
- Your own compatible runtime directory containing the DLLs and code pack below.

The compiler SDK manifest says **10.1.0rc3**, TheRock commit
`b324869fb3116353ac7eec5e6926aa3d452ac987`. Its compiler is AMD clang 24.0.0git,
LLVM commit `d6f6cb691863e9258034ab48c1dfce21689d5606`, with HIP 7.16.26385 headers.
The retained runtime used in the experiments separately reports HIP integer
version `71726405`. Do not equate the SDK package label with the loaded runtime
or graphics driver. No vendor binaries or dependency installers are distributed.
These are the recorded dependencies, not a recommendation to replace a working
driver/SDK or a claim of compatibility with every newer package.

Before acquiring dependencies, review their current advisories. The recorded
[dependency review](../rocblas-upstream-20261114/DEPENDENCY-REVIEW.md) documents
the review performed for these experiments, including its unresolved caveat.
The Python helpers use only the standard library and never download or install.

## 1. Select local paths

Run from this repository's root in the developer shell. Paths can contain
spaces. Use a separate working directory outside this Git repository; each
generated output directory must be new. Existing outputs are never overwritten.

```powershell
$Sdk = Read-Host 'Existing ROCm SDK directory'
$Runtime = Read-Host 'Existing retained runtime directory'
$Source = Read-Host 'Clean rocm-libraries checkout directory'
$Work = Read-Host 'Separate scratch directory outside this repository'
$Ninja = Read-Host 'Existing Ninja executable path'
$Gpu = 'gfx1200' # RX 9060 XT; use gfx1201 for RX 9070 XT
```

The retained runtime must supply `amdhip64_7.dll`, `amd_comgr.dll`,
`rocm_kpack.dll`, `rocblas.dll`, and `.kpack/blas_lib_gfx1200.kpack` or the gfx1201
equivalent. If present, `libhipblaslt.dll`, `libtensilelite-host.dll` and
`origami.dll` are also copied because the retained rocBLAS package imports them.
Supply matching DLLs from one compatible runtime package; the helper does not
search your system for substitutes. It only copies the named files to new
private directories. The recorded retained rocBLAS is 5.8.0.3d8f334b; using a
different runtime is a new comparison and must be labeled as such.

## 2. Build the pinned upstream library

Use a clean checkout at
`4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89`. If you need a checkout, first review
the source/dependencies, then use the upstream guide's
[sparse-checkout instructions](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/CONTRIBUTING.md#option-b-clone-the-monorepo-with-sparse-checkout)
and check out that exact commit. At least `projects/rocblas` must be available.
The optional Tensile logic corpus, docs and Google Test cases are not needed by
this native-BLAS build. An existing checkout requires no download.

```powershell
git -C $Source checkout --detach 4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89
if ($LASTEXITCODE -ne 0) { throw 'Source checkout failed' }
python -B .\reproducer\reproduce.py configure-library --gpu $Gpu --sdk $Sdk --source $Source --ninja $Ninja --out (Join-Path $Work "$Gpu-library")
if ($LASTEXITCODE -ne 0) { throw 'Library configuration failed' }
python -B .\reproducer\reproduce.py compile-library --sdk $Sdk --directory (Join-Path $Work "$Gpu-library") --jobs 4
if ($LASTEXITCODE -ne 0) { throw 'Library build failed' }
```

This builds `staging/rocblas.dll` with Release/O3 and unmodified upstream source.
Tensile, hipBLASLt, pip, clients and automatic CMake dependency fetching are
disabled. There is no install step. The gfx1201 recipe additionally suppresses
the ignored device-export-attribute diagnostic, matching its recorded build.
The native library build can take several minutes. Compiler output stays in
local files named `*.private.*.log`; those files can contain absolute paths.

If you already have the matching isolated upstream DLL, you can reuse it and
skip this build. The DLL's hash is recorded when the runtime is staged.

## 3. Build once and stage the A/B

```powershell
python -B .\reproducer\reproduce.py build --gpu $Gpu --sdk $Sdk --runtime $Runtime --out (Join-Path $Work "$Gpu-retained")
if ($LASTEXITCODE -ne 0) { throw 'Benchmark build failed' }
python -B .\reproducer\reproduce.py stage --directory (Join-Path $Work "$Gpu-retained") --rocblas (Join-Path $Work "$Gpu-library\staging\rocblas.dll") --out (Join-Path $Work "$Gpu-upstream")
if ($LASTEXITCODE -ne 0) { throw 'Runtime staging failed' }
```

Both arms now use exactly the same benchmark executable. Only `rocblas.dll`
differs. Runtime hashes are checked before and after every run. The program
requires HIP/rocBLAS and any loaded optional BLAS DLLs to be adjacent to the
executable. It selects the GPU by exact name rather than assuming a HIP ordinal.

## 4. Correctness, then sequential timing

Use an idle target card and finish each command before starting the next.
The option `--power-limit-percent -30` records the operator's setting; it neither
reads nor changes it. Omit the option if the setting is unknown.

```powershell
python -B .\reproducer\reproduce.py run --directory (Join-Path $Work "$Gpu-retained") --suite smoke --out (Join-Path $Work "$Gpu-retained-smoke") --power-limit-percent -30
if ($LASTEXITCODE -ne 0) { throw 'Retained smoke checks failed' }
python -B .\reproducer\reproduce.py run --directory (Join-Path $Work "$Gpu-upstream") --suite smoke --out (Join-Path $Work "$Gpu-upstream-smoke") --power-limit-percent -30
if ($LASTEXITCODE -ne 0) { throw 'Upstream smoke checks failed' }

$Order = @('retained-01', 'upstream-01', 'upstream-02', 'retained-02', 'retained-03', 'upstream-03')
foreach ($Label in $Order) {
    $Arm = $Label.Split('-')[0]
    python -B .\reproducer\reproduce.py run --directory (Join-Path $Work "$Gpu-$Arm") --suite blas --out (Join-Path $Work "$Gpu-$Label") --power-limit-percent -30
    if ($LASTEXITCODE -ne 0) { throw 'Stop: inspect the failed run before continuing' }
}
```

The six full runs cover all five shapes in both layouts. Every session contains
six balanced-order timing samples after two warm graph replays. A graph contains
eight operations. GPU wall-clock intervals and enclosing host intervals are
recorded, with 100/500/1,000 microsecond calibration checks. This is warm synthetic
operation timing; neither inference tok/s nor cold-memory bandwidth is inferred.

The maximum explicit allocation is 486.551 MiB in the full suite, below the
640 MiB cap. A five-second allocation-free probe requires 2 GiB free, and the
program rechecks at least 1 GiB extra headroom immediately before each allocation.
The helper terminates only its own native child after a bounded timeout (at most
240 seconds). It never requests peer access or compute on the second card.

The standalone helper does **not** include the original machine's ADLX thermal
watchdog. `thermal_telemetry_collected` is therefore false in its result. The
packaging validation used the existing private ADLX guard as a launch adapter;
it is not silently presented as a portable thermal safeguard. These short tests
and process timeouts cannot guarantee recovery from a driver fault.

## 5. Analyze without selecting favorable sessions

```powershell
python -B .\reproducer\analyze.py --retained (Join-Path $Work "$Gpu-retained-01\samples.json") (Join-Path $Work "$Gpu-retained-02\samples.json") (Join-Path $Work "$Gpu-retained-03\samples.json") --upstream (Join-Path $Work "$Gpu-upstream-01\samples.json") (Join-Path $Work "$Gpu-upstream-02\samples.json") (Join-Path $Work "$Gpu-upstream-03\samples.json") --out (Join-Path $Work "$Gpu-comparison.json")
```

The comparison rejects mixed GPUs, executables or non-rocBLAS runtime artifacts.
It preserves all three session medians and their spread. Repeat the entire
sequence separately for the other GPU; these commands do not perform dual-GPU
inference. Original report measurements remain separate from reproduction runs.

The parser produces `samples.json` using explicit public numeric/version fields.
Do not upload the surrounding directory, private diagnostics, binaries or build
configuration. Review any new result before sharing it. The public artifact
verifier checks this repository's fixed allowlist, not arbitrary future outputs.

## Validation and coverage limits

See [validation.json](validation.json). The packaged source compiled for both
targets in paths containing spaces. The new library configuration command passed
for both targets; the already completed upstream DLL builds were reused for
execution rather than rebuilt solely for packaging. The compile-library command
wraps the same CMake build operation used by those completed builds.

One fresh smoke and one full BLAS suite passed per GPU through the public runtime
checks/parser under the private ADLX launch adapter. The helper's ordinary child
launch and timeout path was tested separately with a controlled CPU child. These
four GPU runs validate packaging and do not replace or extend the headline A/B.

CPU-only checks recompute all published session medians, reject incomplete or
corrupted evidence, verify process timeout cleanup, and cover privacy false
positives for generic accounts such as `agent`:

```powershell
python -B .\reproducer\test_reproducer.py
python -B .\verify.py --self-test
```

FP32 tolerance is `abs(error) <= 0.0002 + 0.00002*abs(reference)`. Every output is
compared between implementations; selected first/last rows also use an
independent double-precision CPU reference. Canaries, finite outputs and
post-timing values are checked. Inputs are bounded binary fractions, alpha=1,
beta=0 and unit vector strides. This does not cover the full BLAS API or arbitrary
numerical inputs. Q8 smoke coverage is separate from the FP32 rocBLAS result.

## Publication audit scope

`verify.py` checks current allowlisted files, hashes, relative links, report
accounting and common sensitive patterns. Automatic identity checks use token
boundaries and exclude generic account names; explicit `--deny-term` arguments
can still flag those names deliberately. Machine-path patterns apply regardless
of the account name. Local author information is never saved or printed.

The publication process separately checks committed bytes and the reachable
history of the published branch, using project-name/GitHub no-reply attribution.
This does not inspect deleted Git objects, server caches, forks, or separate
issue/PR attachments. Manual content review remains part of the audit.

The source/helper license and applicable notices are in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
