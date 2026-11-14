"""Windows-only, standard-library build/run helper. No downloads or installation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMIT = '4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89'
TARGETS = {
    'gfx1200': ('RX 9060 XT', '732e187af0d86f9b692b98dfc6384fb90bafba90780fb7bef0eecb753a2a2fdd'),
    'gfx1201': ('RX 9070 XT', '6f01ef696e2bcd018cba2bd42d13b19bf55095133a97fe6e99c99a4d4ea70622'),
}
REQUIRED_DLLS = ('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll', 'rocblas.dll')
OPTIONAL_DLLS = ('libhipblaslt.dll', 'libtensilelite-host.dll', 'origami.dll')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def windows_only():
    require(os.name == 'nt', 'The supplied reproducer targets native Windows only')


def new_output(path):
    path = Path(path).resolve()
    require(not path.exists(), 'Choose a new output directory; existing files are never overwritten')
    require(not path.is_relative_to(ROOT.parent), 'Keep generated output outside the publication repository')
    path.mkdir(parents=True)
    return path


def environment(sdk=None, runtime=None):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('HIP_', 'ROCBLAS_', 'HSA_', 'GGML_', 'TENSORFOLD_', 'TFMC_')) or key in (
                'AMD_LOG_LEVEL', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
            env.pop(key)
    if sdk:
        env.update(HIP_PATH=str(sdk), ROCM_PATH=str(sdk), HIP_PLATFORM='amd')
        env['PATH'] = str(sdk / 'bin') + os.pathsep + str(sdk / 'lib/llvm/bin') + os.pathsep + env['PATH']
        env.update(PYTHONNOUSERSITE='1', PIP_NO_INDEX='1')
    if runtime:
        env['PATH'] = str(runtime) + os.pathsep + env['PATH']
    return env


def sdk_paths(path):
    sdk = Path(path).resolve(strict=True)
    compiler = sdk / 'lib/llvm/bin/clang++.exe'
    for name in ('lib/llvm/bin/clang++.exe', 'lib/amdhip64.lib', 'lib/rocblas.lib',
                 'include/hip/hip_runtime.h', 'lib/llvm/amdgcn/bitcode'):
        require((sdk / name).exists(), 'SDK layout is missing a required compiler/header/library component')
    require(os.environ.get('INCLUDE') and os.environ.get('LIB'),
            'Open an x64 Visual Studio Developer PowerShell first (MSVC and Windows SDK required)')
    return sdk, compiler


def invoke(command, out, label, env=None, timeout=240):
    """Private diagnostics may contain paths; never add generated outputs to Git."""
    started = time.monotonic()
    with (out / (label + '.private.stdout.log')).open('wb') as stdout, \
            (out / (label + '.private.stderr.log')).open('wb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            if child.poll() is None:
                child.kill()  # This exact child only; no process-name termination.
                child.wait(timeout=10)
            raise
    require(code == 0, label + ' failed; inspect its local private diagnostic files')
    return time.monotonic() - started


def build(args):
    windows_only()
    sdk, compiler = sdk_paths(args.sdk)
    source = ROOT / 'sources' / ('benchmark-' + args.gpu + '.hip')
    require(digest(source) == TARGETS[args.gpu][1], 'Benchmark source differs from the tested source')
    runtime = Path(args.runtime).resolve(strict=True)
    rocblas = Path(args.rocblas).resolve(strict=True) if args.rocblas else runtime / 'rocblas.dll'
    copies = {name: (rocblas if name == 'rocblas.dll' else runtime / name) for name in REQUIRED_DLLS}
    copies.update({name: runtime / name for name in OPTIONAL_DLLS if (runtime / name).is_file()})
    pack = '.kpack/blas_lib_' + args.gpu + '.kpack'
    if (runtime / pack).is_file():
        copies[pack] = runtime / pack
    elif not args.rocblas:
        raise ValueError('Retained runtime requires its matching gfx code pack')
    require(all(p.is_file() for p in copies.values()), 'Runtime is missing a required DLL')
    out = new_output(args.out)
    pins = {}
    for name, original in copies.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        pins[name] = digest(dest)
        require(pins[name] == digest(original), 'Runtime copy verification failed')
    command = [str(compiler), '-O2', '-std=c++17', '-ffp-contract=off',
               '-D__HIP_PLATFORM_AMD__=1', '-D_DLL', '-D_MT', '-Xclang', '--dependent-lib=msvcrt',
               '-x', 'hip', '--offload-arch=' + args.gpu,
               '--rocm-device-lib-path=' + (sdk / 'lib/llvm/amdgcn/bitcode').as_posix(),
               '-I' + str(sdk / 'include'), str(source), '-x', 'none',
               str(sdk / 'lib/amdhip64.lib'), str(sdk / 'lib/rocblas.lib'),
               '-o', str(out / 'model-free.exe')]
    seconds = invoke(command, out, 'build', environment(sdk=sdk))
    record = {'schema': 1, 'gpu': args.gpu, 'source_sha256': digest(source),
              'executable_sha256': digest(out / 'model-free.exe'), 'compiler_sha256': digest(compiler),
              'import_library_sha256': {n: digest(sdk / 'lib' / n) for n in ('amdhip64.lib', 'rocblas.lib')},
              'runtime_sha256': pins, 'compile_seconds': seconds, 'gpu_execution': False}
    write_json(out / 'build.json', record)
    print(json.dumps({'status': 'built', 'gpu': args.gpu, 'gpu_execution': False,
                      'source_sha256': record['source_sha256']}))


def configure_library(args):
    windows_only()
    sdk, compiler = sdk_paths(args.sdk)
    source = Path(args.source).resolve(strict=True)
    flags = {'capture_output': True, 'text': True, 'check': True,
             'creationflags': subprocess.CREATE_NO_WINDOW}
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], **flags).stdout.strip()
    require(head == COMMIT, 'Source checkout does not match the reported upstream commit')
    require(not subprocess.run(['git', '-C', str(source), 'status', '--porcelain'], **flags).stdout.strip(),
            'Use a clean source checkout')
    native = source / 'projects/rocblas'
    require((native / 'CMakeLists.txt').is_file(), 'Expected the rocm-libraries checkout root')
    out = new_output(args.out)
    cxx_flags = ('-DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS '
                 '-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -D_DLL -D_MT '
                 '-Xclang --dependent-lib=msvcrt --rocm-device-lib-path="' +
                 (sdk / 'lib/llvm/amdgcn/bitcode').as_posix() + '"')
    if args.gpu == 'gfx1201':
        cxx_flags = '-Wno-ignored-attributes ' + cxx_flags
    command = [args.cmake, '-S', str(native), '-B', str(out), '-G', 'Ninja',
               '-UCOMPILER_HAS_TARGET_ID_*', '-UCXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS',
               '-DCMAKE_MAKE_PROGRAM=' + str(Path(args.ninja).resolve()),
               '-DCMAKE_CXX_COMPILER=' + str(compiler), '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_INSTALL_PREFIX=' + str(out / 'unused-install'),
               '-DCMAKE_PREFIX_PATH=' + str(sdk), '-DROCM_PATH=' + str(sdk),
               '-Dhip_DIR=' + str(sdk / 'lib/cmake/hip'),
               '-DROCmCMakeBuildTools_DIR=' + str(sdk / 'share/rocmcmakebuildtools/cmake'),
               '-DHIP_PLATFORM=amd', '-DGPU_TARGETS=' + args.gpu, '-DAMDGPU_TARGETS=' + args.gpu,
               '-DBUILD_SHARED_LIBS=ON', '-DBUILD_WITH_TENSILE=OFF', '-DBUILD_WITH_HIPBLASLT=OFF',
               '-DBUILD_WITH_HIPBLASLT_ONLY=OFF', '-DBUILD_WITH_PIP=OFF',
               '-DBUILD_CLIENTS_TESTS=OFF', '-DBUILD_CLIENTS_BENCHMARKS=OFF',
               '-DBUILD_CLIENTS_SAMPLES=OFF', '-DBUILD_FORTRAN_CLIENTS=OFF', '-DROCBLAS_ENABLE_CTEST=OFF',
               '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
               '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF',
               '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL', '-DCMAKE_CXX_FLAGS=' + cxx_flags,
               '-DPython3_EXECUTABLE=' + sys.executable]
    seconds = invoke(command, out, 'configure', environment(sdk=sdk))
    write_json(out / 'reproducer-config.json', {'schema': 1, 'gpu': args.gpu,
               'source_commit': COMMIT, 'compiler_sha256': digest(compiler),
               'configure_seconds': seconds, 'gpu_execution': False, 'install': False})
    print(json.dumps({'status': 'configured', 'gpu': args.gpu, 'source_commit': COMMIT,
                      'gpu_execution': False, 'automatic_downloads': False}))


def compile_library(args):
    windows_only()
    sdk, compiler = sdk_paths(args.sdk)
    out = Path(args.directory).resolve(strict=True)
    config = read_json(out / 'reproducer-config.json')
    require(config['source_commit'] == COMMIT and config['gpu'] in TARGETS, 'Unexpected build configuration')
    require(config['compiler_sha256'] == digest(compiler), 'Compiler changed after configuration')
    require(1 <= args.jobs <= 16, 'Build job count must be between 1 and 16')
    seconds = invoke([args.cmake, '--build', str(out), '--target', 'rocblas', '--parallel', str(args.jobs)],
                     out, 'compile', environment(sdk=sdk), timeout=3600)
    write_json(out / 'reproducer-library.json', {**config, 'compile_seconds': seconds,
               'rocblas_sha256': digest(out / 'staging/rocblas.dll')})
    print(json.dumps({'status': 'library-built', 'gpu': config['gpu'], 'gpu_execution': False}))


def check_runtime(directory):
    record = read_json(directory / 'build.json')
    require(record.get('schema') == 1 and record.get('gpu') in TARGETS, 'Unexpected build record')
    require(digest(directory / 'model-free.exe') == record['executable_sha256'], 'Executable changed')
    require(record['source_sha256'] == TARGETS[record['gpu']][1], 'Unrecognized source')
    for name, pin in record['runtime_sha256'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory), 'Runtime path escapes output directory')
        require(digest(path) == pin, 'Runtime artifact changed')
    return record


def stage(args):
    """Reuse one executable across both arms; replace only the requested rocBLAS DLL."""
    directory = Path(args.directory).resolve(strict=True)
    record = check_runtime(directory)
    replacement = Path(args.rocblas).resolve(strict=True)
    require(replacement.is_file(), 'Expected a built rocBLAS DLL')
    out = new_output(args.out)
    for name in (*record['runtime_sha256'], 'model-free.exe'):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(replacement if name == 'rocblas.dll' else directory / name, dest)
    record['runtime_sha256']['rocblas.dll'] = digest(out / 'rocblas.dll')
    write_json(out / 'build.json', record)
    check_runtime(out)
    print(json.dumps({'status': 'staged', 'gpu': record['gpu'], 'same_executable': True,
                      'gpu_execution': False}))


def run(args):
    windows_only()
    require(5 <= args.timeout <= 240, 'Compute timeout must be between 5 and 240 seconds')
    directory = Path(args.directory).resolve(strict=True)
    build_record = check_runtime(directory)
    out = new_output(args.out)
    exe, env = directory / 'model-free.exe', environment(runtime=directory)
    invoke([str(exe), '--list-devices'], out, 'probe', env, timeout=5)
    observed = time.monotonic()
    raw = (out / 'probe.private.stdout.log').read_bytes()
    require(len(raw) <= 8192, 'Probe output exceeded its bound')
    proof = json.loads(raw)
    expected_name = 'AMD Radeon ' + TARGETS[build_record['gpu']][0]
    require(proof.get('kind') == 'memory_probe' and proof.get('name') == expected_name
            and proof.get('arch') == build_record['gpu'] and proof.get('total_mib') == 16304
            and type(proof.get('free_mib')) is int and 2048 <= proof['free_mib'] <= 16304
            and proof.get('allocation_bytes') == 0 and proof.get('kernels_launched') == 0
            and proof.get('runtime_adjacent') is True, 'Invalid GPU/memory proof')
    require(time.monotonic() - observed <= 2, 'Memory proof expired before compute')
    seconds = invoke([str(exe), '--suite', args.suite], out, 'benchmark', env, args.timeout)
    check_runtime(directory)
    from analyze import parse_records
    records = [json.loads(line) for line in (out / 'benchmark.private.stdout.log').read_text().splitlines()]
    clean = parse_records(records, expected_gpu=TARGETS[build_record['gpu']][0], expected_suite=args.suite)
    clean.update({'executable_sha256': build_record['executable_sha256'],
                  'rocblas_sha256': build_record['runtime_sha256']['rocblas.dll'],
                  'runtime_sha256': build_record['runtime_sha256'],
                  'source_sha256': build_record['source_sha256'], 'process_seconds': seconds,
                  'operator_reported_power_limit_percent': args.power_limit_percent,
                  'thermal_telemetry_collected': False})
    write_json(out / 'samples.json', clean)
    print(json.dumps({'status': 'pass', 'gpu': clean['configuration']['gpu'],
                      'suite': args.suite, 'mismatches': 0, 'cases': len(clean['cases'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build', help='Build the benchmark and stage private runtime copies; no GPU execution')
    p.add_argument('--gpu', choices=TARGETS, required=True)
    p.add_argument('--sdk', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--rocblas', help='Override rocblas.dll with a locally built native-BLAS DLL')
    p.add_argument('--out', required=True)
    p.set_defaults(action=build)
    p = commands.add_parser('stage', help='Reuse a built executable with a different private rocBLAS DLL')
    p.add_argument('--directory', required=True)
    p.add_argument('--rocblas', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(action=stage)
    p = commands.add_parser('configure-library', help='Configure a clean pinned upstream checkout; no download/install')
    p.add_argument('--gpu', choices=TARGETS, required=True)
    p.add_argument('--sdk', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cmake', default='cmake')
    p.add_argument('--ninja', required=True, help='Path to an already installed Ninja executable')
    p.set_defaults(action=configure_library)
    p = commands.add_parser('compile-library', help='Build the previously configured native rocBLAS target')
    p.add_argument('--directory', required=True)
    p.add_argument('--sdk', required=True)
    p.add_argument('--cmake', default='cmake')
    p.add_argument('--jobs', type=int, default=4)
    p.set_defaults(action=compile_library)
    p = commands.add_parser('run', help='Run one selected GPU; no dual-GPU or peer operations')
    p.add_argument('--directory', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--suite', choices=['smoke', 'blas'], required=True)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--power-limit-percent', type=int, help='Record the operator setting; never changes it')
    p.set_defaults(action=run)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # Child diagnostics remain local and are never echoed into a publishable record.
        print(json.dumps({'status': 'failed', 'reason': str(error) if isinstance(error, ValueError)
                          else type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
