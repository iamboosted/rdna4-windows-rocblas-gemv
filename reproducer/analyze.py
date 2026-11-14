"""Validate benchmark output and compare three sessions per rocBLAS build."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

SHAPES = {(4096, 256), (65536, 128), (65536, 256), (65536, 512), (248144, 256)}
LAYOUTS = {'row_major_via_transpose', 'column_major_nontransposed'}
FLOAT_VARIANTS = {'rocblas_sgemv', 'custom_wave32_four_rows'}
Q8_VARIANTS = {'eight_waves_one_row', 'useful_waves_one_row', 'useful_waves_two_rows',
               'useful_waves_four_rows', 'useful_waves_eight_rows'}
COUNTS = {'blas': (0, 2695648, 3840), 'smoke': (4048192, 26054, 7168)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def parse_records(records, expected_gpu=None, expected_suite=None):
    require(1 < len(records) <= 1000 and all(isinstance(r, dict) for r in records), 'Invalid record list')
    config, done = records[0], records[-1]
    require(config.get('kind') == 'configuration' and done.get('kind') == 'complete', 'Incomplete run')
    require(sum(r.get('kind') == 'configuration' for r in records) == 1 and
            sum(r.get('kind') == 'complete' for r in records) == 1, 'Duplicate run boundaries')
    gpu, suite = config.get('gpu'), config.get('suite')
    require(gpu in ('RX 9060 XT', 'RX 9070 XT') and suite in COUNTS, 'Unexpected GPU or suite')
    require(expected_gpu is None or gpu == expected_gpu, 'GPU selection mismatch')
    require(expected_suite is None or suite == expected_suite, 'Suite selection mismatch')
    require(config.get('arch') == ('gfx1200' if gpu == 'RX 9060 XT' else 'gfx1201'), 'Architecture mismatch')
    require(config.get('warp') == 32 and config.get('clock_khz') == 100000 and
            config.get('graph_batch_calls') == 8, 'Timing configuration changed')
    require(config.get('second_gpu_compute') is False and config.get('model_free') is True and
            config.get('runtime_adjacent') is True, 'Isolation requirements not recorded')
    require(done.get('suite') == suite and done.get('mismatches') == 0 and
            done.get('second_gpu_compute') is False, 'Correctness or isolation failure')
    require(tuple(done.get(k) for k in ('q8_bit_comparisons', 'numeric_reference_comparisons',
                                      'guard_values_checked')) == COUNTS[suite], 'Correctness coverage changed')
    require(finite(done.get('max_owned_allocation_mib')) and
            0 < done['max_owned_allocation_mib'] <= 640, 'Allocation bound exceeded')
    calibration = [r for r in records if r.get('kind') == 'calibration']
    require([r.get('requested_us') for r in calibration] == [100, 500, 1000], 'Missing calibration')
    for row in calibration:
        require(all(finite(row.get(k)) for k in ('device_ms', 'host_ms', 'hip_event_ms')), 'Nonfinite calibration')
        require(row['requested_us'] / 1000 <= row['device_ms'] < row['requested_us'] / 1000 + .1 and
                row['device_ms'] <= row['host_ms'] + .005, 'Calibration interval invalid')
    middle, end = calibration[1:]
    ratio = (end['host_ms'] - middle['host_ms']) / (end['device_ms'] - middle['device_ms'])
    require(.8 < ratio < 1.2, 'Host/device calibration increments disagree')
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        kind = row.get('kind')
        require(kind in {'configuration', 'complete', 'calibration', 'timing', 'case_pass', 'copy_pass'},
                'Unknown record kind')
        if kind != 'timing':
            continue
        require(row.get('N') == 1 and row.get('calls') == 8 and type(row.get('trial')) is int,
                'Timing dimensions or trial invalid')
        require(all(finite(row.get(k)) and row[k] > 0 for k in ('gpu_us_per_call', 'host_us_per_call')),
                'Nonfinite or nonpositive timing')
        require(row['gpu_us_per_call'] <= row['host_us_per_call'] + .625, 'GPU span exceeds host interval')
        key = (row['suite'], row['M'], row['K'], row['layout'])
        groups[key][row['variant']].append(row)
    fp32 = {k for k in groups if k[0] == 'fp32'}
    expected_shapes = SHAPES if suite == 'blas' else {(4097, 256)}
    require(fp32 == {('fp32', m, k, layout) for m, k in expected_shapes for layout in LAYOUTS},
            'Missing or unexpected FP32 case')
    fp32_passes = [(r.get('M'), r.get('K'), r.get('layout')) for r in records
                  if r.get('kind') == 'case_pass' and r.get('suite') == 'fp32']
    require(len(fp32_passes) == len(fp32) and set(fp32_passes) == {k[1:] for k in fp32},
            'FP32 completion records differ')
    q8 = {k for k in groups if k[0] == 'q8'}
    expected_q8 = {(257, 128), (4097, 256), (248144, 256), (257, 512), (257, 1024)} if suite == 'smoke' else set()
    require(q8 == {('q8', m, k, 'block_q8_0_row_major') for m, k in expected_q8}, 'Q8 smoke coverage differs')
    expected_runtime = {('runtime', 0, 0, 'none')}
    if suite == 'smoke':
        expected_runtime.add(('runtime', 8388608, 0, 'bytes'))
    require(set(groups) == fp32 | q8 | expected_runtime, 'Runtime control coverage differs')
    if suite == 'smoke':
        passes = [r for r in records if r.get('kind') == 'case_pass' and r.get('suite') == 'q8']
        require(len(passes) == 5 and all(r.get('bit_exact') is True for r in passes), 'Q8 correctness missing')
        require([r.get('bytes_checked') for r in records if r.get('kind') == 'copy_pass'] == [8388608],
                'Copy smoke coverage differs')
    cases = []
    for (kind, m, k, layout), variants in groups.items():
        if kind == 'fp32':
            expected_variants = FLOAT_VARIANTS
        elif kind == 'q8':
            expected_variants = Q8_VARIANTS
        else:
            require(kind == 'runtime' and (m, k, layout) in ((0, 0, 'none'), (8388608, 0, 'bytes')),
                    'Unexpected runtime case')
            expected_variants = {'empty_launch'} if m == 0 else {'device_copy_8mib'}
        require(set(variants) == expected_variants, 'Variant coverage changed')
        item = {'suite': kind, 'rows': m, 'reduction': k, 'layout': layout, 'variants': {}}
        for name, rows in variants.items():
            require(sorted(r['trial'] for r in rows) == list(range(6)), 'Missing or duplicate trial')
            rows = sorted(rows, key=lambda r: r['trial'])
            times = [r['gpu_us_per_call'] for r in rows]
            host = [r['host_us_per_call'] for r in rows]
            item['variants'][name] = {'gpu_us': times, 'host_us': host,
                'median_gpu_us': statistics.median(times), 'median_host_us': statistics.median(host)}
        cases.append(item)
    clean_config = {k: config[k] for k in ('suite', 'gpu', 'arch', 'warp', 'clock_khz',
        'runtime_version', 'rocblas', 'free_mib', 'model_free', 'second_gpu_compute',
        'runtime_adjacent', 'graph_batch_calls')}
    clean_done = {k: done[k] for k in ('suite', 'q8_bit_comparisons', 'numeric_reference_comparisons',
        'guard_values_checked', 'mismatches', 'max_owned_allocation_mib', 'second_gpu_compute')}
    clean_calibration = [{k: r[k] for k in ('requested_us', 'device_ms', 'host_ms', 'hip_event_ms',
                                         'event_agrees')} for r in calibration]
    return {'schema': 1, 'configuration': clean_config, 'checks': clean_done,
            'calibrations': clean_calibration, 'calibration_host_device_increment_ratio': ratio,
            'cases': cases}


def compare(retained, upstream):
    require(len(retained) == len(upstream) == 3, 'Supply all three sessions per library')
    runs = retained + upstream
    require(len({r['configuration']['gpu'] for r in runs}) == 1, 'Compare one GPU at a time')
    require(len({r['executable_sha256'] for r in runs}) == 1, 'Benchmark executables differ across arms')
    require(len({r['source_sha256'] for r in runs}) == 1, 'Benchmark sources differ across arms')
    require(len({r['configuration']['runtime_version'] for r in runs}) == 1, 'HIP runtime versions differ')
    common_runtime = [{k: v for k, v in r['runtime_sha256'].items() if k != 'rocblas.dll'} for r in runs]
    require(bool(common_runtime[0]) and all(pins == common_runtime[0] for pins in common_runtime),
            'Non-rocBLAS runtime artifacts differ')
    require(all(r['configuration']['suite'] == 'blas' and r['checks']['mismatches'] == 0 for r in runs),
            'Only successful full BLAS sessions can enter this comparison')
    for arm in (retained, upstream):
        require(len({r['rocblas_sha256'] for r in arm}) == 1, 'Library changed within an arm')
    summary = []
    for m, k in sorted(SHAPES):
        for layout in sorted(LAYOUTS):
            row = {'rows': m, 'reduction': k, 'layout': layout}
            for label, arm in (('retained', retained), ('upstream', upstream)):
                cases = []
                for run in arm:
                    found = [c for c in run['cases'] if (c['suite'], c['rows'], c['reduction'], c['layout'])
                             == ('fp32', m, k, layout)]
                    require(len(found) == 1, 'Missing or duplicate comparison case')
                    cases.append(found[0])
                row[label] = {}
                for variant in sorted(FLOAT_VARIANTS):
                    medians = [statistics.median(c['variants'][variant]['gpu_us']) for c in cases]
                    row[label][variant] = {'session_medians_us': medians,
                        'median_us': statistics.median(medians), 'max_min_session_ratio': max(medians) / min(medians)}
            old = row['retained']['rocblas_sgemv']['median_us']
            new = row['upstream']['rocblas_sgemv']['median_us']
            custom = row['upstream']['custom_wave32_four_rows']['median_us']
            row.update(upstream_over_custom_speed_ratio=new / custom,
                       upstream_time_change_percent=100 * (new / old - 1))
            summary.append(row)
    return {'gpu': runs[0]['configuration']['gpu'], 'summary': summary,
            'scope': 'Warm synthetic FP32 operations; no inference throughput measured'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained', nargs=3, required=True, metavar='SAMPLES_JSON')
    parser.add_argument('--upstream', nargs=3, required=True, metavar='SAMPLES_JSON')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    target = Path(args.out)
    require(not target.exists(), 'Output already exists')
    data = compare([json.loads(Path(p).read_text()) for p in args.retained],
                   [json.loads(Path(p).read_text()) for p in args.upstream])
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'pass', 'gpu': data['gpu'], 'cases': len(data['summary'])}))


if __name__ == '__main__':
    main()
