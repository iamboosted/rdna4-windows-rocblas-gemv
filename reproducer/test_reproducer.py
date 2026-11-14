"""CPU-only failure-path tests and independent replay of published measurements."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import analyze
import reproduce

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('publication_verify', ROOT / 'verify.py')
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)
PROVENANCE = json.loads((ROOT / 'reproducer/provenance.json').read_text())


def saved(gpu='gfx1200'):
    folder = 'rocblas-upstream-20261114' if gpu == 'gfx1200' else 'rocblas-upstream-9070-20261114'
    return json.loads((ROOT / folder / 'RESULTS.json').read_text())


def samples(gpu='gfx1200'):
    data = saved(gpu)
    result = []
    for original in data['runs']:
        if original['configuration']['suite'] != 'blas':
            continue
        row = copy.deepcopy(original)
        row['executable_sha256'] = row['benchmark_sha256']
        row['source_sha256'] = PROVENANCE['benchmark_source_sha256'][gpu]
        row['runtime_sha256'] = dict(PROVENANCE['common_retained_runtime_sha256'],
                                    **{'rocblas.dll': row['rocblas_sha256']})
        result.append(row)
    return [r for r in result if r['backend'] == 'retained'], [r for r in result if r['backend'] == 'upstream']


def records():
    """Reconstruct the public record fields from one published full BLAS session."""
    row = samples()[1][0]
    result = [dict(row['configuration'], kind='configuration', device=0)]
    result.extend(dict(c, kind='calibration') for c in row['calibrations'])
    for case in row['cases']:
        for name, variant in case['variants'].items():
            for trial, (gpu, host) in enumerate(zip(variant['gpu_us'], variant['host_us'])):
                result.append({'kind': 'timing', 'suite': case['suite'], 'M': case['rows'],
                    'K': case['reduction'], 'N': 1, 'layout': case['layout'], 'variant': name,
                    'trial': trial, 'calls': 8, 'gpu_us_per_call': gpu, 'host_us_per_call': host})
        if case['suite'] == 'fp32':
            result.append({'kind': 'case_pass', 'suite': 'fp32', 'M': case['rows'],
                           'K': case['reduction'], 'layout': case['layout']})
    result.append(dict(row['checks'], kind='complete'))
    return result


class EvidenceTests(unittest.TestCase):
    def test_recompute_all_published_session_medians(self):
        for gpu in ('gfx1200', 'gfx1201'):
            with self.subTest(gpu=gpu):
                actual = analyze.compare(*samples(gpu))['summary']
                reference = saved(gpu)['summary']
                self.assertEqual(len(actual), 10)
                for row in actual:
                    expected = next(r for r in reference if
                        (r['rows'], r['reduction'], r['layout']) ==
                        (row['rows'], row['reduction'], row['layout']))
                    for arm in ('retained', 'upstream'):
                        for variant in analyze.FLOAT_VARIANTS:
                            self.assertEqual(row[arm][variant], expected[arm][variant])
                    self.assertTrue(math.isclose(row['upstream_over_custom_speed_ratio'],
                                                 expected['upstream_over_custom_speed_ratio']))

    def test_complete_record_parses_and_drops_gpu_ordinal(self):
        clean = analyze.parse_records(records(), expected_gpu='RX 9060 XT', expected_suite='blas')
        self.assertEqual(clean['checks']['numeric_reference_comparisons'], 2695648)
        self.assertNotIn('device', clean['configuration'])

    def test_truncated_run_rejected(self):
        with self.assertRaises(ValueError):
            analyze.parse_records(records()[:-1])

    def test_mismatches_rejected(self):
        data = records()
        data[-1]['mismatches'] = 1
        with self.assertRaises(ValueError):
            analyze.parse_records(data)

    def test_missing_trial_rejected(self):
        data = records()
        del data[next(i for i, row in enumerate(data) if row['kind'] == 'timing')]
        with self.assertRaises(ValueError):
            analyze.parse_records(data)

    def test_missing_control_rejected(self):
        data = [r for r in records() if r.get('suite') != 'runtime']
        with self.assertRaises(ValueError):
            analyze.parse_records(data)

    def test_nonfinite_timing_rejected(self):
        data = records()
        next(r for r in data if r['kind'] == 'timing')['gpu_us_per_call'] = float('nan')
        with self.assertRaises(ValueError):
            analyze.parse_records(data)

    def test_wrong_gpu_rejected(self):
        with self.assertRaises(ValueError):
            analyze.parse_records(records(), expected_gpu='RX 9070 XT')

    def test_bad_clock_configuration_rejected(self):
        data = records()
        data[0]['clock_khz'] = 1000
        with self.assertRaises(ValueError):
            analyze.parse_records(data)

    def test_changed_executable_rejected(self):
        old, new = samples()
        new[0]['executable_sha256'] = 'different'
        with self.assertRaises(ValueError):
            analyze.compare(old, new)

    def test_changed_runtime_rejected(self):
        old, new = samples()
        new[0]['runtime_sha256']['amdhip64_7.dll'] = 'different'
        with self.assertRaises(ValueError):
            analyze.compare(old, new)


class PrivacyTests(unittest.TestCase):
    def test_common_ci_account_does_not_flag_prose(self):
        for account in ('agent', 'runner', 'root', 'user'):
            self.assertFalse(privacy.privacy_categories('An agent reads user documentation and agentic examples.',
                             privacy.automatic_identity_terms(account, '')))

    def test_private_path_still_detected_for_common_account(self):
        path = 'Z:' + chr(92) + 'Users' + chr(92) + 'agent' + chr(92) + 'file'
        self.assertIn('absolute machine path', privacy.privacy_categories(path,
                      privacy.automatic_identity_terms('agent', '')))

    def test_explicit_term_can_override_common_account_exception(self):
        self.assertIn('private identity term', privacy.privacy_categories('agent', {'agent'}))

    def test_identity_is_not_arbitrary_substring(self):
        self.assertFalse(privacy.privacy_categories('example_identity_suffix', {'example_identity'}))
        self.assertIn('private identity term', privacy.privacy_categories('example_identity', {'example_identity'}))

    def test_privacy_detector_self_checks(self):
        self.assertEqual(privacy.self_test(), 17)


class ProcessTests(unittest.TestCase):
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'marker'
            marker.write_text('preserve')
            with self.assertRaises(ValueError):
                reproduce.new_output(folder)
            self.assertEqual(marker.read_text(), 'preserve')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows process path')
    def test_timeout_stops_only_the_owned_child(self):
        children = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(subprocess.TimeoutExpired):
                reproduce.invoke([sys.executable, '-c', 'import time; time.sleep(5)'],
                                 Path(folder), 'bounded-child', timeout=.05)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == '__main__':
    unittest.main(verbosity=2)
