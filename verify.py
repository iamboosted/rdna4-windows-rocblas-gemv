"""Audit this allowlisted publication snapshot; no network, GPU or package imports."""
import argparse
import getpass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
FILES = {'.gitignore', 'README.md', 'SHA256SUMS', 'verify.py'}
FILES.update('rocblas-upstream-20261114/' + n for n in (
    'REPORT.md', 'MEASUREMENTS.md', 'RESULTS.json', 'PROTOCOL.md', 'DEPENDENCY-REVIEW.md', 'SOURCE-NOTES.md'))
FILES.update('rocblas-upstream-9070-20261114/' + n for n in (
    'REPORT.md', 'MEASUREMENTS.md', 'RESULTS.json', 'TWO-CARD-RESULTS.json', 'PROTOCOL.md', 'DEPENDENCY-REVIEW.md'))
FILES.update('reproducer/' + n for n in (
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING-NOTES.md',
    'provenance.json', 'validation.json', 'reproduce.py', 'analyze.py', 'test_reproducer.py',
    'sources/benchmark-gfx1200.hip', 'sources/benchmark-gfx1201.hip'))
COMMON_ACCOUNTS = {'agent', 'runner', 'root', 'user', 'admin', 'administrator', 'system',
                   'build', 'codex', 'test', 'ubuntu', 'windows', 'github-actions', 'defaultuser0'}
PATTERNS = {
    'absolute machine path': r'(?i)(?<![\w])[A-Z]:[\\/]|/(?:home|Users)/[^\s/]+|\\\\[A-Za-z0-9_.-]+\\',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'access token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b',
    'credential-like value': r'(?i)(?:api[_-]?key|authorization|password|secret)\s*[=:]\s*[\x22\x27]?[A-Za-z0-9_./+=-]{12,}',
    'hardware identifier': r'(?i)(?:PCI|USB)\\(?:VEN_|VID_)|S-1-5-21-[0-9-]+',
    'private key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'URL credentials': r'https?://[^\s/]+@',
    'hardware address': r'\b[A-Fa-f0-9]{2}(?::[A-Fa-f0-9]{2}){5}\b',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def automatic_identity_terms(account=None, hostname=None):
    """Generic CI account names are not distinctive identities in ordinary prose."""
    account = getpass.getuser() if account is None else account
    hostname = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', '')) if hostname is None else hostname
    return {s for s in (account, hostname) if len(s) >= 4 and s.casefold() not in COMMON_ACCOUNTS}


def privacy_categories(text, terms=()):
    found = {label for label, pattern in PATTERNS.items() if re.search(pattern, text)}
    for value in re.findall(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])', text):
        try:
            ipaddress.ip_address(value)
            found.add('network address')
        except ValueError:
            pass
    for term in terms:
        if term and re.search(r'(?i)(?<![\w])' + re.escape(term) + r'(?![\w])', text):
            found.add('private identity term')
    return found


def self_test():
    checks = 0
    def expect(condition):
        nonlocal checks
        require(condition, 'Privacy self-test failed')
        checks += 1
    for account in ('agent', 'runner', 'root', 'user', 'build', 'codex'):
        expect(not privacy_categories('An agent can run a benchmark; agentic tools and user documentation.',
                                      automatic_identity_terms(account, '')))
    expect('private identity term' in privacy_categories('example_identity', {'example_identity'}))
    expect(not privacy_categories('example_identity_suffix', {'example_identity'}))
    expect('private identity term' in privacy_categories('agent', {'agent'}))  # Explicit deny-term still works.
    fake_path = 'Z:' + chr(92) + 'Users' + chr(92) + 'agent' + chr(92) + 'note'
    expect('absolute machine path' in privacy_categories(fake_path, automatic_identity_terms('agent', '')))
    expect('absolute machine path' in privacy_categories('/' + 'home/' + 'agent/note'))
    expect('network address' in privacy_categories('.'.join(('192', '0', '2', '1'))))
    expect('email' in privacy_categories('example' + '@' + 'example.invalid'))
    expect('access token' in privacy_categories('gh' + 'p_' + 'A' * 30))
    expect('credential-like value' in privacy_categories('api' + '_key=' + 'A' * 24))
    expect('hardware identifier' in privacy_categories('PCI' + chr(92) + 'VEN_' + '1234'))
    expect(not privacy_categories('5.8.0.3d8f334b; 5.9.0.4e81d94; gfx1200; https://github.com/ROCm/rocm-libraries'))
    return checks


def numeric_check(data):
    if isinstance(data, dict):
        require(not set(data) & {'pid', 'hostname', 'username', 'serial', 'device', 'adlxIndex',
                                'environment_sha256', 'start_ns', 'end_ns'}, 'Private metadata key')
        for value in data.values():
            numeric_check(value)
    elif isinstance(data, list):
        for value in data:
            numeric_check(value)
    elif isinstance(data, float):
        require(math.isfinite(data), 'Nonfinite result')


def verify(root, extra_terms=()):
    terms = automatic_identity_terms() | set(extra_terms)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '.git' not in p.relative_to(root).parts and '__pycache__' not in p.relative_to(root).parts}
    require(actual == FILES, 'Publication inventory differs from the explicit allowlist')
    hashes = {}
    for name in sorted(FILES):
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe publication path')
        data = path.read_bytes()
        text = data.decode('utf-8')
        categories = privacy_categories(text, terms)
        require(not categories, name + ': ' + ', '.join(sorted(categories)))
        hashes[name] = hashlib.sha256(data).hexdigest()
        if name.endswith('.json'):
            numeric_check(json.loads(text))
        if name.endswith('.md'):
            for target in re.findall(r'\]\(([^)]+)\)', text):
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith('#'):
                    continue
                linked = (path.parent / unquote(parsed.path)).resolve()
                require(linked.is_relative_to(root.resolve()), 'Local link escapes publication')
                require(linked.relative_to(root.resolve()).as_posix() in FILES, 'Local link missing: ' + name)
    listed = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(name not in listed and name in hashes and digest == hashes[name], 'Checksum mismatch')
        listed[name] = digest
    require(set(listed) == FILES - {'SHA256SUMS'}, 'Checksum inventory differs')
    for folder in ('rocblas-upstream-20261114', 'rocblas-upstream-9070-20261114'):
        report = json.loads((root / folder / 'RESULTS.json').read_text())
        require(len(report['runs']) == 8 and len(report['summary']) == 10, 'Report coverage differs')
        require(report['all_run_totals'] == {'q8_bit_comparisons': 8096384,
                'numeric_reference_comparisons': 16225996, 'guard_values_checked': 37376}, 'Report counts differ')
        require(all(r['checks']['mismatches'] == 0 for r in report['runs']), 'Report correctness differs')
    provenance = json.loads((root / 'reproducer/provenance.json').read_text())
    for target in ('gfx1200', 'gfx1201'):
        require(hashes['reproducer/sources/benchmark-' + target + '.hip'] ==
                provenance['benchmark_source_sha256'][target], 'Tested benchmark source differs')
    first = (root / 'reproducer/sources/benchmark-gfx1200.hip').read_bytes()
    second = (root / 'reproducer/sources/benchmark-gfx1201.hip').read_bytes()
    require(second.replace(b'9070', b'9060').replace(b'gfx1201', b'gfx1200') == first,
            'GPU variants differ in more than device identity')
    return hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deny-term', action='append', default=[], help='Extra identity term; never echoed')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    hashes = verify(ROOT, args.deny_term)
    count = self_test() if args.self_test else 0
    print(json.dumps({'status': 'pass', 'publication_files': len(hashes),
                      'privacy_scan': 'pass', 'self_tests': count,
                      'scope': 'Current allowlisted files; not deleted Git objects, caches, forks or attachments'}))
