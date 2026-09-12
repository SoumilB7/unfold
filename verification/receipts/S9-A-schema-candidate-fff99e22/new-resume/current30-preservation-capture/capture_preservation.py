"""Exact30 observation of the unchanged 52-test preservation lane.

No verifier, unfold, Sable, renderer or baseline writer is invoked by this tool.
Only existing calls are delegated once, returning the original object.
Saved reports enumerate all canonical deltas without accepting outputs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback


_ACTIVE = None
_OUTPUT_ENV = 'UNFOLD_S9A30_OUTPUT_CAPTURE'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def serial(value):
    if dataclasses.is_dataclass(value):
        return {field.name: serial(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(serial(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def json_bytes(value):
    return (json.dumps(serial(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def tree_pins(root):
    """Fresh membership and bytes, including packaged non-Python assets."""
    root = Path(root)
    return {str(path.relative_to(root)): {'sha256': sha(path.read_bytes()),
                                          'resolved': str(path.resolve())}
            for path in sorted(root.rglob('*')) if path.is_file()
            and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}}


def source_pins(repo):
    return {name: tree_pins(repo / name) for name in
            ('model_unfolder', 'physics', 'test_support', 'scripts')}


def json_delta(before, after, path=''):
    """Full ordered JSON comparison; additions/removals retain complete subtrees."""
    if type(before) is not type(after):
        return [{'path': path, 'operation': 'replace_type', 'before': before, 'after': after}]
    if isinstance(before, dict):
        rows = []
        for key in sorted(set(before) | set(after)):
            child = path + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if key not in before: rows.append({'path': child, 'operation': 'add', 'after': after[key]})
            elif key not in after: rows.append({'path': child, 'operation': 'remove', 'before': before[key]})
            else: rows.extend(json_delta(before[key], after[key], child))
        return rows
    if isinstance(before, list):
        rows = []
        for i in range(max(len(before), len(after))):
            child = path + '/' + str(i)
            if i >= len(before): rows.append({'path': child, 'operation': 'add', 'after': after[i]})
            elif i >= len(after): rows.append({'path': child, 'operation': 'remove', 'before': before[i]})
            else: rows.extend(json_delta(before[i], after[i], child))
        return rows
    return [] if before == after else [{'path': path, 'operation': 'replace', 'before': before, 'after': after}]


class Capture:
    def __init__(self, output, expected_manifest):
        self.output = Path(output).resolve()
        self.expected_path = Path(expected_manifest).resolve()
        self.repo = self.expected_path.parent.parent
        self.expected_raw = self.expected_path.read_bytes()
        self.expected = json.loads(self.expected_raw)
        self.corpus = self.repo / 'tests/sable_test_corpus'
        self.current = None
        self.phase = 'verify_expected_witness'
        self.pages = []
        self.patches = []
        self.errors = []
        self.cases = []
        self.test_reports = []
        self.collection = []
        self.output.mkdir(parents=True, exist_ok=False)
        self.before = self.pins()
        self.save('pins-before.json', json_bytes(self.before))
        self.save('expected.executed.json', self.expected_raw)
        self.save('capture_preservation.executed.py', Path(__file__).read_bytes())
        self.save('launch.json', json_bytes({'argv': sys.argv, 'pid': os.getpid(),
            'worker': os.environ.get('PYTEST_XDIST_WORKER', 'controller'),
            'scope': 'Observer only; original pytest assertions and verifier findings unchanged.'}))
        try:
            from test_support import preservation
            from model_unfolder.diagram import Diagram
            from model_unfolder import preview
            from model_unfolder.preview import svg_views, _visual_hash
            for module in (preservation, preview, sys.modules[Diagram.__module__]):
                if not Path(module.__file__).resolve().is_relative_to(self.repo):
                    raise ValueError('Imported observation target is outside the pinned checkout')
            self.P = preservation
            self.svg_views, self.visual_hash = svg_views, _visual_hash
            self.raw_meta = preservation.html_meta
            self.patch(preservation, 'verify_expected_witness', self.verify_wrapper)
            for name in ('canonical_surfaces', '_view_hashes', 'gallery_witness'):
                self.patch(preservation, name, lambda original, name=name: self.value_wrapper(original, name))
            self.patch(Diagram, 'to_html', self.html_wrapper)
        except BaseException:
            self.finish()
            raise

    def pins(self):
        return {'production_and_support': source_pins(self.repo),
                'corpus_and_galleries': tree_pins(self.corpus),
                'historical_baseline': tree_pins(self.repo / 'tests/preservation_baseline'),
                'expected_manifest': sha(self.expected_path.read_bytes()),
                'preservation_test': sha((self.repo / 'tests/test_preservation.py').read_bytes()),
                'observer': sha(Path(__file__).read_bytes())}

    def save(self, relative, raw):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {'path': relative, 'sha256': sha(raw), 'bytes': len(raw)}

    def record_error(self):
        detail = traceback.format_exc()
        self.errors.append(detail)
        if self.current is not None:
            self.current['capture_errors'].append(detail)

    def observe(self, callback):
        # Observation failure invalidates its packet, not the original result.
        try:
            callback()
        except Exception:
            self.record_error()

    def patch(self, owner, name, factory):
        original = getattr(owner, name)
        wrapper = factory(original)
        self.patches.append((owner, name, original, wrapper))
        setattr(owner, name, wrapper)

    def verify_wrapper(self, original):
        def wrapped(corpus_dir, manifest_path, slug):
            if Path(manifest_path).resolve() != self.expected_path:
                return original(corpus_dir, manifest_path, slug)
            if self.current is not None:
                raise RuntimeError('Unexpected nested original witness verification')
            case = {'slug': slug, 'capture_errors': [], 'calls': [], 'renders': [], 'packs': [],
                    'original_findings': None, 'original_exception': None}
            self.cases.append(case)
            self.current = case
            prefix = f'cases/{slug}/attempt-{sum(row["slug"] == slug for row in self.cases):02d}'
            case['prefix'] = prefix
            def inputs():
                if slug not in self.expected['witnesses'] or re.fullmatch(r'[a-z0-9-]+', slug) is None:
                    raise ValueError('Unexpected witness address')
                if Path(corpus_dir).resolve() != self.corpus.resolve():
                    raise ValueError('Witness corpus differs from pinned original corpus')
                raw = (self.corpus / f'{slug}.json').read_bytes()
                case['input'] = self.save(prefix + '/input.json', raw)
                case['input_matches_expected'] = self.P.input_sha256(self.corpus / f'{slug}.json') == self.expected['witnesses'][slug]['input_sha256']
            self.observe(inputs)
            try:
                result = original(corpus_dir, manifest_path, slug)
                self.observe(lambda: case.update(original_findings=list(result)))
                return result
            except BaseException as exc:
                case['original_exception'] = {'type': type(exc).__name__, 'message': str(exc)}
                raise
            finally:
                self.observe(lambda: self.finish_case(case))
                self.observe(lambda: self.save(case['prefix'] + '/result.json', json_bytes(case)))
                self.current = None
        return wrapped

    def value_wrapper(self, original, name):
        def wrapped(*args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            old_phase = self.phase
            self.phase = name
            try:
                result = original(*args, **kwargs)
                def retain():
                    index = len(self.current['calls'])
                    artifact = self.save(self.current['prefix'] + f'/call-{index:02d}-{name}.json',
                                         self.P._canon_bytes(result))
                    self.current['calls'].append({'name': name, 'artifact': artifact})
                self.observe(retain)
                return result
            finally:
                self.phase = old_phase
        return wrapped

    def html_wrapper(self, original):
        def wrapped(diagram, *args, **kwargs):
            if self.current is None:
                return original(diagram, *args, **kwargs)
            pending = {'phase': self.phase, 'pack_indices': []}
            self.pages.append(pending)
            try:
                result = original(diagram, *args, **kwargs)
                def retain():
                    index = len(self.current['renders'])
                    prefix = self.current['prefix'] + f'/render-{index:03d}'
                    row = {**pending, 'actual': self.save(prefix + '-actual.html', result.encode())}
                    # Read already materialized contexts; render_events() could
                    # trigger another standalone render and is NOT called.
                    row['events'] = self.save(prefix + '-events.json', json_bytes({
                        str(key): list(context.events)
                        for key, context in diagram._render_contexts.items()}))
                    self.current['renders'].append(row)
                self.observe(retain)
                return result
            finally:
                self.pages.pop()
        return wrapped

    def load(self, artifact):
        raw = (self.output / artifact['path']).read_bytes()
        if sha(raw) != artifact['sha256'] or len(raw) != artifact['bytes']:
            raise ValueError('Captured artifact bytes changed')
        return raw

    def finish_case(self, case):
        expected = self.expected['witnesses'][case['slug']]
        values = {}
        for call in case['calls']:
            values.setdefault(call['name'], []).append(json.loads(self.load(call['artifact'])))
        case['call_counts'] = {key: len(rows) for key, rows in values.items()}
        if any(len(values.get(name, [])) != 1 for name in
               ('canonical_surfaces', '_view_hashes', 'gallery_witness')):
            raise ValueError('Original witness did not return each expected surface producer once')
        docs = values['canonical_surfaces'][0]
        docs['gallery'] = values['gallery_witness'][0]
        case['surface_sha256'] = {name: sha(self.P._canon_bytes(value)) for name, value in docs.items()
                                  if value is not None}
        case['surface_deltas'] = [name for name, value in expected['surfaces'].items()
                                  if case['surface_sha256'].get(name) != value]
        case['views_equal_expected'] = values['_view_hashes'][0] == expected['views']
        case['actual_surfaces'] = self.save(case['prefix'] + '/actual-surfaces.json', self.P._canon_bytes(docs))
        case['views'] = values['_view_hashes'][0]
        case['view_deltas'] = json_delta(expected['views'], case['views'])
        case['surface_comparisons'] = {}
        for surface in sorted(set(expected['surfaces']) | set(docs)):
            baseline_path = self.repo / 'tests/preservation_baseline' / case['slug'] / (surface + '.json')
            expected_sha = expected['surfaces'].get(surface)
            baseline = None
            stored_sha = None
            if baseline_path.is_file():
                baseline_raw = baseline_path.read_bytes()
                baseline = json.loads(baseline_raw)
                stored_sha = sha(self.P._canon_bytes(baseline))
                self.save(case['prefix'] + '/historical-documents/' + surface + '.json', baseline_raw)
            matching = stored_sha == expected_sha and expected_sha is not None
            row = {'expected_sha256': expected_sha,
                   'actual_sha256': case['surface_sha256'].get(surface),
                   'equal_expected': case['surface_sha256'].get(surface) == expected_sha and expected_sha is not None,
                   'stored_document_sha256': stored_sha,
                   'stored_document_matches_expected': matching,
                   'baseline_document_status': ('EXACT_EXPECTED_DOCUMENT' if matching else
                         'HISTORICAL_DOCUMENT_HASH_MISMATCH' if stored_sha else 'DOCUMENT_MISSING')}
            if stored_sha is not None:
                delta = json_delta(baseline, docs.get(surface))
                row['document_delta'] = self.save(case['prefix'] + '/surface-deltas/' + surface + '.json', json_bytes(delta))
                row['document_delta_count'] = len(delta)
                row['document_delta_authority'] = ('COMMITTED_EXPECTED_DOCUMENT' if matching else
                    'HISTORICAL_ONLY_NOT_THE_COMMITTED_EXPECTED_SURFACE')
            case['surface_comparisons'][surface] = row
        for row in case['renders']:
            actual = self.load(row['actual']).decode()
            row['html_meta'] = self.raw_meta(actual)
            row['is_original_html_surface'] = row['html_meta'] == docs['html_meta']
            # Save exact SVG strings from the established production view iterator.
            # Its established visual hash remains a separate field; raw bytes are retained.
            view_rows = []
            for index, (label, svg) in enumerate(self.svg_views(actual)):
                artifact = self.save(row['actual']['path'].replace('-actual.html', f'-view-{index:04d}.svg'), svg.encode())
                view_rows.append({'label': label, 'visual_sha256': self.visual_hash(svg), 'raw': artifact})
            row['all_views'] = self.save(row['actual']['path'].replace('-actual.html', '-all-views.json'), json_bytes(view_rows))
        case['candidate'] = {'input_sha256': self.P.input_sha256(self.corpus / (case['slug'] + '.json')),
                             'surfaces': case['surface_sha256'], 'views': case['views']}
        case['authority'] = 'Unapproved observed candidate. Original verifier findings retained; historical mismatches are not accepted baseline data.'
        self.save(case['prefix'] + '/result.json', json_bytes(case))

    def finish(self):
        for owner, name, original, wrapper in reversed(self.patches):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
            else:
                self.errors.append(f'Observed target changed before restoration: {name}')
        self.patches.clear()
        after = None
        try:
            after = self.pins()
            self.save('pins-after.json', json_bytes(after))
        except Exception:
            self.record_error()
        result = {'capture_complete': not self.errors and before_equal(self.before, after),
                  'before_after_pins_equal': before_equal(self.before, after),
                  'capture_errors': self.errors, 'cases': self.cases,
                  'collection': self.collection, 'pytest_reports': self.test_reports,
                  'authority': 'Observation packet only. Original pytest outcome and raw findings remain authoritative.'}
        self.save('capture-result.json', json_bytes(result))
        return result


def before_equal(before, after):
    return after is not None and before == after


def install_capture(output, expected_manifest):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError('Output observer already installed')
    _ACTIVE = Capture(output, expected_manifest)
    return _ACTIVE


def finish_capture():
    global _ACTIVE
    if _ACTIVE is None:
        return None
    capture, _ACTIVE = _ACTIVE, None
    return capture.finish()


def pytest_configure(config):
    output = os.environ.get(_OUTPUT_ENV)
    if (not output or config.getoption('collectonly') or not any(
            Path(str(arg).split('::')[0]).name == 'test_preservation.py' for arg in config.args)):
        return
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'controller')
    if re.fullmatch(r'[A-Za-z0-9_-]+', worker) is None:
        raise ValueError('Invalid pytest worker namespace')
    repo = Path(config.rootpath)
    install_capture(Path(output) / worker, repo / 'tests/preservation_expected_manifest.json')


def pytest_collection_finish(session):
    if _ACTIVE is not None:
        _ACTIVE.collection = [item.nodeid for item in session.items
                              if Path(item.nodeid.split('::')[0]).name == 'test_preservation.py']


def pytest_runtest_logreport(report):
    if _ACTIVE is None or Path(report.nodeid.split('::')[0]).name != 'test_preservation.py':
        return
    row = {'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
           'longrepr': str(report.longrepr) if report.longrepr else None}
    _ACTIVE.test_reports.append(row)
    _ACTIVE.observe(lambda: _ACTIVE.save(
        f'pytest/{sha(report.nodeid.encode())}-{report.when}.json', json_bytes(row)))


def pytest_sessionfinish(session, exitstatus):
    if _ACTIVE is not None:
        _ACTIVE.observe(lambda: _ACTIVE.save('pytest-session.json', json_bytes({'exitstatus': int(exitstatus)})))
        finish_capture()


def report_capture(output):
    root = Path(output)
    packet = root / 'controller'
    result = json.loads((packet / 'capture-result.json').read_bytes())
    expected = json.loads((packet / 'expected.executed.json').read_bytes())
    cases = result['cases']
    tests = {}
    for row in result['pytest_reports']:
        if row['when'] == 'call' or row['outcome'] in {'failed', 'skipped'}:
            tests.setdefault(row['nodeid'], []).append(row)
    counts = {slug: sum(case['slug'] == slug for case in cases) for slug in expected['witnesses']}
    artifact_checks = []
    def check(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                path = packet / value['path']
                if not path.resolve().is_relative_to(packet.resolve()):
                    raise ValueError('Captured artifact address escapes packet')
                raw = path.read_bytes()
                if sha(raw) != value['sha256'] or len(raw) != value['bytes']:
                    raise ValueError('Captured artifact changed')
                artifact_checks.append(value['path'])
                if value['path'].endswith('-all-views.json'):
                    check(json.loads(raw))
            else:
                for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(cases)
    complete = (result['capture_complete'] and len(result['collection']) == 52 and len(tests) == 52
                and len(cases) == 29 and len(counts) == 29 and all(n == 1 for n in counts.values())
                and all(not c['capture_errors'] and 'candidate' in c for c in cases))
    candidates = {'witness_count': 29, 'witnesses': {c['slug']: c['candidate'] for c in cases if 'candidate' in c},
                  'versions': expected.get('versions'),
                  'authority': 'UNAPPROVED_CANDIDATE_ONLY',
                  'versions_origin': 'Copied expected receipt metadata for context, not a measured current version claim.'}
    (root / 'candidate-manifest.UNAPPROVED.json').write_bytes(json_bytes(candidates))
    report = {'packet_complete': complete, 'collected_count': len(result['collection']),
              'reported_test_count': len(tests), 'witness_counts': counts,
              'original_test_reports': tests, 'cases': cases,
              'raw_preservation_passed': complete and all(not c['original_findings'] and c['original_exception'] is None for c in cases)
                   and all(r['outcome'] == 'passed' for rows in tests.values() for r in rows),
              'verified_artifact_paths': artifact_checks,
              'authority': 'Exact original 52-test outcomes and canonical candidate surfaces, no approval or writer.'}
    (root / 'report.json').write_bytes(json_bytes(report))
    return report
