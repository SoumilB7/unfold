"""Transparent observation of the existing final preservation pytest lane.

No verifier, unfold, Sable, renderer or baseline writer is invoked by this tool.
Only existing calls are delegated once, returning the original object. The
--report mode reads captured artifacts; it cannot generate model observations.
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
_OUTPUT_ENV = 'UNFOLD_S81_OUTPUT_CAPTURE'


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
            from model_unfolder.renderers.html import card_payload
            from model_unfolder import preview
            from model_unfolder.preview import svg_views, _visual_hash
            for module in (preservation, card_payload, preview, sys.modules[Diagram.__module__]):
                if not Path(module.__file__).resolve().is_relative_to(self.repo):
                    raise ValueError('Imported observation target is outside the pinned checkout')
            self.P, self.codec = preservation, card_payload
            self.svg_views, self.visual_hash = svg_views, _visual_hash
            self.raw_meta = preservation.html_meta
            self.patch(preservation, 'verify_expected_witness', self.verify_wrapper)
            for name in ('canonical_surfaces', '_view_hashes', 'gallery_witness'):
                self.patch(preservation, name, lambda original, name=name: self.value_wrapper(original, name))
            self.patch(Diagram, 'to_html', self.html_wrapper)
            self.patch(card_payload, 'pack_card_payloads', self.pack_wrapper)
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

    def pack_wrapper(self, original):
        def wrapped(fragment, mount_id):
            result = original(fragment, mount_id)
            if self.current is not None:
                def retain():
                    index = len(self.current['packs'])
                    prefix = self.current['prefix'] + f'/pack-{index:03d}'
                    row = {'phase': self.phase, 'mount_id': mount_id,
                           'canonical': self.save(prefix + '-canonical.html', fragment.encode()),
                           'actual': self.save(prefix + '-actual.html', result.encode())}
                    row['inverse_exact'] = self.codec.expand_card_payloads(result) == fragment
                    self.current['packs'].append(row)
                    if self.pages:
                        self.pages[-1]['pack_indices'].append(index)
                self.observe(retain)
            return result
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
        eager_by_raw = {}
        canonical_metas = []
        for row in case['renders']:
            actual = self.load(row['actual']).decode()
            eager = actual
            if row['pack_indices']:
                for index in row['pack_indices']:
                    pack = case['packs'][index]
                    packed, original = self.load(pack['actual']).decode(), self.load(pack['canonical']).decode()
                    if actual.count(packed) != 1 or not pack['inverse_exact']:
                        raise ValueError('Page does not contain its uniquely observed packed fragment')
                    eager = eager.replace(packed, original, 1)
                eager_by_raw[sha(actual.encode())] = eager
            elif sha(actual.encode()) in eager_by_raw:
                eager = eager_by_raw[sha(actual.encode())]
                row['binding'] = 'identical returned cached page, previously observed pack input'
            elif self.codec.expand_card_payloads(actual) != actual:
                raise ValueError('Packed page has no observed pack input')
            row['inverse_exact'] = self.codec.expand_card_payloads(actual) == eager
            row['canonical'] = self.save(row['actual']['path'].replace('-actual.html', '-canonical.html'), eager.encode())
            raw_meta, eager_meta = self.raw_meta(actual), self.raw_meta(eager)
            row['raw_html_meta'] = raw_meta
            row['canonical_html_meta'] = eager_meta
            row['raw_meta_sha256'] = sha(self.P._canon_bytes(raw_meta))
            row['canonical_meta_sha256'] = sha(self.P._canon_bytes(eager_meta))
            row['view_ids_removed'] = sorted(set(eager_meta['view_ids']) - set(raw_meta['view_ids']))
            row['view_ids_added'] = sorted(set(raw_meta['view_ids']) - set(eager_meta['view_ids']))
            row['click_targets_removed'] = sorted(set(eager_meta['click_targets']) - set(raw_meta['click_targets']))
            row['click_targets_added'] = sorted(set(raw_meta['click_targets']) - set(eager_meta['click_targets']))
            raw_views = [(name, self.visual_hash(svg), sha(svg.encode()))
                         for name, svg in self.svg_views(actual)]
            eager_views = [(name, self.visual_hash(svg), sha(svg.encode()))
                           for name, svg in self.svg_views(eager)]
            row['all_svg_occurrences_equal'] = raw_views == eager_views
            row['svg_occurrences'] = self.save(row['actual']['path'].replace('-actual.html', '-svg-occurrences.json'), json_bytes(raw_views))
            # Only the exact page whose raw metadata was returned by the
            # original canonical_surfaces call is the baseline HTML surface.
            if raw_meta == docs['html_meta']:
                canonical_metas.append(row['canonical_meta_sha256'])
                row['is_original_html_surface'] = True
        case['canonical_html_matches_old_expected'] = bool(canonical_metas) and all(
            value == expected['surfaces']['html_meta'] for value in canonical_metas)
        case['semantic_parity'] = (case.get('input_matches_expected') is True
            and not (set(case['surface_deltas']) - {'html_meta'})
            and case['views_equal_expected'] and case['canonical_html_matches_old_expected']
            and bool(case['renders']) and all(row['inverse_exact'] and row['all_svg_occurrences_equal']
                                            for row in case['renders'])
            and not case['capture_errors'] and case['original_exception'] is None)
        case['cause'] = ('Observed reversible card-payload transport and lazy interaction script; '
                         'raw structural hash/IDs are retained deltas, not normalized away.'
                         if case['semantic_parity'] and case['surface_deltas'] == ['html_meta'] else
                         'No surface delta.' if case['semantic_parity'] and not case['surface_deltas'] else
                         'Unresolved: do not attribute or approve automatically.')
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
    workers = []
    cases = []
    expected = None
    tests = {}
    artifact_checks = []
    for path in sorted(root.glob('*/capture-result.json')):
        worker = json.loads(path.read_bytes())
        for report in worker['pytest_reports']:
            if report['when'] == 'call' or report['outcome'] == 'skipped':
                previous = tests.setdefault(report['nodeid'], report)
                if previous != report:
                    raise ValueError('Conflicting original pytest outcomes')
        def check_artifacts(value):
            if isinstance(value, dict):
                if set(value) == {'path', 'sha256', 'bytes'}:
                    artifact_path = path.parent / value['path']
                    if not artifact_path.resolve().is_relative_to(path.parent.resolve()):
                        raise ValueError('Artifact path escapes worker namespace')
                    raw = artifact_path.read_bytes()
                    if sha(raw) != value['sha256'] or len(raw) != value['bytes']:
                        raise ValueError('Captured artifact identity mismatch')
                    artifact_checks.append(str(artifact_path.relative_to(root)))
                else:
                    for child in value.values(): check_artifacts(child)
            elif isinstance(value, list):
                for child in value: check_artifacts(child)
        check_artifacts(worker['cases'])
        before = json.loads((path.parent / 'pins-before.json').read_bytes())
        after = json.loads((path.parent / 'pins-after.json').read_bytes())
        if before != after or sha((path.parent / 'expected.executed.json').read_bytes()) != before['expected_manifest']:
            raise ValueError('Worker source/input final pins differ')
        if sha((path.parent / 'capture_preservation.executed.py').read_bytes()) != before['observer']:
            raise ValueError('Executed observer bytes differ from initial pin')
        workers.append({'path': str(path.relative_to(root)), 'sha256': sha(path.read_bytes()),
                        'capture_complete': worker['capture_complete']})
        value = json.loads((path.parent / 'expected.executed.json').read_bytes())
        if expected is not None and expected != value:
            raise ValueError('Worker expected manifests differ')
        expected = value
        cases.extend(worker['cases'])
    if expected is None:
        raise ValueError('No completed capture workers')
    counts = {slug: sum(row['slug'] == slug for row in cases) for slug in expected['witnesses']}
    complete = (all(row['capture_complete'] for row in workers)
                and set(row['slug'] for row in cases) == set(expected['witnesses'])
                and expected['witness_count'] == 29
                and len(cases) == 29 and all(value == 1 for value in counts.values())
                and len(tests) == 52 and all(row['outcome'] != 'skipped' for row in tests.values()))
    result = {'packet_complete': complete, 'witness_counts': counts, 'workers': workers,
              'original_pytest_outcomes': tests, 'original_test_count': len(tests),
              'verified_artifact_paths': artifact_checks,
              'semantic_parity': complete and all(row.get('semantic_parity') for row in cases),
              'raw_baseline_status': ('FAIL' if any(row['original_findings'] or row['original_exception']
                                                   for row in cases) else 'PASS'),
              'original_pytest_status': ('FAIL' if any(row['outcome'] != 'passed'
                                                       for row in tests.values()) else 'PASS'),
              'raw_baseline_findings': {row['slug']: row['original_findings'] for row in cases},
              'cases': cases, 'authority': 'No rebless or new baseline authority. Raw original failures preserved.'}
    (root / 'report.json').write_bytes(json_bytes(result))
    return result


if __name__ == '__main__':
    # Exit zero means a complete review packet with semantic parity, never a
    # replacement PASS for the original raw preservation pytest lane.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Read completed observation artifacts only')
    args = parser.parse_args()
    result = report_capture(args.report)
    print(json.dumps({'packet_complete': result['packet_complete'],
                      'semantic_parity': result['semantic_parity'],
                      'raw_baseline_status': result['raw_baseline_status'],
                      'original_pytest_status': result['original_pytest_status'],
                      'raw_baseline_findings': result['raw_baseline_findings']}, indent=2))
    raise SystemExit(0 if result['packet_complete'] and result['semantic_parity'] else 1)
