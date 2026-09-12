"""Assemble exact current and explicitly inherited coverage rows; never parse models."""
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
FROZEN = Path('/private/tmp/unfold-s8-focused-fb1f871')
CAMPAIGN = Path('/private/tmp/unfold-s8-current-captures-fb1f871')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
from model_unfolder.evidence.coverage import coverage_problems


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


baseline_path = ROOT / 'verification/receipts/S8-final-coverage-7168440/current-coverage.json'
baseline = read(baseline_path)
guards_path = ROOT / 'verification/receipts/S8-coverage-derivation-preparation-fb1f871/14-inputs-and-historical-rows.json'
guards = {r['input']: r for r in read(guards_path)['targets']}
assert len(guards) == 14
assert read(ROOT / 'verification/receipts/S8-global-debt-fb1f871/result.json')['findings'] == []
bloom_final = CAMPAIGN / 'bloom-final-replay-v3/result.json'
assert read(bloom_final)['status'] == 'PASS'
assert read(bloom_final)['semantic_event_sequence_unchanged_except_six_fresh_context_nonces']
rows, phases = [], []
reports = OUT / 'actual-reports'
reports.mkdir(exist_ok=True)
for previous in baseline['models']:
    slug = Path(previous['input']).stem
    input_path = FROZEN / previous['input']
    assert input_path.read_bytes() == (ROOT / previous['input']).read_bytes()
    if previous['input'] in guards:
        guard = guards[previous['input']]
        assert sha(input_path) == guard['input_sha256']
        assert previous == guard['coverage_row']
        row = previous
        phase = 'historical716_actual_row_current_source_input_applicability_guard'
        report_path = None
    else:
        if slug == 'bloom':
            report_path = Path('/private/tmp/unfold-s8-bloom-current-736fdab/capture/sable-report.json')
            phase = 'actual736_sable_row_plus_current_same_IR_geometry_and_semantic_event_derivation'
        elif slug == 'sd-v1-4':
            report_path = CAMPAIGN / 'sd-v1-4-cwd-corrected/sable-report.json'
            phase = 'actual_current_fb1_sable'
        elif (CAMPAIGN / 'example-sable' / slug / 'sable-report.json').exists():
            report_path = CAMPAIGN / 'example-sable' / slug / 'sable-report.json'
            phase = 'actual_current_fb1_sable'
        else:
            report_path = CAMPAIGN / slug / 'sable-report.json'
            phase = 'actual_current_fb1_sable'
        report = read(report_path)
        row = {key: previous[key] for key in ('cohort', 'input', 'model')}
        row.update({key: report['coverage'][key] for key in
                    ('proven', 'flagged', 'silent', 'flagged_findings', 'silent_findings')})
        row['rendered_name'] = report['model']
        (reports / (slug + '.json.gz')).write_bytes(gzip.compress(report_path.read_bytes(), mtime=0))
    assert row == previous, (slug, row, previous)
    rows.append(row)
    phases.append({'input': previous['input'], 'input_sha256': sha(input_path),
                   'phase': phase, 'full_row_equal716': True,
                   'report_path': str(report_path) if report_path else None,
                   'report_sha256': sha(report_path) if report_path else None})
assert len(rows) == 44
assert sum(p['phase'] == 'actual_current_fb1_sable' for p in phases) == 29
assert sum(p['phase'].startswith('historical716') for p in phases) == 14
document = {**baseline, 'models': rows}
assert coverage_problems(document) == []
spec = importlib.util.spec_from_file_location('s8_current_coverage_serializer', FROZEN / 'scripts/coverage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rendered = module._render(document)
assert rendered == (ROOT / 'coverage.json').read_text() == baseline_path.read_text()
write('coverage.json', document)
write('per-model-phases.json', phases)
write('result.json', {'status': 'PASS_derived44_existing_document_validation',
    'actual_current_sable_rows': 29, 'actual736_plus_current_geometry_rows': 1,
    'historical716_current_applicability_guard_rows': 14, 'denominator': 44,
    'proven': sum(r['proven'] for r in rows), 'flagged': sum(r['flagged'] for r in rows),
    'silent': sum(r['silent'] for r in rows), 'all44_complete_rows_equal716': True,
    'existing_coverage_problems_empty': True, 'existing_serializer_bytes_equal': True,
    'baseline_sha256': sha(baseline_path), 'no_new_parse_in_this_derivation': True,
    'fresh44_generator_or_check_execution': False,
    'bloom_raw_event_bytes_equal': False,
    'bloom_semantic_events_equal_except_six_fresh_context_nonces': True})
print('Derived44 PASS: 645 proven / 294 flagged / 0 silent; 29 actual + 1 derived Bloom + 14 guarded historical')
