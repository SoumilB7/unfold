"""Data-only audit: inspect only completed candidate gzip artifacts; no project imports."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
CURRENT = Path('/private/tmp/unfold-s9a-twentysixth/matrix/models')
ACTIVE = ROOT / 'verification/s7/models'
PREVIOUS = ROOT / 'verification/receipts/S9-A-schema-candidate-fff99e22/twentysecond/matrix/models'
SLUGS = ('deepseek-v4-flash', 'gemma-3n-e2b', 'nemotron-h-8b', 'bloom')


def read(path):
    raw = path.read_bytes()
    return raw, json.loads(gzip.decompress(raw))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(payload):
    out = {r['provenance']['instance_path']: r for r in payload['table']['occurrences']}
    assert len(out) == len(payload['table']['occurrences'])
    return out


def placement(row):
    p = row['projection']
    return {'kind': p['kind'], 'parent': p['parent'],
            'fact_keys': sorted(set(p['fact_keys']) | set(p['unqualified_fact_keys'])),
            'decoder_norm_cited': 'decoder.layer.norm_kind' in set(p['fact_keys']) | set(p['unqualified_fact_keys'])}


def kind_changes(a, b):
    return [{'instance_path': k, 'before': placement(a[k]), 'after': placement(b[k])}
            for k in sorted(a.keys() & b.keys())
            if a[k]['projection']['kind'] != b[k]['projection']['kind']]


def changes(a, b, field):
    return [k for k in sorted(a.keys() & b.keys()) if a[k][field] != b[k][field]]


summary = {'candidate': 26, 'status': 'partial data-only audit; no verdict or approval', 'models': [], 'not_ready': []}
for slug in SLUGS:
    candidate = CURRENT / (slug + '.json.gz')
    if not candidate.exists():
        summary['not_ready'].append(slug)
        continue
    # The generator writes these files once. Do not read a file while any
    # process still owns it, then pin stable bytes and require a valid gzip CRC.
    check = subprocess.run(['lsof', '-t', '--', str(candidate)], capture_output=True, text=True)
    if check.returncode != 1 or check.stdout.strip():
        summary['not_ready'].append(slug)
        continue
    before = candidate.stat()
    raw_new, new = read(candidate)
    after = candidate.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    raw_old, old = read(ACTIVE / candidate.name)
    raw_previous, previous = read(PREVIOUS / candidate.name)
    a, b, c = rows(old), rows(previous), rows(new)
    original = kind_changes(a, b)
    triple = [{'instance_path': r['instance_path'], 'active': placement(a[r['instance_path']]),
               'candidate22': placement(b[r['instance_path']]), 'candidate26': placement(c[r['instance_path']])}
              for r in original if r['instance_path'] in c]
    active_delta = kind_changes(a, c)
    previous_delta = kind_changes(b, c)
    report = {'slug': slug, 'status': 'unblessed actual data, source cause review required for deltas',
              'inputs': {'active_sha256': sha(raw_old), 'candidate22_sha256': sha(raw_previous), 'candidate26_sha256': sha(raw_new)},
              'membership': {'active_count': len(a), 'candidate22_count': len(b), 'candidate26_count': len(c),
                             'added_from_active': sorted(c.keys()-a.keys()), 'removed_from_active': sorted(a.keys()-c.keys())},
              'construction_changes_from_active': changes(a,c,'construction'),
              'execution_changes_from_active': changes(a,c,'execution'),
              'runtime_identity_changes_from_active': [k for k in sorted(a.keys() & c.keys()) if
                  any(a[k]['provenance'][field] != c[k]['provenance'][field] for field in
                      ('instance_path','runtime_class','inventory_config_sha256'))],
              'static_identity_changes_from_active': [k for k in sorted(a.keys() & c.keys()) if
                  a[k]['provenance']['meaning']['static_occurrence'] != c[k]['provenance']['meaning']['static_occurrence']],
              'product_layer_schedule_unchanged': old['product_layer_schedule'] == new['product_layer_schedule'],
              'construction_schedules_unchanged': old['construction_schedules'] == new['construction_schedules'],
              'original_returned_projection_changes': triple,
              'projection_kind_changes_from_active': active_delta,
              'projection_kind_changes_from_candidate22': previous_delta,
              'potential_placement_losses_from_active': [r for r in active_delta if r['before']['kind'] in {'rendered','grouped'} and r['after']['kind'] not in {'rendered','grouped'}],
              'norm_citation_membership': {'active': [k for k in sorted(a) if placement(a[k])['decoder_norm_cited']],
                                           'candidate22': [k for k in sorted(b) if placement(b[k])['decoder_norm_cited']],
                                           'candidate26': [k for k in sorted(c) if placement(c[k])['decoder_norm_cited']]}}
    destination = OUT / slug
    destination.mkdir(exist_ok=True)
    files = {'active.json.gz': raw_old, 'candidate22.json.gz': raw_previous, 'candidate26.json.gz': raw_new,
             'audit.json': (json.dumps(report,indent=2)+'\n').encode()}
    for name, content in files.items():
        path=destination/name
        if path.exists():
            assert path.read_bytes()==content, 'immutable audit input/result changed: '+str(path)
        else:
            path.write_bytes(content)
    summary['models'].append({'slug':slug,'audit_sha256':sha(files['audit.json']), 'candidate26_sha256':sha(raw_new),
        'original_returned_kind_deltas':len(original), 'original_rows_restored_to_active_kind':sum(r['active']['kind']==r['candidate26']['kind'] for r in triple),
        'active_kind_deltas':len(active_delta), 'candidate22_kind_deltas':len(previous_delta),
        'potential_losses_from_active':len(report['potential_placement_losses_from_active']),
        'membership_equal':a.keys()==c.keys(),'construction_unchanged':not report['construction_changes_from_active'],
        'execution_unchanged':not report['execution_changes_from_active']})
if not summary['not_ready']:
    summary['status']='all four requested targets captured; no output approval or source-cause verdict implied'
(OUT/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
