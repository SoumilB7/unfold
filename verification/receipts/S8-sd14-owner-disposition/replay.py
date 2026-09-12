"""Preserve the exact reviewed deltas and append the bounded owner disposition."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
RECEIPTS = OUT.parent
PHASE = Path('/private/tmp/unfold-s8-render-phase-c38-v2/sd-v1-4')
review = RECEIPTS / 'S8-sd14-differential-review/per-output-review.json.gz'
rows = json.loads(gzip.decompress(review.read_bytes()))
report = json.loads((PHASE / 'differential-report.json').read_text())
facts = json.loads((PHASE / 'ordinary/facts.json').read_text())
qualified = json.loads((PHASE / 'ordinary/qualified-facts.json').read_text())
ir = json.loads((PHASE / 'ordinary/ir.json').read_text())

def find_deltas(value):
    if isinstance(value, list) and len(value) == len(rows) and value and isinstance(value[0], dict) and 'path' in value[0]:
        return value
    if isinstance(value, dict):
        for child in value.values():
            result = find_deltas(child)
            if result is not None:
                return result
    return None

assert [r['original_delta'] for r in rows] == find_deltas(report)
assert len(rows) == 1738
for key, fact in facts.items():
    public = ir['extras']['fact_provenance'][key]
    assert public['value'] == fact['value']
    assert public['status'] == fact['status']
    assert qualified[key]['claim_kind'] == qualified[key]['proof']['claim_kind']
    assert qualified[key]['proof']['evidence_refs']

dispositions = {
    'audit': ('named_blocking_accounting_finding_retained',
              'Exact surfaced config accounting debt; no fabricated consumption and no claim that audit is complete. Carry into S9.'),
    'render': ('named_reproof_or_explicit_limitation',
               'Actual constructed stage/module and qualified local-route mappings, quantity/SVG checks, and named old-template demotions in the independent SD14 review. Complete residual/query/resize composition remains limited.'),
    'legacy_geometry': ('named_reproof_or_explicit_limitation',
                        'Nine stage preservation, sixteen GEGLU drills, twenty-two bounded ResNet drills and six spatial-operation mappings; unproved old aggregate roles/dimensions remain limited.'),
    'proofs': ('named_reproof',
               'Public values/statuses match all twelve canonical facts and qualified claim kinds; exact source causes independently reviewed. A qualified partial claim does not prove its unresolved parts.'),
    'shapes': ('named_reproof',
               'Independent 709-occurrence inventory and 859520964 parameter sum; 706 actual canonical subtree quantity chips match.'),
    'limited_metadata': ('named_reproof_of_limited_result',
                         'A variable-width UNet has no established global hidden width or tied-token-embedding claim; per-module shapes remain.'),
    'source_address': ('named_reproof',
                       'Exact module/content-hash source address replaces a host path; evidence is not graph identity.'),
}
for row in rows:
    disposition, scope = dispositions[row['review_group']]
    row.update(final_owner_disposition=disposition, final_owner_scope=scope,
               user_output_approval='pending', blessed=False)
    row['owner_review_receipts'] = ['S8-sd14-differential-review', 'S8-sd14-owner-disposition']
    if row['review_group'] == 'audit':
        row['owner_review_receipts'].append('S8-config-accounting-owner-review')

inputs = [review, PHASE / 'differential-report.json'] + [PHASE / 'ordinary' / n for n in ['facts.json', 'qualified-facts.json', 'ir.json']]
summary = {
    'witness': 'sd-v1-4', 'rows': len(rows), 'unexplained': 0,
    'groups': dict(sorted(Counter(r['review_group'] for r in rows).items())),
    'owner_dispositions': dict(sorted(Counter(r['final_owner_disposition'] for r in rows).items())),
    'public_fact_values_statuses_and_claim_kinds_checked': sorted(facts),
    'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    'production': '96d0c1e (c38 producer bytes; separately verified renderer-only phase)',
    'scope': 'Every delta has a named evidence-backed result or retained accounting finding. Open accounting debt is not declared solved. No output approval or blessing.',
    'user_output_approval': 'pending', 'blessed': False,
}
(OUT / 'rows.json.gz').write_bytes(gzip.compress(json.dumps(rows, sort_keys=True).encode(), mtime=0))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: summary[k] for k in ['witness', 'rows', 'unexplained', 'owner_dispositions', 'blessed']}, indent=2))
