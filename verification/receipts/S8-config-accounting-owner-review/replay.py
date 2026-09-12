"""Read-only exact disposition of final UNet config-audit findings."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

from model_unfolder.evidence.ship_findings import (
    accessed_unprojected_findings, standing_unconsumed_findings,
    unsurfaced_findings,
)

ROOT = Path('/private/tmp/unfold-s8-render-phase-c38-v2')
OUT = Path(__file__).parent


class BodyText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results = {}
for witness in ('sdxl', 'sd-v1-4'):
    root = ROOT / witness
    paths = [root / case / name for case in ('legacy', 'ordinary')
             for name in ('ir.json', 'page.html', 'result.json')]
    pins = {str(p): sha(p) for p in paths}
    before = json.loads((root / 'legacy/ir.json').read_text())
    after = json.loads((root / 'ordinary/ir.json').read_text())
    access = after['extras']['config_access']
    text = BodyText()
    text.feed((root / 'ordinary/page.html').read_text())
    actual_text = ' '.join(' '.join(text.parts).split())
    groups = {
        'config_accessed_unprojected': accessed_unprojected_findings(after),
        'config_standing_unconsumed': standing_unconsumed_findings(after),
        'config_audit_incomplete': access.get('audit_incomplete', []),
    }
    records = {}
    for check, messages in groups.items():
        absent_receipts = unsurfaced_findings(after, check, messages)
        absent_text = [m for m in messages if ' '.join(m.split()) not in actual_text]
        assert not absent_receipts, (witness, check, absent_receipts)
        assert not absent_text, (witness, check, absent_text)
        records[check] = {'messages': messages, 'exact_ship_receipts': True,
                          'actual_diagnostic_text_present': True}
    old_consumed = [v for v in before['extras']['config_access']['consumed']
                    if v.startswith('root.denoiser:')]
    new_consumed = [v for v in access['consumed'] if v.startswith('root.denoiser:')]
    assert old_consumed and not new_consumed
    assert access.get('audit_incomplete') == ['root.denoiser']
    violations = [row for row in access.get('migration_claims', []) if row.get('violations')]
    assert not violations, violations
    assert {str(p): sha(p) for p in paths} == pins
    results[witness] = {
        'disposition': 'named_visible_accounting_limitation_not_config_completion',
        'legacy_denoiser_consumed': old_consumed,
        'new_denoiser_consumed': new_consumed,
        'findings': records, 'migration_claim_violations': violations,
        'artifact_sha256': pins, 'artifacts_unchanged': True,
    }
files = [Path(p) for p in (
    'model_unfolder/parser.py', 'model_unfolder/sable.py',
    'model_unfolder/adapters/diffusor/unet.py',
    'model_unfolder/adapters/diffusor/unet_cutover.py',
    'model_unfolder/evidence/ship_findings.py',
)]
results['source_sha256'] = {str(p): sha(p) for p in files}
(OUT / 'result.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
print(json.dumps({w: {k: len(v['messages']) for k, v in results[w]['findings'].items()}
                  for w in ('sdxl', 'sd-v1-4')}, sort_keys=True))
