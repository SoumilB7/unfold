"""Existing validators on archived actual HTML; no model or browser execution."""
import hashlib
import json
from pathlib import Path

from model_unfolder.block_schema import (
    validate_click_coupling,
    validate_no_dotted_arrows,
    validate_no_dotted_boundaries,
    validate_unique_ref_ids,
)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PAGES = ROOT.parent / 'z-docs/12-design/S8/final-f7baef5'
results = []
for name in ['sdxl-ordinary', 'sd14-ordinary']:
    data = (PAGES / (name + '.html')).read_bytes()
    html = data.decode()
    checks = {check.__name__: check(html) for check in (
        validate_click_coupling, validate_no_dotted_arrows,
        validate_no_dotted_boundaries, validate_unique_ref_ids)}
    results.append({'case': name, 'html_sha256': hashlib.sha256(data).hexdigest(),
                    'checks': checks})
    print(name, {key: len(value) for key, value in checks.items()})
(OUT / 'result.json').write_text(json.dumps({
    'scope': 'Read-only existing validators on actual final HTML; no browser, model or pytest run.',
    'results': results}, indent=2) + '\n')
