"""Synthetic transport navigation fixture; it is not a model witness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.checkout.resolve()))
    from model_unfolder.renderers.html import card_payload
    from model_unfolder.renderers.html.interactions import _click_script
    mount = 'uf-a123456789'
    padding = '<!--' + 'Synthetic exact shared transport padding. ' * 400 + '-->'
    def svg(ids):
        rows = []
        for i, identity in enumerate(ids):
            y = 10 + i * 60
            rows.append(f'<g class="uf-node" data-id="{identity}">'
                        f'<rect x="10" y="{y}" width="250" height="45" fill="#e0f2f1" stroke="#00695c"/>'
                        f'<text x="20" y="{y + 28}">{identity}</text></g>')
        return '<svg width="300" height="150">' + ''.join(rows) + '</svg>'
    def card(identity, label, quantity, child=None):
        detail = '<div class="uf-card-svg">' + svg([child]) + '</div>' if child else ''
        return (f'<div class="uf-card-detail" data-card-id="{identity}" data-card-size="compact">'
                f'<h3>{label}</h3><p class="quantity">Own quantity: {quantity}</p>' + detail + padding + '</div>')
    architectures, level2, level3 = [], [], []
    for variant, quantity in (('a', 11), ('b', 29)):
        architectures.append(f'<div class="variant variant-{variant}">' + svg(['shared', 'other']) + '</div>')
        level2.append(f'<div class="variant variant-{variant}">'
                      + card('default', variant + ' default', 0)
                      + card('shared', variant + ' shared', quantity, 'deep_shared')
                      + card('other', variant + ' other', quantity + 1, 'deep_other')
                      + ''.join(card(f'unused-{variant}-{i}', 'Additional retained drill', 7)
                                for i in range(12)) + '</div>')
        level3.append(f'<div class="variant variant-{variant}">'
                      + card('default', variant + ' deep default', 0)
                      + card('deep_shared', variant + ' deep shared', quantity * 10)
                      + card('deep_other', variant + ' deep other', quantity * 10 + 1) + '</div>')
    style = '''<style>
body {font-family: sans-serif; margin: 24px;} .variant {display:none;}
.uf-card-detail {display:none;} .uf-card-detail[data-card-id="default"] {display:block;}
.uf-inspect-panel[data-depth="3"] {display:none;}
.uf-inspect-panel.uf-nested-active[data-depth="3"] {display:block;}
#variant-a:checked ~ .uf-card .variant-a, #variant-b:checked ~ .uf-card .variant-b {display:block;}
.uf-selected rect, .uf-nested-selected rect {stroke-width:4;}
</style>'''
    fragment = (f'<div id="{mount}" class="uf-root">'
                '<input id="variant-a" name="variant" type="radio" checked><label for="variant-a">Variant A</label>'
                '<input id="variant-b" name="variant" type="radio"><label for="variant-b">Variant B</label>'
                '<div class="uf-card"><h1>Synthetic same-depth variant transport control</h1>'
                '<div class="uf-section-arch">' + ''.join(architectures) + '</div>'
                '<div class="uf-inspect-panel" data-depth="2">' + ''.join(level2) + '</div>'
                '<div class="uf-inspect-panel" data-depth="3">' + ''.join(level3) + '</div>'
                '</div>' + _click_script(mount) + '</div>')
    shell = '<!doctype html><html><head><meta charset="utf-8">' + style + '</head><body>{body}</body></html>'
    old_threshold = card_payload._MIN_PACK_BYTES
    try:
        # Explicit fixture-only switch; production threshold remains unchanged.
        card_payload._MIN_PACK_BYTES = 0
        compact_fragment = card_payload.pack_card_payloads(fragment, mount)
    finally:
        card_payload._MIN_PACK_BYTES = old_threshold
    eager, compact = shell.replace('{body}', fragment), shell.replace('{body}', compact_fragment)
    assert 'data-uf-card-payload=' in compact
    assert card_payload.expand_card_payloads(compact) == eager
    (args.output / 'eager.html').write_text(eager)
    (args.output / 'compact.html').write_text(compact)
    report = {'scope': 'synthetic transport control, not a model witness',
              'fixture_only_threshold_override_restored': card_payload._MIN_PACK_BYTES == old_threshold,
              'exact_inverse': True, 'expected_shared_quantities': {'a': 11, 'b': 29},
              'expected_deep_shared_quantities': {'a': 110, 'b': 290},
              'eager_sha256': hashlib.sha256(eager.encode()).hexdigest(),
              'compact_sha256': hashlib.sha256(compact.encode()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'codec_sha256': hashlib.sha256(Path(card_payload.__file__).read_bytes()).hexdigest()}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
