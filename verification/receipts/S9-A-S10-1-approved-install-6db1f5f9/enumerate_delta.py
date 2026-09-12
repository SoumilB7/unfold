"""Enumerate the staged candidate delta against the committed tree. Read-only."""
import difflib, hashlib, json, re, sys
from pathlib import Path
ROOT = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
STAGE = Path('/private/tmp/unfold-s9a-install-6db1f5f9')
sys.path.insert(0, str(ROOT))

def sha(b): return hashlib.sha256(b).hexdigest()
def rd(p): return json.loads(Path(p).read_text())

out = {}
STOP = []

# ---------- 1. preservation expected manifest --------------------------------
old = rd(ROOT / 'tests/preservation_expected_manifest.json')
new = rd(STAGE / 'preservation_expected_manifest.json')
pm = {'old_sha256': sha((ROOT / 'tests/preservation_expected_manifest.json').read_bytes()),
      'new_sha256': sha((STAGE / 'preservation_expected_manifest.json').read_bytes()),
      'witness_count': [old['witness_count'], new['witness_count']],
      'versions_equal': old['versions'] == new['versions'],
      'versions': [old['versions'], new['versions']],
      'witness_set_equal': sorted(old['witnesses']) == sorted(new['witnesses']),
      'per_witness': {}, 'surface_change_counts': {}, 'unchanged_surfaces': []}
if old['witness_count'] != new['witness_count'] or not pm['witness_set_equal']:
    STOP.append('preservation witness set or count changed')
counts = {}
allsurf = set()
for slug in sorted(new['witnesses']):
    o, n = old['witnesses'][slug], new['witnesses'][slug]
    allsurf |= set(o['surfaces']) | set(n['surfaces'])
    if o['input_sha256'] != n['input_sha256']:
        STOP.append(f'input hash changed: {slug}')
    if set(o['surfaces']) != set(n['surfaces']):
        STOP.append(f'surface set changed: {slug}')
    changed = sorted(k for k in n['surfaces'] if o['surfaces'].get(k) != n['surfaces'][k])
    views_changed = o['views'] != n['views']
    if views_changed:
        STOP.append(f'views changed: {slug}')
    for k in changed: counts[k] = counts.get(k, 0) + 1
    illegal = [k for k in changed if k not in ('html_meta', 'ir', 'ledgers', 'sable')]
    if illegal:
        STOP.append(f'out-of-scope surface changed on {slug}: {illegal}')
    pm['per_witness'][slug] = {'changed_surfaces': changed, 'views_changed': views_changed,
                               'view_count': [len(o['views']), len(n['views'])]}
pm['surface_change_counts'] = counts
pm['unchanged_surfaces'] = sorted(allsurf - set(counts))
out['preservation_manifest'] = pm

# ---------- 2. examples -------------------------------------------------------
from model_unfolder.preview import architecture_svg
old_man = rd(ROOT / 'examples/manifest.json')
old_rows = {r['file']: r for r in old_man['examples']}
cand = STAGE / 'examples'
new_rows = {}
ex = {'per_example': {}, 'hunk_causes': {}, 'svg_identical': True}
CSS_MARK = 'Typed annotations remain distinguishable'
OLD_DIGEST = '6f41f3a215a334c4b5bea62fe2d31f1302926233c8692570c1f62f32d5c25b65'
NEW_DIGEST = 'b80fd1f1259cdcbced7eec605e1950d81c5cfb87134cbec8ee225d06e17cd464'
for path in sorted(cand.glob('*.html')):
    name = path.name
    new_html = path.read_text()
    old_html = (ROOT / 'examples' / name).read_text()
    o_svg, n_svg = architecture_svg(old_html), architecture_svg(new_html)
    same_svg = o_svg == n_svg
    if not same_svg: ex['svg_identical'] = False
    causes = {}
    a, b = old_html.splitlines(True), new_html.splitlines(True)
    hunks = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal': continue
        o, n = ''.join(a[i:j]), ''.join(b[k:l])
        cause = None
        # The owner's enumeration already carries SDXL's content-addressed
        # compact-transport digest change; name it exactly, never generically.
        if OLD_DIGEST in o and o.replace(OLD_DIGEST, NEW_DIGEST) == n:
            cause = 'compact_transport_digest'
        elif tag == 'insert' and CSS_MARK in n: cause = 'css'
        elif o.startswith('<div class="uf-card">') and '<details class="uf-unknown-values">' in n: cause = 'html_unknown'
        elif o == n.replace(' data-uf-receipt-node="stats_hidden_size"', '') and 'data-uf-receipt-node' in n: cause = 'banner'
        elif 'data-chip-kind=' in n and 'data-chip-kind=' not in o: cause = 'chip'
        elif 'uf-unknown-values' in n or 'uf-chip-' in n: cause = 'chip_or_unknown_markup'
        causes[cause or 'UNCLASSIFIED'] = causes.get(cause or 'UNCLASSIFIED', 0) + 1
        if cause is None:
            hunks.append({'tag': tag, 'before_lines': [i + 1, j], 'before': o[:400], 'after': n[:400]})
    for c, v in causes.items(): ex['hunk_causes'][c] = ex['hunk_causes'].get(c, 0) + v
    ex['per_example'][name] = {
        'bytes_changed': old_html != new_html,
        'architecture_svg_identical': same_svg,
        'old_sha256': sha(old_html.encode()), 'new_sha256': sha(new_html.encode()),
        'hunk_causes': causes, 'unclassified_hunks': hunks}
    if hunks: STOP.append(f'unclassified example hunk in {name}')
if not ex['svg_identical']: STOP.append('example architecture SVG changed')
# rebuild rows from the staged render (rasterize_hero=False) + the reviewed PNG seal
staged_rows = rd(STAGE / 'stage-result.json')['steps']['examples']['rows']
hero_old = next(r for r in old_man['examples'] if 'hero_png' in r)
hero_new = next(r for r in staged_rows if 'hero_png' in r)
ex['hero'] = {'file': hero_old['hero_png']['file'],
              'source_svg_sha256': [hero_old['hero_png']['source_svg_sha256'], hero_new['hero_png']['source_svg_sha256']],
              'source_svg_unchanged': hero_old['hero_png']['source_svg_sha256'] == hero_new['hero_png']['source_svg_sha256'],
              'reviewed_png_sha256': hero_old['hero_png']['sha256']}
if ex['hero']['source_svg_unchanged']:
    rows = json.loads(json.dumps(staged_rows))
    for r in rows:
        if 'hero_png' in r: r['hero_png']['sha256'] = hero_old['hero_png']['sha256']
    new_man = {'schema': 2, 'examples': rows}
    (STAGE / 'examples-manifest.json').write_text(json.dumps(new_man, indent=2, sort_keys=True) + '\n')
    ex['manifest_new_sha256'] = sha((STAGE / 'examples-manifest.json').read_bytes())
    ex['manifest_old_sha256'] = sha((ROOT / 'examples/manifest.json').read_bytes())
    ex['hero_png_rerasterized'] = False
else:
    STOP.append('hero SVG input changed: the reviewed PNG must be re-rasterized and re-reviewed')
ex['files_changed'] = sorted(n for n, r in ex['per_example'].items() if r['bytes_changed'])
out['examples'] = ex

# ---------- 3. coverage -------------------------------------------------------
oc, nc = rd(ROOT / 'coverage.json'), rd(STAGE / 'coverage.json')
def triple(d): return (sum(r['proven'] for r in d['models']), sum(r['flagged'] for r in d['models']), sum(r['silent'] for r in d['models']))
ot, nt = triple(oc), triple(nc)
cov = {'old': {'proven': ot[0], 'flagged': ot[1], 'silent': ot[2], 'models': len(oc['models'])},
       'new': {'proven': nt[0], 'flagged': nt[1], 'silent': nt[2], 'models': len(nc['models'])},
       'source_environment_equal': oc['source_environment'] == nc['source_environment'],
       'per_model_changes': [], 'presentation_added': 0}
om = {r['model']: r for r in oc['models']}
for r in nc['models']:
    p = om.get(r['model'])
    if p is None: STOP.append(f'coverage model appeared: {r["model"]}'); continue
    d = {k: [p.get(k), r.get(k)] for k in ('proven', 'flagged', 'silent') if p.get(k) != r.get(k)}
    if r.get('presentation') and not p.get('presentation'): cov['presentation_added'] += 1
    if d: cov['per_model_changes'].append({'model': r['model'], **d})
if nt[0] < ot[0]: STOP.append(f'coverage proven dropped {ot[0]} -> {nt[0]}')
if nt[2] != 0: STOP.append(f'coverage silent is not zero: {nt[2]}')
cov['old_sha256'] = sha((ROOT / 'coverage.json').read_bytes())
cov['new_sha256'] = sha((STAGE / 'coverage.json').read_bytes())
out['coverage'] = cov

out['STOP'] = STOP
(STAGE / 'delta-enumeration.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps({'preservation': {'counts': pm['surface_change_counts'], 'unchanged': pm['unchanged_surfaces'],
                                   'witness_count': pm['witness_count'], 'versions_equal': pm['versions_equal']},
                  'examples': {'files_changed': len(ex['files_changed']), 'hunk_causes': ex['hunk_causes'],
                               'svg_identical': ex['svg_identical'], 'hero_source_svg_unchanged': ex['hero']['source_svg_unchanged']},
                  'coverage': {'old': cov['old'], 'new': cov['new'], 'changed_models': len(cov['per_model_changes'])},
                  'STOP': STOP}, indent=1))
