"""Owner's output-delta enumeration: recovered31 baseline packet vs current37 packet (saved data only)."""
import json, hashlib, collections, sys, re
from pathlib import Path
B = Path('/private/tmp/unfold-s9a-thirtyfirst/preservation-baseline'); C = Path('/private/tmp/unfold-s9a-thirtyseventh/preservation-current37')
def sha(b): return hashlib.sha256(b).hexdigest()
def ref(base, r):
    p = base / 'capture/controller' / r['path']; d = p.read_bytes(); assert sha(d) == r['sha256']; return d
def delta(a, b, path=''):
    if type(a) is not type(b): return [(path, 'type', a, b)]
    if isinstance(a, dict):
        rows = []
        for k in sorted(a.keys() | b.keys()):
            ch = f'{path}/{k}'
            if k not in a: rows.append((ch, 'add', None, b[k]))
            elif k not in b: rows.append((ch, 'remove', a[k], None))
            else: rows += delta(a[k], b[k], ch)
        return rows
    if isinstance(a, list):
        rows = []
        for i in range(max(len(a), len(b))):
            ch = f'{path}/{i}'
            if i >= len(a): rows.append((ch, 'add', None, b[i]))
            elif i >= len(b): rows.append((ch, 'remove', a[i], None))
            else: rows += delta(a[i], b[i], ch)
        return rows
    return [] if a == b else [(path, 'replace', a, b)]
PRES = ('presentation_chips', 'presentation_path', 'presentation_aliases', 'unknown_reason', 'presentation_unresolved_values', 'presentation_reference', 'presentation_census')
def classify(path, op, before, after):
    if any('/' + k in path for k in PRES): return 'presentation_or_unknown_metadata'
    if '/extras/fact_provenance/' in path:
        leaf = path.rsplit('/', 1)[-1]
        if leaf in ('status', 'value'): return 'FACT_VALUE_OR_STATUS'
        return 'fact_provenance_metadata'
    if '/extras/config_accounting' in path or '/config_' in path or 'consumption' in path: return 'config_accounting_metadata'
    if '/extras/' in path and op == 'add': return 'extras_added:' + path.split('/extras/')[1].split('/')[0]
    if path.rsplit('/', 1)[-1] in ('source_fact_keys',) : return 'fact_citation_list'
    return 'STRUCTURAL_OR_UNCLASSIFIED'
rb = json.load(open(B / 'capture/report.json')); rc = json.load(open(C / 'capture/report.json'))
cb = {c['slug']: c for c in rb['cases']}; cc = {c['slug']: c for c in rc['cases']}
assert set(cb) == set(cc) and len(cb) == 29
out = {'witnesses': {}, 'totals': collections.Counter(), 'surface_changed_counts': collections.Counter(), 'fact_value_or_status_rows': [], 'structural_rows': []}
for slug in sorted(cb):
    db = json.loads(ref(B, cb[slug]['actual_surfaces'])); dc = json.loads(ref(C, cc[slug]['actual_surfaces']))
    w = {'surfaces_changed': [], 'ir_classes': collections.Counter(), 'ledger_classes': collections.Counter(), 'sable_classes': collections.Counter()}
    for s in sorted(set(db) | set(dc)):
        if db.get(s) != dc.get(s):
            w['surfaces_changed'].append(s); out['surface_changed_counts'][s] += 1
    for s, key in (('ir', 'ir_classes'), ('ledgers', 'ledger_classes'), ('sable', 'sable_classes')):
        if s in db and s in dc and db[s] != dc[s]:
            for path, op, before, after in delta(db[s], dc[s]):
                cls = classify(path, op, before, after)
                w[key][cls] += 1; out['totals'][s + ':' + cls] += 1
                if cls == 'FACT_VALUE_OR_STATUS': out['fact_value_or_status_rows'].append({'witness': slug, 'surface': s, 'path': path, 'op': op, 'before': before, 'after': after})
                elif cls == 'STRUCTURAL_OR_UNCLASSIFIED' and len(out['structural_rows']) < 400: out['structural_rows'].append({'witness': slug, 'surface': s, 'path': path, 'op': op, 'before': str(before)[:120], 'after': str(after)[:120]})
    w = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in w.items()}
    out['witnesses'][slug] = w
out['totals'] = dict(out['totals']); out['surface_changed_counts'] = dict(out['surface_changed_counts'])
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
print('surfaces changed per kind:', out['surface_changed_counts'])
print('leaf classes:', json.dumps(out['totals'], indent=1))
print('fact value/status rows:', len(out['fact_value_or_status_rows'])); print('structural rows:', len(out['structural_rows']))
for r in out['fact_value_or_status_rows'][:12]: print('  FV', r['witness'], r['path'][-90:], r['op'], str(r['before'])[:40], '->', str(r['after'])[:40])
for r in out['structural_rows'][:25]: print('  ST', r['witness'], r['surface'], r['path'][-110:], r['op'])
