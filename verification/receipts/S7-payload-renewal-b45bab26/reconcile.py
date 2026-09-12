"""Fresh S7 shadow on b45bab26 vs candidate37, and vs the committed verification/s7.

Read-only. Writes only renewal-reconciliation.json.
"""
import collections, glob, gzip, hashlib, importlib.util, json, os, sys
from pathlib import Path

R = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
FRESH = Path('/private/tmp/unfold-s9a-s7-renewal-b45bab26/matrix')
C37 = Path('/private/tmp/unfold-s9a-thirtyseventh/matrix')
HEAD = R / 'verification/s7'
OUT = Path('/private/tmp/unfold-s9a-s7-renewal-b45bab26/renewal-reconciliation.json')

sys.path.insert(0, str(R))
spec = importlib.util.spec_from_file_location('s7gen', R / 'scripts/generate_s7_shadow.py')
G = importlib.util.module_from_spec(spec); sys.modules['s7gen'] = G; spec.loader.exec_module(G)

def sha(b): return hashlib.sha256(b).hexdigest()
def load(p): return json.load(gzip.open(p, 'rt', encoding='utf-8'))
def jb(v): return json.dumps(v, sort_keys=True, separators=(',', ':')).encode()

out = {'head_commit': 'b45bab26539eca773c2f03519ebfd72ca7719d93', 'items': {}, 'DISAGREE': []}

# ---------- A. fresh vs candidate37 -------------------------------------------
fresh_m = sorted(os.path.basename(p) for p in glob.glob(str(FRESH / 'models/*.json.gz')))
c37_m = sorted(os.path.basename(p) for p in glob.glob(str(C37 / 'models/*.json.gz')))
a = {'model_set_equal': fresh_m == c37_m, 'models': len(fresh_m),
     'logically_identical': [], 'logically_different': [], 'byte_identical': 0}
for name in fresh_m:
    f, c = load(FRESH / 'models' / name), load(C37 / 'models' / name)
    same = sha(jb(G._semantic_payload(f))) == sha(jb(G._semantic_payload(c)))
    (a['logically_identical'] if same else a['logically_different']).append(name)
    if (FRESH / 'models' / name).read_bytes() == (C37 / 'models' / name).read_bytes():
        a['byte_identical'] += 1
for group in ('observations', 'relations'):
    fs = sorted(os.path.basename(p) for p in glob.glob(str(FRESH / group / '*')))
    cs = sorted(os.path.basename(p) for p in glob.glob(str(C37 / group / '*')))
    same = fs == cs and all((FRESH / group / n).read_bytes() == (C37 / group / n).read_bytes() for n in fs)
    a[group + '_identical'] = same
    a[group + '_count'] = len(fs)
fm, cm = json.loads((FRESH / 'matrix.json').read_text()), json.loads((C37 / 'matrix.json').read_text())
hm = json.loads((HEAD / 'matrix.json').read_text())
a['sources_fresh_equals_candidate37'] = fm['sources'] == cm['sources']
a['sources_fresh_equals_committed_stamp'] = fm['sources'] == hm['sources']
a['sources_entries'] = [len(hm['sources']), len(fm['sources'])]
a['models_rows_equal_candidate37'] = fm['models'] == cm['models']
out['items']['A_fresh_vs_candidate37'] = a
if not a['model_set_equal'] or a['logically_different']:
    out['DISAGREE'].append('fresh S7 is not logically identical to candidate37')
if not (a['observations_identical'] and a['relations_identical']):
    out['DISAGREE'].append('observations/relations differ from candidate37')
if not a['sources_fresh_equals_committed_stamp']:
    out['DISAGREE'].append('fresh sources map differs from the committed stamp')

# ---------- B. fresh vs committed, the owner's enumeration --------------------
def occ_id(o):
    pv = o['provenance']
    return json.dumps(pv.get('instance_path') or pv, sort_keys=True)[:400]

t = collections.Counter()
per = {}
proj_diffs = collections.Counter()
proj_kind_models = collections.defaultdict(collections.Counter)
prov_models, rel_models = [], []
blocking = [0, 0]
for f in sorted(glob.glob(str(HEAD / 'models/*.json.gz'))):
    name = os.path.basename(f)
    h, n = load(f), load(FRESH / 'models' / name)
    A, B = h['table']['occurrences'], n['table']['occurrences']
    t['occ_head'] += len(A); t['occ_fresh'] += len(B)
    row = {'occ': [len(A), len(B)],
           'identity_order_equal': [occ_id(o) for o in A] == [occ_id(o) for o in B],
           'construction_equal': all(x['construction'] == y['construction'] for x, y in zip(A, B)),
           'execution_equal': all(x['execution'] == y['execution'] for x, y in zip(A, B)),
           'provenance_equal': all(x['provenance'] == y['provenance'] for x, y in zip(A, B))}
    d = collections.Counter()
    for x, y in zip(A, B):
        for k in ('kind', 'parent', 'rule', 'block_ids', 'reason', 'reason_class'):
            if x['projection'].get(k) != y['projection'].get(k):
                d[k] += 1
                if k == 'kind':
                    proj_kind_models[name][(x['projection'].get('kind'), y['projection'].get('kind'))] += 1
    row['projection_field_diffs'] = dict(d)
    for k, v in d.items(): proj_diffs[k] += v
    removed = newly = hu = still = newu = kc = hc = 0
    for x, y in zip(A, B):
        px, py = x['projection'], y['projection']
        fx, fy = set(px.get('fact_keys') or []), set(py.get('fact_keys') or [])
        hc += len(fx); removed += len(fx - fy); newly += len(fy - fx)
        ux, uy = set(px.get('unqualified_fact_keys') or []), set(py.get('unqualified_fact_keys') or [])
        hu += len(ux); still += len(ux & uy); newu += len(uy - ux)
        kx, ky = px.get('fact_claim_kinds') or {}, py.get('fact_claim_kinds') or {}
        if isinstance(kx, dict) and isinstance(ky, dict):
            kc += sum(1 for k in kx if k in ky and kx[k] != ky[k])
    row.update(head_cited=hc, removed_citations=removed, newly_cited=newly,
               head_unqualified=hu, head_unqualified_still=still,
               new_unqualified=newu, claim_kind_changed=kc)
    for k in ('head_cited', 'removed_citations', 'newly_cited', 'head_unqualified',
              'head_unqualified_still', 'new_unqualified', 'claim_kind_changed'):
        t[k] += row[k]
    row['relations_equal'] = h['table'].get('relations') == n['table'].get('relations')
    row['relations_count'] = [len(h['table'].get('relations') or []), len(n['table'].get('relations') or [])]
    row['schedules_equal'] = h.get('construction_schedules') == n.get('construction_schedules')
    row['blocking'] = [len(h.get('blocking_findings') or []), len(n.get('blocking_findings') or [])]
    blocking[0] += row['blocking'][0]; blocking[1] += row['blocking'][1]
    if not row['provenance_equal']: prov_models.append(name)
    if not row['relations_equal']: rel_models.append(name)
    per[name] = row

b = {'totals': dict(t), 'blocking_head_to_fresh': blocking,
     'projection_field_diffs': dict(proj_diffs),
     'projection_kind_changes_by_model': {m: {f'{k[0]}->{k[1]}': v for k, v in c.items()}
                                          for m, c in proj_kind_models.items()},
     'models_with_provenance_changes': prov_models,
     'models_with_relation_changes': rel_models,
     'identity_order_equal_all': all(r['identity_order_equal'] for r in per.values()),
     'construction_equal_all': all(r['construction_equal'] for r in per.values()),
     'execution_equal_all': all(r['execution_equal'] for r in per.values()),
     'schedules_equal_all': all(r['schedules_equal'] for r in per.values()),
     'relation_counts_equal_all': all(r['relations_count'][0] == r['relations_count'][1] for r in per.values()),
     'per_model': per}
out['items']['B_fresh_vs_committed'] = b

# ---------- C. reconcile against the owner's matrix_audit.json ----------------
oa = json.load(open(R / 'verification/receipts/S9-A-owner-review-37/matrix_audit.json'))
ot = oa['totals']
o_proj = collections.Counter()
for m in oa['models'].values():
    for k, v in m['projection_field_diffs'].items(): o_proj[k] += v
o_block = [sum(len(m['blocking_head']) for m in oa['models'].values()),
           sum(len(m['blocking_cand']) for m in oa['models'].values())]
def cmp(name, mine, owner, expected=None):
    ok = mine == owner and (expected is None or mine == expected)
    row = {'mine': mine, 'owner': owner, 'expected': expected, 'agree': ok}
    if not ok: out['DISAGREE'].append(f'{name}: mine={mine} owner={owner} expected={expected}')
    return row
c = {
  'occurrences': cmp('occurrences', [t['occ_head'], t['occ_fresh']], [ot['occ_head'], ot['occ_cand']], [44088, 44088]),
  'citations_removed': cmp('citations_removed', t['removed_citations'], ot['removed_citations'], 0),
  'formerly_unqualified': cmp('formerly_unqualified', t['head_unqualified'], ot['head_unqualified'], 16324),
  'still_unqualified': cmp('still_unqualified', t['head_unqualified_still'], ot['head_unqualified_still'], 0),
  'new_unqualified': cmp('new_unqualified', t['new_unqualified'], ot['new_unqualified'], 481),
  'claim_kinds_changed': cmp('claim_kinds_changed', t['claim_kind_changed'], ot['claim_kind_changed'], 0),
  'newly_cited': cmp('newly_cited', t['newly_cited'], ot['newly_cited']),
  'blocking_findings': cmp('blocking_findings', blocking, o_block, [70186, 54257]),
  'projection_field_diffs': cmp('projection_field_diffs', dict(proj_diffs), dict(o_proj)),
  'models_with_relation_changes': cmp('models_with_relation_changes', sorted(rel_models),
                                      sorted(n for n, m in oa['models'].items() if not m['relations_equal'])),
  'models_with_provenance_changes': cmp('models_with_provenance_changes', sorted(prov_models),
                                        sorted(n for n, m in oa['models'].items() if not m['provenance_equal'])),
  'identity_order_construction_execution_schedules': cmp(
      'identity_order_construction_execution_schedules',
      [b['identity_order_equal_all'], b['construction_equal_all'], b['execution_equal_all'], b['schedules_equal_all']],
      [all(m['identity_order_equal'] for m in oa['models'].values()),
       all(m['construction_equal'] for m in oa['models'].values()),
       all(m['execution_equal'] for m in oa['models'].values()),
       all(m['schedules_equal'] for m in oa['models'].values())],
      [True, True, True, True]),
}
out['items']['C_reconciliation_vs_owner_audit'] = c

# ---------- hard STOP conditions ----------------------------------------------
if t['removed_citations']: out['DISAGREE'].append('citations were removed')
if not b['construction_equal_all']: out['DISAGREE'].append('construction axis changed')
if not b['execution_equal_all']: out['DISAGREE'].append('execution axis changed')
if proj_diffs.get('kind'): out['DISAGREE'].append(
    f"projection kind changed on {proj_diffs['kind']} occurrences: "
    + json.dumps(b['projection_kind_changes_by_model']))

OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps({'A': {k: v for k, v in a.items() if k != 'logically_identical'},
                  'B_totals': dict(t), 'B_blocking': blocking,
                  'B_projection_field_diffs': dict(proj_diffs),
                  'B_projection_kind_changes': b['projection_kind_changes_by_model'],
                  'C_agree': {k: v['agree'] for k, v in c.items()},
                  'DISAGREE': out['DISAGREE']}, indent=1))
