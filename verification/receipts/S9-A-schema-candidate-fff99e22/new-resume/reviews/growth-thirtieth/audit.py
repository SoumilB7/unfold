"""Independent saved-source accounting; standard library only, no package imports."""
import ast
from collections import Counter
import difflib
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
BASE = ROOT / '.claude/worktrees/verify-s10-1-census-fix'
CANDIDATE = ROOT / '.claude/worktrees/verify-s9-a-thirtieth'
OUT = Path(__file__).resolve().parent
ROOTS = ('model_unfolder', 'physics', 'tests', 'test_support', 'scripts')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def inventory(tree):
    return {str(p.relative_to(tree)): sha(p.read_bytes())
            for prefix in ROOTS for p in sorted((tree / prefix).rglob('*.py'))
            if '__pycache__' not in p.parts}


def category(name):
    return ('production' if name.startswith(('model_unfolder/', 'physics/')) else
            'test_support' if name.startswith('test_support/') else
            'tests' if name.startswith('tests/') else 'scripts')


def changes(old, new):
    added = deleted = 0
    for tag, i, j, k, l in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != 'equal':
            added += l - k
            deleted += j - i
    return added, deleted


def debt(tree):
    parsed = ast.parse((tree / 'model_unfolder/evidence/structural_debt.py').read_text())
    declaration = next(n for n in parsed.body if isinstance(n, ast.AnnAssign)
                       and isinstance(n.target, ast.Name) and n.target.id == 'STRUCTURAL_DEBT')
    constants = {target.id: n.value for n in parsed.body if isinstance(n, ast.Assign)
                 for target in n.targets if isinstance(target, ast.Name)}
    def literal(n):
        if isinstance(n, ast.Name): return literal(constants[n.id])
        if isinstance(n, ast.JoinedStr):
            return ''.join(str(literal(item.value)) if isinstance(item, ast.FormattedValue)
                           else str(literal(item)) for item in n.values)
        return ast.literal_eval(n)
    rows = []
    for n in declaration.value.elts:
        assert isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        rows.append({'factory': n.func.id, 'args': [literal(a) for a in n.args],
                     'kwargs': {k.arg: literal(k.value) for k in n.keywords}})
    return rows


def main():
    source_manifest = Path('/private/tmp/unfold-s9a-thirtieth/source-manifest.json')
    assert sha(source_manifest.read_bytes()) == 'e4636c47c40267792d6ab489818bcd1c02deeb7c52ffe8cc44f2e8d60e6b9d43'
    for r in json.loads(source_manifest.read_text()):
        p = CANDIDATE / r['path']
        assert (sha(p.read_bytes()) if p.exists() else None) == r['sha256'], r['path']
    baseline_manifest = ROOT / 'verification/receipts/S10-1-unet-deletion-fff99e22/raw/deletion/census-fix/source-manifest.json.gzip'
    baseline_raw = gzip.decompress(baseline_manifest.read_bytes())
    for name, expected in json.loads(baseline_raw).items():
        p = BASE / name
        assert (sha(p.read_bytes()) if p.exists() else None) == expected, name
    oldpins, newpins = inventory(BASE), inventory(CANDIDATE)
    write('baseline-source-inventory.json', oldpins)
    write('candidate-source-inventory.json', newpins)
    rows = []
    counts = {c: {'added': 0, 'deleted': 0, 'files_added': 0, 'files_deleted': 0,
                   'files_changed': 0} for c in ('production', 'tests', 'test_support', 'scripts')}
    for name in sorted(oldpins.keys() | newpins.keys()):
        if oldpins.get(name) == newpins.get(name):
            continue
        a, b = BASE / name, CANDIDATE / name
        plus, minus = changes(a.read_text().splitlines() if a.exists() else [],
                              b.read_text().splitlines() if b.exists() else [])
        cat = category(name)
        counts[cat]['added'] += plus
        counts[cat]['deleted'] += minus
        counts[cat]['files_added'] += name not in oldpins
        counts[cat]['files_deleted'] += name not in newpins
        counts[cat]['files_changed'] += name in oldpins and name in newpins
        rows.append({'path': name, 'category': cat, 'added': plus, 'deleted': minus,
                     'before_sha256': oldpins.get(name), 'after_sha256': newpins.get(name)})
    write('line-growth.json', rows)
    before_debt, after_debt = debt(BASE), debt(CANDIDATE)
    assert before_debt == after_debt
    write('debt-rows.json', after_debt)
    stable_names = ('model_unfolder/evidence/structural_debt.py',
                    'model_unfolder/evidence/legacy_reader_quarantine.py',
                    'docs/U3_CURRENT_READER_INVENTORY.md')
    stable = {}
    for name in stable_names:
        old, new = (BASE/name).read_bytes(), (CANDIDATE/name).read_bytes()
        assert old == new
        stable[name] = sha(new)
    producers = []
    for name in newpins:
        if not name.startswith('model_unfolder/evidence/'):
            continue
        tree = ast.parse((CANDIDATE/name).read_text())
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in n.decorator_list:
                fun = dec.func if isinstance(dec, ast.Call) else dec
                if not isinstance(fun, ast.Name) or fun.id != 'retained_claim_reader':
                    continue
                claims = ()
                if isinstance(dec, ast.Call):
                    claims = next((ast.literal_eval(k.value) for k in dec.keywords
                                   if k.arg == 'intended_claims'), ())
                producers.append({'path': name, 'symbol': n.name, 'line': n.lineno,
                                  'fixed_intended_claims': claims})
    write('retained-producers.json', producers)
    fixture = ROOT / 'verification/receipts/S9-A-schema-candidate-fff99e22/new-resume/fixture-isolation'
    fm = json.loads((fixture/'manifest.json').read_text())
    moved = []
    for move in fm['moves']:
        held_tree = ast.parse((fixture/'source'/move['destination']).read_text())
        live_tree = ast.parse((CANDIDATE/move['destination']).read_text())
        def definitions(tree):
            out = {}
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)): out[n.name] = n
                elif isinstance(n, ast.Assign):
                    out.update({target.id: n for target in n.targets if isinstance(target, ast.Name)})
            return out
        expected, current = definitions(held_tree), definitions(live_tree)
        for name in move['definitions']:
            assert ast.dump(expected[name], include_attributes=False) == ast.dump(current[name], include_attributes=False)
            n = current[name]
            start = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', ())])
            moved.append({'from': move['source'], 'to': move['destination'], 'name': name,
                          'retained_definition_lines': n.end_lineno-start+1})
    write('fixture-relocation.json', moved)
    # Source identity correction of an existing latency receipt is metadata only.
    budget_old = json.loads((BASE/'verification/latency_budgets.json').read_text())
    budget_new = json.loads((CANDIDATE/'verification/latency_budgets.json').read_text())
    def deltas(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict):
            return [d for k in sorted(a.keys() | b.keys()) for d in deltas(a.get(k), b.get(k), (*path,k))]
        if isinstance(a, list) and isinstance(b, list) and len(a)==len(b):
            return [d for i,(x,y) in enumerate(zip(a,b)) for d in deltas(x,y,(*path,i))]
        return [] if a==b else [{'path':path,'before':a,'after':b}]
    latency = deltas(budget_old,budget_new)
    write('latency-metadata-deltas.json', latency)
    # Full input inventories and each published source-manifest remain stable.
    assert oldpins == inventory(BASE) and newpins == inventory(CANDIDATE)
    result = {'scope':'Immutable30 versus separately held S10 census-fix; stdlib saved-source audit, no imports/tests',
      'method':'All Python sources under model_unfolder, physics, tests, test_support, scripts; SequenceMatcher autojunk=False (reproduces prior26 counts exactly). Generated receipts, examples, and data are excluded from source line totals; latency metadata is accounted separately.',
      'source_manifest_sha256':sha(source_manifest.read_bytes()),
      's10_source_manifest_gzip_sha256':sha(baseline_manifest.read_bytes()),
      's10_source_manifest_raw_sha256':sha(baseline_raw),
      'baseline_full_python_inventory_sha256':sha((OUT/'baseline-source-inventory.json').read_bytes()),
      'candidate_full_python_inventory_sha256':sha((OUT/'candidate-source-inventory.json').read_bytes()),
      'input_fingerprints_identical_before_after':True,
      'line_growth':counts,'total_added':sum(c['added'] for c in counts.values()),
      'total_deleted':sum(c['deleted'] for c in counts.values()),
      'product_modules_added':[r['path'] for r in rows if r['category']=='production' and r['before_sha256'] is None],
      'product_modules_deleted':[r['path'] for r in rows if r['category']=='production' and r['after_sha256'] is None],
      'fixture_relocation':{'definitions':len(moved),'definition_lines':sum(r['retained_definition_lines'] for r in moved),'destination_modules':len({r['to'] for r in moved}),
         'note':'Moved authority remains present. Gross source additions/deletions include relocation; never credit these deleted test lines or moved uncertainty definitions as authority removal.'},
      'debt_register':{'before':len(before_debt),'after':len(after_debt),'added':0,'eliminated':0,'factories':dict(Counter(r['factory'] for r in after_debt)),
          'limits':'Fresh runtime gate must assess deletion predicates. Fact-level S7 findings are a different census.'},
      'unchanged_inventory_source_pins':stable,
      'retained_producers':{'functions':len(producers),'modules':len({r['path'] for r in producers}),'fixed_intended_triples':sum(len(r['fixed_intended_claims']) for r in producers)},
      'latency_delta_count':len(latency),
      'limits':['No runtime/model import/acceptance/blessing.','S10 UNet deletion and original latency measurement receive no S9-A credit.','Current source counts do not count qualified facts or replace matrix/page evidence.']}
    result['authority_accounting'] = 'authority-bridges-debt.json'
    result['test_combined'] = {
        'added': counts['tests']['added'] + counts['test_support']['added'],
        'deleted': counts['tests']['deleted'] + counts['test_support']['deleted'],
        'note': 'tests + test_support combined; test_support separately includes relocated shared fixtures.'}
    write('summary.json',result)
    print(json.dumps({'counts':counts,'added':result['total_added'],'deleted':result['total_deleted'],'modules':result['product_modules_added'],'debt':result['debt_register'],'producers':result['retained_producers'],'fixture':result['fixture_relocation'],'latency_delta_count':len(latency)}))

if __name__ == '__main__': main()
