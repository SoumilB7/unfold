"""Exact source/input/artifact pins; stdlib only, no product imports."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
FILES={'capture_preservation.py','capture_contract.py','run-capture.py','run-tests.py','plan.json','contract.json','saved-baseline-inventory.json','compare-recovered.py','composition-audit.json','harness.diff'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
def inputs(tree):
    return {'source':{k:files(tree/k) for k in ('model_unfolder','physics','test_support','scripts')},
            'corpus':files(tree/'tests/sable_test_corpus'),
            'historical_baseline':files(tree/'tests/preservation_baseline'),
            'expected_manifest':sha(tree/'tests/preservation_expected_manifest.json'),
            'preservation_test':sha(tree/'tests/test_preservation.py')}
def validate(lane, source_pin):
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    if set(manifest['files'])!=FILES: raise ValueError('Held harness membership differs')
    for name,h in manifest['files'].items():
        if sha(HERE/name)!=h: raise ValueError('Held harness changed: '+name)
    plan=json.loads((HERE/'plan.json').read_bytes())
    if source_pin!=plan['source_manifest_sha256'] or sha(Path(plan['source_manifest']))!=source_pin:
        raise ValueError('Explicit frozen35 source manifest pin differs')
    selected=plan['lanes'][lane]; tree=Path(selected['tree'])
    if tree.resolve()!=tree: raise ValueError('Worktree address differs')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    if head!=selected['head']: raise ValueError('Worktree HEAD differs')
    rows=json.loads(Path(plan['source_manifest']).read_bytes()); sources={}
    if lane=='current35':
        for row in rows:
            relative=row['path']; path=tree/relative
            if path.resolve().relative_to(tree).as_posix()!=relative or relative in sources:
                raise ValueError('Noncanonical/duplicate frozen source address')
            expected=row['sha256']
            if expected is None:
                if path.exists() or path.is_symlink(): raise ValueError('Expected deleted source exists')
            elif sha(path)!=expected: raise ValueError('Frozen source changed: '+relative)
            sources[relative]=expected
    actual=inputs(tree)
    if actual!=selected['frozen_inputs']: raise ValueError('Source/input/artifact census differs from held plan')
    raw=(tree/'tests/preservation_expected_manifest.json').read_bytes()
    if sha(tree/'tests/preservation_expected_manifest.json')!=selected['expected_manifest_sha256']:
        raise ValueError('Expected manifest changed')
    committed=subprocess.check_output(['git','show',head+':tests/preservation_expected_manifest.json'],cwd=tree)
    if raw!=committed: raise ValueError('Expected manifest differs from committed HEAD')
    expected=json.loads(raw)
    if expected['witness_count']!=29 or len(expected['witnesses'])!=29:
        raise ValueError('Expected29 witness set differs')
    return {'lane':lane,'head':head,'source_manifest_sha256':source_pin,'changed_source_rows':sources,
            'frozen_inputs':actual,'scripts':{n:sha(HERE/n) for n in sorted(FILES|{'manifest.json'})}}
