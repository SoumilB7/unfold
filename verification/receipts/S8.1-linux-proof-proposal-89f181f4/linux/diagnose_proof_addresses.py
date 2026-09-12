"""Record actual SDXL proof source closures, without renewing artifacts."""
from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
out = Path('/private/tmp/unfold-s81-proof-address-diagnostic')
out.mkdir(exist_ok=False)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert commit == '89f181f4306e21122d1ac4cc98573454a5937427'
spec = importlib.util.spec_from_file_location('proof_verifier', repo / 'scripts/verify_commit.py')
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
tree = v._add_worktree(commit, 'proof-address', 's81-linux-' + commit[:8])
v._stage_external_artifacts(repo, tree)
os.environ['UNFOLD_EVIDENCE_CACHE_DIR'] = str(out / 'cache')
code = r'''
import dataclasses, hashlib, json, sys
from pathlib import Path
from scripts import generate_s7_shadow as g
from model_unfolder.evidence.program_index import portable_source_index_fingerprint, _aggregate_fingerprint
target = next(t for t in g._targets() if t['slug'] == 'stable-diffusion-xl-base-1-0')
_, config = g._read_payload(g.ROOT / target['input'])
config_hash = g._sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
inventory = g._s6_inventory(target['slug'], config_hash)
assert inventory is not None and inventory.status == 'ok'
context, _, _, _, ir = g._source_inputs(config, inventory.inventory)
indices = {}
facts = []
for key, fact in context.facts.typed_records().items():
    proof = fact.claim_evidence
    if proof is None:
        continue
    summary = proof.summary()
    if not summary.index_fingerprints:
        continue
    facts.append({'key': key, 'summary': dataclasses.asdict(summary)})
    candidates = [getattr(proof, 'index', None)]
    for attr in ('bindings', 'graph'):
        owner = getattr(proof, attr, None)
        candidates.append(getattr(owner, 'index', None))
    for index in candidates:
        if index is None or index.fingerprint in indices:
            continue
        sources = [n.source_id for n in index.source_nodes] + [f.source for f in index.parse_failures]
        assert _aggregate_fingerprint(sources) == index.fingerprint
        linux_sources = []
        for sid in sources:
            path = sid.canonical_path
            if '/site-packages/' in path:
                path = '/opt/hostedtoolcache/Python/3.12.10/x64/lib/python3.12/site-packages/' + path.split('/site-packages/', 1)[1]
            linux_sources.append(dataclasses.replace(sid, canonical_path=path))
        indices[index.fingerprint] = {
            'portable': portable_source_index_fingerprint(index),
            'sources': [dataclasses.asdict(s) for s in sources],
            'linux_path_relocation_only_raw_fingerprint': _aggregate_fingerprint(linux_sources),
        }
assert facts and indices
payload = {'target': target, 'indices': indices, 'facts': facts}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps({'facts':len(facts),'indices':{k:{'portable':v['portable'],'linux_raw':v['linux_path_relocation_only_raw_fingerprint'],'sources':len(v['sources'])} for k,v in indices.items()}},indent=2))
'''
lane = v.Lane('proof-address-diagnostic', (sys.executable, '-c', code, str(out / 'proofs.json')))
result = v._run_lane(lane, tree, out)
record = {'commit': commit, 'worktree': str(tree), 'status': 'PASS' if result.passed else 'FAIL',
          'lane': {**dataclasses.asdict(result), 'log_path': str(result.log_path), 'passed': result.passed}}
(out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
assert result.passed
