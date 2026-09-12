"""Run the actual SDXL producer and enumerate a scratch-only S7 proposal."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from scripts import generate_s7_shadow as g

if sys.flags.optimize:
    raise RuntimeError('Proposal guards require assertions')

out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
target = next(t for t in g._targets() if t['slug'] == 'stable-diffusion-xl-base-1-0')
relative = 'models/stable-diffusion-xl-base-1-0.json.gz'
before = json.loads(gzip.decompress((g.OUTPUT / relative).read_bytes()))
actual, observation, relations = g._one(target, write_relations=False)
g._write_gzip_json(out / 'actual-model.json.gz', actual)
g._write_gzip_json(out / 'actual-observation.json.gz', observation)
g._write_gzip_json(out / 'actual-relations.json.gz', relations)

def differences(a, b, path='$'):
    if type(a) is not type(b):
        return [{'path': path, 'before': a, 'after': b}]
    if isinstance(a, dict):
        assert a.keys() == b.keys(), ('keys', path)
        return [row for key in sorted(a) for row in differences(a[key], b[key], path + '.' + key)]
    if isinstance(a, list):
        assert len(a) == len(b), ('length', path)
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in differences(x, y, path + '[' + str(i) + ']')]
    return [] if a == b else [{'path': path, 'before': a, 'after': b}]

delta = differences(g._semantic_payload(before), g._semantic_payload(actual))
(out / 'full-semantic-delta.json').write_text(json.dumps(delta, indent=2) + '\n')
expected = {
    '09fc7b6a95f66c670862b0660094bce1078a3b341e2c963b5da40d3e19673f9d': 'da6dcc0fb04a1b1751240838c08858850b61ec18c40b3413f7a12b004e490680',
    '14f56c8ae9acd61f6cf43b673a4143dcb3593dfa2124757f63bdcc635d5a637e': '524247a453189c6da1e8ab7775de0fed439cab4aa9442998e32a06ebd8d61382',
}
pattern = re.compile(r'^\$\.table\.occurrences\[\d+\]\.projection\.fact_claim_proofs\[\d+\]\.index_fingerprints\[0\]$')
assert len(delta) == 200, len(delta)
assert all(pattern.fullmatch(row['path']) and expected.get(row['before']) == row['after'] for row in delta)
candidate = copy.deepcopy(before)
changed = 0
for occurrence in candidate['table']['occurrences']:
    for proof in occurrence['projection'].get('fact_claim_proofs', []):
        if proof.get('index_fingerprints'):
            assert len(proof['index_fingerprints']) == 1
            old = proof['index_fingerprints'][0]
            proof['index_fingerprints'] = [expected[old]]
            changed += 1
assert changed == 200
assert g._semantic_payload(candidate) == g._semantic_payload(actual)
assert differences(before, candidate) == delta
g._assert_logical_payload_matches(slug=target['slug'], kind='observation', actual=observation,
    expected_path=g.OUTPUT / 'observations/stable-diffusion-xl-base-1-0.json.gz')
g._assert_logical_payload_matches(slug=target['slug'], kind='relation', actual=relations,
    expected_path=g.OUTPUT / 'relations/stable-diffusion-xl-base-1-0.json.gz')
proposal = out / 'proposal'
shutil.copytree(g.OUTPUT, proposal)
g._write_gzip_json(proposal / relative, candidate)
matrix_before = json.loads((g.OUTPUT / 'matrix.json').read_text())
matrix = copy.deepcopy(matrix_before)
matrix['sources'] = g._source_hashes(g._targets())
matrix['artifacts'][relative] = g._sha256((proposal / relative).read_bytes())
matrix['logical_artifacts'][relative] = g._sha256(g._json_bytes(g._semantic_payload(candidate)))
g._write_json(proposal / 'matrix.json', matrix)
g.check(proposal)
matrix_delta = differences(matrix_before, matrix)
(out / 'matrix-delta.json').write_text(json.dumps(matrix_delta, indent=2) + '\n')
renewed = []
unchanged = []
for path in sorted(g.OUTPUT.rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(g.OUTPUT).as_posix()
    dst = proposal / rel
    row = {'path': rel, 'before_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'after_sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}
    (unchanged if row['before_sha256'] == row['after_sha256'] else renewed).append(row)
assert {row['path'] for row in renewed} == {relative, 'matrix.json'}
payloads_unchanged = [r for r in unchanged if r['path'].startswith(('models/', 'observations/', 'relations/')) and r['path'].endswith('.json.gz')]
assert len(payloads_unchanged) == 116
result = {'status': 'PASS_SCRATCH_PROPOSAL_REQUIRES_OWNER_APPROVAL',
          'actual_semantic_comparison': 'exactly 200 proof index seals; zero other differences',
          'fresh_observation_and_relation_semantics': 'unchanged against committed artifacts',
          'changed_summary_seals': changed, 'renewed': renewed,
          'unchanged_payloads': len(payloads_unchanged), 'unchanged_files': unchanged,
          'matrix_summary_rows_unchanged': matrix['models'] == matrix_before['models'],
          'normalizer_unchanged': True,
          'approval': 'NOT_REQUESTED_OR_GRANTED_FOR_THIS_NEW_PAYLOAD_DELTA'}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'unchanged_files'}, indent=2))
