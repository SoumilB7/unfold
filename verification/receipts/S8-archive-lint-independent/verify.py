"""Read-only receipt/source-archive verification; no models or test runner."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CORRECTION=ROOT/'verification/receipts/S8-archive-lint-correction-4df71c8'
RED=ROOT/'verification/receipts/S8-final-broad-4df71c8'
COMMIT='4df71c8ca919161e6995192eae0d49deffecb6b7'
def old(path):return subprocess.check_output(['git','show',COMMIT+':'+path],cwd=ROOT)
def sha(data):return hashlib.sha256(data).hexdigest()
r=json.loads((RED/'coordinator/receipt.json').read_text());outer=json.loads((RED/'outer/receipt.json').read_text())
assert r['commit']==outer['commit']==COMMIT and outer['exit_code']==1
assert r['failed_lanes']==['static'] and r['missing_lanes']==['full','preservation']
assert r['lane_schedule']=='serial'
assert r['coordinator_fingerprint_before']==r['coordinator_fingerprint_after']
assert r['source_artifacts_before']==r['source_artifacts_after']
for name,lane in r['lanes'].items():
 assert lane['fingerprint_before']==lane['fingerprint_after'] and lane['artifacts_before']==lane['artifacts_after']
 assert lane['passed']==(name!='static') and lane['returncode']==(1 if name=='static' else 0)
 nested=Path('/private/tmp/model-unfolder-verification/ee416b74b1')
 for item in ('receipt.json','focused.log','u2-authority.log','collect.log','static.log'):
  assert (nested/item).read_bytes()==(RED/'coordinator'/item).read_bytes()
rows=json.loads((CORRECTION/'source-map.json').read_text());assert len(rows)==3
removed=[]
for row in rows:
 original=old(row['original_path']);archive=(ROOT/row['historical_archive']).read_bytes();clean=(ROOT/row['cleaned_runnable_path']).read_bytes()
 assert original==archive and sha(archive)==row['original_executed_sha256']==row['historical_archive_sha256']
 assert sha(clean)==row['cleaned_sha256'] and row['cleaned_copy_executed_in_historical_campaign'] is False
 expected=copy.deepcopy(ast.parse(original));names=set(row['removed_unused_standard_library_imports']);seen=[]
 for node in expected.body:
  if isinstance(node,ast.Import):
   seen.extend(a.name for a in node.names if a.name in names)
   node.names=[a for a in node.names if a.name not in names]
 expected.body=[n for n in expected.body if not isinstance(n,ast.Import) or n.names]
 assert set(seen)==names and len(seen)==len(names)
 assert ast.dump(expected,include_attributes=False)==ast.dump(ast.parse(clean),include_attributes=False)
 removed.extend(seen)
assert sorted(removed)==['gzip','hashlib','hashlib','html','re']
idx='verification/receipts/S8-final-render-7168440/artifact-sha256.json'
assert old(idx)==(CORRECTION/'original-render-index.json').read_bytes()==(ROOT/Path(idx).parent/'artifact-sha256.executed-4df71c8.json').read_bytes()
current=json.loads((ROOT/idx).read_text())
assert all(sha((ROOT/Path(idx).parent/name).read_bytes())==digest for name,digest in current.items())
for path in ('scripts/verify_commit.py','scripts/pytest_file_bracket.py'):
 assert old(path)==(ROOT/path).read_bytes()
assert not subprocess.check_output(['git','diff','--','model_unfolder','physics','scripts','tests','test_support'],cwd=ROOT)
static=json.loads((CORRECTION/'static-result.json').read_text());assert static['exit_code']==0
print(json.dumps({'verdict':'ACCEPT bounded archival correction; original broad remains FAIL',
 'failed_commit':COMMIT,'focused_passed':439,'authority_passed':44,'collected':4476,
 'missing_lanes':r['missing_lanes'],'lane_and_global_pins_equal':True,
 'red_archive_equals_actual_logs':True,'exact_historical_scripts_preserved':3,
 'only_removed_imports':removed,'original_artifact_index_preserved':True,
 'current_packaging_index_valid':True,'production_tests_gate_unchanged':True,
 'cleaned_copies_claimed_historically_executed':False,
 'static_correction_result_scope':'executor recorded unchanged21-file static check; not independently rerun',
 'fresh_whole_coordinator_still_required':True},indent=2,sort_keys=True))
