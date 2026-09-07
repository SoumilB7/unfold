"""Final source-pinned derivation over owner-reviewed unchanged matrix payloads."""
from pathlib import Path
import gzip,hashlib,importlib.util,json,os,subprocess,sys,time
root=Path('/private/tmp/unfold-s8-final-verification-7168440');base=Path('/private/tmp/unfold-s8-final-verification-5227e9c/verification/s7');out=Path('/private/tmp/unfold-s8-derived-matrix-7168440');out.mkdir(exist_ok=True);sys.path.insert(0,str(root))
owner=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-matrix-scope-owner-7168440');replay=Path('/private/tmp/unfold-s8-matrix-replay-7168440');dest=root/'verification/s7'
def read(p):
 b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def write(name,v):(out/name).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(root)):sha(p) for area in ('model_unfolder','physics','scripts') for p in (root/area).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
before=manifest();write('implementation-before.json',before)
ruling=read(owner/'result.json');targets=read(owner/'targets.json');assert ruling['status']=='ACCEPT_bounded_table_reuse_source_input_scope' and len(targets)==38
for path,versions in ruling['changed_production_files'].items():assert before[path]==versions['7168440']
for path,expected in ruling['unchanged_generation_sources'].items():assert before[path]==expected
assert read(replay/'source-before.json')==read(replay/'source-after.json')==before
assert read(replay/'summary.json')['no_new_model_or_execution']
model=read(replay/'model.json.gz');slug='stable-diffusion-xl-base-1-0';assert model==read(base/'models'/f'{slug}.json.gz')
for group,file in [('observations','observation.json.gz'),('relations','relations.json.gz')]:assert read(replay/file)==read(base/group/f'{slug}.json.gz')
spec=importlib.util.spec_from_file_location('derived_s7_final716',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
all_targets=g._targets();assert len(all_targets)==39
by_slug={r['slug']:r for r in targets};pins={};phases=[]
for target in all_targets:
 key=target['slug']
 if key!=slug:
  approved=by_slug[key];assert approved['artifact_sha256']==sha(base/'models'/f'{key}.json.gz')
  assert approved['input_sha256']==sha(root/target['input'])
  assert not approved['required_summary_fact_keys_in_projection']
 for group in ('models','observations','relations'):
  rel=Path(group)/f'{key}.json.gz';pins[str(rel)]=sha(base/rel);assert sha(dest/rel)==pins[str(rel)]
def one(target,**kwargs):
 key=target['slug'];phases.append({'slug':key,'phase':'actual_saved_evidence_reconciliation_7168440' if key==slug else 'owner_source_input_proven_unchanged_payload','new_model_or_execution':False})
 return tuple(read(base/group/f'{key}.json.gz') for group in ('models','observations','relations'))
def forbidden(*args,**kwargs):raise AssertionError('final matrix derivation must not execute a runtime model')
g._one=one;g.inventory_in_subprocess=forbidden;g.observe_relations_in_subprocess=forbidden;g._run_signature_recipe=forbidden
start=time.monotonic();matrix=g.generate(output=dest)
assert len(phases)==39 and matrix['models']==read(base/'matrix.json')['models']
assert all(sha(dest/p)==value for p,value in pins.items())
assert before==manifest()
write('per-target-phases.json',phases);write('all39-original-artifact-sha256.json',pins);write('owner-review-pins.json',{str(p):sha(p) for p in owner.iterdir() if p.is_file()});write('generation-receipt.json',{'elapsed_seconds':round(time.monotonic()-start,3),'denominator':39,'all_model_summaries_equal_5227':True,'all117_artifact_bytes_equal_5227':True,'new_model_or_execution':False,'source_manifest_refreshed_by_real_generator':True,'lineage':'25original96 +1separatePixArt +1savedSDXL +12actual5227;716derivation adds38source-input no-op proofs +1savedSDXL reconciliation'})
command=['python3','scripts/generate_s7_shadow.py','--check']
with (out/'check.log').open('w') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
write('check-receipt.json',{'command':command,'exit_code':r.returncode});assert not r.returncode;assert before==manifest();write('implementation-after.json',manifest());print('Final derived39 and unchanged check PASS; all117 payload files byte-identical',flush=True)
