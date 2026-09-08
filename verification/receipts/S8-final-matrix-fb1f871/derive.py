"""Derived current source manifest over exact reviewed matrix payloads; no runtime."""
import gzip,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/private/tmp/unfold-s8-focused-fb1f871');sys.path.insert(0,str(root))
repo=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
prep=repo/'verification/receipts/S8-final-source-derivation-preparation-fb1f871'
replay=Path('/private/tmp/unfold-s8-matrix-replay-fb1f871')
out=Path('/private/tmp/unfold-s8-derived-matrix-fb1f871');out.mkdir(exist_ok=True)
dest=root/'verification/s7'
def read(p):
 b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def manifest():return {str(p.relative_to(root)):sha(p) for folder in ('model_unfolder','physics','scripts') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
before=manifest();write('source-before.json',before)
assert before==read(replay/'external-before.json')==read(replay/'external-after.json')
assert read(replay/'summary.json')['no_new_model_or_execution']
assert all(row['logical_equal'] for row in read(replay/'full716-comparison.json'))
guards=read(prep/'38-target-guards.json');assert len(guards['targets'])==38
for row in guards['source_dispatch_projector_reconciliation_generator_checks']:assert before[row['path']]==row['sha256']
spec=importlib.util.spec_from_file_location('derived_current_s7',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
targets=g._targets();assert len(targets)==39
by_slug={t['slug']:t for t in guards['targets']};pins={};values={};phases=[];prior=read(dest/'matrix.json');slug='stable-diffusion-xl-base-1-0'
for target in targets:
 key=target['slug']
 if key!=slug:
  row=by_slug[key];assert sha(root/target['input'])==row['input_sha256'];assert sha(dest/'models'/f'{key}.json.gz')==row['model_sha256'];assert not row['new_primitive_receipt_key_present']
 for folder in ('models','observations','relations'):
  rel=Path(folder)/f'{key}.json.gz';pins[str(rel)]=sha(dest/rel);values[(key,folder)]=read(dest/rel)
assert values[(slug,'models')]==read(replay/'model.json.gz')
assert values[(slug,'observations')]==read(replay/'observation.json.gz')
assert values[(slug,'relations')]==read(replay/'relations.json.gz')
def one(target,**kwargs):
 key=target['slug'];phases.append({'slug':key,'phase':'actual_saved_evidence_current_reconciliation' if key==slug else 'exact_retained_input_source_applicability_guards','new_model_execution':False});return tuple(values[(key,folder)] for folder in ('models','observations','relations'))
def forbidden(*args,**kwargs):raise AssertionError('runtime forbidden in derived matrix')
g._one=one;g.inventory_in_subprocess=forbidden;g.observe_relations_in_subprocess=forbidden;g._run_signature_recipe=forbidden
start=time.monotonic();current=g.generate(output=dest)
assert len(phases)==39 and current['models']==prior['models']
assert {k:v for k,v in current.items() if k!='sources'}=={k:v for k,v in prior.items() if k!='sources'}
assert all(sha(dest/path)==value for path,value in pins.items())
write('payload-pins.json',pins);write('per-target-phases.json',phases)
write('summary.json',{'denominator':39,'model_summary_equality':True,'all_matrix_fields_except_sources_equal':True,'117_payload_bytes_equal':True,'new_model_execution':False,'lineage':'25original96 +1separatePixArt +1savedSDXL +12actual5227;716 derived38guards+SDXL;fb1 derived38exactguard checks+actual savedSDXL currentreconciliation','elapsed_seconds':time.monotonic()-start,'matrix_sha256':sha(dest/'matrix.json')})
command=[sys.executable,'scripts/generate_s7_shadow.py','--check']
with (out/'check.log').open('w') as log:r=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT)
write('check.json',{'command':command,'exit_code':r.returncode});assert r.returncode==0
assert before==manifest();write('source-after.json',manifest());print('Derived39/check PASS;117 payload bytes unchanged',flush=True)
