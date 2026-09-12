"""Explicitly attributed 26 preserved + 1 saved-evidence replay + 12 fresh matrix."""
import gzip,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);replay=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('derived_s7',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
original=Path('/private/tmp/unfold-s8-final-verification-96d0c1e/verification/s7');destination=root/'verification/s7'
def read(p):
 data=p.read_bytes();return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():return {str(p.relative_to(root)):sha(p) for area in ('model_unfolder','physics','scripts') for p in (root/area).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
before=sources();write(out/'implementation-before.json',before)
proof=read(Path('/private/tmp/unfold-s8-completed-matrix-premises-v2/completed26-premises.json'));assert len(proof)==26
for row in proof:
 assert not row['unet_cutover_selected']
 if not row['same_context_bundle']:
  assert row['target']['slug']=='pixart-sigma-xl-2-1024-ms'
  for phase in ('before','after'):
   audit=read(Path('/private/tmp/unfold-s8-pixart-address-replay')/phase/'summary.json')
   assert audit['model_equal_original'] and audit['observations_exact'] and audit['relations_exact'] and audit['no_new_model_or_execution']
assert read(replay/'source-after.json')==read(replay/'source-before.json')==before
assert read(replay/'summary.json')['no_new_model_or_execution']
targets=g._targets();cached={t['slug'] for t in targets[:26]};assert {r['target']['slug'] for r in proof}==cached
original_pins={}
for target in targets[:26]:
 for group in ('models','observations','relations'):
  rel=Path(group)/f"{target['slug']}.json.gz";source=original/rel;dest=destination/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);original_pins[str(rel)]=sha(source)
for group,name in [('observations','observation.json.gz'),('relations','relations.json.gz')]:
 dest=destination/group/'stable-diffusion-xl-base-1-0.json.gz';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(replay/name,dest)
write(out/'preserved26-artifact-sha256.json',original_pins)
real_one=g._one;phases=[]
def one(target,**kwargs):
 slug=target['slug'];started=time.monotonic()
 if slug in cached:
  answer=tuple(read(original/group/f'{slug}.json.gz') for group in ('models','observations','relations'));phase=('preserved_96d0c1e_separately_replayed_address' if slug=='pixart-sigma-xl-2-1024-ms' else 'preserved_generation_96d0c1e')
 elif slug=='stable-diffusion-xl-base-1-0':
  answer=tuple(read(replay/name) for name in ('model.json.gz','observation.json.gz','relations.json.gz'));phase='saved_evidence_final_source_reconciliation'
 else:answer=real_one(target,**kwargs);phase='fresh_final_source_generation'
 phases.append({'slug':slug,'phase':phase,'elapsed_seconds':round(time.monotonic()-started,3)})
 write(out/'per-target-phases.json',phases);return answer
g._one=one
started=time.monotonic();matrix=g.generate(output=destination)
assert before==sources()
assert all(sha(destination/rel)==digest for rel,digest in original_pins.items())
write(out/'generate-receipt.json',{'elapsed_seconds':round(time.monotonic()-started,3),'exit_code':0,'denominator':39,'phase_counts':{p:sum(r['phase']==p for r in phases) for p in sorted({r['phase'] for r in phases})},'original26_artifact_bytes_unchanged':True,'implementation_unchanged':True,'fresh39_model_run':False})
command=['python3','scripts/generate_s7_shadow.py','--check']
with (out/'check.log').open('w') as log:checked=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
write(out/'check-receipt.json',{'command':command,'exit_code':checked.returncode});assert checked.returncode==0
accepted=json.loads(subprocess.check_output(['git','show','83140f1:verification/s7/matrix.json'],cwd=root))
old={r['slug']:r for r in accepted['models']};new={r['slug']:r for r in matrix['models']};assert old.keys()==new.keys() and len(new)==39
comparison=[{'slug':slug,'before':old[slug],'after':new[slug],'changed_fields':{k:{'before':old[slug].get(k),'after':new[slug].get(k)} for k in sorted(old[slug].keys()|new[slug].keys()) if old[slug].get(k)!=new[slug].get(k)}} for slug in sorted(new)]
write(out/'per-model-comparison.json',comparison);write(out/'implementation-after.json',sources());print('Derived matrix and unchanged --check complete',flush=True)
