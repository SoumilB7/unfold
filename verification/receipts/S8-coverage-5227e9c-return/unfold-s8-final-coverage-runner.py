import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/unfold-s8-final-verification-5227e9c')
out=Path('/private/tmp/unfold-s8-coverage-5227e9c');out.mkdir(exist_ok=True)
def write(name,data):(out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def manifest():
 return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for base in ('model_unfolder','physics','scripts') for p in (root/base).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
before=manifest();write('implementation-before.json',before)
helpers=[Path(__file__),Path('/private/tmp/unfold-s8-coverage-capture.py')]
helper_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers};write('capture-helper-pins.json',helper_pins)
accepted=json.loads(subprocess.check_output(['git','show','83140f1:coverage.json'],cwd=root));write('accepted-coverage.json',accepted)
for name,cmd in [('generate',['python3','-u','/private/tmp/unfold-s8-coverage-capture.py',str(root),str(out/'actual-generate')]),('check',['python3','-u','/private/tmp/unfold-s8-coverage-capture.py',str(root),str(out/'actual-check'),'--check'])]:
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
 receipt={'command':cmd,'exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'implementation_unchanged':manifest()==before};write(name+'-receipt.json',receipt);print(name,receipt,flush=True)
 if name=='generate':
  current=json.loads((root/'coverage.json').read_text());write('current-coverage.json',current)
  old={r['input']:r for r in accepted['models']};new={r['input']:r for r in current['models']};assert old.keys()==new.keys() and len(new)==44
  rows=[{'input':p,'model':new[p]['model'],'cohort':new[p]['cohort'],'before':old[p],'after':new[p],'delta':{k:new[p][k]-old[p][k] for k in ('proven','flagged','silent')},'status':'owner_review_required' if old[p]!=new[p] else 'unchanged'} for p in sorted(new)]
  write('per-model-comparison.json',rows);write('totals.json',{v:{k:sum(r[k] for r in doc['models']) for k in ('proven','flagged','silent')} for v,doc in [('accepted',accepted),('current',current)]})
 if result.returncode or not receipt['implementation_unchanged']:raise SystemExit(1)
assert helper_pins=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
write('implementation-after.json',manifest())
