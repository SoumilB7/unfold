from pathlib import Path
import json, hashlib, subprocess, shutil
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
r=root/'verification/receipts/S9-A-schema-candidate-fff99e22/new-resume'
out=Path('/private/tmp/unfold-s9a-thirtyseventh');out.mkdir(exist_ok=False)
base=root/'.claude/worktrees/verify-s9-a-thirtysixth'; tree=root/'.claude/worktrees/verify-s9-a-thirtyseventh'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
base_manifest=json.loads(Path('/private/tmp/unfold-s9a-thirtysixth/source-manifest.json').read_text())
expected={v['path']:v['sha256'] for v in base_manifest}
allowed={p:{s:['frozen36']} for p,s in expected.items()}; manifests=[]
names=['typed-chip-wrap-proposal-r2']
for n in names:
 m=r/n/'manifest.json';d=json.loads(m.read_text());manifests.append({'path':str(m.relative_to(root)),'sha256':h(m)})
 fs=d.get('files',[])
 if isinstance(fs,dict): fs=[{'path':p,'sha256':s} for p,s in fs.items()]
 for v in fs:
  p=v['path']
  if not p.startswith('proposed/'):continue
  p=p[len('proposed/'):]
  for prefix in ['held/','source/']:
   if p.startswith(prefix): p=p[len(prefix):];break
  if not p.startswith(('model_unfolder/','physics/','tests/','test_support/')):continue
  allowed.setdefault(p,{}).setdefault(v['sha256'],[]).append(n)
paths=set(expected)|set(allowed)
composed=[]
for p in sorted(paths):
 actual=h(root/p)
 assert actual in allowed[p], (p,actual,allowed[p])
 if p in expected: assert h(base/p)==expected[p], ('base moved',p)
 composed.append({'path':p,'sha256':actual,'accepted_snapshots':allowed[p][actual],'frozen36_sha256':expected.get(p),'actual_predecessor_sha256':h(base/p)})
# Every authorized source/test edit must be covered, including added fixture files.
status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=root,text=True)
for line in status.splitlines():
 p=line[3:]
 if p.startswith(('model_unfolder/','physics/','tests/','test_support/')): assert p in paths, ('unaccounted',p)
subprocess.run(['git','worktree','add','--quiet','--detach',str(tree),'fff99e2230064762598748463accd170e1e34e22'],cwd=root,check=True)
for v in composed:
 p=v['path']; dest=tree/p
 if v['sha256'] is None:
  if dest.exists():dest.unlink()
 else:
  dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/p,dest)
 assert h(dest)==v['sha256'],p
# Persisted S10 receipt inputs needed by the unchanged latency contract.
s10='verification/receipts/S10-1-unet-deletion-fff99e22'
inputs=[]
for p in sorted((base/s10).rglob('*')):
 if p.is_file():
  rel=p.relative_to(base);dst=tree/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);inputs.append({'path':str(rel),'sha256':h(p)});assert h(dst)==h(p)
(out/'source-manifest.json').write_text(json.dumps([{'path':v['path'],'sha256':v['sha256']} for v in composed],indent=2)+'\n')
(out/'composition.json').write_text(json.dumps({'base_manifest_sha256':h(Path('/private/tmp/unfold-s9a-thirtysixth/source-manifest.json')),'handoffs':manifests,'files':composed},indent=2)+'\n')
(out/'s10-receipt-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
print(json.dumps({'paths':len(composed),'changed_from36':sum(v['sha256']!=v['frozen36_sha256'] for v in composed),'manifest':h(out/'source-manifest.json'),'s10_inputs':len(inputs),'tree':str(tree)},indent=2))
