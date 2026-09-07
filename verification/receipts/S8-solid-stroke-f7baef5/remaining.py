from pathlib import Path
import subprocess,json,hashlib
root=Path(__file__).parent;source_root=Path('/private/tmp/unfold-s8-render-phase-c38-v2')
cases={**{'sdxl-'+n:source_root/'sdxl'/n for n in ('sparse','misleading','rewrite','unchanged','changed','missing')},**{n:source_root/'generalization'/n/'ordinary' for n in ('sdxl-refiner','sdxl-inpainting','ddpm-cifar10')}}
for name,source in cases.items():
 case=root/name;case.mkdir()
 for phase,checkout in [('before','/private/tmp/unfold-s8-component-label-review-96d0c1e'),('after','/private/tmp/unfold-s8-solid-review-f7baef5')]:
  cmd=['python3',str(root/'worker.py'),'--checkout',checkout,'--source',str(source),'--output',str(case/phase)]
  with (case/(phase+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:print((case/(phase+'.log')).read_text(),flush=True);raise SystemExit(r.returncode)
 r=subprocess.run(['python3',str(root/'compare.py'),str(case)]);print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
# The raw current renderer pages must retain the original control invariant.
hashes={name:hashlib.sha256((root/name/'after/page.html').read_bytes()).hexdigest() for name in ('sdxl-ordinary','sdxl-rewrite','sdxl-unchanged')};assert len(set(hashes.values()))==1;(root/'raw-control-identity.json').write_text(json.dumps(hashes,indent=2)+'\n');print('Raw ordinary/rewrite/unchanged identity',hashes,flush=True)
