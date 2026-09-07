import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/unfold-s8-final-verification-96d0c1e')
out=Path('/private/tmp/unfold-s8-family-96d0c1e');out.mkdir(exist_ok=True)
def manifest():
 return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for base in ('model_unfolder','physics','scripts') for p in (root/base).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
before=manifest();(out/'implementation-before.json').write_text(json.dumps(before,indent=2,sort_keys=True))
for name in ('sdxl','sd-v1-4'):
 case=out/name;case.mkdir(exist_ok=True)
 source=root/'verification/receipts/S8-family-4637ef4'/name/'input.json'
 (case/'input.json').write_bytes(source.read_bytes())
 command=['python3','scripts/verify_s8_unet_family.py','--input',str(case/'input.json'),'--output',str(case)]
 if name=='sdxl':command+=['--accepted-table',str(root/'verification/s7/models/stable-diffusion-xl-base-1-0.json.gz')]
 started=time.monotonic()
 with (case/'run.log').open('w') as log:
  result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
 record={'command':command,'elapsed_seconds':round(time.monotonic()-started,3),'exit_code':result.returncode,'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'implementation_unchanged':manifest()==before}
 (case/'run-receipt.json').write_text(json.dumps(record,indent=2))
 print(name,json.dumps(record),flush=True)
 if result.returncode or not record['implementation_unchanged']:raise SystemExit(1)
(out/'implementation-after.json').write_text(json.dumps(manifest(),indent=2,sort_keys=True))
