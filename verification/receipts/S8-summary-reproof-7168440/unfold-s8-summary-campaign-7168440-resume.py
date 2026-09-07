import subprocess,json,hashlib,time,os
from pathlib import Path
root=Path('/private/tmp/unfold-s8-final-verification-7168440');out=Path('/private/tmp/unfold-s8-summary-proofs-7168440-resume');out.mkdir(exist_ok=True)
script=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-typed-terminal-focused/reprove_summary.py')
cases={f'sdxl-{c}':Path('/private/tmp/unfold-s8-final-c38b008/sdxl')/c for c in ('ordinary','sparse','misleading','rewrite')}
cases.update({f'sdxl-{c}':Path('/private/tmp/unfold-s8-render-phase-c38-v2/sdxl')/c for c in ('unchanged','changed','missing')})
cases['sd14-ordinary']=Path('/private/tmp/unfold-s8-render-phase-c38-v2/sd-v1-4/ordinary')
cases.update({c:Path('/private/tmp/unfold-s8-render-phase-c38-v2/generalization')/c/'ordinary' for c in ('sdxl-refiner','sdxl-inpainting','ddpm-cifar10')})
(out/'campaign-pin.json').write_text(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'cases':{k:str(v) for k,v in cases.items()}},indent=2)+'\n')
rows=[]
for name,case in cases.items():
 if name in ('sdxl-ordinary','sdxl-sparse','sdxl-misleading'):continue
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(['python3','-u',str(script),str(root),str(case),str(out/name)],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
 row={'case':name,'exit_code':r.returncode,'elapsed_seconds':round(time.monotonic()-start,3)};rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
