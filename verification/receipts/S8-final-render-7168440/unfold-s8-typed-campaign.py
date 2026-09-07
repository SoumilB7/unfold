from pathlib import Path
import subprocess,sys,json,hashlib
root=Path('/private/tmp/unfold-s8-typed-phase-7168440')
cases={f'sdxl-{c}':f'sdxl/{c}' for c in ('sparse','misleading','rewrite','unchanged','changed','missing')}
cases['sd14-ordinary']='sd-v1-4/ordinary'
cases.update({c:f'generalization/{c}/ordinary' for c in ('sdxl-refiner','sdxl-inpainting','ddpm-cifar10')})
for case,path in cases.items():
 proofbase='/private/tmp/unfold-s8-summary-proofs-7168440' if case in ('sdxl-sparse','sdxl-misleading') else '/private/tmp/unfold-s8-summary-proofs-7168440-resume'
 argv=[sys.executable,'/private/tmp/unfold-s8-final-typed-render.py','--checkout','/private/tmp/unfold-s8-final-verification-7168440','--base-case','/private/tmp/unfold-s8-context-phase-5227e9c/'+case,'--fact-case','/private/tmp/unfold-s8-render-phase-c38-v2/'+path,'--summary-proof-dir',proofbase+'/'+case,'--output',str(root/case)]
 print('START '+case,flush=True)
 with (root/(case+'.log')).open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps(json.loads((root/case/'result.json').read_text())),flush=True)
control={c:hashlib.sha256((root/('sdxl-'+c)/'page.html').read_bytes()).hexdigest() for c in ('ordinary','rewrite','unchanged')}
(root/'raw-control-identity.json').write_text(json.dumps(control,indent=2)+'\n');assert len(set(control.values()))==1
print(json.dumps({'raw_control_identity':control}),flush=True)
