import json,os,subprocess,time
from pathlib import Path
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
out=Path('/private/tmp/unfold-s8-final-broad-7168440');out.mkdir(exist_ok=True)
assert not (out/'receipt.json').exists(), 'refuse overwrite completed broad receipt'
manifest=json.loads((root/'verification/receipts/S8-final-lanes-7168440/manifest.json').read_text())
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
command=[commit if x=='<commit_with_verified_s7_artifacts>' else x for x in manifest['broad_command']]
(out/'command.json').write_text(json.dumps({'commit':commit,'command':command},indent=2)+'\n')
start=time.monotonic()
with (out/'coordinator.log').open('w') as log:
 result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
receipt={'commit':commit,'command':command,'exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-start,3)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
raise SystemExit(result.returncode)
