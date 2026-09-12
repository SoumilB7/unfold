import sys,json,subprocess,time,hashlib,shutil
from pathlib import Path
checkout=Path('/private/tmp/unfold-s8-final-boundary-review');root=Path('/private/tmp/unfold-s8-generalization-293c2d9');pin=json.loads(Path('/private/tmp/unfold-s8-final-293c2d9/campaign-pin.json').read_text());(root/'campaign-pin.json').write_text(json.dumps(pin,indent=2)+'\n')
sys.path.insert(0,str(checkout/'scripts'));sys.path.insert(0,str(checkout));from report_s8_demonstration import report
review=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/generalization-293c2d9');review.mkdir(exist_ok=True)
records=[]
for name in ['sdxl-refiner','sdxl-inpainting','ddpm-cifar10']:
 p=root/name;cmd=['python3','scripts/demonstrate_s8_unet.py','--input',str(p/'input.json'),'--output',str(p),'--condition','ordinary','--expect-source-sha256',pin['production']['sha256']];assert not (p/'ordinary').exists();start=time.monotonic();print(json.dumps({'event':'started','witness':name}),flush=True)
 with (p/'run.log').open('w') as log:result=subprocess.run(cmd,cwd=checkout,stdout=log,stderr=subprocess.STDOUT)
 row={'witness':name,'command':cmd,'returncode':result.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'commit':pin['commit'],'scope':'Exploratory published denoiser-only config; not final S8 acceptance.'};page=p/'ordinary/page.html'
 if page.exists():
  raw=page.read_bytes();row.update(html_bytes=len(raw),html_sha256=hashlib.sha256(raw).hexdigest());shutil.copyfile(page,review/(name+'.html'))
 if result.returncode==0:
  try:r=report(p);row['report_failures']=[c for c in r['conditions']['ordinary']['checks'] if c['status']=='FAIL']
  except Exception as e:row['report_error']=repr(e)
 else:row['failure_log_tail']=(p/'run.log').read_text()[-4000:]
 (p/'run.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row);(root/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'event':'completed',**row}),flush=True)
 links=''.join('<li><a href="'+x['witness']+'.html">'+x['witness']+'</a> — exit '+str(x['returncode'])+'</li>' for x in records if 'html_bytes'in x);(review/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>293c2d9 exploratory UNet generalization</h1><p>Published denoiser configs only; no fabricated pipeline components. Actual outputs, not final acceptance.</p><ul>'+links+'</ul>')
