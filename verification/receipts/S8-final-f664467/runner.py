"""Serial frozen-candidate demonstration runner; never modifies source or blesses."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, time
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--witness',choices=['sdxl','sd-v1-4'],required=True);p.add_argument('--conditions',nargs='+',required=True);a=p.parse_args()
root=a.checkout;out=a.output;out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
sys.path[:0]=[str(root/'scripts'),str(root)]
from demonstrate_s8_unet import _tree_sources
from report_s8_demonstration import report
production=_tree_sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();inputs={w:root/'verification/receipts/S8-final-run-preparation/inputs'/(w+'.json') for w in ['sdxl','sd-v1-4']}
pin={'commit':commit,'production':production,'scripts':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'scripts/demonstrate_s8_unet.py',root/'scripts/report_s8_demonstration.py',root/'scripts/verify_s8_unet_family.py']},'inputs':{w:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for w,p in inputs.items()},'native_gallery_script_sha256':hashlib.sha256(Path('/private/tmp/unfold-s8-native-gallery.py').read_bytes()).hexdigest(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert not pin['git_status'], 'candidate must be clean'
if (out/'campaign-pin.json').exists():
 saved=json.loads((out/'campaign-pin.json').read_text());assert saved==pin,'candidate pin changed between lanes'
else:(out/'campaign-pin.json').write_text(json.dumps(pin,indent=2)+'\n')
review=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8')/('final-'+commit[:7]);review.mkdir(exist_ok=True);shutil.copy2(out/'campaign-pin.json',review/'campaign-pin.json')
for condition in a.conditions:
 target=out/a.witness/condition
 assert not target.exists(),'never overwrite a condition attempt'
 if condition not in ['ordinary','legacy']:assert (out/a.witness/'ordinary/result.json').exists(),'ordinary required first'
 cmd=['python3','scripts/demonstrate_s8_unet.py','--input',str(inputs[a.witness]),'--output',str(out/a.witness),'--condition',condition,'--expect-source-sha256',production['sha256']]
 start=time.monotonic();print(json.dumps({'event':'started','witness':a.witness,'condition':condition,'commit':commit}),flush=True)
 with (out/'logs'/f'{a.witness}-{condition}.log').open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 receipt={'command':cmd,'condition':condition,'witness':a.witness,'elapsed_seconds':round(time.monotonic()-start,3),'returncode':r.returncode,'commit':commit}
 assert _tree_sources()==production,'production source changed during campaign'
 assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in pin['scripts'].items()),'campaign scripts changed'
 if (target/'page.html').is_file():
  data=(target/'page.html').read_bytes();receipt.update(html_bytes=len(data),html_sha256=hashlib.sha256(data).hexdigest());shutil.copy2(target/'page.html',review/f'{a.witness}-{condition}.html')
 (out/'logs'/f'{a.witness}-{condition}-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
 links=''.join('<li><a href="'+page.name+'">'+page.stem+'</a></li>' for page in sorted(review.glob('*.html')) if page.name!='index.html')
 (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S8 candidate '+commit[:7]+'</title><h1>S8 candidate '+commit[:7]+' — campaign in progress</h1><p>Actual generated pages; provisional review outputs, no blessing or acceptance. Native reviews and condition reports remain required.</p><p><a href="campaign-pin.json">Exact source pin</a></p><ul>'+links+'</ul>')
 print(json.dumps({'event':'completed',**receipt}),flush=True)
 if r.returncode:
  print((out/'logs'/f'{a.witness}-{condition}.log').read_text()[-7000:],flush=True);raise SystemExit(r.returncode)
 if condition!='legacy':
  reports=report(out/a.witness);failures=[row for row in reports['conditions'][condition]['checks'] if row['status']=='FAIL']
  print(json.dumps({'event':'condition_checks','witness':a.witness,'condition':condition,'failures':failures}),flush=True)
  if failures:raise SystemExit(2)
