from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,shutil
p=argparse.ArgumentParser();p.add_argument('--witness',required=True);p.add_argument('--conditions',nargs='+',required=True);a=p.parse_args()
root=Path('/private/tmp/unfold-s8-compiler-final');out=Path('/private/tmp/unfold-s8-render-phase-c38-v2');scripts=out/'scripts'
sys.path[:0]=[str(scripts),str(root)]
import demonstrate_s8_unet as demo
from report_s8_demonstration import report
demo.ROOT=root
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
production=demo._tree_sources();pin={'production':production,'production_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'phase_scripts':{str(p):sha(p) for p in [*sorted(scripts.glob('*.py')),out/'phase-entry.py',Path(__file__)]},'raw_campaign_pin_sha256':sha(Path('/private/tmp/unfold-s8-final-c38b008/campaign-pin.json'))}
if (out/'phase-pin.json').exists():assert json.loads((out/'phase-pin.json').read_text())==pin
else:(out/'phase-pin.json').write_text(json.dumps(pin,indent=2)+'\n')
(out/'logs').mkdir(exist_ok=True)
review=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design/S8/final-c38b008/render-phase');review.mkdir(parents=True,exist_ok=True)
for condition in a.conditions:
 target=out/a.witness/condition;assert not target.exists()
 source_input=root/'verification/receipts/S8-final-run-preparation/inputs'/(a.witness+'.json')
 cmd=['python3',str(out/'phase-entry.py'),'--checkout',str(root),'--scripts',str(scripts),'--input',str(source_input),'--output',str(out/a.witness),'--condition',condition,'--expect-source-sha256',production['sha256']]
 print(json.dumps({'event':'started','witness':a.witness,'condition':condition}),flush=True);start=time.monotonic()
 with (out/'logs'/f'{a.witness}-{condition}.log').open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=root)
 receipt={'command':cmd,'returncode':run.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'input_sha256':sha(source_input),'phase_pin_sha256':sha(out/'phase-pin.json')}
 assert production==demo._tree_sources();assert all(sha(Path(path))==value for path,value in pin['phase_scripts'].items())
 if (target/'page.html').exists():
  receipt['html_sha256']=sha(target/'page.html');receipt['html_bytes']=(target/'page.html').stat().st_size;shutil.copy2(target/'page.html',review/f'{a.witness}-{condition}.html')
 (out/'logs'/f'{a.witness}-{condition}-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'event':'completed','condition':condition,**receipt}),flush=True)
 if run.returncode:print((out/'logs'/f'{a.witness}-{condition}.log').read_text()[-6000:],flush=True);raise SystemExit(run.returncode)
 if condition!='legacy':
  summary=report(out/a.witness);failures=[row for row in summary['conditions'][condition]['checks'] if row['status']=='FAIL'];print(json.dumps({'event':'checks','condition':condition,'failures':failures}),flush=True)
  if failures:raise SystemExit(2)
