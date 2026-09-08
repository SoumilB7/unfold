from pathlib import Path
import subprocess,json,time,sys
r=Path('/private/tmp/unfold-s8-current-captures-fb1f871'); progress=[]
for entry in json.loads((r/'final-render-manifest.json').read_text()):
 c=entry['case'];argv=[sys.executable,str(r/'driver/scripts/replay_final.py'),'--checkout','/private/tmp/unfold-s8-current-capture-fb1f871','--source-manifest',str(r/'source-manifest.json'),'--source',entry['base'],'--output',str(r/'final-render'/c)]
 print('START final renderer '+c,flush=True);start=time.monotonic()
 with (r/(c+'-final-render.log')).open('w') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 record={'case':c,'argv':argv,'exit_code':p.returncode,'elapsed_seconds':round(time.monotonic()-start,3)};progress.append(record);(r/'final-render-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
 print('COMPLETE final renderer '+c,flush=True)
