from pathlib import Path
import subprocess,json,time,sys
root=Path('/private/tmp/unfold-s8-current-captures-fb1f871')
checkout='/private/tmp/unfold-s8-current-capture-fb1f871'
slugs=['deepseek-v3', 'granite-3-0-8b-instruct', 'hunyuanvideo', 'llama-7b', 'musicgen-small', 'pixart-sigma-xl-2-1024-ms', 'qwen2-vl-7b-instruct', 'stable-diffusion-xl-base-1-0']
progress=[]
for slug in slugs:
 out=root/'example-sable'/slug
 argv=[sys.executable,str(root/'driver/scripts/capture_current.py'),'--checkout',checkout,'--source-manifest',str(root/'source-manifest.json'),'--slug',slug,'--output',str(out)]
 print('START example Sable '+slug,flush=True)
 started=time.monotonic()
 with (root/(slug+'-sable.log')).open('w') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=checkout)
 record={'slug':slug,'argv':argv,'exit_code':r.returncode,'elapsed_seconds':round(time.monotonic()-started,3)}
 (root/(slug+'-sable-process.json')).write_text(json.dumps(record,indent=2)+'\n')
 progress.append(record);(root/'example-sable-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 if r.returncode:raise SystemExit(r.returncode)
 print('COMPLETE example Sable '+slug,flush=True)
