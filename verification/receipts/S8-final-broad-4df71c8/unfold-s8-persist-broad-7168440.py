import gzip,hashlib,json,re,shutil
from pathlib import Path
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
outer=Path('/private/tmp/unfold-s8-final-broad-7168440');lane=Path('/private/tmp/model-unfolder-verification/ee416b74b1')
assert (outer/'receipt.json').exists() and (lane/'receipt.json').exists()
out=root/'verification/receipts/S8-final-broad-4df71c8';out.mkdir(exist_ok=True)
sources=[('outer',outer),('coordinator',lane)]
full=lane/'full.log'
if full.exists():
 match=re.search(r'^batch logs: (.+)$',full.read_text(),re.M)
 if match:sources.append(('file-bracket',Path(match.group(1))))
rows=[]
for prefix,base in sources:
 for p in sorted(base.rglob('*')):
  if not p.is_file():continue
  rel=Path(prefix)/p.relative_to(base);data=p.read_bytes();target=out/rel
  encoded=data
  if len(data)>500000 and p.suffix!='.gz':target=Path(str(target)+'.gz');encoded=gzip.compress(data,mtime=0)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
  rows.append({'original':str(p),'artifact':str(target.relative_to(out)),'original_sha256':hashlib.sha256(data).hexdigest(),'artifact_sha256':hashlib.sha256(encoded).hexdigest()})
for name in ('unfold-s8-final-broad-runner-7168440.py','unfold-s8-persist-broad-7168440.py'):
 shutil.copy2(Path('/private/tmp')/name,out/name)
(out/'artifact-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(out)
