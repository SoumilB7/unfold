from pathlib import Path
import json,gzip,hashlib,time
if __debug__ is not True:raise RuntimeError('Assertions required')
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8.1-owner-resubmission-actual-results')
sha=lambda b:hashlib.sha256(b).hexdigest()
start=time.perf_counter();mraw=(root/'artifact-map.json').read_bytes();m=json.loads(mraw);praw=(root/'released-plan.json').read_bytes();plan=json.loads(praw)
assert sha(praw)==m['plan_sha256'];assert sha((root/'archive.py').read_bytes())==m['archive_sha256']
seen={};total=0
for name,item in m['entries'].items():
 f=root/item['stored'];packed=f.read_bytes()
 assert len(packed)==item['stored_bytes'] and sha(packed)==item['stored_sha256'],name
 raw=gzip.decompress(packed)
 assert len(raw)==item['raw_bytes'] and sha(raw)==item['raw_sha256'],name
 assert raw==Path(item['original']).read_bytes(),name
 total+=len(raw);seen[item['stored']]=item['stored_sha256']
assert {str(p.relative_to(root)) for p in (root/'content').iterdir()}==set(seen)
for s in plan['scopes']:
 base=Path(s['source']);files={};dirs=[]
 if base.is_file():files={'':base.stat().st_size}
 else:
  def visit(p,rel):
   dirs.append(str(rel))
   for f in sorted(p.iterdir()):
    r=rel/f.name
    if str(r) in s.get('excludes',[]):continue
    assert not f.is_symlink()
    if f.is_dir():visit(f,r)
    else:assert f.is_file();files[str(r)]=f.stat().st_size
  visit(base,Path('.'))
 assert files==s['files'] and dirs==s['directories'],s['namespace']
 assert {s['namespace']+('/'+r if r else '') for r in files}=={n for n in m['entries'] if n==s['namespace'] or n.startswith(s['namespace']+'/')}
 assert m['directories'][s['namespace']]==dirs
assert (root/'artifact-map.json').read_bytes()==mraw
out={'status':'PASS_ROOT_ARCHIVE_BYTE_AUDIT','artifact_map_sha256':sha(mraw),'streams':len(m['entries']),'unique_stored_streams':len(seen),'raw_bytes':total,'elapsed_seconds':time.perf_counter()-start,'checks':['stored hashes/sizes','decompressed raw hashes/sizes','every original byte equality','source and logical membership','exact stored membership'],'acceptance':'S8.1 RETURNED; artifact approval PENDING; no Linux/push verdict'}
Path('/private/tmp/unfold-s81-cap3-root-archive-audit/result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
