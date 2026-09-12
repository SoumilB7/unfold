from pathlib import Path
import collections,gzip,hashlib,json,tarfile
r=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');b=r/'verification/receipts/S9-A-schema-candidate-fff99e22';raw=Path('/private/tmp/unfold-s9a-thirtysecond/child-native-observation');out=b/'thirtysecond/child-native-observation-r1';out.mkdir(exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def diff(a,b,path=''):
 if type(a)is not type(b):return[{'path':path,'kind':'type_changed','before_type':type(a).__name__,'after_type':type(b).__name__,'before':a,'after':b}]
 if isinstance(a,dict):
  rows=[]
  for k in sorted(set(a)|set(b)):
   q=path+'/'+k.replace('~','~0').replace('/','~1')
   if k not in a or k not in b:rows.append({'path':q,'kind':'added'if k not in a else'deleted','before':a.get(k),'after':b.get(k)})
   else:rows.extend(diff(a[k],b[k],q))
  return rows
 if isinstance(a,list):
  rows=[]
  for i in range(max(len(a),len(b))):
   q=path+'/'+str(i)
   if i>=len(a)or i>=len(b):rows.append({'path':q,'kind':'added'if i>=len(a)else'deleted','before':a[i]if i<len(a)else None,'after':b[i]if i<len(b)else None})
   else:rows.extend(diff(a[i],b[i],q))
  return rows
 return []if a==b else[{'path':path,'kind':'value_changed','before':a,'after':b}]
files={p.relative_to(raw).as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}for p in sorted(raw.rglob('*'))if p.is_file()}
with tarfile.open(out/'exact-capture.tar.gz','w:gz')as archive:
 for name in files:archive.add(raw/name,arcname=name,recursive=False)
with tarfile.open(out/'exact-capture.tar.gz','r:gz')as archive:
 assert {m.name for m in archive.getmembers()}==set(files)
 for m in archive.getmembers():
  data=archive.extractfile(m).read();assert sha(data)==files[m.name]['sha256']and len(data)==files[m.name]['bytes']
modelrows=[];capture=json.loads((raw/'capture/result.json').read_bytes());prior=Path('/private/tmp/unfold-s9a-thirtysecond/current-pages/capture')
for model in capture['models']:
 slug=model['slug'];current=raw/'capture'/slug/'root-ir.json';old=prior/slug/'ir-before-render.json.gz';a=json.loads(gzip.decompress(old.read_bytes()));z=json.loads(current.read_bytes());d=diff(a,z)
 modelrows.append({'slug':slug,'saved_json_recursive_delta_count':len(d),'raw_deltas':d,'saved_current_sha256':sha(current.read_bytes()),'saved_prior_gzip_sha256':sha(old.read_bytes()),'r1_reported_live_vs_json_equal':model['raw_root_ir_equal_saved_actual32'],'original_calls':model['projector_calls'],'observation_errors':model['errors']})
assert sum(v['saved_json_recursive_delta_count']for v in modelrows)==0
summary={'runtime_verdict':'RED_PRESERVED','lane':json.loads((raw/'result.json').read_bytes()),'models':len(modelrows),'callbacks':sum(len(r['original_calls'])for r in modelrows),'saved_json_ir_delta_count':0,'native_child_snapshot_limit':'R1 did not persist live child before/after objects before raising. Exact native tuple/type paths cannot be established from this packet. Source compares live records.to_dict() with a JSON-roundtripped copy, which is an invalid mutation comparison when tuples occur. R2 must capture actual types and compare consistent representations.','unverified_native_joins':105,'original_product_return_objects_unchanged':all(c['return_identity_preserved']for m in modelrows for c in m['original_calls']),'no_product_or_test_runtime_performed_by_this_audit':True,'scope':'Saved JSON diff is the exact shared serialization, not an output normalization or architectural proof.'}
save(out/'artifact-map.json',files);save(out/'saved-ir-deltas.json',modelrows);save(out/'summary.json',summary);(out/'audit-source.py').write_bytes(Path('/private/tmp/archive-child-observer32-r1.py').read_bytes());save(out/'manifest.json',{'files':{p.name:sha(p.read_bytes())for p in sorted(out.iterdir())if p.is_file()}});print(json.dumps({'manifest_sha256':sha((out/'manifest.json').read_bytes()),'files':len(files),'raw_bytes':sum(v['bytes']for v in files.values()),'json_ir_deltas':0},indent=2))
