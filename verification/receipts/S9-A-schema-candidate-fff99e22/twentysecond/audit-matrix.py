from pathlib import Path
import json,gzip,hashlib,collections
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
out=Path('/private/tmp/unfold-s9a-twentysecond');matrix=out/'matrix'
assert (matrix/'matrix.json').is_file(), 'wait for complete generation'
old=json.loads((root/'verification/s7/matrix.json').read_text());new=json.loads((matrix/'matrix.json').read_text())
rows=[]
for target in new['models']:
 name=target['slug']+'.json.gz';pa=root/'verification/s7/models'/name;pb=matrix/'models'/name
 a=json.loads(gzip.decompress(pa.read_bytes()));b=json.loads(gzip.decompress(pb.read_bytes()))
 def indexed(d):
  seq=d['table']['occurrences']; result={r['provenance']['instance_path']:r for r in seq}
  assert len(result)==len(seq);return result
 before,after=indexed(a),indexed(b);delta=[];props=collections.Counter()
 for path in sorted(before.keys()|after.keys()):
  x,y=before.get(path),after.get(path)
  if x is None or y is None:delta.append({'instance_path':path,'kind':'occurrence_added_or_removed'});continue
  changed=[]
  for axis in ('construction','execution','provenance'):
   if x[axis]!=y[axis]:changed.append(axis);props[axis]+=1
  for key in sorted(x['projection'].keys()|y['projection'].keys()):
   if x['projection'].get(key)!=y['projection'].get(key):changed.append('projection.'+key);props['projection.'+key]+=1
  if changed:delta.append({'instance_path':path,'changed_fields':changed})
 rows.append({'slug':target['slug'],'old_sha256':hashlib.sha256(pa.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(pb.read_bytes()).hexdigest(),'old_occurrences':len(before),'new_occurrences':len(after),'changed_fields':dict(props),'occurrence_deltas':delta,'relations_equal':a['table']['relations']==b['table']['relations'],'input_failures_equal':a['table']['input_failures']==b['table']['input_failures'],'product_layer_schedule_equal':a['product_layer_schedule']==b['product_layer_schedule'],'old_blocking':len(a['blocking_findings']),'new_blocking':len(b['blocking_findings'])})
record={'scope':'Actual unblessed S7 matrix versus active S7; field changes enumerated, not approved or normalized','old_matrix_sha256':hashlib.sha256((root/'verification/s7/matrix.json').read_bytes()).hexdigest(),'new_matrix_sha256':hashlib.sha256((matrix/'matrix.json').read_bytes()).hexdigest(),'models':rows}
(out/'matrix-delta-inventory.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'models':len(rows),'changes':dict(sum((collections.Counter(r['changed_fields']) for r in rows),collections.Counter())),'relations_equal':all(r['relations_equal'] for r in rows),'schedules_equal':all(r['product_layer_schedule_equal'] for r in rows),'old_blocking':sum(r['old_blocking'] for r in rows),'new_blocking':sum(r['new_blocking'] for r in rows)},indent=2))
