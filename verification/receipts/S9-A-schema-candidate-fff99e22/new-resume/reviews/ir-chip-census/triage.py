"""Classify saved markers; finite typed-null count, stdlib data only."""
from pathlib import Path
import collections, gzip, hashlib, json
OUT=Path(__file__).resolve().parent
CAPTURE=Path('/private/tmp/unfold-s9a-twentysixth/current-pages/capture')
FROZEN=Path('.claude/worktrees/verify-s9-a-twentysixth')
patterns={}; bymodel=[]
def pattern(path):return '.'.join('*' if type(p) is int else p for p in path)
def slots(doc):
 def blocks(items,p):
  for i,b in enumerate(items if isinstance(items,list) else []):
   if isinstance(b,dict):
    yield (*p,i),b
    yield from blocks(b.get('children'),(*p,i,'children'))
 for i,l in enumerate(doc.get('layers') or []):yield from blocks(l.get('blocks'),('layers',i,'blocks'))
 r=(doc.get('extras') or {}).get('render') or {}
 for n in ('model_blocks','loop_blocks'):yield from blocks(r.get(n),('extras','render',n))
 for n in ('loop_region','opaque_layer_block'):
  if isinstance(r.get(n),dict):yield ('extras','render',n),r[n]
def typed_nulls(l,p):
 out=[]
 def add(d,p,names):
  if isinstance(d,dict):
   out.extend([(*p,n) for n in names if n in d and d[n] is None])
 for n in ('attention','cross_attention'):
  a=l.get(n)
  if not isinstance(a,dict):continue
  q=(*p,n);add(a,q,('kind','mask','qk_norm','cached','output_projection','bias','scores_scaled'))
  if a.get('kind') in (None,'mha','gqa','mqa'):add(a,q,('projection_mode',))
  if a.get('cross_attention') is True:add(a,q,('cross_kv_source_kind',))
 f=l.get('ffn')
 if isinstance(f,dict):
  q=(*p,'ffn');add(f,q,('kind','activation','gated'))
  if f.get('kind') in (None,'dense','moe'):add(f,q,('projection_mode',))
  if f.get('kind')=='moe':add(f,q,('expert_projection_mode',))
 if l.get('residual_topology') in ('parallel','fused_parallel'):add(l,p,('parallel_norm_count',))
 return out
def nested(spec,p):
 if not isinstance(spec,dict):return
 for i,g in enumerate(spec.get('groups') or []):
  if isinstance(g,dict):yield from typed_nulls(g,(*p,'groups',i))
 for i,c in enumerate(spec.get('sub_models') or []):yield from nested(c,(*p,'sub_models',i))
for m in json.loads((OUT/'summary.json').read_text())['models']:
 slug=m['slug'];report=json.loads((OUT/(slug+'.json')).read_text());phase=report['phases']['ir-after-render.json.gz']
 for row in phase['outside_finite_coverage_markers']:
  p=row['path'];pat=pattern(p)
  if 'source_evidence' in p:category='diagnostic_copy_excluded'
  elif p[-1]=='opaque_layer_block':category='opaque_architectural_envelope'
  elif 'modalities' in p:category='canonical_modality_unknown'
  else:category='typed_drill_or_submodel_snapshot'
  v=patterns.setdefault(pat,{'count':0,'category':category,'models':{},'example_path':p})
  v['count']+=1;v['models'][slug]=v['models'].get(slug,0)+1
 raw=(CAPTURE/slug/'ir-after-render.json.gz').read_bytes();doc=json.loads(gzip.decompress(raw));nulls=[]
 for p,b in slots(doc):
  d=b.get('detail')
  if isinstance(d,dict):
   nulls.extend(typed_nulls({n:d[n] for n in ('attention','cross_attention','ffn') if n in d},(*p,'detail')))
   nulls.extend(nested(d.get('sub_model'),(*p,'detail','sub_model')))
 bymodel.append({'slug':slug,'typed_snapshot_null_count':len(nulls),'typed_snapshot_null_paths':[list(p) for p in nulls],
   'opaque_selected_by_zero_layer_branch': not doc.get('layers') and isinstance(((doc.get('extras') or {}).get('render') or {}).get('opaque_layer_block'),dict)})
counts=collections.Counter()
for p,v in patterns.items():counts[v['category']]+=v['count']
result={'scope':'frozen26 saved transport addresses, not mechanisms or construction occurrences',
        'explicit_marker_categories':dict(counts),'patterns':patterns,'models':bymodel,
        'typed_snapshot_null_count':sum(m['typed_snapshot_null_count'] for m in bymodel)}
(OUT/'finite-triage.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'categories':counts,'patterns':len(patterns),'additional_typed_nulls':result['typed_snapshot_null_count'],'selected_opaque_models':[m['slug'] for m in bymodel if m['opaque_selected_by_zero_layer_branch']]},indent=2))
