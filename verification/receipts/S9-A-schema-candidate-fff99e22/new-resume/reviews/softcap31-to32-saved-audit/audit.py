"""Independent saved JSON/HTML inspection only. No product imports or normalization."""
from pathlib import Path
import json,hashlib,gzip,difflib,collections
HERE=Path(__file__).resolve().parent
ROOTS={g:Path('/private/tmp/unfold-s9a-'+g+'/softcap-diagnostic') for g in ('thirtyfirst','thirtysecond')}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def delta(a,b,path=''):
 if type(a)is not type(b):return [{'path':path,'before':a,'after':b}]
 if isinstance(a,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   q=path+'/'+k.replace('~','~0').replace('/','~1')
   if k not in a:out.append({'path':q,'added':b[k]})
   elif k not in b:out.append({'path':q,'removed':a[k]})
   else:out.extend(delta(a[k],b[k],q))
  return out
 if isinstance(a,list):
  out=[]
  for i in range(max(len(a),len(b))):
   q=path+'/'+str(i)
   if i>=len(a):out.append({'path':q,'added':b[i]})
   elif i>=len(b):out.append({'path':q,'removed':a[i]})
   else:out.extend(delta(a[i],b[i],q))
  return out
 return [] if a==b else [{'path':path,'before':a,'after':b}]
def pins(root):return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}
inputs={g:pins(root) for g,root in ROOTS.items()};cases=('explicit_17','explicit_none','gemma_default','injected_missing','no_mechanism','plain_missing');report={}
for g,root in ROOTS.items():
 lane=json.loads((root/'result.json').read_bytes());assert lane['passed'] and lane['returncode']==0 and lane['before']==lane['after'] and lane['artifacts_before']==lane['artifacts_after']
 assert sorted(p.name for p in (root/'capture').iterdir() if p.is_dir())==sorted(cases)
for case in cases:
 out=HERE/case;out.mkdir(exist_ok=False);dirs=[r/'capture'/case for r in ROOTS.values()];documents={};counts={}
 for name in sorted(set(p.name for d in dirs for p in d.iterdir())):
  ps=[d/name for d in dirs];assert all(p.is_file() for p in ps)
  if name.endswith('.json'):
   vals=[json.loads(p.read_bytes()) for p in ps];documents[name]=vals;changes=delta(*vals);write(out/(name+'.delta.json'),changes);counts[name]=len(changes)
 for key in ('input.json','prepared-document.json','context-defaults-before.json','native-facts.json'):
  assert documents[key][0]==documents[key][1],(case,key)
 if case!='gemma_default':assert (dirs[0]/'model.py').read_bytes()==(dirs[1]/'model.py').read_bytes()
 results=documents['result.json'];assert results[0]['native_softcap']==results[1]['native_softcap'] and results[0]['caps']==results[1]['caps'] and results[0]['layers']==results[1]['layers'] and results[0]['consumed_softcap_events']==results[1]['consumed_softcap_events']
 receipts=[]
 for g,contexts in zip(ROOTS,documents['render-contexts.json']):
  events=[e for group in contexts.values() for e in group];rr=[(e,r) for e in events for r in e['receipts']];soft=[(e,r) for e,r in rr if r['fact_key']=='logit_softcap'];bad=[{'view':e['view'],'receipt':r,'event_node_ids':e['node_ids']} for e,r in rr if r['surface']=='opgraph' and not set(r['node_ids'])<=set(e['node_ids'])]
  receipts.append({'generation':g,'total_receipts':len(rr),'softcap_receipts':[r for e,r in soft],'softcap_graph_fact_events':sum('decoder.attention.logit_softcap' in e['facts_projected'] for e in events),'invalid_opgraph_receipt_nodes':bad})
 assert not receipts[1]['invalid_opgraph_receipt_nodes'],case
 pages=[gzip.decompress((d/'page.html.gz').read_bytes()).decode() for d in dirs]
 (out/'page.raw.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='actual31',tofile='actual32')).encode(),mtime=0))
 html={'sha256':[sha(p.encode()) for p in pages],'bytes':[len(p.encode()) for p in pages],'operand_reason_counts':[p.count('softcap_operand_unavailable') for p in pages],'softcap_node_text_counts':[p.count('attn_softcap') for p in pages]}
 if case=='explicit_17':
  assert [len(x['softcap_receipts']) for x in receipts]==[1,0] and [x['softcap_graph_fact_events'] for x in receipts]==[1,0]
  assert results[1]['native_softcap']['has_proof'] and results[1]['native_softcap']['value']==17.0
  assert html['softcap_node_text_counts']==[0,0]
 elif case=='gemma_default':
  assert [len(x['softcap_receipts']) for x in receipts]==[2,2] and [x['softcap_graph_fact_events'] for x in receipts]==[2,2]
  assert all(r['node_ids']==['attn_softcap'] for r in receipts[1]['softcap_receipts'])
 else:assert all(not x['softcap_receipts'] and x['softcap_graph_fact_events']==0 for x in receipts)
 if case in ('plain_missing','injected_missing'):
  f=results[1]['native_softcap'];assert f['value'] is None and f['status']=='unknown' and not f['has_proof'] and f['unknown_reason']=={'reason_class':'investigation_missing','concrete_reason':'softcap_operand_unavailable','investigation':None}
  assert html['operand_reason_counts']==[1,1] and 'unresolved' in pages[1]
 report[case]={'json_delta_counts':counts,'native_softcap':results[1]['native_softcap'],'layers':results[1]['layers'],'caps_and_consumption_unchanged':True,'receipts':receipts,'html':html}
assert inputs=={g:pins(root) for g,root in ROOTS.items()}
write(HERE/'input-pins.json',inputs)
write(HERE/'verdict.json',{'status':'PASS_SAVED_SIX_CASE_SCOPE','cases':report,'authority':'Actual saved native facts, caps, layers and consumption unchanged in all six cases. Explicit17 undrawn op receipt and graph fact claim removed; Gemma two actual softcap receipts preserved. Both missing cases display one real HTML reason and remain unqualified class1 None. No model/test/product import executed; no output normalization, blessing or occurrence proof inferred.','input_pins_sha256':sha((HERE/'input-pins.json').read_bytes())})
write(HERE/'manifest.json',{'files':[{ 'path':p.relative_to(HERE).as_posix(),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file()]})
print(json.dumps({'verdict_sha256':sha((HERE/'verdict.json').read_bytes()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes()),'cases':{k:{'native_fact_deltas':v['json_delta_counts']['native-facts.json'],'softcap_receipts':[len(r['softcap_receipts']) for r in v['receipts']],'html_reason_counts':v['html']['operand_reason_counts']} for k,v in report.items()}},indent=2))
