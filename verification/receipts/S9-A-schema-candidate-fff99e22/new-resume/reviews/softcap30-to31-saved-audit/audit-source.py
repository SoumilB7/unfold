from pathlib import Path
import ast,collections,difflib,gzip,hashlib,json,re
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';O=B/'new-resume/reviews/softcap30-to31-saved-audit';O.mkdir(exist_ok=False)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def delta(a,b,path=''):
 if type(a)is not type(b):return [{'path':path,'kind':'replace_type','before':a,'after':b}]
 if isinstance(a,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   q=path+'/'+k.replace('~','~0').replace('/','~1')
   if k not in a:out.append({'path':q,'kind':'add','after':b[k]})
   elif k not in b:out.append({'path':q,'kind':'remove','before':a[k]})
   else:out.extend(delta(a[k],b[k],q))
  return out
 if isinstance(a,list):
  out=[]
  for i in range(max(len(a),len(b))):
   q=path+'/'+str(i)
   if i>=len(a):out.append({'path':q,'kind':'add','after':b[i]})
   elif i>=len(b):out.append({'path':q,'kind':'remove','before':a[i]})
   else:out.extend(delta(a[i],b[i],q))
  return out
 return [] if a==b else [{'path':path,'kind':'replace','before':a,'after':b}]
roots={g:B/g/'softcap-diagnostic' for g in ('thirtieth','thirtyfirst')};archive={}
for g,r in roots.items():
 m=r/'archive-map.json';rows=json.loads(m.read_bytes())
 if g=='thirtyfirst':assert sha(m.read_bytes())=='a98fcfdc879331d9405da388b941fdf3b4f311a9443c9c01edd5263186339ba2'
 for row in rows:
  raw=(r/row['path']).read_bytes();assert sha(raw)==row['sha256'] and len(raw)==row['bytes']
 result=json.loads((r/'result.json').read_bytes());assert result['passed'] and result['returncode']==0 and result['before']==result['after'] and result['artifacts_before']==result['artifacts_after']
 archive[g]={'map_sha256':sha(m.read_bytes()),'files_verified':len(rows),'lane_result':result}
# Actual capture helper source changed only fixed output address; helper pipeline source unchanged.
scripts=[roots[g]/'softcap-diagnostic.py' for g in roots];a,b=[p.read_text() for p in scripts]
assert a.replace('thirtieth','thirtyfirst')==b
fixture_before=R/'.claude/worktrees/verify-s9-a-thirtieth/tests/test_attention_softcap.py';fixture_after=R/'.claude/worktrees/verify-s9-a-thirtyfirst/tests/test_attention_softcap.py'
fn=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
assert fn(fixture_before.read_text())['_pipeline']==fn(fixture_after.read_text())['_pipeline']
summary={}
for case in ('injected_missing','plain_missing','explicit_17','explicit_none','no_mechanism','gemma_default'):
 dirs=[roots[g]/'capture'/case for g in roots];out=O/case;out.mkdir()
 snapshots={};files={}
 for name in sorted(set(p.name for d in dirs for p in d.iterdir())):
  paths=[d/name for d in dirs];raws=[p.read_bytes() if p.is_file() else None for p in paths]
  row={'before_sha256':sha(raws[0]) if raws[0] is not None else None,'after_sha256':sha(raws[1]) if raws[1] is not None else None,'byte_equal':raws[0]==raws[1]}
  if name.endswith('.json') and all(raw is not None for raw in raws):
   vals=[json.loads(raw) for raw in raws];snapshots[name]=vals;changes=delta(*vals);write(out/(name+'.delta.json'),changes)
   row['json_delta_count']=len(changes);row['delta_path']=case+'/'+name+'.delta.json'
  files[name]=row
 results=snapshots['result.json'];events=snapshots['events.json'];renders=snapshots['render-contexts.json'];facts=snapshots['native-facts.json'];irs=snapshots['ir-after.json']
 assert snapshots['input.json'][0]==snapshots['input.json'][1]
 assert snapshots['prepared-document.json'][0]==snapshots['prepared-document.json'][1]
 if case!='gemma_default':assert (dirs[0]/'model.py').read_bytes()==(dirs[1]/'model.py').read_bytes()
 assert results[0]['caps']==results[1]['caps'] and results[0]['layers']==results[1]['layers']
 other_facts=[{k:v for k,v in f.items() if k!='decoder.attention.logit_softcap'} for f in facts];assert other_facts[0]==other_facts[1]
 eventrows=[];receiptrows=[]
 for gen,ev,render in zip(roots,events,renders):
  matching=[e for e in ev if e['fact_key']=='logit_softcap'];operand=[e for e in ev if e['config_path'] in ('attention_cap','attn_logit_softcapping')]
  eventrows.append({'generation':gen,'fact_addressed_intents':dict(collections.Counter(e['intent'] for e in matching)), 'operand_events':operand})
  rr=[]
  for items in render.values():
   for event in items:
    for receipt in event['receipts']:
     if receipt['fact_key']=='logit_softcap':rr.append({'event_node_ids':event['node_ids'],'event_drawn_ops':event['drawn_ops'],'receipt':receipt,'receipt_nodes_subset_of_event_nodes':set(receipt['node_ids'])<=set(event['node_ids'])})
  receiptrows.append({'generation':gen,'softcap_receipts':rr})
 pages=[gzip.decompress((d/'page.html.gz').read_bytes()).decode() for d in dirs]
 raw_diff=''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='thirtieth/page.html',tofile='thirtyfirst/page.html'))
 (out/'page.raw.diff.gz').write_bytes(gzip.compress(raw_diff.encode(),mtime=0))
 mounts=[re.search(r'uf-[0-9a-f]{6,32}\b',p).group() for p in pages]
 replaced=pages[0].replace(mounts[0],mounts[1]); residual=''.join(difflib.unified_diff(replaced.splitlines(True),pages[1].splitlines(True),fromfile='before-with-exact-paired-mount-substitution',tofile='actual-after'))
 (out/'page.mount-accounted-residual.diff.gz').write_bytes(gzip.compress(residual.encode(),mtime=0))
 html={'raw_byte_equal':pages[0]==pages[1],'bytes':[len(p.encode()) for p in pages],'raw_sha256':[sha(p.encode()) for p in pages],'paired_mounts':mounts,'exact_mount_occurrences':[p.count(m) for p,m in zip(pages,mounts)],'finite_mount_replacement_explains_all_bytes':replaced==pages[1],'reason_occurrences':[p.count('softcap_operand_unavailable') for p in pages],'attn_softcap_text_occurrences':[p.count('attn_softcap') for p in pages], 'authority':'Raw diff retained. Paired mount substitution is a finite cause audit only, not acceptance normalization.'}
 if case in ('injected_missing','plain_missing'):
  assert results[0]['native_softcap'] is None
  n=results[1]['native_softcap'];assert n['value']is None and n['status']=='unknown' and not n['has_proof'] and n['unknown_reason']=={'reason_class':'investigation_missing','concrete_reason':'softcap_operand_unavailable','investigation':None}
  assert html['reason_occurrences']==[0,1] and all(not r['softcap_receipts'] for r in receiptrows)
  cause='Previously omitted question is now one null/unknown native applied_function fact plus unresolved presentation markers and one displayed operand-unavailable reason; no stronger proof, consumed softcap event or softcap op receipt.'
 else:
  assert facts[0]==facts[1]
  cause='No native fact value/status/proof or layer/cap changes; raw source addresses and local context/mount identities remain separately enumerated.'
 summary[case]={'files':files,'native_softcap_before_after':[x['native_softcap'] for x in results],'layers':results[1]['layers'],'caps_unchanged':True,'other_native_facts_exactly_equal':True,'events':eventrows,'receipts':receiptrows,'html':html,'cause':cause}
# Pin the exact source emitter and finite proof scope, without editing source.
refs={}
for name in ('model_unfolder/renderers/html/block_views/attention.py','model_unfolder/evidence/receipts.py','model_unfolder/renderers/html/graph_engine.py','model_unfolder/renderers/html/render_context.py','model_unfolder/opgraph.py'):
 p=R/'.claude/worktrees/verify-s9-a-thirtyfirst'/name;refs[name]=sha(p.read_bytes())
issue={'case':'explicit_17','exists_in_both30_and31':True,'actual_receipt_type':'ProjectionReceipt(surface=opgraph, projection_kind=op, structural_target=attn_softcap, node_ids=[attn_softcap])','actual_render_event':{'node_ids':['block','region_out'],'drawn_ops':['opaque','port']},'actual_HTML_attn_softcap_occurrences':0,
 'emitter':'model_unfolder/renderers/html/block_views/attention.py:69-78 selects nonnull fact without graph node membership;188-193 hardcodes op receipt node attn_softcap. receipts_from_projects preserves descriptor; render_graph carries receipt into actual RenderEvent.',
 'classification':'S9-A projection qualification/accounting defect; rejecting receipt for undrawn op needs no new mechanism proof. Restoring a visible supported partial softcap graph is separate C/design mechanism work.',
 'distinction':'Explicit17 native applied_function proof and bound/consumed config events remain established independently; they do not prove the op was drawn.', 'source_sha256':refs}
write(O/'synthetic-explicit17-receipt-finding.json',issue)
report={'status':'SAVED_DATA_AUDIT_COMPLETE_WITH_EXISTING_PROJECTION_RECEIPT_FINDING','archive_validation':archive,'protocol_check':'Capture scripts equal except generation output directory; _pipeline helper AST unchanged; saved input/prepared document/model source exactly equal per case. Test fixture byte hash changes are recorded by original captures.', 'cases':summary,'receipt_finding':issue,'scope':'No runtime models/tests/product imports or production edits. Missing-case no-false-receipt finding is bounded to those cases. Raw changes are retained, never approved or normalized away.'}
write(O/'report.json',report)
write(O/'manifest.json',{'files':[{'path':str(p.relative_to(O)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(O.rglob('*')) if p.is_file()]})
print(json.dumps({'report_sha256':sha((O/'report.json').read_bytes()),'manifest_sha256':sha((O/'manifest.json').read_bytes()),'cases':{k:{'nativefact_deltas':v['files']['native-facts.json']['json_delta_count'],'ir_before_deltas':v['files']['ir-before.json']['json_delta_count'],'ir_after_deltas':v['files']['ir-after.json']['json_delta_count'],'mount_only_HTML':v['html']['finite_mount_replacement_explains_all_bytes'],'reason_counts':v['html']['reason_occurrences']} for k,v in summary.items()}},indent=2))
