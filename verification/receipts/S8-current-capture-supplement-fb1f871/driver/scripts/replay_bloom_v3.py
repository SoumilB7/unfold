"""Actual renderer-only replay of exact captured inputs; preserve old bytes."""
from pathlib import Path
import argparse,copy,gzip,json,sys,time,hashlib,dataclasses,traceback
p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.checkout).resolve();base=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(root))
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):
 if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
 if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
 return v
def write(p,v):p.write_text(json.dumps(canonical(v),indent=2,sort_keys=True)+'\n')
def manifest():return {str(p.relative_to(root)):sha(p) for folder in ('model_unfolder','physics','scripts') for p in sorted((root/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
start=time.monotonic();before=manifest();write(out/'source-before.json',before);assert before==read(Path(a.source_manifest));pins={str(p):sha(p) for p in (Path(__file__),base/'ir.json',base/'render-input.json',base/'page.html',base/'render-events.json.gz')};write(out/'input-pins.json',pins)
try:
 from model_unfolder.ir import ModelIR,EvidenceWarning,LayerSpec,AttentionSpec,FFNSpec,CrossLayerEdge
 from model_unfolder.diagram import Diagram
 from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
 from model_unfolder.lint import lint_labels
 from model_unfolder.sable import _projection_audit_findings
 import model_unfolder.diagram as dm
 assert Path(dm.__file__).resolve().is_relative_to(root)
 raw=read(base/'ir.json');capture=read(base/'render-input.json');typed=copy.deepcopy(raw)
 for layer in typed['layers']:
  layer['attention']=AttentionSpec(**layer['attention']);layer['ffn']=FFNSpec(**layer['ffn'])
  if layer.get('cross_attention') is not None:layer['cross_attention']=AttentionSpec(**layer['cross_attention'])
 typed['layers']=[LayerSpec(**layer) for layer in typed['layers']];typed['cross_layer_edges']=[CrossLayerEdge(**edge) for edge in typed['cross_layer_edges']]
 ir=ModelIR(**typed);ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==read(base/'ir.json')
 d=Diagram(ir);d._mount_id=capture['mount_id'];assert canonical(d.param_count())==capture['parameters'];assert d.to_ir()==capture['ir']
 page=d.to_html();(out/'page.html').write_text(page);events=canonical(d.render_events());eventbytes=(json.dumps(events,sort_keys=True,separators=(',',':'))+'\n').encode();(out/'render-events.json.gz').write_bytes(gzip.compress(eventbytes,mtime=0));write(out/'ir.json',ir.to_dict());write(out/'render-input.json',{'ir':d.to_ir(),'parameters':canonical(d.param_count()),'warnings':capture['warnings'],'mount_id':capture['mount_id']})
 old=read(base/'render-events.json.gz');left=copy.deepcopy(old);right=copy.deepcopy(events);nonce_paths=[];oldtokens=set();newtokens=set()
 for i,(aevent,bevent) in enumerate(zip(left,right)):
  for j,(arec,brec) in enumerate(zip(aevent['receipts'],bevent['receipts'])):
   if 'context_token' in arec:
    oldtokens.add(arec['context_token']);newtokens.add(brec['context_token']);nonce_paths.append(f'/{i}/receipts/{j}/context_token');arec['context_token']='<fresh-context-nonce>';brec['context_token']='<fresh-context-nonce>'
 assert len(nonce_paths)==6 and len(oldtokens)==len(newtokens)==1 and oldtokens!=newtokens
 assert left==right,'Semantic event values/order changed beyond the six fresh context nonces';added=[]
 write(out/'fresh-context-nonce-comparison.json',{'paths':nonce_paths,'old_tokens':sorted(oldtokens),'fresh_tokens':sorted(newtokens),'all_other_event_fields_and_order_exact':True,'raw_events_equal':False,'actual_event_bytes_unchanged':True})
 write(out/'event-deltas.json',{'prior_count':len(old),'new_count':len(events),'added':[],'semantic_events_exact_order_except_six_fresh_context_nonces':True})
 checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)};checks.update(validate_block_tree=canonical(validate_block_tree(ir)),label_lint=lint_labels(d.to_ir()),projection_audit=_projection_audit_findings(d.to_ir(),d.render_events()));write(out/'checks.json',checks)
 import re
 oldpage=(base/'page.html').read_text();svgre=re.compile(r'<svg\b[\s\S]*?</svg>');oldsvgs=svgre.findall(oldpage);newsvgs=svgre.findall(page);assert len(oldsvgs)==len(newsvgs)
 assert svgre.sub('<SVG-REPLAY-SLOT>',oldpage)==svgre.sub('<SVG-REPLAY-SLOT>',page),'Non-SVG page bytes changed'
 deltas=[]
 for i,(left,right) in enumerate(zip(oldsvgs,newsvgs)):
  if left==right:continue
  (out/f'old-{i:03d}.svg').write_text(left);(out/f'current-{i:03d}.svg').write_text(right)
  deltas.append({'index':i,'old_sha256':hashlib.sha256(left.encode()).hexdigest(),'current_sha256':hashlib.sha256(right.encode()).hexdigest(),'cause':'Reviewed empty Parallel lane geometry; exact IR and canonical events unchanged','approval':'pending actual review'})
 write(out/'svg-deltas.json',deltas)

 assert ir.to_dict()==read(base/'ir.json') and d.to_ir()==capture['ir'];assert canonical(d.param_count())==capture['parameters'];assert all(not v for v in checks.values()),checks
 after=manifest();write(out/'source-after.json',after);assert before==after;assert pins=={p:sha(p) for p in pins}
 result={'status':'PASS','case':base.name,'source_case':str(base),'production_commit':'fb1f871','html_sha256':sha(out/'page.html'),'prior_html_sha256':sha(base/'page.html'),'raw_html_equal':page==oldpage,'changed_svgs':len(deltas),'exact_ir_warning_parameter_input':True,'semantic_event_sequence_unchanged_except_six_fresh_context_nonces':True,'raw_events_equal':False,'added_primitive_events':len(added),'checks':checks,'elapsed_seconds':round(time.monotonic()-start,3),'no_model_or_pytest':True,'blessed':False};write(out/'result.json',result);print(json.dumps(result),flush=True)
except BaseException as error:
 write(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise

finally:
 after=manifest();write(out/'source-after-finally.json',after);actual={p:sha(p) for p in pins};write(out/'pin-check-finally.json',{'source_equal':after==before,'input_equal':pins==actual});assert after==before and pins==actual
