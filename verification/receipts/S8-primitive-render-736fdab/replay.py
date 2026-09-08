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
 from model_unfolder.ir import ModelIR,EvidenceWarning
 from model_unfolder.diagram import Diagram
 from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
 from model_unfolder.lint import lint_labels,_walk_blocks
 from model_unfolder.sable import _projection_audit_findings
 import model_unfolder.diagram as dm
 assert Path(dm.__file__).resolve().is_relative_to(root)
 raw=read(base/'ir.json');capture=read(base/'render-input.json');ir=ModelIR(**copy.deepcopy(raw));ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==read(base/'ir.json')
 d=Diagram(ir);d._mount_id=capture['mount_id'];assert canonical(d.param_count())==capture['parameters'];assert d.to_ir()==capture['ir']
 page=d.to_html();(out/'page.html').write_text(page);events=canonical(d.render_events());eventbytes=(json.dumps(events,sort_keys=True,separators=(',',':'))+'\n').encode();(out/'render-events.json.gz').write_bytes(gzip.compress(eventbytes,mtime=0));write(out/'ir.json',ir.to_dict());write(out/'render-input.json',{'ir':d.to_ir(),'parameters':canonical(d.param_count()),'warnings':capture['warnings'],'mount_id':capture['mount_id']})
 old=read(base/'render-events.json.gz');assert not any(e['view']=='constructed_primitive_label' for e in old)
 added=[e for e in events if e['view']=='constructed_primitive_label'];retained=[e for e in events if e['view']!='constructed_primitive_label'];write(out/'event-deltas.json',{'prior_count':len(old),'new_count':len(events),'added':added,'prior_events_exact_order':retained==old})
 assert retained==old,'Existing event values/order changed'
 blocks=list(_walk_blocks(d.to_ir()))
 for e in added:
  assert e['facts_projected']==['root.denoiser.runtime_primitives'] and len(e['node_ids'])==1 and e['block_path'][-1]==e['node_ids'][0]
  assert any(b.get('id')==e['node_ids'][0] and b.get('source_component')=='root' and 'root.denoiser.runtime_primitives' in b.get('source_fact_keys',()) for b in blocks)
 checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)};checks.update(validate_block_tree=canonical(validate_block_tree(ir)),label_lint=lint_labels(d.to_ir()),projection_audit=_projection_audit_findings(d.to_ir(),d.render_events()));write(out/'checks.json',checks)
 assert (out/'page.html').read_bytes()==(base/'page.html').read_bytes(),'Actual raw HTML differs'
 assert ir.to_dict()==read(base/'ir.json') and d.to_ir()==capture['ir'];assert canonical(d.param_count())==capture['parameters'];assert all(not v for v in checks.values()),checks
 after=manifest();write(out/'source-after.json',after);assert before==after;assert pins=={p:sha(p) for p in pins}
 result={'status':'PASS','case':base.name,'source_case':str(base),'production_commit':'736fdab','html_sha256':sha(out/'page.html'),'prior_html_sha256':sha(base/'page.html'),'raw_html_equal':True,'exact_ir_warning_parameter_input':True,'old_event_sequence_unchanged':True,'added_primitive_events':len(added),'checks':checks,'elapsed_seconds':round(time.monotonic()-start,3),'no_model_or_pytest':True,'blessed':False};write(out/'result.json',result);print(json.dumps(result),flush=True)
except BaseException as error:
 write(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
