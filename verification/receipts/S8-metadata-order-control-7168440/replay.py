from pathlib import Path
import sys,json,gzip,copy,hashlib,dataclasses,time
ROOT=Path('/private/tmp/unfold-s8-final-verification-7168440');sys.path.insert(0,str(ROOT))
from model_unfolder.ir import ModelIR,EvidenceWarning
from model_unfolder.diagram import Diagram
from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
from model_unfolder.lint import lint_labels
from model_unfolder.sable import _projection_audit_findings
base=Path('/private/tmp/unfold-s8-typed-phase-7168440/sdxl-ordinary');source=Path('/private/tmp/unfold-s8-matrix-replay-7168440');out=Path('/private/tmp/unfold-s8-metadata-order-7168440');out.mkdir(exist_ok=False)
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):
 if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
 if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
 return v
def write(n,v):(out/n).write_text(json.dumps(canonical(v),indent=2,sort_keys=True)+'\n')
def manifest():return {str(p.relative_to(ROOT)):sha(p) for d in ('model_unfolder','physics') for p in sorted((ROOT/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
start=time.monotonic();before=manifest();write('source-before.json',before)
pins={str(p):sha(p) for p in (Path(__file__),base/'ir.json',base/'render-input.json',base/'page.html',base/'render-events.json.gz',source/'parsed-ir.json.gz',source/'actual-parser-vs-reviewed-render-deltas.json')};write('input-pins.json',pins)
raw=read(base/'ir.json');parsed=read(source/'parsed-ir.json.gz');capture=read(base/'render-input.json');key='root.denoiser.declared_constructor_defaults'
old=raw['extras']['render']['loop_blocks'][6]['detail']['fact_display_lines'][key];new=parsed['extras']['render']['loop_blocks'][6]['detail']['fact_display_lines'][key]
assert old!=new and len(old)==len(set(old))==len(new)==len(set(new))==3 and set(old)==set(new)
assert raw['extras']['render']['loop_blocks'][6]['id']==parsed['extras']['render']['loop_blocks'][6]['id']=='denoiser'
assert raw['extras']['render']['loop_blocks'][6]['facts']==parsed['extras']['render']['loop_blocks'][6]['facts']
raw['extras']['render']['loop_blocks'][6]['detail']['fact_display_lines'][key]=copy.deepcopy(new);assert raw==parsed,'source/reviewed IR differs beyond exact metadata list permutation'
ir=ModelIR(**copy.deepcopy(raw));ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==parsed
d=Diagram(ir);d._mount_id=capture['mount_id'];assert canonical(d.param_count())==capture['parameters'];page=d.to_html();(out/'page.html').write_text(page);write('ir.json',ir.to_dict());write('render-input.json',{'ir':d.to_ir(),'parameters':canonical(d.param_count()),'warnings':capture['warnings'],'mount_id':capture['mount_id']})
events=(json.dumps(canonical(d.render_events()),sort_keys=True,separators=(',',':'))+'\n').encode();(out/'render-events.json.gz').write_bytes(gzip.compress(events,mtime=0))
assert page.encode()==(base/'page.html').read_bytes();assert events==gzip.decompress((base/'render-events.json.gz').read_bytes())
checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)};checks['validate_block_tree']=canonical(validate_block_tree(ir));checks['label_lint']=lint_labels(d.to_ir());checks['projection_audit']=_projection_audit_findings(d.to_ir(),d.render_events());assert all(not v for v in checks.values())
after=manifest();write('source-after.json',after);assert before==after;assert pins=={p:sha(Path(p)) for p in pins}
write('result.json',{'status':'PASS','production_commit':'7168440c81a3e79b1e220437e2f9fa77e25bc296','source_order':new,'saved_fact_json_order':old,'both_lists_unique_same_exact_strings':True,'actual_card_facts_order_unchanged':True,'actual_parser_ir_exact_after_only_metadata_permutation':True,'actual_html_byte_identical':True,'actual_render_event_bytes_identical':True,'actual_html_sha256':sha(out/'page.html'),'event_uncompressed_sha256':hashlib.sha256(events).hexdigest(),'parameters_and_warnings_preserved':True,'checks':checks,'no_model_or_pytest':True,'elapsed_seconds':round(time.monotonic()-start,3),'blessed':False});print(json.dumps(read(out/'result.json')))
