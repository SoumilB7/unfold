"""Exact captured-input render only. No model or evidence-reader execution."""
from pathlib import Path
import argparse,json,hashlib,sys,dataclasses,gzip,time,subprocess
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
sys.path.insert(0,str(a.checkout))
from model_unfolder.ir import ModelIR,EvidenceWarning
from model_unfolder.diagram import Diagram
from model_unfolder.block_schema import validate_no_dotted_arrows,validate_no_dotted_boundaries
assert Path(sys.modules[Diagram.__module__].__file__).resolve().is_relative_to(a.checkout.resolve())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(a.checkout)):sha(p) for folder in ('model_unfolder','physics') for p in sorted((a.checkout/folder).rglob('*')) if p.is_file() and p.suffix in {'.py','.yaml','.yml'}}
def canonical(value):
 if dataclasses.is_dataclass(value):return canonical(dataclasses.asdict(value))
 if isinstance(value,dict):return {key:canonical(v) for key,v in value.items()}
 if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
 if isinstance(value,(set,frozenset)):return sorted(canonical(v) for v in value)
 return value
before=manifest();pins={name:sha(a.source/name) for name in ('ir.json','render-input.json','facts.json','qualified-facts.json','result.json')};capture=json.loads((a.source/'render-input.json').read_text());raw=json.loads((a.source/'ir.json').read_text());assert not raw['layers'] and not raw['cross_layer_edges'];ir=ModelIR(**raw);ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==json.loads((a.source/'ir.json').read_text());d=Diagram(ir);d._mount_id=capture['mount_id'];assert d.to_ir()==capture['ir'];assert canonical(d.param_count())==capture['parameters'];start=time.monotonic();html=d.to_html();events=canonical(d.render_events());event_bytes=(json.dumps(events,sort_keys=True,separators=(',',':'))+'\n').encode();(a.output/'render-events.json.gz').write_bytes(gzip.compress(event_bytes,mtime=0));(a.output/'page.html').write_text(html);after=manifest();assert before==after;assert pins=={name:sha(a.source/name) for name in pins}
result={'source_case':str(a.source),'production_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.checkout,text=True).strip(),'source_before':before,'source_after':after,'source_artifact_pins':pins,'exact_ir_warning_parameter_roundtrip':True,'actual_html_sha256':hashlib.sha256(html.encode()).hexdigest(),'render_event_sha256':hashlib.sha256(event_bytes).hexdigest(),'render_event_count':len(events),'dotted_arrows':validate_no_dotted_arrows(html),'dotted_boundaries':validate_no_dotted_boundaries(html),'worker_sha256':sha(Path(__file__)),'elapsed_seconds':round(time.monotonic()-start,3),'blessed':False}
(a.output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ('actual_html_sha256','render_event_sha256','render_event_count','elapsed_seconds')}))
