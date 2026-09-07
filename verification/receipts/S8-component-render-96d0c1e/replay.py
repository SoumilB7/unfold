"""Actual renderer-only replay; exact captured inputs and bounded SVG delta."""
from pathlib import Path
import argparse,json,sys,hashlib,shutil,time,xml.etree.ElementTree as E
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-pipeline-control',action='store_true');a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
sys.path.insert(0,str(a.checkout))
from model_unfolder.ir import ModelIR,EvidenceWarning
from model_unfolder.diagram import Diagram
from model_unfolder.preview import svg_views,_visual_hash
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(a.checkout)):sha(p) for folder in ('model_unfolder','physics') for p in sorted((a.checkout/folder).rglob('*')) if p.is_file() and p.suffix in {'.py','.yaml','.yml'}}
assert Path(sys.modules[Diagram.__module__].__file__).resolve().is_relative_to(a.checkout.resolve())
source_before=manifest();pins={p.name:sha(p) for p in a.source.iterdir() if p.is_file()}
capture=json.loads((a.source/'render-input.json').read_text());raw=json.loads((a.source/'ir.json').read_text());assert not raw['layers'] and not raw['cross_layer_edges']
ir=ModelIR(**raw);ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==json.loads((a.source/'ir.json').read_text())
diagram=Diagram(ir);diagram._mount_id=capture['mount_id'];assert diagram.to_ir()==capture['ir'];assert json.loads(json.dumps(diagram.param_count()))==capture['parameters'];started=time.monotonic();html=diagram.to_html();(a.output/'page.html').write_text(html)
old=(a.source/'page.html').read_text();before=list(svg_views(old));after=list(svg_views(html));assert [k for k,v in before]==[k for k,v in after];deltas=[]
for (name,x),(_,y) in zip(before,after):
 if _visual_hash(x)!=_visual_hash(y):deltas.append(name)
assert set(deltas)<=({'architecture'} if not a.full_pipeline_control else set()),deltas
if a.full_pipeline_control:assert html==old,'full supplied pipeline bytes changed'
else:
 def nodes(svg):return {n.get('data-id') for n in E.fromstring(svg).iter() if n.get('data-id')}
 old_svg=next(x for k,x in before if k=='architecture');new_svg=next(x for k,x in after if k=='architecture');assert nodes(old_svg)==nodes(new_svg);(a.output/'architecture.svg').write_text(new_svg)
assert manifest()==source_before;assert {p.name:sha(p) for p in a.source.iterdir() if p.is_file()}==pins
record={'status':'review_required','blessed':False,'source_case':str(a.source),'original_artifacts':pins,'renderer_sources_before':source_before,'renderer_sources_after':manifest(),'replay_script_sha256':sha(Path(__file__)),'input_roundtrip_exact':True,'parameter_record_exact':True,'warning_metadata_exact':True,'mount_id':capture['mount_id'],'original_html_sha256':hashlib.sha256(old.encode()).hexdigest(),'actual_html_sha256':hashlib.sha256(html.encode()).hexdigest(),'changed_svg_labels':deltas,'all_other_baked_svgs_identical':True,'full_pipeline_raw_bytes_identical':html==old if a.full_pipeline_control else None,'elapsed_seconds':round(time.monotonic()-started,3)}
(a.output/'render-phase.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ('actual_html_sha256','changed_svg_labels','elapsed_seconds')}))
