from pathlib import Path
import sys,json,copy,hashlib,importlib.util,dataclasses,gzip
from types import SimpleNamespace
ROOT=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');sys.path.insert(0,str(ROOT))
from model_unfolder.adapters.diffusor.unet_projection import project_unet
from model_unfolder.ir import ModelIR,EvidenceWarning
from model_unfolder.diagram import Diagram
from model_unfolder.preview import svg_views
from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
spec=importlib.util.spec_from_file_location('model_unfolder.adapters.diffusor.old_projector','/private/tmp/unfold-s8-solid-review-f7baef5/model_unfolder/adapters/diffusor/unet_projection.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(ROOT)):sha(p) for folder in ('model_unfolder','physics') for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
def walk(v):
 if isinstance(v,dict):
  if 'id' in v:yield v
  for x in v.values():yield from walk(x)
 elif isinstance(v,list):
  for x in v:yield from walk(x)
def without_refs(v):
 if isinstance(v,dict):return {k:[without_refs(c) for c in x if c.get('role')!='context_input'] if k=='children' and v.get('view')=='runtime_context_connection' else without_refs(x) for k,x in v.items()}
 if isinstance(v,list):return [without_refs(x) for x in v]
 return v
def canonical(v):
 if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
 if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [canonical(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
 return v
for case,source in [('sdxl-ordinary','sdxl'),('sd14-ordinary','sd-v1-4')]:
 out=Path('/private/tmp/unfold-s8-context-click-audit')/case;out.mkdir(exist_ok=True);src=Path('/private/tmp/unfold-s8-render-phase-c38-v2')/source/'ordinary';before=manifest();pins={n:sha(src/n) for n in ('ir.json','render-input.json','facts.json','qualified-facts.json','result.json')};raw=json.loads((src/'ir.json').read_text());capture=json.loads((src/'render-input.json').read_text());facts={k:SimpleNamespace(value=v['value']) for k,v in json.loads((src/'facts.json').read_text()).items()};args=dict(facts=facts,handoffs={},name=raw['name'],architecture=raw['architecture']);prior=old.project_unet(**args).to_dict();current=project_unet(**args).to_dict();assert without_refs(current)==prior
 refs={b['id']:next(c for c in b['children'] if c.get('role')=='context_input') for b in walk(current) if b.get('view')=='runtime_context_connection'};rootrefs={b['id']:b for b in walk(raw) if b.get('role')=='context_input'};added=[]
 for block in walk(raw):
  if block.get('view')=='runtime_context_connection':
   ref=refs[block['id']];assert ref==rootrefs[block['detail']['source']];assert not any(c['id']==ref['id'] for c in block.get('children',[]));block.setdefault('children',[]).append(copy.deepcopy(ref));added.append(block['id'])
 assert len(added)==len(refs);assert without_refs(raw)==json.loads((src/'ir.json').read_text());ir=ModelIR(**raw);ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==raw;d=Diagram(ir);d._mount_id=capture['mount_id'];assert without_refs(d.to_ir())==capture['ir'];assert canonical(d.param_count())==capture['parameters'];page=d.to_html();checks={f.__name__:f(page) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)};checks['validate_block_tree']=validate_block_tree(ir);assert all(not x for x in checks.values()),checks
 priorpage=Path('/private/tmp/unfold-s8-solid-f7baef5')/case/'after/page.html';assert list(svg_views(page))==list(svg_views(priorpage.read_text()))
 # The added card at panel9 is the only actual HTML change.
 import re
 original=priorpage.read_text();at=page.rfind('data-card-id="unet_context_0"');start=page.rfind('<div',0,at);depth=0;end=None
 for token in re.finditer(r'<div\b[^>]*>|</div>',page[start:]):
  depth += -1 if token.group()=='</div>' else 1
  if depth==0:end=start+token.end();break
 assert end is not None
 fragment=page[start:end];assert 'Root input: encoder_hidden_states' in fragment
 assert page[:start]+page[end:]==original
 events=(json.dumps(canonical(d.render_events()),sort_keys=True,separators=(',',':'))+'\n').encode();old_events=gzip.decompress((Path('/private/tmp/unfold-s8-solid-f7baef5')/case/'after/render-events.json.gz').read_bytes());(out/'render-events.json.gz').write_bytes(gzip.compress(events,mtime=0));(out/'page.html').write_text(page);(out/'ir.json').write_text(json.dumps(ir.to_dict(),indent=2,sort_keys=True)+'\n');capture['ir']=d.to_ir();(out/'render-input.json').write_text(json.dumps(capture,indent=2,sort_keys=True)+'\n');assert before==manifest();assert pins=={n:sha(src/n) for n in pins};result={'case':case,'checks':checks,'added_context_reference_parent_ids':added,'canonical_facts_unchanged':True,'projection_only_reference_children_added':True,'captured_complete_ir_only_reference_children_added':True,'all_actual_svgs_byte_identical':True,'actual_html_only_one_context_card_added_at_depth9':True,'actual_html_sha256':sha(out/'page.html'),'prior_html_sha256':sha(priorpage),'render_events_identical':events==old_events,'before_event_count':len(json.loads(old_events)),'after_event_count':len(json.loads(events)),'source_before':before,'source_after':manifest(),'saved_input_pins':pins,'blessed':False};(out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ('case','checks','actual_html_sha256','render_events_identical','before_event_count','after_event_count')}),flush=True)
