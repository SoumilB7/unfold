"""Re-derive only the three construction-summary proofs from exact case evidence."""
import dataclasses,gzip,hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);case=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
from physics.instance_inventory import InventoryResult
from physics.source_override import SourceOverride
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.models import SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from model_unfolder.evidence.runtime_inventory import request_from_resolved_source
from model_unfolder.evidence.instance_population_claim import read_instance_population
from model_unfolder.evidence.instance_shape_claim import read_instance_shapes
from model_unfolder.evidence.unet_claims import read_unet_stage_relations
from model_unfolder.evidence.construction_summary import project_construction_summary,construction_summary_problems
from model_unfolder.ir import ModelIR

def read(p):
 data=p.read_bytes();return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)
def write(name,v): (out/name).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(root)):sha(p) for base in ('model_unfolder','physics') for p in (root/base).rglob('*.py')}
before=manifest();write('source-before.json',before);start=time.monotonic()
config=read(case/'input.json');inventory=InventoryResult.from_dict({'status':'ok','inventory':read(case/'inventory.json')}).inventory;oldfacts=read(case/'facts.json')
context=ParseContext.build(config);bundle=context.source_bundle;source_overrides=()
if (case/'source-control.json').exists():
 control=read(case/'source-control.json');original=Path(control['original_path']);scratch=Path(control['scratch_path']);assert sha(scratch)==control['scratch_sha256']
 source_overrides=(SourceOverride(control['module'],str(scratch),control['scratch_sha256']),)
 replacements={}
 for entry in bundle.import_roots['root']:
  base=Path(entry.path)
  if original.is_relative_to(base):
   suffix=original.relative_to(base);new=scratch
   for _ in suffix.parts:new=new.parent
   replacements[base]=new
  else:
   raise AssertionError('source control root mapping needs an explicit preserved path')
 def copied(path):
  p=Path(path);matches=[str(new/p.relative_to(old)) for old,new in replacements.items() if p.is_relative_to(old)];assert len(matches)==1;return matches[0]
 root_files=set(bundle.component_files['root'])
 bundle=dataclasses.replace(bundle,files=tuple(copied(p) if p in root_files else p for p in bundle.files),component_files={**bundle.component_files,'root':tuple(copied(p) for p in bundle.component_files['root'])},supporting_files={**bundle.supporting_files,**({'root':tuple(copied(p) for p in bundle.supporting_files['root'])} if 'root' in bundle.supporting_files else {})},import_roots={**bundle.import_roots,'root':tuple(SourceImportRoot(e.package,str(replacements[Path(e.path)])) for e in bundle.import_roots['root'])})
 document=prepare_document(config,merge=False)
else:document=prepare_document(config,merge=False)
index=build_program_index(bundle);selected=resolve_component_root(index,bundle,'root')
request=request_from_resolved_source(document,bundle,selected,index=index,source_overrides=source_overrides)
digest=hashlib.sha256(json.dumps(dict(request.config),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert digest==inventory.provenance.config_sha256
assert request.factory_module+'.'+request.factory_qualname==inventory.provenance.requested_factory
root_sha=sha(Path(selected.graph.root.symbol.source.canonical_path));result=read(case/'result.json');assert root_sha==result['static_source_sha256']==result['runtime_source_sha256']
topology=read_diffusion_root_topology(index,selected).require_value()
construction=read_unet_stage_construction(index,bundle,selected,topology).require_value()
execution=read_unet_stage_execution(construction,bundle,selected).require_value()
table=reconcile(model=str(case),inventory=inventory,observations=(),config_document=document,program_index=execution.index)
bindings=RuntimeSourceBindings(table,inventory,execution.index)
facts={f.ledger_key():f for f in (read_instance_population(bindings),read_instance_shapes(bindings),read_unet_stage_relations(execution,bindings))}
for key,fact in facts.items():assert fact.value==oldfacts[key]['value'],key
archived={row['sha256'] for row in read(case/'cited-source.json') if row['status']=='archived_exact_bytes'}
source_rows=[]
for node in execution.index.source_nodes:
 sid=node.source_id
 required=sid.component_key=="root" or sid.external
 if required:assert sid.content_fingerprint in archived,(sid.canonical_path,sid.content_fingerprint)
 source_rows.append({"path":sid.canonical_path,"sha256":sid.content_fingerprint,"component":sid.component_key,"external":sid.external,"archival_required":required})
write('indexed-sources.json',source_rows)
summary=project_construction_summary(facts);assert summary is not None
ir=ModelIR('summary-check','',0,None,None,None,[],construction_summary=summary);assert not construction_summary_problems(ir,facts)
write('summary.json',summary.to_dict());write('proofs.json',{k:{'value':v.value,'summary':dataclasses.asdict(v.claim_evidence.summary())} for k,v in facts.items()});write('request.json',request.to_dict())
assert before==manifest();write('source-after.json',manifest());write('result.json',{'elapsed_seconds':round(time.monotonic()-start,3),'case':str(case),'inventory_sha256':sha(case/'inventory.json'),'facts_sha256':sha(case/'facts.json'),'input_sha256':sha(case/'input.json'),'root_sha256':root_sha,'three_fact_values_exact':True,'indexed_source_nodes':len(execution.index.source_nodes),'root_and_external_nodes_archived':sum(row['archival_required'] for row in source_rows),'no_model_or_execution':True,'summary':summary.to_dict()});print(json.dumps(read(out/'result.json')))
