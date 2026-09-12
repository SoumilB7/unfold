"""Replay exact saved SDXL construction/execution through the real matrix caller."""
import dataclasses,gzip,hashlib,importlib.util,json,sys,time
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
from physics.instance_inventory import InventoryResult
from model_unfolder.evidence.runtime_inventory import request_from_resolved_source
from model_unfolder.evidence.execution_recipe import RecipeAttemptBundle,RecipeResolution
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reconciliation import _fact_source_keys
from model_unfolder.adapters.diffusor import unet_cutover

def read(p):
 data=p.read_bytes();return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)
def write(p,v):
 data=(json.dumps(v,indent=2,sort_keys=True)+'\n').encode();p.write_bytes(gzip.compress(data,mtime=0) if p.suffix=='.gz' else data)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {str(p.relative_to(root)):sha(p) for area in ('model_unfolder','physics','scripts') for p in (root/area).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
source_before=manifest();write(out/'source-before.json',source_before)
spec=importlib.util.spec_from_file_location('matrix_replay_generator',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
slug='stable-diffusion-xl-base-1-0';target=next(t for t in g._targets() if t['slug']==slug)
original=Path('/private/tmp/unfold-s8-final-verification-96d0c1e/verification/s7')
saved_path=Path('/private/tmp/unfold-s8-render-phase-c38-v2/sdxl/ordinary/inventory.json')
saved=InventoryResult.from_dict({'status':'ok','inventory':read(saved_path)})
observations=read(original/'observations'/f'{slug}.json.gz');relations=read(original/'relations'/f'{slug}.json.gz')
package_root=Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages')
source_checks=[]
for row in saved.inventory.provenance.source_files:
 path=package_root.joinpath(*row.module.split('.')).with_suffix('.py')
 if not path.exists():path=package_root.joinpath(*row.module.split('.'),'__init__.py')
 assert sha(path)==row.sha256,(row.module,path)
 source_checks.append({'path':str(path),'sha256':row.sha256})
write(out/'saved-source-checks.json',source_checks)
state={}
def cached_build(document,bundle,selected_root,**kwargs):
 request=request_from_resolved_source(document,bundle,selected_root,**kwargs)
 digest=hashlib.sha256(json.dumps(dict(request.config),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
 assert digest==saved.inventory.provenance.config_sha256
 assert request.factory_module+'.'+request.factory_qualname==saved.inventory.provenance.requested_factory
 assert request.factory_method=='from_config' and request.capture_framework_primitives and not request.source_overrides
 actual={b.attribute:b for module in saved.inventory.modules if module.path=='' for b in module.attribute_bindings}
 for demand in request.attribute_lookups:
  assert demand.owner_path=='' and demand.attribute in actual
  if demand.storage_attributes and actual[demand.attribute].kind == "property_getter":
   available={r.attribute for r in actual[demand.attribute].storage_attribute_lookups}
   assert set(demand.storage_attributes)<=available, (demand.attribute, demand.storage_attributes, sorted(available))
 state['construction_request']=request.to_dict();return saved
original_source_inputs=g._source_inputs
def source_inputs(config,inventory):
 result=original_source_inputs(config,inventory);context,index,resolved,claims,ir=result
 state['context']=context;state['index']=index;state['ir']=ir
 base=build_program_index(context.source_bundle)
 fact=context.facts.typed_records()['root.denoiser.ffn_mechanisms']
 try:_fact_source_keys(fact,base)
 except ValueError as exc:state['original_source_join_red']=str(exc)
 else:raise AssertionError('initial index unexpectedly closes nested claim')
 state['closed_ffn_sources']=_fact_source_keys(fact,index)
 state['all_fact_source_keys']={k:_fact_source_keys(v,index) for k,v in context.facts.typed_records().items()}
 write(out/'parsed-ir.json.gz',ir.to_dict());write(out/'parsed-facts.json.gz',context.facts.to_dict());write(out/'source-join.json',{'original_red':state['original_source_join_red'],'closed':state['all_fact_source_keys']})
 return result

def cached_signature(request,resolution):
 assert resolution==RecipeResolution.from_dict(observations['resolution']),g._payload_differences(json.loads(json.dumps(dataclasses.asdict(resolution))),observations['resolution'])
 assert request.to_dict()==g._s6_request(slug,saved.inventory.provenance.config_sha256).to_dict()
 state['signature_resolution_exact']=True
 return RecipeAttemptBundle.from_dict(observations)
def forbidden(*args,**kwargs):raise AssertionError('no new runtime invocation is authorized in saved-evidence replay')
started=time.monotonic()
with patch.object(unet_cutover,'build_resolved_instance',cached_build),patch.object(g,'_source_inputs',source_inputs),patch.object(g,'_run_signature_recipe',cached_signature),patch.object(g,'inventory_in_subprocess',forbidden),patch.object(g,'observe_relations_in_subprocess',forbidden):
 artifact,obs,rels=g._one(target,write_relations=False,output=original)
assert json.loads(json.dumps(obs))==observations and json.loads(json.dumps(rels))==relations
assert source_before==manifest()
write(out/'model.json.gz',artifact);write(out/'observation.json.gz',obs);write(out/'relations.json.gz',rels)
write(out/'ir.json.gz',state['ir'].to_dict());write(out/'facts.json.gz',state['context'].facts.to_dict())
write(out/'construction-request.json',state['construction_request'])
summary={'elapsed_seconds':round(time.monotonic()-started,3),'original_source_join_red':state['original_source_join_red'],'closed_ffn_sources':state['closed_ffn_sources'],'all_fact_source_keys':state['all_fact_source_keys'],'signature_resolution_exact':True,'observations_exact':True,'relations_exact':True,'sources_unchanged':True,'saved_inventory_sha256':sha(saved_path),'retained_index_fingerprint':state['index'].fingerprint,'source_count':len(state['index'].source_nodes),'model_occurrences':len(artifact['table']['occurrences']),'blocking_findings':artifact['blocking_findings'],'no_new_model_or_execution':True}
write(out/'summary.json',summary);write(out/'source-after.json',manifest());print(json.dumps(summary,indent=2))
