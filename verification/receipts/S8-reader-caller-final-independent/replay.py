from pathlib import Path
import sys,ast,json,hashlib,subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch
R=Path('/private/tmp/unfold-s8-compiler-final');O=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert commit=='c38b008af8153302d85bbaf2a8e3593841d87e1a'
old=json.loads((R/'verification/receipts/S8-reader-caller-audit/caller-ledger.json').read_text())
original=subprocess.check_output(['git','ls-tree','-r','--name-only','83140f1','model_unfolder/evidence'],cwd=R,text=True).splitlines();original=[p for p in original if Path(p).name.startswith('unet_') and p.endswith('.py')];assert len(original)==14==len(old['modules'])
files=[p for root in ('model_unfolder','physics','scripts')for p in (R/root).rglob('*.py')];sources={str(p.relative_to(R)):p.read_text() for p in files};trees={p:ast.parse(s) for p,s in sources.items()}
def refs(name,exclude=''):
 return [{'path':p,'line':n.lineno,'source':sources[p].splitlines()[n.lineno-1].strip()} for p,t in trees.items() if p!=exclude for n in ast.walk(t) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id==name]
def line(p,s):return next(i+1 for i,l in enumerate(sources[p].splitlines()) if s in l)
rows=[]
for row in old['modules']:
 p=row['module'];assert p in original and p in sources
 names=row['entry'].split(' / ') if 'selected_constructor.py' not in p else ['constructor_environments','selected_constructor_environment','selected_constructor_environments']
 entries=[]
 for name in names:
  calls=[r for r in refs(name,p) if r['path'].startswith('model_unfolder/')];assert calls,(p,name)
  entries.append({'name':name,'definition_line':next(n.lineno for n in trees[p].body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),'production_references':calls})
 rows.append({'module':p,'status':'retained with production caller','entries':entries})
new={
'model_unfolder/evidence/unet_runtime.py':['investigate_unet_runtime'],
'model_unfolder/evidence/unet_claims.py':['read_unet_ffn_claims','read_unet_join_claims'],
'model_unfolder/evidence/unet_cell_connections.py':['read_unet_cell_connections'],
'model_unfolder/evidence/unet_primary_ports.py':['read_unet_primary_ports'],
'model_unfolder/evidence/unet_call_binding.py':['demanded_root_lookups','extend_lookup_sources','read_root_invocations'],
'model_unfolder/evidence/unet_lookup_closure.py':['read_lookup_closure','close_parent_helpers'],
'model_unfolder/evidence/unet_wrapper_binding.py':['read_wrapper_binding'],
'model_unfolder/evidence/unet_iteration_binding.py':['read_iteration_binding'],
'model_unfolder/evidence/local_port_routes.py':['read_local_port_route'],
'model_unfolder/evidence/runtime_inventory.py':['build_resolved_instance'],
'model_unfolder/evidence/runtime_source.py':['RuntimeSourceBindings','RuntimePrimitiveClaimProof'],
'model_unfolder/evidence/instance_population_claim.py':['read_instance_population','read_constructor_defaults'],
'model_unfolder/evidence/instance_shape_claim.py':['read_instance_shapes'],
'physics/attribute_bindings.py':['witness_attribute_bindings','snapshot_instance_storage','capture_attribute_lookup_types'],
'physics/framework_primitives.py':['witness_framework_type','capture_framework_types'],
'physics/source_override.py':['source_overrides'],
}
newrows=[]
for p,names in new.items():
 entries=[]
 for name in names:
  calls=[r for r in refs(name,p) if r['path'].startswith(('model_unfolder/','physics/'))];assert calls,(p,name)
  entries.append({'name':name,'production_references':calls})
 newrows.append({'module':p,'entries':entries})
from model_unfolder.adapters.diffusor import parser,unet_cutover
from model_unfolder.adapters.diffusor.unet_differential import legacy_unet_comparison,legacy_comparison_enabled
assert not legacy_comparison_enabled()
try:parser._parse_unet_model({},'unused',[])
except RuntimeError as e:assert 'restricted to differential' in str(e)
else:raise AssertionError('unguarded legacy body')
try:
 with legacy_unet_comparison():
  assert legacy_comparison_enabled()
  with legacy_unet_comparison():assert legacy_comparison_enabled()
  assert legacy_comparison_enabled();raise ValueError('deliberate comparison failure')
except ValueError:pass
assert not legacy_comparison_enabled()
config={'_class_name':'Address','legacy_unet_comparison':True,'UNFOLD_UNET_LEGACY':True};ctx=NS(source_overrides=());sentinel=object();limited=NS(warnings=[]);calls=[]
def oldfail(*a,**k):raise AssertionError('legacy fallback reached')
with patch.object(parser,'_shadow_diffusion_root_topology',lambda c:NS(has_value=True,value=NS(kind='u_shaped'))),patch.object(parser,'_projected_pipeline_handoffs',lambda *a,**k:{}),patch.object(parser,'_parse_unet_model',oldfail):
 with patch.object(unet_cutover,'build_unet_cutover',lambda *a,**k:NS(ir=sentinel)):
  assert parser.parse(config,context=ctx)is sentinel
 for failure in (NS(detail='deliberate inventory failure'),None):
  def limited_projection(*a,**k):calls.append(a[-1]);return limited
  with patch.object(unet_cutover,'build_unet_cutover',lambda *a,**k:NS(ir=None,inventory_result=NS(failure=failure))),patch.object(parser,'_parse_projected_denoiser',limited_projection):
   assert parser.parse(config,context=ctx)is limited
 assert len(calls)==2 and all(not r.has_value and r.failures for r in calls)
 assert any('deliberate inventory failure' in w for w in limited.warnings)
 assert any('exact selected source reader did not close' in w for w in limited.warnings)
scoped=[r for r in refs('legacy_unet_comparison','model_unfolder/adapters/diffusor/unet_differential.py') if r['path'].startswith(('model_unfolder/','scripts/','physics/'))];assert len(scoped)==1 and scoped[0]['path']=='scripts/demonstrate_s8_unet.py' and 'args.condition == "legacy"' in scoped[0]['source']
pinned=set(original)|set(new)|{'model_unfolder/adapters/diffusor/parser.py','model_unfolder/adapters/diffusor/unet_cutover.py','model_unfolder/adapters/diffusor/unet_projection.py','model_unfolder/adapters/diffusor/unet_differential.py','model_unfolder/evidence/program_index.py','model_unfolder/evidence/execution_recipe.py','physics/instance_inventory.py','scripts/demonstrate_s8_unet.py','scripts/verify_s8_unet_family.py'}
pins={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in sorted(pinned)}
result={'checkpoint':commit,'original_commit':'83140f1','original_readers':rows,'new_live_modules':newrows,'legacy_enabling_callers':scoped,'routing_probes':{'old_direct_guard':True,'nested_comparison_failure_resets':True,'misleading_config_uses_new_path':True,'inventory_failure_typed_limited_without_legacy':True,'source_closure_failure_typed_limited_without_legacy':True},'source_sha256':pins,'limits':'Static caller/deletion and controlled parser routing audit only. No models, pytest, classifications, broad gates, output blessings or source-mechanism audit.'}
(O/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
assert pins=={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in pins};print(json.dumps({'original_retained_live':len(rows),'new_live_modules':len(newrows),'routing_probes':result['routing_probes'],'source_pins_unchanged':True}))
