from pathlib import Path
from types import SimpleNamespace
import sys,importlib.util,tempfile,json,gzip,hashlib,dataclasses
REVIEW=Path(sys.argv[1]).resolve();sys.path.insert(0,str(REVIEW));OUT=Path(__file__).resolve().parent
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
from model_unfolder.evidence.unet_call_binding import read_root_invocations
from physics.instance_inventory import InventoryResult,BuildRequest,inventory_model
from physics.attribute_bindings import AttributeLookupRequest,capture_attribute_lookup_types
import model_unfolder.evidence.unet_cell_connections as cell
import model_unfolder.evidence.unet_iteration_binding as iteration
# Observe the actual production computation; neither wrapper supplies a value.
real_stable=cell._member_stays_bound;real_iteration=iteration.read_iteration_binding
stable_rows=[];iteration_rows=[]
def stable(*args,**kwargs):
 result=real_stable(*args,**kwargs);stable_rows.append({'line':args[2].span.line,'member':args[2].callee.name,'result':result});return result
def reader(*args,**kwargs):
 result=real_iteration(*args,**kwargs)
 row={'loop_line':args[2].span.line,'container':args[3].child_path,'parent_stable':kwargs['parent_stable'],'kind':result.kind,'reason':result.reason,'slots':[dataclasses.asdict(x) for x in result.slots],'targets':[{'line':x.call.span.line,'slot_index':x.slot_index,'path':x.occurrence_path,'guard_kinds':[s.kind for s in x.call.guard]} for x in result.targets]}
 for target in result.targets:
  assert target.call in args[0].calls_in(args[1].symbol)
  assert result.slots[target.slot_index].occurrence_path==target.occurrence_path
 iteration_rows.append(row);return result
cell._member_stays_bound=stable;iteration.read_iteration_binding=reader

def run(bindings,forward):
 stable_rows.clear();iteration_rows.clear()
 witness=next(x for x in bindings._modules[''].attribute_bindings if x.attribute=='forward')
 wrapper=read_wrapper_binding(bindings.index,witness,forward.symbol)
 targets,spans,investigations=read_root_invocations(bindings,forward,wrapper)
 return {'wrapper':wrapper.kind,'body_closures':[{'method':m.symbol.qualified_name,'closed':p.body_closed} for m,p in investigations],'stability_computations':list(stable_rows),'iteration_calls':list(iteration_rows),'targets':[{'line':r['call'].span.line,'targets':r.get('targets'),'conditions':r['conditions'],'guard_kinds':[g.kind for g in r['call'].guard]} for r in targets.values() if r['kind']=='constructed_target']}

controls={}
spec=importlib.util.spec_from_file_location('independent_iteration_existing_controls',REVIEW/'tests/test_s8_iteration_binding.py');tests=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tests;spec.loader.exec_module(tests)
for name in sorted(vars(tests)):
 if name.startswith('test_'):
  with tempfile.TemporaryDirectory() as d:
   getattr(tests,name)(Path(d));controls[name]='PASS'
assert len(controls)==7
synthetic={}
for mutating in (False,True):
 source='''import torch.nn as nn
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        shared=nn.Identity()
        self.stages=nn.ModuleList([shared,shared,None])
    def helper(self):
        self.stages=nn.ModuleList([])
    def forward(self,sample):
'''+('        self.helper()\n' if mutating else '')+'''        for index,block in enumerate(self.stages):
            if block is not None:
                sample=block(sample)
        return sample
'''
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'integration.py';path.write_text(source)
  spec=importlib.util.spec_from_file_location('iteration_integrated_'+str(mutating),path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  owner=module.Cell();assert owner.stages[0] is owner.stages[1]
  request=BuildRequest({},'custom',module.__name__,'Cell',attribute_lookups=tuple(AttributeLookupRequest('',x) for x in ('forward','helper','stages')))
  inv=inventory_model(owner,request,'direct',attribute_lookup_types=capture_attribute_lookup_types())
  index=build_program_index(SourceBundle(source='path',component_files={'root':(str(path),)}));bindings=SimpleNamespace(index=index,_modules={r.path:r for r in inv.modules});forward=next(x for x in index.callables if x.symbol.qualified_name=='Cell.forward')
  result=run(bindings,forward);result['source']=source;synthetic['mutation' if mutating else 'ordinary_alias_none']=result
  rows=[r for r in result['iteration_calls'] if r['container']=='stages']
  if mutating:
   assert not rows and not result['targets']
   assert result['body_closures'] and result['body_closures'][0]['closed'] is False
  else:
   assert len(rows)==1
   assert rows[0]['parent_stable'] is True and rows[0]['kind']=='iteration'
   assert [x['occurrence_path'] for x in rows[0]['slots']]==['stages.0','stages.1',None]
   assert result['targets'][0]['targets']==['stages.0','stages.1']
   assert result['targets'][0]['guard_kinds']
   assert any(c.get('kind')=='per_iteration' and c['slots'][2]['kind']=='none' for c in result['targets'][0]['conditions'])
receipt=REVIEW/'verification/receipts/S8-actual-lookup-positive-v2'
inv=InventoryResult.from_dict(json.loads(gzip.decompress((receipt/'inventory-result.json.gz').read_bytes()))).inventory
sources=json.loads((receipt/'indexed-sources.json').read_text())
for row in sources:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
index=build_program_index(SourceBundle(source='path',component_files={'root':tuple(x['path'] for x in sources)}));bindings=SimpleNamespace(index=index,_modules={r.path:r for r in inv.modules});forward=next(x for x in index.callables if x.symbol.qualified_name=='UNet2DConditionModel.forward')
actual=run(bindings,forward)
positive=[r for r in actual['iteration_calls'] if r['kind']=='iteration']
assert len(positive)==2 and all(r['parent_stable'] is True and len(r['slots'])==3 and len(r['targets'])==6 for r in positive)
assert all(t['guard_kinds'] for r in positive for t in r['targets'])
FILES=['model_unfolder/evidence/'+x for x in ('unet_iteration_binding.py','unet_call_binding.py','unet_lookup_closure.py','unet_cell_connections.py','unet_wrapper_binding.py','program_index.py')]
hashes={p:hashlib.sha256((REVIEW/p).read_bytes()).hexdigest() for p in FILES}
(OUT/'results.json').write_text(json.dumps({'checkout':str(REVIEW),'hashes':hashes,'seven_controls':controls,'synthetic':synthetic,'actual':actual},sort_keys=True,indent=2)+'\n')
print(json.dumps({'controls':controls,'synthetic_ordinary':'PASS alias+None retained, actual stability True','synthetic_mutation':'PASS computed body closure False rejects before iteration, no targets','actual_positive_loops':[{'container':r['container'],'line':r['loop_line'],'slots':len(r['slots']),'targets':len(r['targets']),'computed_parent_stable':r['parent_stable']} for r in positive]},indent=2))
