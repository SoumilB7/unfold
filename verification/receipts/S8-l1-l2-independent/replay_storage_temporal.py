from pathlib import Path
import sys,tempfile,importlib.util,json
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from physics.attribute_bindings import AttributeLookupRequest,capture_attribute_lookup_types
from physics.instance_inventory import BuildRequest,inventory_model
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure,close_parent_helpers
SOURCE='''import torch.nn as nn
POISON = None
class Config:
    def __init__(self):
        self.flag = False
class Poison:
    def __init__(self, owner):
        self.owner = owner
    def __bool__(self):
        self.owner.child = nn.Linear(3, 3)
        return False
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(2, 2)
        self._storage = Config()
    @property
    def config(self):
        return self._storage
    def helper(self, value):
        alias = self.config
        alias.flag = POISON
        if self.config.flag:
            pass
        return value
    def forward(self, value):
        value = self.helper(value)
        return self.child(value)
'''
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'temporal.py';path.write_text(SOURCE)
 spec=importlib.util.spec_from_file_location('independent_storage_temporal',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 owner=module.Cell();module.POISON=module.Poison(owner)
 requests=(AttributeLookupRequest('','forward'),AttributeLookupRequest('','helper'),AttributeLookupRequest('','child'),AttributeLookupRequest('','config',storage_attributes=('flag',)))
 inv=inventory_model(owner,BuildRequest({},'custom',module.__name__,'Cell',attribute_lookups=requests),'direct',attribute_lookup_types=capture_attribute_lookup_types())
 index=build_program_index(SourceBundle(source='path',component_files={'root':(str(path),)}));bindings=SimpleNamespace(index=index,_modules={r.path:r for r in inv.modules})
 lookups=[read_lookup_closure(bindings,w) for w in inv.modules[0].attribute_bindings]
 method=next(x for x in index.callables if x.symbol.qualified_name=='Cell.forward')
 closure=close_parent_helpers(bindings,method,lookups)
 prop=next(x for x in inv.modules[0].attribute_bindings if x.attribute=='config')
 before=list(owner.child.weight.shape);owner.helper(7);after=list(owner.child.weight.shape)
 out={'source':SOURCE,'scalar_premise':[{'kind':p.kind,'value_kind':p.value_kind} for p in prop.storage_attribute_lookups if p.storage_name=='_storage' and p.attribute=='flag'],'closed_helpers':[c.callee.name for c in closure.closed_helpers],'unresolved':[(s.line,r) for s,r in closure.unresolved],'conditions_count':len(closure.conditions),'before_child_shape':before,'after_child_shape':after}
 (OUT/'storage-temporal-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
