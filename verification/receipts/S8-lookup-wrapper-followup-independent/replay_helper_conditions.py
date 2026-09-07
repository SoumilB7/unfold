from pathlib import Path
import sys,tempfile,importlib.util,json,dataclasses
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from physics.attribute_bindings import AttributeLookupRequest,capture_attribute_lookup_types
from physics.instance_inventory import BuildRequest,inventory_model
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_wrapper_binding import read_wrapper_binding
from model_unfolder.evidence.unet_call_binding import read_root_invocations
SOURCE='''import torch.nn as nn
enabled = False
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Identity()
    @property
    def opaque(self):
        return unknown()
    def helper(self, value):
        if enabled:
            ignored = self.opaque
            return self.child(value)
        return value
    def forward(self, value):
        value = self.helper(value)
        return self.child(value)
'''
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'helper_fixture.py';path.write_text(SOURCE)
 spec=importlib.util.spec_from_file_location('independent_helper_conditions',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 owner=module.Cell();request=BuildRequest({},'custom',module.__name__,'Cell',attribute_lookups=tuple(AttributeLookupRequest('',x) for x in ('forward','helper','opaque','child')))
 inventory=inventory_model(owner,request,'direct',attribute_lookup_types=capture_attribute_lookup_types())
 index=build_program_index(SourceBundle(source='path',component_files={'root':(str(path),)}));bindings=SimpleNamespace(index=index,_modules={r.path:r for r in inventory.modules})
 forward=next(x for x in index.callables if x.symbol.qualified_name=='Cell.forward')
 witness=next(x for x in inventory.modules[0].attribute_bindings if x.attribute=='forward')
 wrapper=read_wrapper_binding(index,witness,forward.symbol)
 targets,spans,investigations=read_root_invocations(bindings,forward,wrapper)
 out={'source':SOURCE,'wrapper_kind':wrapper.kind,'targets':[{'kind':r['kind'],'call_line':r['call'].span.line,'method':r['call'].enclosing_callable.qualified_name,'guards':[{'kind':g.kind,'predicate':g.test.source_segment if g.test else None} for g in r['call'].guard],'conditions':r['conditions']} for r in targets.values()],'actual_helper_false_result':owner.helper(7)}
 (OUT/'helper-condition-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
