from pathlib import Path
import sys,tempfile,importlib.util,json,dataclasses
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from physics.attribute_bindings import AttributeLookupRequest,capture_attribute_lookup_types
from physics.instance_inventory import BuildRequest,inventory_model
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure
SOURCE='''import torch.nn as nn
SWITCH = False
class ConfigProbe:
    def __init__(self, owner):
        self.owner = owner
    def __getattr__(self, name):
        if SWITCH:
            self.owner.child = nn.Linear(3, 3)
        raise AttributeError(name)
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(2, 2)
        self._storage = ConfigProbe(self)
    def __getattr__(self, name):
        is_in_config = "_storage" in self.__dict__ and hasattr(self.__dict__["_storage"], name)
        is_attribute = name in self.__dict__
        if is_in_config and not is_attribute:
            return self._storage[name]
        return super().__getattr__(name)
'''
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'lookup_fixture.py';path.write_text(SOURCE)
 spec=importlib.util.spec_from_file_location('independent_lookup_receiver',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 owner=module.Cell();original=owner._modules['child']
 request=BuildRequest({},'custom',module.__name__,'Cell',attribute_lookups=(AttributeLookupRequest('','child'),))
 inventory=inventory_model(owner,request,'direct',attribute_lookup_types=capture_attribute_lookup_types())
 index=build_program_index(SourceBundle(source='path',component_files={'root':(str(path),)}))
 bindings=SimpleNamespace(index=index,_modules={r.path:r for r in inventory.modules})
 witness=inventory.modules[0].attribute_bindings[0]
 result=read_lookup_closure(bindings,witness)
 module.SWITCH=True
 selected=owner.child
 out={'source':SOURCE,'worker_kind':witness.kind,'observed_state_changed':witness.observed_state_changed,'registered_slots_unchanged':witness.registered_slots_unchanged,'reader_kind':result.kind,'reason':result.reason,'conditions':[{'return_line':c.return_span.line,'guards':[{'kind':s.kind,'predicate':s.test.source_segment if s.test else None} for s in c.branch]} for c in result.conditions],'selected_is_original':selected is original,'frozen_shape':list(inventory.modules[1].parameters[0].shape),'actual_shape':list(selected.weight.shape),'earlier_config_return_taken':False,'source_sha256':index.callables[0].symbol.source.content_fingerprint}
 (OUT/'lookup-receiver-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2))
