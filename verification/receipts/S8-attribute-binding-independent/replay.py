from pathlib import Path
import importlib.util
import json
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch.nn as nn
from physics.attribute_bindings import AttributeLookupRequest, capture_attribute_lookup_types
from physics.instance_inventory import BuildRequest, inventory_model

class ChangingFallback(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 2)
    def __getattr__(self, name):
        if name == 'first':
            self.second = nn.Linear(3, 3)
            raise RuntimeError('changed before unsuccessful observation')
        return super().__getattr__(name)

class ChangingSlots(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.stages = nn.ModuleList([nn.Linear(2, 2)])
    def __getattr__(self, name):
        if name == 'first':
            self._modules['stages'][0] = nn.Linear(3, 3)
        return super().__getattr__(name)

def probe(cls, second):
    model=cls()
    request=BuildRequest({}, 'custom', __name__, cls.__qualname__, attribute_lookups=(AttributeLookupRequest('', 'first'), AttributeLookupRequest('', second)))
    result=inventory_model(model, request, 'direct', attribute_lookup_types=capture_attribute_lookup_types())
    lookup=result.modules[0].attribute_bindings
    constructed={r.path:[list(p.shape) for p in r.parameters] for r in result.modules}
    import dataclasses
    return {'constructed_shapes':constructed,'lookups':[dataclasses.asdict(x) for x in lookup],
      'actual_shapes':{p:list(v.shape) for p,v in model.named_parameters()}}

def main():
    spec=importlib.util.spec_from_file_location('independent_attribute_controls', ROOT/'tests/test_s8_attribute_bindings.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    controls={}
    for name in sorted(vars(module)):
        if name.startswith('test_'):
            try: getattr(module,name)();controls[name]='PASS'
            except Exception as exc: controls[name]=type(exc).__name__+': '+str(exc)
    out={'hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('physics/attribute_bindings.py','physics/instance_inventory.py')},'controls':controls,'failed_lookup_contamination':probe(ChangingFallback,'second'),'slot_contamination':probe(ChangingSlots,'stages')}
    Path(__file__).with_name('results.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'controls':controls,'failed_lookup':[(r['kind'],r['reason'],r['child_path'],r['observed_state_changed'],r['registered_slots_unchanged']) for r in out['failed_lookup_contamination']['lookups']], 'slot_lookup':[(r['kind'],r['reason'],r['child_path'],r['observed_state_changed'],r['registered_slots_unchanged']) for r in out['slot_contamination']['lookups']]},sort_keys=True,indent=2))
if __name__=='__main__': main()
