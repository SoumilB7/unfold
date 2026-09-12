from pathlib import Path
import sys,json,dataclasses
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch.nn as nn
from diffusers import UNet2DConditionModel
from physics.attribute_bindings import AttributeLookupRequest,capture_attribute_lookup_types,witness_attribute_bindings
captured=capture_attribute_lookup_types()
owner=object.__new__(UNet2DConditionModel)
nn.Module.__init__(owner)
owner.conv_in=nn.Linear(2,2)
owner.down_blocks=nn.ModuleList([owner.conv_in,None])
owner._internal_dict={}
rows=witness_attribute_bindings(owner,'',tuple(AttributeLookupRequest('',a) for a in ('forward','config','conv_in','down_blocks')),captured)
out=[dataclasses.asdict(r) for r in rows]
Path(__file__).with_name('installed-class.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
assert rows[0].kind=='plain_bound_method'
assert rows[1].kind=='property_getter'
assert all(r.kind=='observed_registered_child' for r in rows[2:])
assert all(r.super_target is not None for r in rows[2:])
assert rows[3].iteration.slots[0].occurrence_path=='down_blocks.0'
assert rows[3].iteration.slots[1].occurrence_path is None
print(json.dumps({r.attribute:{'kind':r.kind,'function':r.function.qualname if r.function else None,'super_anchor':r.super_target.anchor_mro_index if r.super_target else None} for r in rows},indent=2))
