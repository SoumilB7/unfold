from pathlib import Path
import importlib.util,json,dataclasses,sys
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');sys.path.insert(0,str(root))
from diffusers.configuration_utils import FrozenDict
from physics.attribute_bindings import _class_slot
s=importlib.util.spec_from_file_location('storage_controls',root/'tests/test_s8_attribute_bindings.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
model=m.StoragePropertyCell();model._storage=FrozenDict({'sample_size':128})
w,rows=m.storage_observation(model,('child','sample_size','items'))
result={'scope':'Actual installed FrozenDict object, tiny fixture; no full model/render.', 'python':sys.version,'actual_lookup_is_dict_primitive':_class_slot(FrozenDict,'__getattribute__') is dict.__getattribute__, 'rows':{name:dataclasses.asdict(row) for name,row in rows.items()}}
assert rows['child'].kind == rows['sample_size'].kind == 'plain_attribute_lookup'
assert rows['items'].reason == 'stored_receiver_requested_attribute_is_descriptor'
m.test_storage_lookup_refuses_custom_receiver_without_executing_it()
result['custom_receiver_negative']='PASS; custom getattr/getattribute not invoked'
out=root/'verification/receipts/S8-storage-lookup-refinement';(out/'actual-storage-positive.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
