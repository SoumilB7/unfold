from pathlib import Path
from dataclasses import replace
import runpy,json
OUT=Path(__file__).resolve().parent
state=runpy.run_path(str(OUT/'replay_saved_helpers.py'))
bindings=state['bindings'];root=bindings._modules[''];forward=state['forward'];wrapper=state['wrapper']
from model_unfolder.evidence.unet_lookup_closure import read_lookup_closure,close_parent_helpers
key='encoder_hid_dim_type'
original=next(w for w in root.attribute_bindings if w.attribute=='config')
assert any(p.attribute==key and p.storage_name=='_internal_dict' and p.value_kind=='scalar' for p in original.storage_attribute_lookups)
changed=replace(original,storage_attribute_lookups=tuple(p for p in original.storage_attribute_lookups if not(p.attribute==key and p.storage_name=='_internal_dict')))
bindings._modules['']=replace(root,attribute_bindings=tuple(changed if w.attribute=='config' else w for w in root.attribute_bindings))
lookups=[read_lookup_closure(bindings,w) for w in bindings._modules[''].attribute_bindings]
parent=close_parent_helpers(bindings,forward,lookups,function_witness=wrapper.function)
closed=[c.callee.name for c in parent.closed_helpers]
assert 'process_encoder_hidden_states' not in closed
out={'removed_exact_premise':['_internal_dict',key],'before_closed_helpers':[c.callee.name for c in state['parent'].closed_helpers],'after_closed_helpers':closed,'unresolved':[(s.line,r) for s,r in parent.unresolved]}
(OUT/'missing-nested-premise-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
