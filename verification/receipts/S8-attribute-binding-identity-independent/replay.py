from pathlib import Path
import sys,importlib.util,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
FILES=('physics/attribute_bindings.py','physics/instance_inventory.py')
def hashes():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}
before=hashes()
expected=json.loads((ROOT/'verification/receipts/S8-attribute-binding-identity-correction/source-manifest.json').read_text())
assert all(before[p]==expected[p] for p in FILES)
old=ROOT/'verification/receipts/S8-attribute-binding-independent/replay.py'
spec=importlib.util.spec_from_file_location('original_independent_worker_probes',old);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
failed=module.probe(module.ChangingFallback,'second')
slots=module.probe(module.ChangingSlots,'stages')
first,second=failed['lookups']
assert first['kind']=='unresolved' and first['reason']=='custom_fallback_raised_RuntimeError'
assert first['lookup_state_before'] is not None and first['lookup_state_after'] is not None
assert first['lookup_getattr'] is not None
assert first['registered_slots_unchanged'] is False
assert second['kind']=='unresolved' and 'contaminated' in second['reason'] and second['child_path'] is None
assert failed['constructed_shapes']['second'][0]==[2,2]
assert failed['actual_shapes']['second.weight']==[3,3]
first,second=slots['lookups']
assert second['kind']=='observed_registered_child' and second['child_path']=='stages'
assert second['iteration'] is None and second['iteration_reason']=='modulelist_slots_differ_from_construction'
assert slots['constructed_shapes']['stages.0'][0]==[2,2]
assert slots['actual_shapes']['stages.0.weight']==[3,3]
after=hashes();assert before==after
(OUT/'results.json').write_text(json.dumps({'verdict':'PASS within original correction scope','before':before,'after':after,'failed_lookup_contamination':failed,'nested_slot_contamination':slots},indent=2,sort_keys=True)+'\n')
print('PASS: original failed-lookup and nested-slot contamination probes; hashes identical')
