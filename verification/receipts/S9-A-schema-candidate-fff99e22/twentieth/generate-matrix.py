from pathlib import Path
import importlib.util,json,sys
root=Path.cwd()
spec=importlib.util.spec_from_file_location('s9a_s7_candidate_generator',root/'scripts/generate_s7_shadow.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
out=Path('/private/tmp/unfold-s9a-twentieth/matrix')
assert not out.exists()
result=module.generate(output=out)
print(json.dumps({'candidate_only':True,'output':str(out),'models':len(result['models']),'active_matrix_renewed':False},indent=2),flush=True)
