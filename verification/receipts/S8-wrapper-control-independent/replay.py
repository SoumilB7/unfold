from pathlib import Path
import sys,runpy,importlib.util,contextlib,io,json,hashlib,ast
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
PATH='model_unfolder.evidence.unet_wrapper_binding'
current=ROOT/'model_unfolder/evidence/unet_wrapper_binding.py'
old=ROOT/'verification/receipts/S8-wrapper-control-stop/proposed-source/unet_wrapper_binding-before-refusal.py'
probe=ROOT/'verification/receipts/S8-wrapper-control-stop/reproduce.py'
results={}
legacy_name='model_unfolder.evidence.unet_lookup_closure'
legacy_path=ROOT/'verification/receipts/S8-wrapper-control-stop/proposed-source/unet_lookup_closure.py'
spec=importlib.util.spec_from_file_location(legacy_name,legacy_path);module=importlib.util.module_from_spec(spec);sys.modules[legacy_name]=module;spec.loader.exec_module(module)
for label,path in [('before_refusal',old),('current_refusal',current)]:
 spec=importlib.util.spec_from_file_location(PATH,path);module=importlib.util.module_from_spec(spec);sys.modules[PATH]=module;spec.loader.exec_module(module)
 capture=io.StringIO();code=0
 try:
  with contextlib.redirect_stdout(capture):runpy.run_path(str(probe),run_name='__main__')
 except SystemExit as exc:code=exc.code
 results[label]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exit_code':code,'output':json.loads(capture.getvalue())}
assert results['before_refusal']['output']['result_kind']=='conditional_wrapper'
assert results['current_refusal']['output']['result_kind']=='unresolved'
installed=Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/diffusers/utils/peft_utils.py')
raw=installed.read_bytes();tree=ast.parse(raw)
outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_lora_scale')
dec=next(n for n in outer.body if isinstance(n,ast.FunctionDef))
wrapper=next(n for n in dec.body if isinstance(n,ast.FunctionDef))
trys=[n for n in ast.walk(wrapper) if isinstance(n,ast.Try)]
assert len(trys)==1
node=trys[0]
results['installed_wrapper']={'path':str(installed),'sha256':hashlib.sha256(raw).hexdigest(),'wrapper_lines':[wrapper.lineno,wrapper.end_lineno],'try_lines':[node.lineno,node.end_lineno], 'handlers':len(node.handlers),'else_count':len(node.orelse),'finally_count':len(node.finalbody),'delegation_calls':[{'line':n.lineno,'callee':n.func.id} for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]}
results['program_index_sha256']=hashlib.sha256((ROOT/'model_unfolder/evidence/program_index.py').read_bytes()).hexdigest()
(OUT/'results.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')
print(json.dumps({'red':results['before_refusal']['output']['result_kind'],'current':results['current_refusal']['output']['result_kind'],'installed':results['installed_wrapper']},indent=2))
