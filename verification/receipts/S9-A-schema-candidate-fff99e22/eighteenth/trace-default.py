from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
cfg=json.loads(Path('/private/tmp/unfold-s9a-sparse-eighteenth-llama/llama-7b/sparse-valid/input.json').read_text())
rows=[]
def trace(frame,event,arg):
 if not frame.f_code.co_filename.endswith('/evidence/class_default_value.py'):
  return None
 if frame.f_code.co_name not in ('_declaration','_constructor_keeps_default'):
  return None
 if event=='return':
  row={'function':frame.f_code.co_name,'line':frame.f_lineno,'return_type':type(arg).__name__,'return_value':arg if arg is None or isinstance(arg,bool) else 'object'}
  for key in ('constructor','binding','target','access','alias'):
   if key in frame.f_locals:row[key]=repr(frame.f_locals[key])[:4000]
  rows.append(row)
 return trace
context=ParseContext.build(cfg)
sys.settrace(trace)
try: ir=config_to_ir(cfg,parse_context=context)
finally: sys.settrace(None)
out={'trace':rows,'hidden_size':ir.hidden_size,'width_fact':context.facts.to_dict().get('model.hidden_size')}
Path('/private/tmp/unfold-s9a-eighteenth/actual-default-trace.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'hidden_size':ir.hidden_size,'trace':[{'function':r['function'],'line':r['line'],'return':r['return_value']} for r in rows]},indent=2))
