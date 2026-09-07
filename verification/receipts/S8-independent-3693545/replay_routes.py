"""Bounded source-port controls, no pytest or checkpoint execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path.cwd()))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.local_port_routes import read_local_port_route
cases = {
 'loop_result': 'for item in items:\n   state = step(state)',
 'loop_update': 'for item in items:\n   state = step(state)\n   if enabled:\n    state += extra',
 'loop_if_else': 'for item in items:\n   if enabled:\n    state = first(state)\n   else:\n    state = second(state)',
 'loop_break': 'for item in items:\n   state = step(state)\n   if enabled:\n    break',
 'loop_continue': 'for item in items:\n   if enabled:\n    continue\n   state = step(state)',
 'loop_else': 'for item in items:\n   state = step(state)\n  else:\n   state = finish(state)',
 'loop_else_break': 'for item in items:\n   state = step(state)\n   if enabled:\n    break\n  else:\n   state = finish(state)',
 'optional_after_loop': 'for item in items:\n   state = step(state)\n  if enabled:\n   state = finish(state)',
 'conditional': 'if enabled:\n   state = first(state)\n  else:\n   state = second(state)',
 'optional_update': 'if enabled:\n   state += extra',
 'receiver': 'state = state.flatten()',
}
reader=Path('model_unfolder/evidence/local_port_routes.py')
before=hashlib.sha256(reader.read_bytes()).hexdigest()
rows={}
for key, body in cases.items():
 source='class Cell:\n def forward(self, state, items, enabled, extra):\n  '+body+'\n  consume(state)\n'
 with tempfile.TemporaryDirectory(prefix='s8-port-review-') as temporary:
  path=Path(temporary)/'source.py'; path.write_text(source)
  index=build_program_index(SourceBundle(source='path', files=(str(path),), component_files={'root':(str(path),)}))
  forward=next(row for row in index.callables if row.symbol.qualified_name=='Cell.forward')
  call=next(row for row in index.calls_in(forward.symbol) if row.callee.name=='consume')
  route=read_local_port_route(index,forward,call.args[0],call.span,call.guard)
  rows[key]={'source':source,'route':route.value, 'binding_guards': [{'line':row.span.line,'guard_kinds':[step.kind for step in row.guard]} for row in index.bindings_in(forward.symbol)]}
  if key == 'loop_else_break':
   captured=[]
   env={'step':lambda value:value+1,'finish':lambda value:value+100,'consume':captured.append}
   exec(source,env)
   env['Cell']().forward(0,[1],True,0)
   rows[key]['concrete_break_value']=captured[0]
   rows[key]['finish_would_return']=101
   rows[key]['false_positive']=route.value['kind']=='call_result'
   assert captured==[1]
   assert route.value['kind']=='unresolved'
after=hashlib.sha256(reader.read_bytes()).hexdigest()
assert before==after
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'reader_sha256':before,'unchanged':True,'results':rows},indent=2,sort_keys=True))
