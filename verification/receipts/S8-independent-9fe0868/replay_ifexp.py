"""Conditional expression source-port controls; no model or pytest."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path.cwd()))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.local_port_routes import read_local_port_route
cases={
 'ordinary_conditioning':'selected = emb + aug if enabled else emb',
 'opaque_branches':'selected = first(emb) if enabled else second(aug)',
 'guard_walrus':'selected = emb if (emb := aug) else aug',
 'branch_walrus':'selected = ((emb := aug), emb) if enabled else (emb, emb)',
}
rows={}
for key,body in cases.items():
 source='class Cell:\n def forward(self,emb,aug,enabled):\n  '+body+'\n  consume(selected)\n'
 with tempfile.TemporaryDirectory(prefix='s8-ifexp-review-') as folder:
  path=Path(folder)/'source.py';path.write_text(source)
  index=build_program_index(SourceBundle(source='path',files=(str(path),),component_files={'root':(str(path),)}))
  forward=next(row for row in index.callables if row.symbol.qualified_name=='Cell.forward')
  call=next(row for row in index.calls_in(forward.symbol) if row.callee.name=='consume')
  route=read_local_port_route(index,forward,call.args[0],call.span,call.guard)
  rows[key]={'source':source,'route':route.value}
  if key=='branch_walrus':
   captured=[];environment={'consume':captured.append};exec(source,environment)
   environment['Cell']().forward(11,23,True)
   assert captured==[(23,23)]
   rows[key]['concrete_true_arm']=captured[0]
   rows[key]['original_emb']=11
assert rows['ordinary_conditioning']['route']['kind']=='conditional'
assert rows['ordinary_conditioning']['route']['when_true']['kind']=='source_operation'
assert rows['ordinary_conditioning']['route']['when_false']=={'kind':'formal','formal':'emb'}
assert rows['guard_walrus']['route']['kind']=='unresolved'
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'results':rows},indent=2,sort_keys=True))
