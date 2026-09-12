"""Read-only working-producer audit. Review artifacts only, no model/pytest."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
paths=[Path('model_unfolder/evidence/unet_primary_ports.py'),Path('model_unfolder/evidence/local_port_routes.py')]
contents={str(p):p.read_bytes() for p in paths}
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_primary_ports import read_primary_regions
cases={
 'formal_rebound_before_region':'''class Cell:
 def forward(self,state,conditioning):
  conditioning = normalize(conditioning)
  state = block(state, conditioning)
  return state
''',
 'formal_rebound_between_regions':'''class Cell:
 def forward(self,state,conditioning):
  state = first(state)
  conditioning = normalize(conditioning)
  state = block(state, conditioning)
  return state
''',
 'loop_seed':'''class Cell:
 def forward(self,state,items):
  state = first(state)
  for item in items:
   state = block(state)
  return state
''',
 'unsupported_rebind_between_regions':'''class Cell:
 def forward(self,state,manager):
  state = first(state)
  with manager() as state:
   pass
  state = block(state)
  return state
''',
 'loop_target_between_regions':'''class Cell:
 def forward(self,state,items):
  state = first(state)
  for state in items:
   pass
  state = block(state)
  return state
''',
}
results={}
for key,source in cases.items():
 with tempfile.TemporaryDirectory(prefix='s8-primary-probe-') as folder:
  path=Path(folder)/'source.py';path.write_text(source)
  index=build_program_index(SourceBundle(source='path',files=(str(path),),component_files={'root':(str(path),)}))
  forward=next(row for row in index.callables if row.symbol.qualified_name=='Cell.forward')
  rows,spans=read_primary_regions(index,forward,'state')
  results[key]={'source':source,'regions':rows}
for p in paths: assert p.read_bytes()==contents[str(p)],'working source changed during probe'
out=ROOT/'verification/receipts/S8-primary-ports-early-independent'
for p in paths:(out/(p.name+'.txt')).write_bytes(contents[str(p)])
print(json.dumps({'source_sha256':{key:hashlib.sha256(value).hexdigest() for key,value in contents.items()},'source_unchanged_during_probe':True,'results':results},indent=2,sort_keys=True))
