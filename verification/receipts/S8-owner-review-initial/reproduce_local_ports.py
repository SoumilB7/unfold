"""Independent source-only probes of the working local-port reader.

Run from its checkout. No pytest, model execution, or checkout writes. These
assertions reproduce the reviewed defects; they should fail after correction.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.local_port_routes import read_local_port_route

reader_path = ROOT / 'model_unfolder/evidence/local_port_routes.py'
before = hashlib.sha256(reader_path.read_bytes()).hexdigest()
cases = {
    'carried_local': '''class Cell:
 def forward(self, seed, items):
  state = seed
  for item in items:
   consume(state)
   state = update(item)
''',
    'loop_target_overwrites_formal': '''class Cell:
 def forward(self, state, items):
  for state in items:
   consume(state)
''',
    'with_target_overwrites_formal': '''class Cell:
 def forward(self, seed):
  with manager() as seed:
   consume(seed)
''',
    'starred_unpack': '''class Cell:
 def forward(self, seed):
  first, *middle, last = helper(seed)
  consume(last)
''',
    'ordinary_alias_control': '''class Cell:
 def forward(self, seed):
  state = seed
  consume(state)
''',
    'guarded_assignment_control': '''class Cell:
 def forward(self, seed, flag):
  state = seed
  if flag:
   state = replace(seed)
   return consume(state)
  return state
''',
    'opaque_call_control': '''class Cell:
 def forward(self, seed):
  result = helper(seed)
  consume(result)
''',
}
results = {}
for name, source in cases.items():
    with tempfile.TemporaryDirectory(prefix='s8-review-port-') as directory:
        path = Path(directory) / 'model.py'
        path.write_text(source)
        index = build_program_index(SourceBundle(
            source='path', files=(str(path),), component_files={'root': (str(path),)}))
        forward = next(row for row in index.callables if row.symbol.qualified_name == 'Cell.forward')
        call = next(row for row in index.calls_in(forward.symbol) if row.callee.name == 'consume')
        route = read_local_port_route(index, forward, call.args[0], call.span, call.guard)
        results[name] = {'source': source, 'route': route.value}

assert results['carried_local']['route'] == {'kind': 'formal', 'formal': 'seed'}
assert results['loop_target_overwrites_formal']['route'] == {'kind': 'formal', 'formal': 'state'}
assert results['with_target_overwrites_formal']['route'] == {'kind': 'formal', 'formal': 'seed'}
assert results['starred_unpack']['route']['result_slot'] == [2]
assert results['ordinary_alias_control']['route'] == {'kind': 'formal', 'formal': 'seed'}
assert results['guarded_assignment_control']['route']['kind'] == 'call_result'
assert results['opaque_call_control']['route']['mechanism'] == 'unresolved'
after = hashlib.sha256(reader_path.read_bytes()).hexdigest()
assert before == after
print(json.dumps({
    'checkout_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'reviewed_source': str(reader_path.relative_to(ROOT)),
    'source_sha256_before': before, 'source_sha256_after': after,
    'scope': 'working reader source identified by hash; not an immutable whole-tree verdict',
    'results': results,
}, indent=2, sort_keys=True))
