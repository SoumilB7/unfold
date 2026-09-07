"""Independent correction replay; run from immutable 4637ef4 checkout.

This collects corrected controls and remaining counterexamples, without
pytest, model execution, or production writes. Initial review helper functions
are loaded as definitions; their old defect assertions are removed solely to
collect the new values. Acceptance assertions below are explicit.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
initial = ROOT / 'verification/receipts/S8-owner-review-initial/reproduce.py'
parsed = ast.parse(initial.read_text())
body = []
for node in parsed.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'before' for target in node.targets):
        break
    body.append(node)
class CollectValues(ast.NodeTransformer):
    def visit_Assert(self, node):
        return ast.copy_location(ast.Pass(), node)
namespace = {}
exec(compile(ast.fix_missing_locations(CollectValues().visit(ast.Module(body=body, type_ignores=[]))), str(initial), 'exec'), namespace)
fingerprint = namespace['fingerprint']
before = fingerprint()
results = {'R1_direct_alias': namespace['alias_escape'](),
           'R2_spoofed_name': namespace['forged_primitive']()}
try:
    namespace['unrelated_concat']()
    results['R3_wrong_concat'] = {'rejected': False}
except ValueError as error:
    results['R3_wrong_concat'] = {'rejected': True, 'reason': str(error)}

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.local_port_routes import read_local_port_route
from model_unfolder.evidence.unet_cell_connections import _member, _member_stays_bound

def route(source, *, member=False):
    with tempfile.TemporaryDirectory(prefix='s8-correction-review-') as directory:
        path = Path(directory) / 'model.py'
        path.write_text(source)
        index = build_program_index(SourceBundle(source='path', files=(str(path),), component_files={'root': (str(path),)}))
        forward = next(row for row in index.callables if row.symbol.qualified_name == 'Cell.forward')
        if member:
            call = next(row for row in index.calls_in(forward.symbol) if _member(row.callee) == 'second')
            bindings = SimpleNamespace(_modules={'cell': SimpleNamespace(init_attributes={}),
                                                'cell.first': object(), 'cell.second': object()})
            value = {'member_stays_bound': _member_stays_bound(index, forward, call, 'cell', bindings)}
        else:
            call = next(row for row in index.calls_in(forward.symbol) if row.callee.name == 'consume')
            value = read_local_port_route(index, forward, call.args[0], call.span, call.guard).value
        return {'source': source, 'value': value}

results['R1_container_escape'] = route('''class Cell:
 def forward(self, value):
  holder = [self]
  helper(holder)
  value = self.first(value)
  return self.second(value)
''', member=True)
ports_source = ROOT / 'verification/receipts/S8-owner-review-initial/reproduce_local_ports.py'
cases = next(ast.literal_eval(node.value) for node in ast.parse(ports_source.read_text()).body
             if isinstance(node, ast.Assign) and node.targets[0].id == 'cases')
results['R7_R8_original_controls'] = {key: route(source) for key, source in cases.items()}
results['R7_after_with'] = route('''class Cell:
 def forward(self, seed):
  with manager() as seed:
   pass
  consume(seed)
''')
results['R7_after_loop'] = route('''class Cell:
 def forward(self, state, items):
  for state in items:
   pass
  consume(state)
''')
results['R4_opaque_helper'] = route('''def ignore(argument):
 return 7
class Cell:
 def forward(self, side):
  result = ignore(side)
  consume(result)
''')
from physics.framework_primitives import capture_framework_types, witness_framework_type
import torch
captured = capture_framework_types()
canonical = witness_framework_type(torch.nn.SiLU(), captured)
class Custom(torch.nn.Module):
    def forward(self, value): return value + 10
Custom.__module__ = 'torch.nn.modules.activation'
Custom.__qualname__ = 'SiLU'
results['R2_worker_membership'] = {'canonical_key': canonical.key,
                                   'spoof_rejected': witness_framework_type(Custom(), captured) is None}
assert results['R1_direct_alias']['member_stays_bound'] is False
assert results['R1_container_escape']['value']['member_stays_bound'] is False
assert results['R2_spoofed_name']['claimed_meaning'] is None
assert results['R2_worker_membership']['spoof_rejected']
assert results['R3_wrong_concat']['rejected']
assert results['R7_R8_original_controls']['carried_local']['value']['kind'] == 'loop_carried'
assert results['R7_R8_original_controls']['loop_target_overwrites_formal']['value']['kind'] == 'unresolved'
assert results['R7_R8_original_controls']['with_target_overwrites_formal']['value']['kind'] == 'unresolved'
assert results['R7_R8_original_controls']['starred_unpack']['value']['kind'] == 'unresolved'
assert results['R7_after_with']['value']['kind'] == 'unresolved'
assert results['R7_after_loop']['value']['kind'] == 'unresolved'
assert results['R4_opaque_helper']['value']['mechanism'] == 'unresolved'
results['R1_known_holder_attribute'] = route('''class Cell:
 def forward(self, value):
  holder = Namespace()
  holder.parent = self
  helper(holder.parent)
  value = self.first(value)
  return self.second(value)
''', member=True)
results['R1_alias_attribute_write'] = route('''class Cell:
 def forward(self, value, replacement):
  alias = self
  alias.second = replacement
  value = self.first(value)
  return self.second(value)
''', member=True)
assert results['R1_known_holder_attribute']['value']['member_stays_bound'] is False
assert results['R1_alias_attribute_write']['value']['member_stays_bound'] is False
after = fingerprint()
assert before == after
print(json.dumps({'checkpoint': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'tracked_fingerprint_before': before, 'tracked_fingerprint_after': after,
                  'verdict': 'PASS: reviewed R1 alias/holder controls rejected; R7 original controls retained; R5 separate replay',
                  'results': results}, indent=2, sort_keys=True))
