"""Read-only neutral-record inspection for the installed SDXL root route.

This records source premises, not a new architecture reader or schema.
Run from the immutable reviewed code checkout; output is a review artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
raw = args.source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == '052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26'
index = build_program_index(SourceBundle(source='path', files=(str(args.source),),
                                         component_files={'root': (str(args.source),)}))
forward = next(row for row in index.callables if row.symbol.qualified_name == 'UNet2DConditionModel.forward')

def expression(value):
    if value is None: return None
    result = {'kind': value.kind}
    if value.name: result['name'] = value.name
    if value.operator: result['operator'] = value.operator
    if value.kind == 'constant': result['value'] = value.const_value
    if value.children: result['children'] = [expression(child) for child in value.children]
    if value.keyword_children: result['keywords'] = {key: expression(child) for key, child in value.keyword_children}
    return result

def guards(rows):
    return [{'kind': row.kind, 'line': row.span.line, 'end_line': row.span.end_line,
             'test': expression(row.test)} for row in rows]

lines = {1110, 1137, 1145, 1155, 1157, 1159, 1170, 1175, 1184, 1192, 1195,
         1201, 1202, 1210, 1221, 1230, 1231, 1232}
bindings = [{'line': row.span.line, 'end_line': row.span.end_line,
             'kind': row.assignment_kind, 'targets': [expression(target) for target in row.targets],
             'value': expression(row.value), 'guards': guards(row.guard),
             'dataflow_operators': [edge.op for edge in index.dataflow
                                    if edge.enclosing_callable == forward.symbol and edge.span == row.span]}
            for row in index.bindings_in(forward.symbol) if row.span.line in lines]
loops = [{'line': row.span.line, 'end_line': row.span.end_line,
          'target': expression(row.target), 'iterable': expression(row.iterable),
          'guards': guards(row.guard)} for row in index.loops_in(forward.symbol)]
unsupported = [{'kind': row.construct_kind, 'line': row.span.line, 'end_line': row.span.end_line,
                'reason': row.reason} for row in index.unsupported_execution_in(forward.symbol)]
print(json.dumps({'checkpoint': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'source_file': str(args.source), 'source_sha256': digest,
                  'forward': forward.symbol.qualified_name, 'bindings': bindings, 'loops': loops,
                  'unsupported_regions': unsupported,
                  'limit': 'Neutral syntax inventory; not an execution, target-binding or complete mechanism proof'},
                 indent=2, sort_keys=True))
