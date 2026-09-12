"""Independent R5 replay against a fresh persisted worker inventory.

Run from the immutable review checkout. --inventory and --config identify
persisted evidence, not model downloads. This rebuilds static source evidence
and applies explicit DTO poisons; it does not execute the checkpoint model.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.unet_runtime import investigate_unet_runtime
from model_unfolder.evidence.unet_claims import read_unet_ffn_claims
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from physics.instance_inventory import _inventory_from_dict, ResolvedClass
from physics.framework_primitives import capture_framework_types, witness_framework_type

parser = argparse.ArgumentParser()
parser.add_argument('--inventory', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
raw = args.inventory.read_bytes()
inventory = _inventory_from_dict(json.loads(gzip.decompress(raw) if args.inventory.suffix == '.gz' else raw))
config = json.loads(args.config.read_text())
context = ParseContext.build(config)
evidence = investigate_unet_runtime(model='independent fresh R5 witness', inventory=inventory,
    observations=(), document=prepare_document(config, merge=False),
    bundle=context.source_bundle, index=context.program_index())
attempts = evidence.value('nested_ffns')
before = read_unet_ffn_claims(attempts, evidence.bindings)
assert before is not None and before.value, 'fresh ordinary must preserve positive FFN mechanisms'
path = next(iter(before.value))
input_path = before.value[path]['input_projection']
output_path = before.value[path]['output_projection']
dropouts = [row.path for row in inventory.modules if row.path.startswith(path + '.')
            and row.framework_primitive is not None and row.framework_primitive.key == 'dropout']
targets = list(dict.fromkeys([path, input_path.rpartition('.')[0], input_path, output_path, *dropouts]))
results = {}
for target in targets:
    changed = replace(inventory, modules=tuple(replace(row,
        init_attributes={**row.init_attributes, 'forward': {'type': 'builtins.function', 'value': None}})
        if row.path == target else row for row in inventory.modules))
    after = read_unet_ffn_claims(attempts, replace(evidence.bindings, inventory=changed))
    results[target + ':forward'] = {'claim_survives': path in after.value, 'remaining_ffns': len(after.value)}
    assert path not in after.value

import torch
relu_witness = witness_framework_type(torch.nn.ReLU(), capture_framework_types())
for target in (input_path, output_path, *dropouts):
    cls = ResolvedClass('torch.nn.modules.activation', 'ReLU')
    changed = replace(inventory, modules=tuple(replace(row, class_ref=cls, origin_module=cls.module,
        mro_entries=(cls, ResolvedClass('torch.nn.modules.module', 'Module'), ResolvedClass('builtins', 'object')),
        framework_primitive=relu_witness) if row.path == target else row for row in inventory.modules))
    table = reconcile(model='R5 coherent changed descendant', inventory=changed, observations=(),
                      config_document=prepare_document(config, merge=False), program_index=evidence.bindings.index)
    after = read_unet_ffn_claims(attempts, RuntimeSourceBindings(table, changed, evidence.bindings.index))
    results[target + ':type'] = {'claim_survives': path in after.value, 'remaining_ffns': len(after.value),
                               'class_ref': 'torch.nn.modules.activation.ReLU', 'framework_primitive': relu_witness.key}
    assert path not in after.value

unwitnessed = replace(inventory, modules=tuple(replace(row, framework_primitive=None) for row in inventory.modules))
limited = read_unet_ffn_claims(attempts, replace(evidence.bindings, inventory=unwitnessed))
assert limited is None or not limited.value
source_hash = evidence.bindings.symbol_at('').source.content_fingerprint
runtime_hashes = [row.sha256 for row in inventory.provenance.source_files
                  if row.module == inventory.provenance.resolved_class.module]
assert runtime_hashes == [source_hash]
print(json.dumps({
    'checkpoint': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'inventory_sha256': hashlib.sha256(raw).hexdigest(),
    'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
    'framework_witness_count': sum(row.framework_primitive is not None for row in inventory.modules),
    'ordinary_ffn_count': len(before.value), 'selected_path': path, 'ordinary_fact': before.value[path],
    'source_hash': source_hash, 'runtime_hashes': runtime_hashes,
    'missing_worker_witness_is_limited': True, 'results': results,
    'scope': 'fresh positive static investigation plus explicit DTO poisons; no checkpoint execution by reviewer',
}, indent=2, sort_keys=True))
