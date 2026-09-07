"""Re-read SDXL source, then poison the FFN instance callable witness.

Run from immutable initial checkout. Uses that checkout's persisted ordinary
inventory, rebuilds source evidence, and does not construct or execute a model.
No cached pickle, pytest, production mutation, or baseline write is involved.
"""
from dataclasses import replace
import gzip
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
from physics.instance_inventory import _inventory_from_dict

ordinary = ROOT / 'verification/receipts/S8-class-lookup-stop/ordinary'
config = json.loads((ordinary / 'input.json').read_text())
inventory = _inventory_from_dict(json.loads(gzip.decompress((ordinary / 'inventory.json.gz').read_bytes())))
context = ParseContext.build(config)
evidence = investigate_unet_runtime(
    model='independent SDXL FFN callable override poison', inventory=inventory, observations=(),
    document=prepare_document(config, merge=False), bundle=context.source_bundle,
    index=context.program_index())
attempts = evidence.value('nested_ffns')
before = read_unet_ffn_claims(attempts, evidence.bindings)
path = next(iter(before.value))
module = next(row for row in inventory.modules if row.path == path)
results = {}
for target in (path, path + '.net.0', path + '.net.0.proj', path + '.net.2'):
    changed_inventory = replace(inventory, modules=tuple(
        replace(row, init_attributes={**row.init_attributes, 'forward': {'type': 'builtins.function', 'value': None}})
        if row.path == target else row for row in inventory.modules))
    after = read_unet_ffn_claims(attempts, replace(evidence.bindings, inventory=changed_inventory))
    results[target] = {'kind': 'recorded_forward_override', 'claim_survives': path in after.value}
from physics.instance_inventory import ResolvedClass
from physics.framework_primitives import capture_framework_types, witness_framework_type
import torch
relu_witness = witness_framework_type(torch.nn.ReLU(), capture_framework_types())
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
for target in (path + '.net.0.proj', path + '.net.2'):
    cls = ResolvedClass('torch.nn.modules.activation', 'ReLU')
    changed_inventory = replace(inventory, modules=tuple(
        replace(row, class_ref=cls, origin_module=cls.module, mro_entries=(cls, ResolvedClass('torch.nn.modules.module','Module'), ResolvedClass('builtins','object')), framework_primitive=relu_witness)
        if row.path == target else row for row in inventory.modules))
    table = reconcile(model='independent FFN projection type poison', inventory=changed_inventory,
                      observations=(), config_document=prepare_document(config, merge=False),
                      program_index=evidence.bindings.index)
    after = read_unet_ffn_claims(attempts, RuntimeSourceBindings(table, changed_inventory, evidence.bindings.index))
    results[target + ':type'] = {'kind': 'actual_runtime_class_ReLU', 'framework_primitive_key': relu_witness.key, 'framework_forward_hash': relu_witness.forward_source_sha256, 'claim_survives': path in after.value}
result = {
    'checkpoint': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    'source_hash': evidence.bindings.symbol_at('').source.content_fingerprint,
    'inventory_root_hashes': [row.sha256 for row in inventory.provenance.source_files
                             if row.module == inventory.provenance.resolved_class.module],
    'evidence': 'fresh static investigation against persisted ordinary SDXL inventory, then explicit DTO poisons; class changes regenerate reconciliation',
    'ordinary_inventory_framework_witnesses': sum(row.framework_primitive is not None for row in inventory.modules), 'ordinary_ffn_count': len(before.value), 'path': path, 'ordinary_fact': before.value[path],
    'results': results,
}
assert not results[path]['claim_survives']
assert not results[path + '.net.0']['claim_survives']
assert results[path + '.net.0.proj']['claim_survives']
assert results[path + '.net.2']['claim_survives']
assert results[path + '.net.0.proj:type']['claim_survives']
assert results[path + '.net.2:type']['claim_survives']
print(json.dumps(result, indent=2, sort_keys=True))
