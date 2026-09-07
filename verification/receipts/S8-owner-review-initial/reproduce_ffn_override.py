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
altered = replace(module, init_attributes={**module.init_attributes,
                  'forward': {'type': 'builtins.function', 'value': None}})
changed_inventory = replace(inventory, modules=tuple(
    altered if row.path == path else row for row in inventory.modules))
after = read_unet_ffn_claims(attempts, replace(evidence.bindings, inventory=changed_inventory))
result = {
    'checkout_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'evidence': 'fresh static investigation against persisted ordinary SDXL inventory; then a DTO poison',
    'path': path, 'override_recorded': True, 'claim_survives': path in after.value,
    'facts_identical': before.value == after.value, 'before': before.value[path],
    'source_hash': evidence.bindings.symbol_at('').source.content_fingerprint,
    'inventory_root_hashes': [row.sha256 for row in inventory.provenance.source_files
                             if row.module == inventory.provenance.resolved_class.module],
}
assert result['claim_survives'] and result['facts_identical']
print(json.dumps(result, indent=2, sort_keys=True))
