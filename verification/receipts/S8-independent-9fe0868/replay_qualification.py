"""Actual installed SDXL source proof → fact ledger → citation → archive.

Uses an existing typed inventory only to select/check source identity. No model
construction, execution, pytest or acceptance of a family execution receipt.
"""
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path.cwd();sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.reconciliation import reconcile,ProjectionFactCitation,FACT_CLAIM_REQUIREMENTS
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from model_unfolder.evidence.unet_primary_ports import read_unet_primary_ports,UNetPrimaryPortProof
from physics.instance_inventory import _inventory_from_dict
from demonstrate_s8_unet import _qualified_facts
receipt=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-independent-9fe0868')
fresh=ROOT/'verification/receipts/S8-authority-followup/independent/fresh-evidence'
raw=(fresh/'inventory.json.gz').read_bytes(); inventory=_inventory_from_dict(json.loads(gzip.decompress(raw)))
config=json.loads((fresh/'input.json').read_text());context=ParseContext.build(config)
bundle=context.source_bundle;index=context.program_index();root=resolve_component_root(index,bundle,'root')
topology=read_diffusion_root_topology(index,root).require_value()
construction=read_unet_stage_construction(index,bundle,root,topology).require_value()
graph=read_unet_stage_execution(construction,bundle,root).require_value()
table=reconcile(model='independent actual source primary proof',inventory=inventory,observations=(),config_document=prepare_document(config,merge=False),program_index=graph.index)
bindings=RuntimeSourceBindings(table,inventory,graph.index)
fact=read_unet_primary_ports(graph,bindings);assert fact is not None
summary=fact.claim_evidence.summary();assert summary.evidence_refs==tuple(sorted(set(summary.evidence_refs)))
assert len(summary.evidence_refs)>1
context.facts.record_typed(fact);citation=ProjectionFactCitation(fact);assert citation.summary==summary
assert FACT_CLAIM_REQUIREMENTS[fact.key]==fact.claim_kind=='connection'
try:dataclasses.replace(fact,claim_kind='existence')
except ValueError as error:kind_poison=str(error)
else:raise AssertionError('wrong claim kind accepted')
changed=dataclasses.replace(inventory,modules=tuple(dataclasses.replace(row,init_attributes={**row.init_attributes,'forward':{'type':'builtins.function','value':None}}) if row.path=='' else row for row in inventory.modules))
assert read_unet_primary_ports(graph,dataclasses.replace(bindings,inventory=changed)) is None
archive=receipt/'actual-primary-archive';archive.mkdir(exist_ok=False)
qualified=_qualified_facts(context,archive)
key=fact.ledger_key();record=qualified[key];assert not record['source_archive']['unmatched_hashes']
for fingerprint,path in record['source_archive']['artifacts'].items():assert hashlib.sha256(gzip.decompress((archive/path).read_bytes())).hexdigest()==fingerprint
source_hash=bindings.symbol_at('').source.content_fingerprint
runtime_hashes=[row.sha256 for row in inventory.provenance.source_files if row.module==inventory.provenance.resolved_class.module]
assert runtime_hashes==[source_hash]
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inventory_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':source_hash,'runtime_hashes':runtime_hashes,'primary_regions':len(fact.value['regions']),'reference_count':len(summary.evidence_refs),'summary':dataclasses.asdict(summary),'archive':record['source_archive'],'claim_kind_poison':kind_poison,'root_forward_override_refused':True,'scope':'actual static source proof and saved inventory identity only; no model or family execution acceptance'},indent=2,sort_keys=True))
