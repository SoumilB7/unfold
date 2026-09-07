"""UNet family cutover: isolated instance, existing readers, canonical facts/IR."""
from dataclasses import dataclass

from ...evidence.component_owner import resolve_component_root
from ...evidence.document import DocumentBinding, prepare_document
from ...evidence.instance_population_claim import read_instance_population, read_constructor_defaults
from ...evidence.instance_shape_claim import read_instance_shapes
from ...evidence.primitive_semantics import read_runtime_primitives
from ...evidence.runtime_inventory import build_resolved_instance
from ...evidence.unet_claims import (
    read_unet_ffn_claims, read_unet_join_claims, read_unet_stage_relations,
    read_unet_context_connections,
    read_unet_cell_arithmetic,
    read_unet_spatial_claims,
)
from ...evidence.unet_runtime import investigate_unet_runtime
from ...evidence.unet_cell_connections import read_unet_cell_connections
from ...evidence.unet_primary_ports import read_unet_primary_ports
from ...ir import ModelIR
from .unet_projection import project_unet


@dataclass(frozen=True)
class UNetCutoverResult:
    """Review handles accompany the canonical IR; no alternative IR is built."""

    ir: ModelIR | None
    inventory_result: object
    evidence: object = None


def build_unet_cutover(cfg, context, *, handoffs, name, source_overrides=()):
    binding = context.prepared_documents.get("root")
    if binding is None:
        binding = DocumentBinding("root", (), prepare_document(cfg, merge=False))
        context.prepared_documents["root"] = binding
    root = resolve_component_root(context.program_index(), context.source_bundle, "root")
    result = build_resolved_instance(binding.prepared, context.source_bundle, root,
                                     source_overrides=source_overrides)
    if result.status != "ok":
        return UNetCutoverResult(None, result)
    evidence = investigate_unet_runtime(
        model=name, inventory=result.inventory, observations=(),
        document=binding.prepared, bundle=context.source_bundle,
        index=context.program_index())
    execution = evidence.value("execution")
    if execution is None:
        return UNetCutoverResult(None, result, evidence)
    facts = [read_instance_population(evidence.bindings),
             read_constructor_defaults(evidence.bindings, binding.prepared),
             read_unet_spatial_claims(evidence.value("spatial"), evidence.bindings),
             read_instance_shapes(evidence.bindings),
             read_runtime_primitives(evidence.bindings),
             read_unet_context_connections(evidence.value("attention_sources"), evidence.bindings),
             read_unet_cell_arithmetic(evidence.value("mechanisms"), evidence.bindings,
                                      evidence.value("child_execution")),
             read_unet_cell_connections(evidence.value("mechanisms"), evidence.bindings,
                                        evidence.value("child_execution")),
             read_unet_stage_relations(execution, evidence.bindings),
             read_unet_primary_ports(execution, evidence.bindings),
             read_unet_ffn_claims(evidence.value("nested_ffns") or (), evidence.bindings),
             read_unet_join_claims(evidence.value("stage_joins") or (), evidence.bindings)]
    projected = {}
    for reader, reader_result in evidence.reader_results:
        context.reader_results[("root.denoiser.unet." + reader, ())] = reader_result
    for fact in facts:
        if fact is not None:
            context.facts.record_typed(fact)
            projected[fact.ledger_key()] = fact
    limitations = {}
    for module in result.inventory.modules:
        if evidence.bindings.symbol_at(module.path) is None and module.framework_primitive is None:
            limitations[module.path] = "investigation_missing · mechanism source unavailable in the indexed closure · owner: S8"
    qualified_ffns = projected.get("root.denoiser.ffn_mechanisms")
    for attempt in evidence.value("nested_ffns") or ():
        for path in attempt.instance_paths:
            if qualified_ffns is not None and path in qualified_ffns.value:
                continue
            reason = ("invoked affine/transparent callable witness unavailable or modified"
                      if attempt.result.has_value else
                      "; ".join(failure.detail for failure in attempt.result.failures))
            limitations[path] = "investigation_missing · mechanism investigation: " + reason + " · owner: S8"
    ir = project_unet(facts=projected, handoffs=handoffs, name=name,
                      architecture=result.inventory.provenance.resolved_class.qualname,
                      table=evidence.bindings.table, mechanism_findings=limitations)
    return UNetCutoverResult(ir, result, evidence)
