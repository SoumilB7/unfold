"""Exact multimodal projector/merger evidence from qualified HF source."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .component_owner import resolve_component_root
from .models import ProjectorEvidence, SourceBundle, SourceOp
from .sources import resolve_source_files
from .program_index import ProgramIndex
from .projector_lineage import projector_lineage_result
from .projector_chain import ACTIVATION_REGISTRY_PROTOCOLS
from .construction_calls import resolve_import_reference
from .expression_eval import constructor_argument_env
from .ffn_mechanism import ffn_mechanism_at_block
from .repeated_projector import (
    repeated_projector_pipeline_at_owner,
    repeated_projector_pipeline_for_candidate,
)
from .projector_width import ProjectorWidthEvidence, WidthOperand, projector_width_evidence
from .reader_result import ReaderFailure, ReaderResult

@dataclass(frozen=True)
class ProjectorEvidenceInventory:
    """Per-destination projector facts proven from exact fusion operands."""

    projectors: tuple[ProjectorEvidence, ...]

    def __post_init__(self):
        if not self.projectors:
            raise ValueError("a resolved projector inventory is non-empty")
        modalities = tuple(
            modality for item in self.projectors for modality in item.modalities)
        if any(item.status != "proven" or not item.modalities
               for item in self.projectors):
            raise ValueError("inventory entries are proven and destination-qualified")
        if len(modalities) != len(set(modalities)):
            raise ValueError("one destination modality has one projector mechanism")

    def to_dict(self):
        return {"projectors": [item.to_dict() for item in self.projectors]}


def projector_evidence(target: Any, *, source: str = "local",
                       bundle: SourceBundle | None = None,
                       index: ProgramIndex | None = None,
                       parse_context=None,
                       config_selector=None) -> ProjectorEvidence:
    """Compatibility projection of the exact producer-lineage result.

    This public entry point no longer has a second whole-file AST authority.
    Supplying only a config still builds the same ProgramIndex and exact owner
    graph used by parser/conformance.
    """
    if parse_context is not None:
        from .context import ParseContext
        if not isinstance(parse_context, ParseContext):
            raise TypeError("parse_context must be a ParseContext")
        return _projector_result_value(projector_result_for_context(
            parse_context,
            config_document=target,
            config_selector=config_selector or _document_selector(target)))
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return ProjectorEvidence("oracle_missing", reason="no modeling source")
    from .program_index import build_program_index
    result = projector_result(
        index or build_program_index(bundle), bundle,
        config_document=target,
        config_selector=config_selector or _document_selector(target))
    return _projector_result_value(result)


def projector_result_for_context(context, *, config_document=None,
                                 config_selector=None):
    """The one call-local projector result shared by parser and conformance."""
    from .context import ParseContext
    if not isinstance(context, ParseContext):
        raise TypeError("projector_result_for_context requires a ParseContext")
    key = ("root.projector", ())
    result = context.reader_results.get(key)
    if result is None:
        if config_selector is None and config_document is not None:
            config_selector = _document_selector(config_document)
        result = projector_result(
            context.program_index(), context.source_bundle,
            config_document=config_document,
            config_selector=config_selector)
        context.reader_results[key] = result
    return result


def projector_result_for_target(target, *, source="local", bundle=None,
                                index=None):
    """Build the same typed result when no call-local parse context exists."""
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return ReaderResult.failed(None, (ReaderFailure(
            "source_missing", "no modeling source"),))
    if index is None:
        from .program_index import build_program_index
        index = build_program_index(bundle)
    return projector_result(
        index, bundle, config_document=target,
        config_selector=_document_selector(target))


def projector_result(index: ProgramIndex, bundle: SourceBundle, *,
                     config_document=None, config_selector=None):
    """Resolve per-destination projector facts from exact fusion operands."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("projector_result requires ProgramIndex + SourceBundle")
    lineage = projector_lineage_result(
        index, bundle, config_selector=config_selector)
    if lineage.status == "absent":
        return ReaderResult.absent(lineage.owner)
    if lineage.status == "ambiguous":
        return ReaderResult.ambiguous(lineage.owner, lineage.ambiguity)
    if lineage.status == "failed":
        return ReaderResult.failed(lineage.owner, lineage.failures)

    root = resolve_component_root(index, bundle, "root")
    candidates = lineage.value.candidates
    groups = {}
    for candidate in candidates:
        signature = tuple(
            (op.kind, op.label, op.fn) for op in candidate.chain.operations)
        groups.setdefault(signature, []).append(candidate)
    projectors = tuple(
        _projector_evidence_for_candidates(
            index, root, tuple(group),
            nested_config_addresses=lineage.value.nested_config_addresses,
            config_document=config_document,
            config_selector=config_selector)
        for _signature, group in groups.items())
    inventory = ProjectorEvidenceInventory(projectors)
    if lineage.status == "incomplete":
        return ReaderResult.incomplete(
            lineage.owner, inventory, failures=lineage.failures,
            provenance=lineage.provenance)
    return ReaderResult.resolved(
        lineage.owner, inventory, provenance=lineage.provenance)


def _projector_evidence_for_candidates(
    index, root, candidates, *, nested_config_addresses=(),
    config_document=None, config_selector=None,
):
    chains = tuple(item.chain for item in candidates)
    widths = tuple(projector_width_evidence(
                       index, item.owner_graph, item,
                       nested_config_addresses=nested_config_addresses)
                   for item in candidates)
    width = ProjectorWidthEvidence(
        _common_width_operand(widths, "input"),
        _common_width_operand(widths, "output"),
    )
    caller_names = tuple(dict.fromkeys(
        root.graph.node_for(item.caller_occurrence).symbol.qualified_name
        for item in candidates))
    fields = tuple(dict.fromkeys(item.field for item in candidates))
    projector_names = tuple(dict.fromkeys(
        (item.chain.owner_symbol.qualified_name
         if item.constructed_occurrence is not None
         # A primitive has no constructed class occurrence of its own.  Keep
         # the compatibility/display field at the operation's code-derived
         # label; the fully-qualified primitive remains in SourceOp.fn.
         else item.chain.operations[-1].label)
        for item in candidates))
    repeated = tuple(
        (repeated_projector_pipeline_at_owner(
            index, root, item.constructed_occurrence,
            config_document=config_document,
            config_selector=config_selector)
         if item.constructed_occurrence is not None
         else repeated_projector_pipeline_for_candidate(
             index, root, item,
             config_document=config_document,
             config_selector=config_selector))
        for item in candidates)
    repeated_values = tuple(
        item.value for item in repeated if item.status == "resolved")
    if repeated_values and len(repeated_values) != len(candidates):
        return ProjectorEvidence(
            "ambiguous",
            modalities=tuple(dict.fromkeys(
                modality for item in candidates
                for modality in item.destination_modalities)),
            reason=("equivalent affine producers disagree on whether an exact "
                    "repeated stage follows them"),
        )
    if repeated_values:
        signatures = tuple(_repeated_signature(item) for item in repeated_values)
        if any(item != signatures[0] for item in signatures[1:]):
            return ProjectorEvidence(
                "ambiguous",
                modalities=tuple(dict.fromkeys(
                    modality for item in candidates
                    for modality in item.destination_modalities)),
                reason="destination-equivalent repeated projectors disagree",
            )
        ops = _repeated_projector_ops(
            index, root, repeated_values[0],
            config_document=config_document,
            config_selector=config_selector)
        learned_queries = all(
            item.learned_query is not None for item in repeated_values)
    else:
        ops = chains[0].operations
        learned_queries = False
    return ProjectorEvidence(
        "proven",
        modalities=tuple(dict.fromkeys(
            modality for item in candidates
            for modality in item.destination_modalities)),
        owner_class=caller_names[0] if len(caller_names) == 1 else "",
        field_name=fields[0] if len(fields) == 1 else "",
        projector_class=(
            root.graph.node_for(repeated_values[0].owner_occurrence)
                .symbol.qualified_name
            if repeated_values
            else projector_names[0] if len(projector_names) == 1
            else "Code-defined projector"),
        source_file=ops[0].source_file,
        line=ops[0].line,
        ops=ops,
        kind=("perceiver_resampler" if learned_queries
              else "repeated_attention_projector" if repeated_values
              else _derive_kind(list(ops))),
        learned_queries=learned_queries,
        out_width_source=width.output.source,
        out_width_path=width.output.path,
        out_width_value=width.output.value,
        in_width_source=width.input.source,
        in_width_path=width.input.path,
        in_width_value=width.input.value,
    )


def _repeated_signature(pipeline):
    mechanism = pipeline.mechanisms
    attention = mechanism.attention.compute.protocol
    ffn = mechanism.ffn
    return (
        tuple((item.kind, item.label, item.fn) for item in pipeline.operations),
        attention,
        getattr(ffn, "gated", None),
        mechanism.block_norm_kind,
        mechanism.final_norm_kind,
        mechanism.count_config_path,
        pipeline.learned_query is not None,
    )


def _repeated_projector_ops(
    index, root, pipeline, *, config_document=None, config_selector=None,
):
    mechanism = pipeline.mechanisms
    count = "N"
    if mechanism.count_config_path and config_selector is not None:
        present, value, _status = config_selector(mechanism.count_config_path)
        if present and isinstance(value, int) and not isinstance(value, bool) \
                and value > 0:
            count = value
    ffn = mechanism.ffn
    ffn_label = "gated FFN" if getattr(ffn, "gated", None) is True else "FFN"
    attention_label = "attention"
    description = (
        f"Repeated {attention_label} + {ffn_label} stage with "
        f"{_norm_display(mechanism.block_norm_kind)} normalization; "
        "the exact repeat count is bound only when its source-named config "
        "operand is present."
    )
    stage_span = mechanism.spans[0]
    stage = SourceOp(
        "opaque", f"Repeated {attention_label} + {ffn_label} stage",
        mechanism.stage_symbol.qualified_name,
        stage_span.source.canonical_path, stage_span.line,
        repeat=count, description=description,
    )
    final_span = mechanism.spans[-1]
    final_norm = SourceOp(
        "norm", _norm_display(mechanism.final_norm_kind),
        mechanism.stage_symbol.qualified_name,
        final_span.source.canonical_path, final_span.line,
    )
    prefix_ops = tuple(
        operation
        for prefix in pipeline.prefixes
        for operation in _qualified_prefix_ops(
            index, root, prefix,
            config_document=config_document,
            config_selector=config_selector))
    return (*prefix_ops, stage, final_norm)


def _qualified_prefix_ops(
    index, root, prefix, *, config_document=None, config_selector=None,
):
    """Retain exact gated-FFN lane labels for an affine prefix."""
    ops = list(prefix.operations)
    if prefix.callee_occurrence is None:
        return tuple(ops)

    def select_value(path):
        if config_selector is None:
            return None
        selected = config_selector(path)
        return selected[1] if selected[0] else None

    mechanism = ffn_mechanism_at_block(
        index, root, prefix.callee_occurrence,
        config_selector=select_value if config_selector is not None else None)
    if mechanism.status != "resolved" \
            or getattr(mechanism.value, "gated", None) is not True:
        return tuple(ops)
    linear = [position for position, item in enumerate(ops)
              if item.kind == "linear"]
    kinds = tuple(item.kind for item in ops)
    if len(linear) != 3 or "activation" not in kinds \
            or "elementwise" not in kinds:
        return tuple(ops)
    for position, label in zip(
            linear, ("Linear (gate)", "Linear (up)", "Linear (out)")):
        ops[position] = _replace_op_label(ops[position], label)
    # A gated FFN is a two-lane graph, not a misleading serial list.  The exact
    # dataflow proof above establishes gate/up/output roles; present both input
    # projections before the gate activation and merge.
    activation_positions = [position for position, item in enumerate(ops)
                            if item.kind == "activation"]
    merge_positions = [position for position, item in enumerate(ops)
                       if item.kind == "elementwise"]
    if len(activation_positions) == 1 and len(merge_positions) == 1:
        gate, up, output = (ops[position] for position in linear)
        activation_op = ops[activation_positions[0]]
        merge = ops[merge_positions[0]]
        ops = [gate, up, activation_op, merge, output]
    activation = _constructor_dispatched_activation(
        index, root, prefix.callee_occurrence, config_document)
    if activation is not None:
        for position, item in enumerate(ops):
            if item.kind == "activation":
                ops[position] = SourceOp(
                    item.kind, activation, item.class_name,
                    item.source_file, item.line, fn=activation.lower(),
                    repeat=item.repeat, description=item.description,
                    op_id=item.op_id, inputs=item.inputs)
    return tuple(ops)


def _constructor_dispatched_activation(
    index, root, occurrence, config_document,
):
    """Resolve an exact ACT2FN constructor operand without field vocabulary."""
    if config_document is None:
        return None
    node = root.graph.node_for(occurrence)
    if node is None:
        return None
    env = constructor_argument_env(
        index, root.graph, occurrence, config_document)
    forward = type(node.symbol)(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    invoked = {
        _expr_self_field(call.callee) for call in index.calls_in(forward)
        if _expr_self_field(call.callee) is not None
    }
    values = []
    for assignment in index.field_assigns_of(node.symbol):
        expression = assignment.value
        if assignment.field not in invoked or expression.kind != "subscript" \
                or len(expression.children) != 2:
            continue
        proof = resolve_import_reference(
            index, node.symbol.source, assignment.enclosing_callable,
            expression.children[0])
        operand = expression.children[1]
        if proof is None \
                or proof.qualified_target not in ACTIVATION_REGISTRY_PROTOCOLS \
                or operand.kind != "name" or operand.name not in env:
            continue
        resolved = env[operand.name]
        if isinstance(resolved.value, str) and resolved.value:
            values.append(resolved.value)
    return values[0] if values and len(set(values)) == 1 else None


def _replace_op_label(op, label):
    return SourceOp(
        op.kind, label, op.class_name, op.source_file, op.line,
        fn=op.fn, repeat=op.repeat, description=op.description,
        op_id=op.op_id, inputs=op.inputs)


def _expr_self_field(expression):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return expression.name \
        if base.kind == "name" and base.name == "self" else None


def _norm_display(kind):
    return "RMSNorm" if kind == "rmsnorm" else "LayerNorm"


def _common_width_operand(widths, attribute):
    if not widths:
        return WidthOperand("unavailable")
    values = tuple(getattr(item, attribute) for item in widths)
    return values[0] if all(item == values[0] for item in values[1:]) \
        else WidthOperand("unavailable")


def _document_selector(target):
    """Read one exact path for branch selection; no mechanism semantics."""
    def select(path):
        current = target
        for part in tuple(path):
            if isinstance(current, dict):
                if part not in current:
                    return False, None, ""
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return False, None, ""
        return True, current, "config_declared"
    return select


def _projector_result_value(result):
    if result.status == "resolved":
        if len(result.value.projectors) == 1:
            return result.value.projectors[0]
        return ProjectorEvidence(
            "ambiguous", reason="multiple destination-specific projectors")
    if result.status == "incomplete":
        projectors = result.value.projectors
        return ProjectorEvidence(
            "ambiguous",
            owner_class=(projectors[0].owner_class
                         if len(projectors) == 1 else ""),
            source_file=(projectors[0].source_file
                         if len(projectors) == 1 else ""),
            reason="; ".join(item.detail for item in result.failures))
    if result.status == "ambiguous":
        return ProjectorEvidence(
            "ambiguous", reason="multiple non-equivalent exact projector producers")
    if result.status == "absent":
        return ProjectorEvidence(
            "ambiguous", reason="no affine producer reaches a proven fusion operand")
    return ProjectorEvidence(
        "oracle_missing", reason="; ".join(item.detail for item in result.failures))


def _derive_kind(ops: list[SourceOp], *, learned_queries: bool = False) -> str:
    kinds = [op.kind for op in ops]
    if learned_queries and any(op.repeat is not None for op in ops):
        return "perceiver_resampler"
    if "reshape" in kinds and (kinds.count("linear") or "norm" in kinds):
        return "patch_merger"
    if kinds.count("linear") >= 2:
        return "mlp_projector"
    if kinds.count("linear") == 1 and set(kinds) <= {"norm", "linear"}:
        return "linear_projector"
    return "code_defined_projector"



__all__ = [
    "ProjectorEvidenceInventory", "projector_evidence", "projector_result",
    "projector_result_for_context", "projector_result_for_target",
]
