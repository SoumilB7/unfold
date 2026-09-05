"""Exact repeated auxiliary next-token predictor evidence.

The config count is only a repetition operand.  A positive result requires one
exact model-stage container whose repeated element executes this complete
source path:

    norm(hidden) || norm(shared embedding) -> concat -> projection
      -> exact constructed block -> exact shared output head

Both sharing claims are address proofs: the predictor receives the model
stage's constructed embedding output and stores/calls the stage's constructed
output-head field.  Names, model families and count fields never classify the
mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

from .construction_calls import resolve_import_reference
from .container_inventory import ContainerAddress, resolve_container_inventory
from .decoder_stage import DecoderStagePath, decoder_stage_for_config
from .execution_flow import RepeatedInvocationTemplate, resolve_addressed_invocations
from .program_index import (
    CallObservation, ProgramIndex, SourceSpan, SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR = frozenset({"torch.nn.Linear", "torch.nn.modules.linear.Linear"})
_EMBEDDING = frozenset({
    "torch.nn.Embedding", "torch.nn.modules.sparse.Embedding",
})
_NORM = frozenset({
    "torch.nn.LayerNorm", "torch.nn.modules.normalization.LayerNorm",
    "torch.nn.RMSNorm", "torch.nn.modules.normalization.RMSNorm",
})
_CONCAT = frozenset({"torch.cat", "torch.concat", "torch.concatenate"})


@dataclass(frozen=True)
class MTPModuleProof:
    container: ContainerAddress
    invocation: RepeatedInvocationTemplate
    hidden_norm: CallObservation
    embedding_norm: CallObservation
    concat: CallObservation
    projection: CallObservation
    block: CallObservation
    output_head: CallObservation
    hidden_norm_kind: str
    embedding_norm_kind: str
    block_symbol: SymbolId
    shares_embedding: bool
    shares_output_head: bool
    count_path: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.invocation.container != self.container:
            raise ValueError("MTP invocation cites its exact container")
        calls = (self.hidden_norm, self.embedding_norm, self.concat,
                 self.projection, self.block, self.output_head)
        if len(set(calls)) != len(calls):
            raise ValueError("MTP operation calls are distinct")
        if not self.count_path or any(not item for item in self.count_path):
            raise ValueError("MTP proof carries one exact count operand")
        if any(call.span not in self.spans for call in calls):
            raise ValueError("MTP provenance cites every operation")
        if self.hidden_norm_kind not in {"LayerNorm", "RMSNorm"} \
                or self.embedding_norm_kind not in {"LayerNorm", "RMSNorm"}:
            raise ValueError("MTP norms carry exact framework kinds")
        if not isinstance(self.block_symbol, SymbolId):
            raise TypeError("MTP block carries an exact class symbol")
        if not isinstance(self.shares_embedding, bool) \
                or not isinstance(self.shares_output_head, bool):
            raise TypeError("MTP sharing facts are exact booleans")


@dataclass(frozen=True)
class MTPConstructionEvidence:
    stage: DecoderStagePath
    modules: MTPModuleProof
    shares_embedding: bool
    shares_output_head: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DecoderStagePath) \
                or not isinstance(self.modules, MTPModuleProof):
            raise TypeError("MTP evidence carries exact stage and module proofs")
        if self.modules.container.owner_occurrence != self.stage.stage_occurrence:
            raise ValueError("the MTP container belongs to the exact stage")
        if (self.shares_embedding, self.shares_output_head) != (
                self.modules.shares_embedding,
                self.modules.shares_output_head):
            raise ValueError("MTP construction copies the module sharing proof")

    @property
    def count_path(self) -> tuple[str, ...]:
        return self.modules.count_path


def decoder_mtp_construction_for_path(
    index: ProgramIndex, bundle, config_path: tuple[str, ...], *,
    allow_root_stage: bool,
) -> ReaderResult[MTPConstructionEvidence]:
    stage_result = decoder_stage_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if stage_result.status != "resolved":
        if stage_result.status == "ambiguous":
            return ReaderResult.ambiguous(
                stage_result.owner, stage_result.ambiguity,
                provenance=stage_result.provenance)
        if stage_result.status == "absent":
            return ReaderResult.absent(
                stage_result.owner, provenance=stage_result.provenance)
        return ReaderResult.failed(
            stage_result.owner, stage_result.failures or (ReaderFailure(
                "incomplete_graph", "model-stage address is unresolved"),),
            provenance=stage_result.provenance)
    stage = stage_result.value
    root = stage.component_root
    stage_node = root.graph.node_for(stage.stage_occurrence)
    inventory = resolve_container_inventory(
        index, root, stage.stage_occurrence)
    if stage_node is None or inventory.status == "failed":
        return _failed(stage, "stage_inventory_unavailable")
    if inventory.status == "absent" or inventory.rivals:
        return ReaderResult.absent(stage.stage_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, stage.stage_occurrence, inventory)
    if invocations.status != "resolved":
        return _failed(stage, "stage_invocations_unavailable")

    proofs = []
    for template in invocations.templates:
        proof = _module_proof(
            index, stage_node, template, invocations.templates,
            config_path)
        if proof is not None:
            proofs.append(proof)
    if not proofs:
        return ReaderResult.absent(stage.stage_occurrence)
    if len(proofs) != 1:
        return _failed(stage, "rival_auxiliary_predictor_containers")
    proof = proofs[0]
    value = MTPConstructionEvidence(
        stage, proof, proof.shares_embedding, proof.shares_output_head)
    spans = tuple(dict.fromkeys((*stage.address_spans, *proof.spans)))
    return ReaderResult.resolved(
        stage.stage_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=("exact repeated auxiliary module: two shared-input norms, "
                    "concat, projection, block and shared output head")),))


def _failed(stage, detail):
    return ReaderResult.failed(stage.stage_occurrence, (
        ReaderFailure("incomplete_graph", detail),))


def _module_proof(index, stage_node, template, all_templates, prefix):
    container = template.container
    if container.count_config_path is None:
        return None
    count_parts = tuple(segment.name
                        for segment in container.count_config_path.segments)
    if not count_parts or any(segment.dynamic
                              for segment in container.count_config_path.segments):
        return None
    count_path = (*prefix, *count_parts)
    site = template.element_template
    if len(site.candidates) != 1 or site.candidates[0].symbol is None:
        return None
    child_symbol = site.candidates[0].symbol
    child_nodes = [node for node in stage_node.children
                   if node.via_site == site.site_id and node.symbol == child_symbol]
    if len(child_nodes) != 1:
        return None
    forward = _unique_method(index, child_symbol, "forward")
    initializer = _unique_method(index, child_symbol, "__init__")
    if forward is None or initializer is None:
        return None
    formals = tuple(item.name for item in forward.params if item.name != "self")
    if len(formals) < 2 or len(template.call.args) < len(formals):
        return None

    sites = tuple(index.construction_sites_of(child_symbol))
    protocol = {item.target: _site_protocol(index, item) for item in sites}
    calls = tuple(index.calls_in(forward.symbol))
    bindings = tuple(index.bindings_in(forward.symbol))
    norm_calls = [call for call in calls
                  if protocol.get(_self_field(call.callee)) in _NORM
                  and len(call.args) == 1]
    if len(norm_calls) != 2:
        return None
    norm_formals = []
    for call in norm_calls:
        names = [name for name in formals if _contains_name(call.args[0], name)]
        if len(names) != 1:
            return None
        norm_formals.append(names[0])
    if len(set(norm_formals)) != 2:
        return None
    norm_kinds = {
        call: _norm_kind(protocol.get(_self_field(call.callee)))
        for call in norm_calls
    }
    if any(kind is None for kind in norm_kinds.values()):
        return None

    concats = [call for call in calls if _import_target(index, call) in _CONCAT
               and all(_call_reaches(bindings, norm, call) for norm in norm_calls)]
    if len(concats) != 1:
        return None
    concat = concats[0]
    internal_sites = tuple(
        item for item in sites
        if len(item.candidates) == 1
        and item.candidates[0].symbol is not None)
    internal_fields = {item.target for item in internal_sites}
    blocks = [call for call in calls
              if _self_field(call.callee) in internal_fields
              and call.lexical_order > concat.lexical_order
              and _call_reaches(bindings, concat, call)]
    if len(blocks) != 1:
        return None
    block = blocks[0]
    projections = [call for call in calls
                   if protocol.get(_self_field(call.callee)) in _LINEAR
                   and concat.lexical_order < call.lexical_order
                   < block.lexical_order
                   and _call_reaches(bindings, concat, call)
                   and _call_reaches(bindings, call, block)]
    if len(projections) != 1:
        return None
    projection = projections[0]

    alias_fields = _parameter_alias_fields(index, initializer)
    stage_head_fields = _stage_output_head_fields(index, stage_node.symbol)
    heads = []
    for call in calls:
        field = _self_field(call.callee)
        if call.lexical_order <= block.lexical_order \
                or not _call_reaches(bindings, block, call):
            continue
        shared = field in alias_fields
        own = protocol.get(field) in _LINEAR
        if shared or own:
            heads.append((call, shared))
    if len(heads) != 1 or not any(
            _expr_reaches_call(bindings, item.value, heads[0][0], item.span.line)
            for item in index.return_observations_in(forward.symbol)
            if item.value is not None):
        return None
    head, shares_head = heads[0]
    if shares_head:
        head_param = alias_fields[_self_field(head.callee)]
        init_params = tuple(item.name for item in initializer.params
                            if item.name != "self")
        if head_param not in init_params:
            return None
        head_position = init_params.index(head_param)
        if head_position >= len(site.args):
            return None
        stage_head_field = _self_field(site.args[head_position])
        if stage_head_field is None or stage_head_field not in stage_head_fields:
            return None

    # The norm formal whose repeated-call argument descends from the stage's
    # exact embedding construction is the shared embedding lane.
    block_sites = tuple(
        item for item in internal_sites if item.target == _self_field(block.callee))
    if len(block_sites) != 1:
        return None
    block_symbol = block_sites[0].candidates[0].symbol
    main_templates = tuple(
        item for item in all_templates
        if item is not template
        and item.element_template.candidates[0].symbol == block_symbol)
    if len(main_templates) != 1 or not any(
            child.via_site == main_templates[0].element_template.site_id
            and child.symbol == block_symbol for child in stage_node.children):
        return None
    main_call = main_templates[0].call
    stage_bindings = tuple(index.bindings_in(template.call.enclosing_callable))
    stage_returns = tuple(index.return_observations_in(
        template.call.enclosing_callable))
    if not any(
            item.value is not None
            and _expr_reaches_call(
                stage_bindings, item.value, template.call, item.span.line)
            for item in stage_returns):
        return None
    embedding_pairs = []
    for formal, norm in zip(norm_formals, norm_calls):
        position = formals.index(formal)
        argument = template.call.args[position]
        field = _stage_call_field(index, template.call.enclosing_callable,
                                  argument, template.call.span.line)
        if field is not None and _stage_field_protocol(
                index, stage_node.symbol, field, _EMBEDDING):
            embedding_pairs.append((formal, norm, True))
            continue
        own_field = _self_call_field_reaching(
            bindings, norm.args[0], norm.span.line)
        if own_field is not None and protocol.get(own_field) in _EMBEDDING:
            embedding_pairs.append((formal, norm, False))
    if len(embedding_pairs) != 1:
        return None
    embedding_formal, embedding_norm, shares_embedding = embedding_pairs[0]
    hidden_norm = norm_calls[1] if norm_calls[0] == embedding_norm else norm_calls[0]
    hidden_position = formals.index(
        norm_formals[norm_calls.index(hidden_norm)])
    if not _expr_reaches_call(
            stage_bindings,
            template.call.args[hidden_position], main_call,
            template.call.span.line):
        return None
    spans = tuple(dict.fromkeys((
        container.record.span, site.span, template.call.span,
        hidden_norm.span, embedding_norm.span, concat.span, projection.span,
        block.span, head.span,
    )))
    if None in spans:
        return None
    return MTPModuleProof(
        container, template, hidden_norm, embedding_norm, concat,
        projection, block, head, norm_kinds[hidden_norm],
        norm_kinds[embedding_norm], block_symbol,
        shares_embedding, shares_head, count_path, spans)


def _unique_method(index, owner, leaf):
    values = tuple(item for item in index.callables_of(owner)
                   if item.symbol.qualified_name.rsplit(".", 1)[-1] == leaf)
    return values[0] if len(values) == 1 else None


def _parameter_alias_fields(index, initializer):
    params = {item.name for item in initializer.params if item.name != "self"}
    out = {}
    for item in index.field_assigns_of(initializer.owner):
        if item.enclosing_callable != initializer.symbol:
            continue
        if item.value.kind == "name" and item.value.name in params:
            out[item.field] = item.value.name
    return out


def _site_protocol(index, site):
    if len(site.candidates) != 1:
        return None
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return proof.qualified_target if proof is not None else None


def _stage_field_protocol(index, owner, field, protocols):
    sites = tuple(item for item in index.construction_sites_of(owner)
                  if item.target == field)
    return len(sites) == 1 and _site_protocol(index, sites[0]) in protocols


def _stage_output_head_fields(index, owner):
    """Exact stage fields that are framework Linear calls in a return value."""
    forward = _unique_method(index, owner, "forward")
    if forward is None:
        return frozenset()
    constructed = {
        site.target for site in index.construction_sites_of(owner)
        if _site_protocol(index, site) in _LINEAR}
    fields = set()
    for returned in index.return_observations_in(forward.symbol):
        if returned.value is None:
            continue
        for call in _calls_in_expr(returned.value):
            field = (_self_field(call.children[0])
                     if call.kind == "call" and call.children else None)
            if field in constructed:
                fields.add(field)
    return frozenset(fields)


def _calls_in_expr(expression):
    out = []
    if expression.kind == "call":
        out.append(expression)
    for child in expression.children:
        if child is not None:
            out.extend(_calls_in_expr(child))
    return tuple(out)


def _self_call_field_reaching(bindings, expression, before):
    """Return the unique direct ``self.field(...)`` producer reaching expr."""
    resolved = _latest_expression(bindings, expression, before)
    calls = _calls_in_expr(resolved)
    fields = tuple(dict.fromkeys(
        field for call in calls
        if call.children
        and (field := _self_field(call.children[0])) is not None))
    return fields[0] if len(fields) == 1 else None


def _norm_kind(protocol):
    if protocol in {
            "torch.nn.LayerNorm", "torch.nn.modules.normalization.LayerNorm"}:
        return "LayerNorm"
    if protocol in {
            "torch.nn.RMSNorm", "torch.nn.modules.normalization.RMSNorm"}:
        return "RMSNorm"
    return None


def _stage_call_field(index, callable_symbol, expression, before):
    resolved = _latest_expression(
        tuple(index.bindings_in(callable_symbol)), expression, before)
    if resolved.kind != "call" or not resolved.children:
        return None
    return _self_field(resolved.children[0])


def _latest_expression(bindings, expression, before):
    if expression.kind != "name":
        return expression
    matches = [item for item in bindings if item.span.line < before
               and expression.name in _target_names(item.targets)]
    return (_latest_expression(bindings, matches[-1].value,
                               matches[-1].span.line)
            if matches and matches[-1].value is not None else expression)


def _target_names(targets):
    out = set()
    def walk(expr):
        if expr.kind == "name" and expr.name:
            out.add(expr.name)
        for child in expr.children:
            if child is not None:
                walk(child)
    for target in targets:
        walk(target)
    return out


def _call_reaches(bindings, producer, consumer):
    return any(_expr_reaches_call(
        bindings, arg, producer, consumer.span.line) for arg in consumer.args)


def _expr_reaches_call(bindings, expression, producer, before):
    if expression is None:
        return False
    if expression.kind == "call" and expression.span == producer.span:
        return True
    if expression.kind == "name":
        matches = [item for item in bindings if item.span.line < before
                   and expression.name in _target_names(item.targets)]
        if matches and matches[-1].value is not None:
            return _expr_reaches_call(
                bindings, matches[-1].value, producer, matches[-1].span.line)
    return any(child is not None and _expr_reaches_call(
        bindings, child, producer, before) for child in expression.children)


def _import_target(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else None


def _self_field(expression):
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = expression.children[0]
        if base is not None and base.kind == "name" and base.name == "self":
            return expression.name
    return None


def _contains_name(expression, name):
    return (expression.kind == "name" and expression.name == name) or any(
        child is not None and _contains_name(child, name)
        for child in expression.children)


__all__ = [
    "MTPModuleProof", "MTPConstructionEvidence",
    "decoder_mtp_construction_for_path",
]
