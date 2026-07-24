"""U3-F — Q/K normalization from one exact attention occurrence.

This reader starts at the shared selected-config -> decoder-block address and
the code-proven attention child.  It proves only positive Q/K normalization:

* an exact norm construction is classified from its implementation/protocol;
* two exact norm *application sites* must reach the score's Q/K operands;
* each norm input must itself descend from an exact Linear construction; and
* every construction/application guard must resolve to exact config paths.

Failure to find that positive chain is unknown evidence, never proof that Q/K
normalization is absent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    attention_child_evidence,
)
from .attention_storage import (
    attention_projection_storage_for_child_evidence,
    producer_sources_reaching_expressions,
)
from .construction_calls import (
    ConstructionCallResolution,
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .primitive_semantics import classify_primitive_call
from .program_index import (
    CallObservation,
    CallSiteId,
    GuardStep,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import (
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_SDPA_PROTOCOLS = frozenset({
    "torch.nn.functional.scaled_dot_product_attention",
})
_SOFTMAX_PROTOCOLS = frozenset({
    "torch.nn.functional.softmax",
    "torch.softmax",
})
_DOT_PROTOCOLS = frozenset({
    "torch.bmm",
    "torch.einsum",
    "torch.matmul",
})


@dataclass(frozen=True, order=True)
class QKNormGateAtom:
    """One exact config field controlling the positive Q/K-norm chain."""

    field: str
    config_path: tuple[str, ...]
    per_layer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise TypeError("a QK-norm gate atom names one non-empty config field")
        if not isinstance(self.config_path, tuple) or not self.config_path \
                or any(not isinstance(part, str) or not part
                       for part in self.config_path):
            raise TypeError(
                "a QK-norm gate atom carries one exact non-empty config path")
        if self.config_path[-1] != self.field:
            raise ValueError(
                "the QK-norm gate field is the exact path's final segment")
        if not isinstance(self.per_layer, bool):
            raise TypeError("per_layer is a boolean code-shape fact")


@dataclass(frozen=True)
class QKNormCodeEvidence:
    """A positive unconditional or config-gated Q/K normalization proof."""

    present: bool | None
    gate: tuple[QKNormGateAtom, ...] = ()

    def __post_init__(self) -> None:
        if self.present not in {True, None}:
            raise ValueError(
                "owner-bound QK evidence proves present/gated, never absent")
        if self.present is True and self.gate:
            raise ValueError("unconditional QK normalization carries no gate")
        if self.present is None and not self.gate:
            raise ValueError("gated QK normalization carries >=1 exact atom")
        if any(not isinstance(atom, QKNormGateAtom) for atom in self.gate):
            raise TypeError("QK gates contain QKNormGateAtom values")
        if tuple(sorted(set(self.gate))) != self.gate:
            raise ValueError("QK gate atoms are unique and canonical")


@dataclass(frozen=True)
class _NormApplication:
    call_site: CallSiteId
    call: CallObservation
    construction: ConstructionCallResolution
    primitive: str


def decoder_qk_norm_evidence_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[QKNormCodeEvidence]:
    """Prove Q/K normalization for one parser-selected decoder config path."""
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return _forward_failure(block, "decoder block address")

    root = block.value.component_root
    child = attention_child_evidence(
        index, root, block.value.block_occurrence)
    if child.status != "resolved":
        return _forward_failure(child, "attention child address")
    return _qk_norm_at_attention(index, root, child.value)


def _qk_norm_at_attention(
    index: ProgramIndex,
    root,
    child: AttentionChildEvidence,
) -> ReaderResult[QKNormCodeEvidence]:
    owner = child.child_occurrence
    storage = attention_projection_storage_for_child_evidence(
        index, root, child.block_occurrence, child)
    if storage.status != "resolved":
        return _forward_failure(storage, "attention projection storage")
    storage_projections = frozenset(storage.value.projections)

    consumer_result = _attention_score_consumers(index, child)
    if consumer_result is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the attention proof does not expose exact Q/K score operands"),))
    callable_symbol, score_lanes = consumer_result

    calls = tuple(index.calls_in(callable_symbol))
    construction_calls = []
    for call in calls:
        if _self_field(call.callee) is None:
            continue
        resolution = resolve_construction_call(
            index, root, owner, call)
        if resolution.status == "resolved":
            construction_calls.append((call, resolution))

    linear_calls: dict[ConstructionOccurrenceId, CallObservation] = {}
    norm_calls: list[_NormApplication] = []
    for call, resolution in construction_calls:
        selected = resolution.selected
        if selected.kind == "external" \
                and selected.external_reference.qualified_target \
                in _LINEAR_PROTOCOLS:
            linear_calls[selected.occurrence] = call
        primitive = classify_primitive_call(index, resolution)
        if primitive.status == "resolved" \
                and primitive.value in {"layernorm", "rmsnorm"}:
            norm_calls.append(_NormApplication(
                CallSiteId.of(call), call, resolution, primitive.value))

    if not norm_calls or not linear_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact attention callable has no complete norm-from-Linear "
            "candidate chain"),))

    norm_by_site = {item.call_site: item.call for item in norm_calls}
    lane_producers = {**linear_calls, **norm_by_site}
    live_sites = set()
    score_spans = tuple(lane[0] for lane in score_lanes)
    for score_span, q_expressions, k_expressions in score_lanes:
        q_reaching, _, _, q_uncertain = \
            producer_sources_reaching_expressions(
            index, callable_symbol,
            ((score_span, q_expressions),), lane_producers,
            preserve_local_tuple_lanes=True)
        k_reaching, _, _, k_uncertain = \
            producer_sources_reaching_expressions(
            index, callable_symbol,
            ((score_span, k_expressions),), lane_producers,
            preserve_local_tuple_lanes=True)
        q_sites = frozenset(set(q_reaching).intersection(norm_by_site))
        k_sites = frozenset(set(k_reaching).intersection(norm_by_site))
        if not q_sites or not k_sites:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "both exact score operands must independently descend from "
                "a norm application",
                score_span),))
        if not set(q_sites).isdisjoint(k_sites):
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "the Q and K score operands share a norm application; two "
                "independent normalization lanes are not proven",
                score_span),))
        # Conditional reaching definitions are expected for a config-gated
        # mechanism.  They are accepted only provisionally here; _gate_atoms
        # below must resolve every construction/application guard to exact
        # config evidence before the result can become consumable.
        if q_uncertain and any(
                not norm_by_site[site].guard for site in q_sites):
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "an uncertain Q-lane norm has no exact application guard",
                score_span),))
        if k_uncertain and any(
                not norm_by_site[site].guard for site in k_sites):
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "an uncertain K-lane norm has no exact application guard",
                score_span),))
        live_sites.update(q_sites)
        live_sites.update(k_sites)
    live = tuple(item for item in norm_calls if item.call_site in live_sites)
    if len(live) < 2:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "Q and K do not reach two distinct exact norm applications "
            f"(applications={len(live)})"),))

    for application in live:
        if not application.call.args:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "a live Q/K norm application has no exact data operand",
                application.call.span),))
        upstream, _, _, uncertain = producer_sources_reaching_expressions(
            index, callable_symbol,
            ((application.call.span, (application.call.args[0],)),),
            linear_calls)
        if uncertain or not upstream \
                or not set(upstream).issubset(storage_projections):
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "a live Q/K norm input is not proven to descend from an exact "
                "Q/K/V storage projection",
                application.call.span),))
        downstream_linear_consumers = tuple(
            (call.span, tuple(call.args))
            for call in linear_calls.values()
            if call.span != application.call.span
            and any(_span_before(call.span, score_span)
                    for score_span in score_spans))
        downstream, _, _, downstream_uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                downstream_linear_consumers,
                {application.call_site: application.call})
        if downstream_uncertain or application.call_site in downstream:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "a live norm result feeds another exact Linear application "
                "before the score and is therefore an intermediate/latent norm",
                application.call.span),))

    config_prefix = (
        tuple(root.config_path)
        if hasattr(root, "config_path") else ()
    )
    atoms, gate_spans, gate_failure = _gate_atoms(
        index, owner, live, config_prefix)
    if gate_failure is not None:
        return ReaderResult.failed(owner, (gate_failure,))
    value = QKNormCodeEvidence(
        True if not atoms else None,
        tuple(sorted(atoms)),
    )
    spans = tuple(dict.fromkeys(
        span for span in (
            *(item.call.span for item in live),
            *(item.construction.selected.site.span for item in live),
            *gate_spans,
            *child.compute.spans,
        ) if isinstance(span, SourceSpan)))
    config_paths = tuple(atom.config_path for atom in value.gate)
    provenance_kind = "code_and_config" if config_paths else "source"
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            provenance_kind,
            spans=spans,
            config_paths=config_paths,
            detail=(
                "two exact norm applications descend from exact Linear "
                "constructions and reach the code-proven Q/K score operands"),
        ),),
    )


def _attention_score_consumers(index, child):
    compute = child.compute
    score_lanes = _score_lanes_in_callable(index, compute)
    if not score_lanes:
        return None
    entry_callable = compute.entry_call.enclosing_callable
    if compute.callable_symbol == entry_callable:
        return entry_callable, score_lanes

    # The compute protocol lives in an exactly bound free-function fallback
    # (Transformers eager-attention dispatch).  First prove which function
    # parameters reach the score operands, then bind only those parameters back
    # to the exact entry-call arguments.  Parameter spelling is an address
    # inside one exact callable, never the semantic Q/K proof.
    record = index.callable_by_symbol(compute.callable_symbol)
    if record is None:
        return None
    params = tuple(
        param for param in record.params
        if param.kind in {"positional", "posonly", "keyword_only"})
    param_sources = {
        param.name: ("parameter", compute.callable_symbol, position)
        for position, param in enumerate(params)
    }
    positional = tuple(compute.entry_call.args)
    keywords = dict(compute.entry_call.kwargs)
    positional_params = tuple(
        param for param in record.params
        if param.kind in {"positional", "posonly"})

    def entry_argument(position):
        param = params[position]
        if param.name in keywords:
            return keywords[param.name]
        try:
            positional_position = positional_params.index(param)
        except ValueError:
            return None
        if positional_position >= len(positional):
            return None
        return positional[positional_position]

    rebound = []
    for score_span, q_expressions, k_expressions in score_lanes:
        lane_arguments = []
        for expressions in (q_expressions, k_expressions):
            reaching, _, _, uncertain = producer_sources_reaching_expressions(
                index, compute.callable_symbol,
                ((score_span, expressions),), {},
                initial_sources=param_sources)
            selected = tuple(sorted({
                source[2] for source in reaching
                if isinstance(source, tuple) and len(source) == 3
                and source[0] == "parameter"
                and source[1] == compute.callable_symbol
            }))
            if uncertain or not selected:
                return None
            arguments = tuple(entry_argument(position)
                              for position in selected)
            if any(argument is None for argument in arguments):
                return None
            # A lane may also depend on the exact attention receiver (for
            # example ``repeat_kv(key, module.num_key_value_groups)``).  That
            # receiver is address/context evidence, not a competing tensor
            # argument.  Every other dependency is a possible data carrier;
            # exactly one must remain.  A union such as ``query * scale`` has
            # two carriers and is therefore unknowable here—selecting whichever
            # one happens to be normalized would launder the other.
            carriers = tuple(
                argument for argument in arguments
                if not _is_exact_self_reference(argument))
            if len(carriers) != 1:
                return None
            lane_arguments.append(carriers)
        if set(lane_arguments[0]) == set(lane_arguments[1]):
            return None
        rebound.append((
            compute.entry_call.span,
            lane_arguments[0],
            lane_arguments[1],
        ))
    return entry_callable, tuple(rebound)


def _score_lanes_in_callable(index, compute):
    """Return exact score-operation Q/K operands, never a producer union."""
    callable_symbol = compute.callable_symbol
    protocol_calls = (
        tuple(compute.input_calls)
        if compute.entry_call.enclosing_callable == callable_symbol
        else tuple(index.calls_in(callable_symbol))
    )
    if compute.protocol == "scaled_dot_product_attention":
        calls = tuple(
            call for call in protocol_calls
            if _exact_target(index, call) in _SDPA_PROTOCOLS
            and len(call.args) >= 2)
        return tuple((call.span, (call.args[0],), (call.args[1],))
                     for call in calls)

    softmax = tuple(
        call for call in protocol_calls
        if _exact_target(index, call) in _SOFTMAX_PROTOCOLS)
    if not softmax:
        return ()
    first_softmax = min(softmax, key=lambda item: item.lexical_order)
    score_dots = tuple(
        call for call in protocol_calls
        if _exact_target(index, call) in _DOT_PROTOCOLS
        and (_span_within(call.span, first_softmax.span)
             or _span_before(call.span, first_softmax.span))
        and len(call.args) >= 2)
    if score_dots:
        return tuple((call.span, (call.args[0],), (call.args[1],))
                     for call in score_dots)
    # ``scores = q @ k.T`` is structurally observed, but this unit does not
    # have an exact two-lane reaching-definition boundary for that binop.
    return ()


def _gate_atoms(index, owner, applications, config_prefix):
    node = applications[0].construction.caller_symbol
    if node is None:
        return set(), (), ReaderFailure(
            "incomplete_graph", "the norm construction has no exact owner symbol")
    init = next((
        item for item in index.callables_of(node)
        if item.symbol.qualified_name.endswith(".__init__")
    ), None)
    params = frozenset(
        param.name for param in (init.params if init is not None else ())
        if param.name != "self")
    fields = {
        application.construction.field for application in applications
    }
    flag_values = {}
    for assignment in index.field_assigns_of(node):
        flag_values.setdefault(assignment.field, []).append(assignment.value)

    guards = []
    for application in applications:
        guards.extend(application.construction.selected.site.guard)
        guards.extend(application.call.guard)
    atoms: set[QKNormGateAtom] = set()
    spans: list[SourceSpan] = []
    for step in guards:
        if not isinstance(step, GuardStep) \
                or step.kind not in {"if", "elif"} or step.test is None:
            return set(), (), ReaderFailure(
                "unsupported_syntax",
                "Q/K norm construction/application has an unresolved guard",
                getattr(step, "span", None))
        conjuncts = (
            tuple(step.test.children)
            if step.test.kind == "boolop" and step.test.operator == "and"
            else (step.test,)
        )
        for conjunct in conjuncts:
            if _is_exact_hasattr_guard(conjunct, fields):
                spans.append(step.span)
                continue
            atom = _config_atom(conjunct, params, config_prefix)
            if atom is None:
                field = _self_field(conjunct)
                values = flag_values.get(field, ()) if field else ()
                if len(values) == 1:
                    atom = _config_atom(
                        values[0], params, config_prefix)
            if atom is None:
                return set(), (), ReaderFailure(
                    "unsupported_syntax",
                    "Q/K norm guard is not an exact config-field predicate",
                    step.span)
            atoms.add(atom)
            spans.append(step.span)
    return atoms, tuple(dict.fromkeys(spans)), None


def _config_atom(expr, params, config_prefix):
    current = expr
    if current.kind == "call" and len(current.children) == 2:
        callee = current.children[0]
        if callee.kind == "name" and callee.name in {"bool", "int"}:
            current = current.children[1]
    per_layer = False
    if current.kind == "subscript" and len(current.children) == 2:
        index_expr = current.children[1]
        if index_expr.kind != "name" or index_expr.name not in params:
            return None
        per_layer = True
        current = current.children[0]
    field = _config_field(current)
    return (
        QKNormGateAtom(
            field, (*config_prefix, field), per_layer)
        if field else None
    )


def _config_field(expr):
    if expr.kind != "attribute" or len(expr.children) != 1:
        return None
    root = expr.children[0]
    if root.kind == "name" and root.name in {"config", "cfg"}:
        return expr.name
    if root.kind == "attribute" and root.name == "config" \
            and len(root.children) == 1:
        base = root.children[0]
        if base.kind == "name" and base.name == "self":
            return expr.name
    return None


def _is_exact_hasattr_guard(expr, fields):
    if expr.kind != "call" or len(expr.children) != 3:
        return False
    callee, receiver, field = expr.children
    return (
        callee.kind == "name" and callee.name == "hasattr"
        and receiver.kind == "name" and receiver.name == "self"
        and field.kind == "constant" and field.const_value in fields
    )


def _self_field(expr):
    if expr.kind != "attribute" or len(expr.children) != 1:
        return None
    root = expr.children[0]
    return expr.name if root.kind == "name" and root.name == "self" else None


def _is_exact_self_reference(expr):
    return expr.kind == "name" and expr.name == "self"


def _exact_target(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else None


def _span_within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (
        (outer.line, outer.col)
        <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


def _span_before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (
        first.end_line or first.line,
        first.end_col or first.col,
    ) <= (second.line, second.col)


def _forward_failure(result, label):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(
            result.owner, result.ambiguity,
            provenance=result.provenance)
    details = tuple(result.failures) or (ReaderFailure(
        "incomplete_graph", f"{label} is {result.status}"),)
    return ReaderResult.failed(
        result.owner, details, provenance=result.provenance)


__all__ = [
    "QKNormCodeEvidence",
    "QKNormGateAtom",
    "decoder_qk_norm_evidence_for_path",
]
