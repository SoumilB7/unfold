"""Exact fixed-sinusoidal pre-stack position evidence."""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .call_arguments import bind_addressed_invocation
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .models import SourceBundle
from .position_absolute import pre_stack_additions_for_producer
from .position_coordinate import coordinate_origin
from .program_index import (
    BindingObservation,
    CallObservation,
    CallSiteId,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult
from .reader_claims import ReaderFactProjection, ReaderClaimUnavailable, retained_claim_reader


@dataclass(frozen=True)
class FixedSinusoidalTableProducer:
    owner_occurrence: OwnerOccurrenceId
    forward_call: CallObservation
    coordinate_spans: tuple[SourceSpan, ...]
    buffer_call: CallObservation
    builder_call: CallObservation
    sinusoid_binding: BindingObservation
    cosine_call: CallObservation
    sine_call: CallObservation
    table_return: ReturnObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a fixed table producer names an exact occurrence")
        calls = (self.forward_call, self.buffer_call, self.builder_call,
                 self.cosine_call, self.sine_call)
        if any(not isinstance(item, CallObservation) or item.span is None
               for item in calls):
            raise TypeError("fixed-table evidence carries exact calls")
        if not self.coordinate_spans or any(
                not isinstance(span, SourceSpan) for span in self.coordinate_spans):
            raise ValueError("fixed-table coordinates carry exact provenance")
        if not isinstance(self.sinusoid_binding, BindingObservation) \
                or self.sinusoid_binding.value is None \
                or self.sinusoid_binding.guard:
            raise ValueError("the cosine/sine table binding is unconditional")
        if not isinstance(self.table_return, ReturnObservation) \
                or self.table_return.value is None or self.table_return.guard:
            raise ValueError("the fixed table has one unconditional return")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("fixed-table evidence carries exact source spans")
        required = {
            *(item.span for item in calls),
            *self.coordinate_spans,
            self.sinusoid_binding.span,
            self.table_return.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("fixed-table provenance cites every decisive operation")


@dataclass(frozen=True)
class FixedAbsolutePositionEvidence:
    owner: OwnerOccurrenceId
    producer: FixedSinusoidalTableProducer
    stage_call: CallObservation
    addition: BindingObservation
    repeated_call_sites: tuple[CallSiteId, ...]
    provenance_spans: tuple[SourceSpan, ...]
    kind: str = "fixed_absolute"
    application: str = "sinusoidal_add"

    def __post_init__(self) -> None:
        if self.kind != "fixed_absolute" or self.application != "sinusoidal_add":
            raise ValueError("fixed-absolute evidence has a closed kind")
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or self.producer.owner_occurrence.root != self.owner.root:
            raise ValueError("fixed-absolute evidence belongs to one owner graph")
        if not isinstance(self.stage_call, CallObservation) \
                or self.stage_call.span is None:
            raise TypeError("fixed-absolute evidence cites its exact stage call")
        if not isinstance(self.addition, BindingObservation) \
                or self.addition.value is None \
                or self.addition.value.kind != "binop" \
                or self.addition.value.operator != "+" or self.addition.guard:
            raise ValueError("fixed position is unconditionally added pre-stack")
        if not self.repeated_call_sites \
                or len(set(self.repeated_call_sites)) != len(self.repeated_call_sites):
            raise ValueError("fixed position reaches exact repeated calls")
        if not self.provenance_spans or any(
                not isinstance(span, SourceSpan) for span in self.provenance_spans):
            raise ValueError("fixed-absolute evidence carries exact provenance")
        required = {
            self.stage_call.span, self.addition.span,
            *(site.span for site in self.repeated_call_sites),
            *self.producer.spans,
        }
        if None in required or not required <= set(self.provenance_spans):
            raise ValueError("fixed-absolute provenance closes producer to stack")


@dataclass(frozen=True)
class FixedAbsolutePositionClaimWitness:
    """Retained source origin and positive producer→addition→stack relation.

    This does not establish absence of opaque rival positional mechanisms.
    The legacy evidence DTO alone is deliberately insufficient for issuance.
    """
    index: ProgramIndex
    path: DecoderBlockPath
    inventory: object
    invocations: object
    invocation: AddressedInvocation
    value: FixedAbsolutePositionEvidence
    origin_calls: tuple[CallObservation, ...] | None
    reader_symbol = "model_unfolder.evidence.position_fixed.decoder_fixed_absolute_position_for_path"

    def validate_result(self, result):
        self.path.__post_init__()
        self.value.__post_init__()
        if result.status != "resolved" or result.value is not self.value \
                or result.owner != self.path.stage_occurrence \
                or self.value.owner != self.path.stage_occurrence:
            raise ValueError("fixed-position claim belongs to another result or owner")
        root = self.path.component_root
        if self.inventory != resolve_container_inventory(self.index, root, self.path.stage_occurrence) \
                or self.invocations != resolve_addressed_invocations(
                    self.index, root, self.path.stage_occurrence, self.inventory):
            raise ValueError("fixed-position claim changed its source invocation census")
        if not any(item is self.invocation for item in self.invocations.addressed) \
                or self.value.stage_call is not self.invocation.call:
            raise ValueError("fixed-position claim needs the actual addressed invocation")
        producer = _fixed_producer(self.index, root, self.invocation)
        if producer is None or producer != self.value.producer:
            raise ValueError("fixed-position producer differs from actual source derivation")
        origin = _admitted_fixed_origin(self.index, root, self.invocation, producer)
        if (origin is None) != (self.origin_calls is None) or (origin is not None and (
                len(origin) != len(self.origin_calls)
                or any(a is not b for a, b in zip(origin, self.origin_calls)))):
            raise ValueError("fixed-position constructor or trigonometric input origin changed")
        repeated = tuple(dict.fromkeys(
            proof.template.call for proof in self.path.repeated_child.proofs))
        if tuple(CallSiteId.of(call) for call in repeated) != self.value.repeated_call_sites:
            raise ValueError("fixed-position claim changed its repeated-child sinks")
        additions = pre_stack_additions_for_producer(
            self.index, repeated[0].enclosing_callable, self.invocation.call_site,
            self.invocation.call, repeated)
        if self.value.addition not in additions:
            raise ValueError("fixed-position addition no longer reaches the exact stack")
        required = tuple(dict.fromkeys((*self.value.provenance_spans,
                                        *(call.span for call in origin or ()))))
        actual = tuple(dict.fromkeys(span for item in result.provenance for span in item.spans))
        if actual != required:
            raise ValueError("fixed-position provenance must close origin, addition and sink")

    def declared_kind(self, owner, key):
        if (owner, key) != ("decoder.input", "position_addition"):
            raise ValueError("fixed-position proof cannot author another fact projection")
        return "connection"

    def project(self, owner, key, document):
        self.declared_kind(owner, key)
        if self.origin_calls is None:
            raise ReaderClaimUnavailable("fixed-position constructor/table/returned-value origin is unsupported")
        return ReaderFactProjection(owner, key, "connection", {
            "position_kind": "fixed_absolute", "position_application": "embedding_add",
        }, "code_proven", completeness="presence_only")


@retained_claim_reader
def decoder_fixed_absolute_position_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[FixedAbsolutePositionEvidence]:
    if not isinstance(index, ProgramIndex):
        raise TypeError("fixed-absolute evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("fixed-absolute evidence requires a SourceBundle")
    path = decoder_block_path_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if path.status != "resolved":
        return ReaderResult.failed(path.owner, (ReaderFailure(
            "incomplete_graph", "the exact decoder path is unresolved"),))
    return read_fixed_absolute_position(index, path.value)


def read_fixed_absolute_position(index, path):
    if not isinstance(index, ProgramIndex) or not isinstance(path, DecoderBlockPath):
        raise TypeError("fixed-absolute reader needs ProgramIndex + DecoderBlockPath")
    root = path.component_root
    owner = path.stage_occurrence
    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "stage invocation census is unavailable"),))
    repeated_calls = tuple(dict.fromkeys(
        proof.template.call for proof in path.repeated_child.proofs))
    if not repeated_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the decoder path retained no repeated call"),))
    callable_symbol = repeated_calls[0].enclosing_callable
    if any(item.enclosing_callable != callable_symbol for item in repeated_calls):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "repeated calls do not share one stage"),))

    qualified = []
    retained_invocations = {}
    for invocation in invocations.addressed:
        producer = _fixed_producer(index, root, invocation)
        if producer is None:
            continue
        for addition in pre_stack_additions_for_producer(
                index, callable_symbol, invocation.call_site,
                invocation.call, repeated_calls):
            spans = tuple(dict.fromkeys((
                invocation.call.span, addition.span, *producer.spans,
                *(call.span for call in repeated_calls))))
            evidence = FixedAbsolutePositionEvidence(
                owner, producer, invocation.call, addition,
                tuple(CallSiteId.of(call) for call in repeated_calls), spans)
            qualified.append(evidence)
            retained_invocations[id(evidence)] = invocation
    by_identity = {
        (item.stage_call.span, item.addition.span,
         item.producer.buffer_call.span): item for item in qualified}
    if len(by_identity) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(sorted(
                (item.addition.span for item in by_identity.values()),
                key=_span_key))))
    if not by_identity:
        return ReaderResult.absent(owner)
    value = next(iter(by_identity.values()))
    selected_invocation = retained_invocations[id(value)]
    origin = _admitted_fixed_origin(index, root, selected_invocation, value.producer)
    witness = FixedAbsolutePositionClaimWitness(
        index, path, inventory, invocations, selected_invocation, value, origin)
    # Unsupported origin shapes retain the original visible value and status,
    # but never acquire claim authority from the older, weaker DTO census.
    spans = tuple(dict.fromkeys((*value.provenance_spans,
                                 *(call.span for call in origin or ()))))
    return ReaderResult.resolved(
        owner, value, claim_witness=witness,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact generated cosine/sine buffer lookup -> coordinate -> "
                "pre-stack hidden-stream addition")),))


def _admitted_fixed_origin(index, root, invocation, producer):
    forward = index.callable_by_symbol(producer.forward_call.enclosing_callable)
    if invocation.guard or producer.forward_call.guard or forward is None \
            or not _fixed_forward_callable(index, forward) or invocation.call.kwargs \
            or len(invocation.call.args) != len(forward.params) - 1 \
            or any(param.kind != "positional" for param in forward.params) \
            or bind_addressed_invocation(index, root, invocation).status != "resolved":
        return None
    return _fixed_claim_origin(index, producer)


def _synchronous_callable(index, record):
    declarations = tuple(statement for module in index.modules
                         if module.source.source_id == record.symbol.source
                         for statement in module.statements if statement.span == record.span)
    return (len(declarations) == 1 and declarations[0].kind == "FunctionDef"
            and not declarations[0].guard
            and not any(item.kind in {"yield", "yield_from"}
                        for item in index.control_transfers_in(record.symbol)))


def _fixed_forward_callable(index, record):
    if not _synchronous_callable(index, record):
        return False
    for decorator in record.decorators:
        if decorator.kind != "call" or len(decorator.children) != 1 or decorator.keyword_children:
            return False
        target = resolve_import_reference(index, record.symbol.source, None, decorator.children[0])
        if target is None or target.qualified_target != "torch.no_grad":
            return False
    return True


def _exact_positional_method_call(index, call, target, *, static=False):
    """Finite explicit positional binding; unsupported Python call forms decline."""
    record = index.callable_by_symbol(target)
    if record is None or not _synchronous_callable(index, record) or call.kwargs \
            or _self_field(call.callee) != target.qualified_name.rpartition(".")[2]:
        return False
    params = record.params
    if static:
        if len(record.decorators) != 1 or record.decorators[0].kind != "name" \
                or record.decorators[0].name != "staticmethod" \
                or any(item.name == "staticmethod" for item in index.module_bindings_in(target.source)):
            return False
        owner = index.class_by_symbol(record.owner)
        if owner is None or any(item.attr == "staticmethod" for item in owner.body_assigns):
            return False
    else:
        if record.decorators or not params or params[0].name != "self" \
                or params[0].kind not in {"posonly", "positional"}:
            return False
        params = params[1:]
    return (len(params) == len(call.args)
            and all(param.kind in {"posonly", "positional"} for param in params)
            and all(arg.kind not in {"starred", "unsupported"} for arg in call.args))


def _nodes(expression):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        if child is not None:
            yield from _nodes(child)
    for _, child in expression.keyword_children:
        if child is not None:
            yield from _nodes(child)


def _tensor_return_chain(index, callable_symbol, expression, origin):
    """Return only an exact origin or a finite Tensor value-preserving wrapper."""
    wrappers = []
    while not origin(expression):
        if expression is None or expression.kind != "call" or not expression.children \
                or expression.keyword_children:
            return None
        callee, *args = expression.children
        if callee.kind != "attribute" or len(callee.children) != 1:
            return None
        if callee.name in {"detach", "contiguous"}:
            if args:
                return None
        elif callee.name in {"view", "reshape"}:
            if not args or any(node.kind not in {
                    "name", "constant", "binop", "unaryop", "unary"}
                    for arg in args for node in _nodes(arg)):
                return None
        elif callee.name == "to":
            # The actual fixed-table builder uses this exact dtype operation.
            if len(args) != 1 or args[0].kind != "call" \
                    or len(args[0].children) != 1 or args[0].keyword_children:
                return None
            target = resolve_import_reference(
                index, callable_symbol.source, callable_symbol, args[0].children[0])
            if target is None or target.qualified_target != "torch.get_default_dtype":
                return None
            wrappers.append(args[0].span)
        else:
            return None
        wrappers.append(expression.span)
        expression = callee.children[0]
    return tuple(wrappers)


def _bare_receiver_escape(expression):
    if expression is None:
        return False
    if expression.kind == "name":
        return expression.name == "self"
    if expression.kind == "attribute" and len(expression.children) == 1 \
            and expression.children[0].kind == "name" and expression.children[0].name == "self":
        return False
    return any(_bare_receiver_escape(child) for child in expression.children) or any(
        _bare_receiver_escape(child) for _, child in expression.keyword_children)


def _statements(index, scope):
    return tuple(statement for module in index.modules if module.source.source_id == scope.source
                 for statement in module.statements if statement.enclosing_scope == scope
                 and not (statement.kind == "Expr" and statement.value is not None
                          and statement.value.kind == "constant" and type(statement.value.const_value) is str))


def _fixed_constructor_lineage(index, producer, installed, installer, builder):
    scope = installed.enclosing_callable
    constructor = index.callable_by_symbol(scope)
    owner = index.class_by_symbol(constructor.owner)
    method_records = index.callables_of(owner.symbol)
    methods = {item.symbol.qualified_name.rpartition(".")[2] for item in method_records}
    if owner.body_assigns or len(method_records) != 4 or methods != {"__init__", "forward",
            installer.qualified_name.rpartition(".")[2], builder.qualified_name.rpartition(".")[2]}:
        return None
    formals = {param.name for param in constructor.params[1:]
               if param.kind in {"posonly", "positional"}}
    if any(arg.kind != "name" or arg.name not in formals for arg in installed.args):
        return None
    if any(item.name == "super" for item in index.module_bindings_in(scope.source)) \
            or any(item.name == "super" and item.context in {"parameter", "store", "del"}
                   for item in index.identifiers_in(scope)):
        return None
    allowed_calls = {installed.span}
    scalar_fields = set()
    statements = _statements(index, scope)
    if not statements or statements[-1].kind != "Expr" or statements[-1].value is None \
            or statements[-1].value.span != installed.span:
        return None
    initialized_module = False
    for statement in statements:
        if statement.kind == "Assign" and len(statement.targets) == 1:
            target = statement.targets[0]
            field = _self_field(target)
            if field is None or field.startswith("_") or field in methods \
                    or field in {producer.buffer_call.args[0].const_value, "register_buffer"} \
                    or statement.value is None or statement.value.kind != "name" \
                    or statement.value.name not in formals or field in scalar_fields or not initialized_module:
                return None
            scalar_fields.add(field)
            continue
        if statement.kind != "Expr" or statement.value is None or statement.value.kind != "call":
            return None
        call = statement.value
        if call.span == installed.span:
            if not initialized_module:
                return None
            continue
        if initialized_module or len(call.children) != 1 or call.keyword_children:
            return None
        callee = call.children[0]
        if callee.kind != "attribute" or callee.name != "__init__" or len(callee.children) != 1:
            return None
        inherited = callee.children[0]
        if inherited.kind != "call" or len(inherited.children) != 1 or inherited.keyword_children \
                or inherited.children[0].kind != "name" or inherited.children[0].name != "super":
            return None
        allowed_calls.update((call.span, inherited.span))
        initialized_module = True
    if {call.span for call in index.calls_in(scope)} != allowed_calls:
        return None
    return frozenset(scalar_fields)


def _fixed_installation_lineage(index, producer):
    buffer, builder = producer.buffer_call, producer.builder_call
    scope = buffer.enclosing_callable
    if buffer.args[1].kind != "name":
        return False
    name = buffer.args[1].name
    buffer_name = buffer.args[0].const_value
    allowed_statements = {buffer.span}
    allowed_calls = {buffer.span, builder.span}
    initial = False
    for binding in index.bindings_in(scope):
        if _single_target(binding) != name or binding.value is None:
            return False
        value = binding.value
        if not initial and value.span == builder.span and not binding.guard:
            initial = True
        elif initial and binding.guard and value.kind == "call" and len(value.children) == 1:
            callee = value.children[0]
            if callee.kind != "attribute" or callee.name != "to" or len(callee.children) != 1 \
                    or callee.children[0].kind != "name" or callee.children[0].name != name \
                    or len(value.keyword_children) != 2 or {key for key, _ in value.keyword_children} != {"dtype", "device"}:
                return False
            for key, actual in value.keyword_children:
                if actual.kind != "attribute" or actual.name != key or len(actual.children) != 1 \
                        or _self_field(actual.children[0]) != buffer_name:
                    return False
            allowed_calls.add(value.span)
        else:
            return False
        allowed_statements.add(binding.span)
    if not initial:
        return False
    for module in index.modules:
        if module.source.source_id != scope.source:
            continue
        for statement in module.statements:
            if statement.enclosing_scope != scope:
                continue
            if statement.span in allowed_statements and statement.kind in {"Assign", "Expr"}:
                continue
            if statement.kind == "If":
                test = statement.value
                if test is None or test.kind != "call" or len(test.children) != 3 or test.keyword_children \
                        or test.children[0].kind != "name" or test.children[0].name != "hasattr" \
                        or test.children[1].kind != "name" or test.children[1].name != "self" \
                        or test.children[2].kind != "constant" or test.children[2].const_value != buffer_name:
                    return False
                if any(item.name == "hasattr" for item in index.module_bindings_in(scope.source)) \
                        or any(item.name == "hasattr" and item.context in {"parameter", "store", "del"}
                               for item in index.identifiers_in(scope)):
                    return False
                allowed_calls.add(test.span)
                continue
            return False
    return {call.span for call in index.calls_in(scope)} == allowed_calls


def _fixed_return_lineage(index, producer, cat, installer):
    """Close returned tensors and refuse intervening mutation or escape.

    The builder suffix is finite: optional guarded zero-padding assignments,
    then the exact return. The forward may read the buffer or rebuild it via
    the same admitted installer, but cannot alias, replace or mutate it.
    """
    if not _fixed_installation_lineage(index, producer):
        return False
    binding = producer.sinusoid_binding
    name = _single_target(binding)
    returned = producer.table_return
    if name is None:
        return False
    chain = _tensor_return_chain(index, binding.enclosing_callable, returned.value,
                                 lambda expr: expr is not None and expr.kind == "name" and expr.name == name)
    if chain is None:
        return False
    allowed_calls = set(chain)
    allowed_assignments = set()
    for later in index.bindings_in(binding.enclosing_callable):
        if later.span is None or not (_span_key(binding.span) < _span_key(later.span)
                                    < _span_key(returned.span)):
            continue
        if not later.guard or _single_target(later) != name or later.value is None:
            return False
        calls = tuple(call for call in index.calls_in(binding.enclosing_callable)
                      if _expr_contains_span(later.value, call.span))
        cats = tuple(call for call in calls if _protocol(index, binding.enclosing_callable, call) == "torch.cat")
        zeros = tuple(call for call in calls if _protocol(index, binding.enclosing_callable, call) == "torch.zeros")
        if len(cats) != 1 or len(zeros) != 1 or len(calls) != 2:
            return False
        append, zero = cats[0], zeros[0]
        if later.value.span != append.span or len(append.args) != 1 \
                or append.args[0].kind not in {"list", "tuple"} \
                or len(append.args[0].children) != 2 \
                or len(append.kwargs) != 1 or append.kwargs[0][0] != "dim" \
                or _int_literal(append.kwargs[0][1]) != 1 or zero.kwargs:
            return False
        original, padding = append.args[0].children
        if original.kind != "name" or original.name != name or padding.span != zero.span \
                or any(node.kind not in {"name", "constant", "binop", "unaryop", "unary"}
                       for arg in zero.args for node in _nodes(arg)):
            return False
        allowed_calls.update((append.span, zero.span))
        allowed_assignments.add(later.span)
    for module in index.modules:
        if module.source.source_id != binding.enclosing_callable.source:
            continue
        for statement in module.statements:
            if statement.enclosing_scope != binding.enclosing_callable \
                    or _span_key(statement.span) <= _span_key(binding.span):
                continue
            if statement.span == returned.span and statement.kind == "Return":
                continue
            if statement.span in allowed_assignments and statement.kind == "Assign":
                continue
            if statement.kind == "If" and _span_key(statement.span) < _span_key(returned.span):
                continue
            return False
    end = (binding.span.end_line or binding.span.line,
           binding.span.end_col or binding.span.col)
    suffix_calls = {call.span for call in index.calls_in(binding.enclosing_callable)
                    if (call.span.line, call.span.col) >= end}
    if suffix_calls != allowed_calls:
        return False

    forward = producer.forward_call.enclosing_callable
    returns = index.return_observations_in(forward)
    if len(returns) != 1 or returns[0].guard:
        return False
    forward_chain = _tensor_return_chain(index, forward, returns[0].value,
        lambda expr: expr is not None and expr.kind == "call" and expr.span == producer.forward_call.span)
    if forward_chain is None:
        return False
    owner = producer.buffer_call.owner
    constructor = SymbolId(owner.source, owner.qualified_name + ".__init__")
    installed = tuple(call for call in index.calls_in(constructor)
                      if _self_field(call.callee) == installer.qualified_name.rpartition(".")[2])
    if len(installed) != 1:
        return False
    builder = SymbolId(owner.source, owner.qualified_name + "." + _self_field(producer.builder_call.callee))
    scalar_fields = _fixed_constructor_lineage(index, producer, installed[0], installer, builder)
    if scalar_fields is None:
        return False
    buffer_name = producer.buffer_call.args[0].const_value
    allowed_self = set()
    installer_calls = set()
    buffer_calls = {producer.forward_call.span, *forward_chain}
    def admit_receiver(expression):
        allowed_self.update(node.span for node in _nodes(expression)
                            if node.kind == "name" and node.name == "self")
    admit_receiver(producer.forward_call.callee)
    for call in index.calls_in(forward):
        if call.callee.kind == "attribute" and call.callee.name == "size" \
                and len(call.callee.children) == 1 \
                and _self_field(call.callee.children[0]) == buffer_name \
                and len(call.args) == 1 and _int_literal(call.args[0]) == 0 and not call.kwargs:
            buffer_calls.add(call.span)
            admit_receiver(call.callee)
        method = _self_field(call.callee)
        if method is not None:
            if method != installer.qualified_name.rpartition(".")[2] \
                    or not _exact_positional_method_call(index, call, installer):
                return False
            installer_calls.add(call.span)
            admit_receiver(call.callee)
            for actual in call.args:
                if actual.kind in {"name", "constant"}:
                    if _bare_receiver_escape(actual):
                        return False
                elif _self_field(actual) in scalar_fields:
                    admit_receiver(actual)
                else:
                    return False
    for statement in _statements(index, forward):
        if statement.kind in {"Assign", "AnnAssign"}:
            if any(node.kind != "name" for target in statement.targets
                   for node in _nodes(target) if node.kind not in {"tuple", "list"}) \
                    or any(node.kind == "name" and node.name == "self"
                           for node in _nodes(statement.value)):
                return False
        elif statement.kind == "Expr":
            if statement.value is None or statement.value.span not in installer_calls:
                return False
        elif statement.kind == "Return":
            if statement.span != returns[0].span:
                return False
        elif statement.kind != "If":
            return False
        # Every occurrence of the receiver is admitted at an exact buffer-read,
        # installer-call, or constructor-proven scalar argument address. Alias
        # and namespace/reflection uses have no such address and are refused.
        if any(node.kind == "name" and node.name == "self" and node.span not in allowed_self
               for expression in (*statement.targets, statement.value) for node in _nodes(expression)):
            return False
    for call in index.calls_in(forward):
        if call.span in buffer_calls or call.span in installer_calls:
            continue
        if any(node.kind == "name" and node.name == "self"
               for expression in (call.callee, *call.args, *(value for _, value in call.kwargs))
               for node in _nodes(expression)):
            return False
        if _protocol(index, forward, call) == "torch.arange":
            continue
        if call.callee.kind == "attribute" and call.callee.name in {
                "size", "view", "reshape", "to", "clone", "contiguous", "expand",
                "expand_as", "float", "long", "type_as", "unsqueeze", "detach"}:
            continue
        return False
    return True


def _fixed_claim_origin(index, producer):
    """Finite, source-addressed admission beyond the legacy table DTO.

    Admit a direct unconditional constructor→installer call and the literal
    cos/sin pair supplied to the returned table concatenation. No method-name
    semantics, arbitrary expression evaluator, or absent-origin inference.
    """
    buffer = producer.buffer_call
    builder_call = producer.builder_call
    installer = buffer.enclosing_callable
    if buffer.guard or builder_call.guard or installer != builder_call.enclosing_callable \
            or len(buffer.args) != 2 or len(buffer.kwargs) != 1 \
            or buffer.kwargs[0][0] != "persistent":
        return None
    installer_record = index.callable_by_symbol(installer)
    if installer_record is None or installer_record.decorators or buffer.owner is None:
        return None
    # The buffer protocol must be the imported nn.Module operation, not an
    # owner-defined method with the same spelling or an unknown inherited one.
    owner_class = index.class_by_symbol(buffer.owner)
    if owner_class is None or len(owner_class.bases) != 1 \
            or owner_class.decorators or owner_class.keywords:
        return None
    base = resolve_import_reference(index, installer.source, None,
                                    owner_class.bases[0])
    if base is None or base.qualified_target not in {
            "torch.nn.Module", "torch.nn.modules.module.Module"} \
            or index.callable_by_symbol(SymbolId(
                installer.source, buffer.owner.qualified_name + ".register_buffer")) is not None:
        return None
    constructor = SymbolId(installer.source, buffer.owner.qualified_name + ".__init__")
    constructor_record = index.callable_by_symbol(constructor)
    if constructor_record is None or constructor_record.decorators \
            or not _synchronous_callable(index, constructor_record):
        return None
    method_name = installer.qualified_name.rpartition(".")[2]
    calls = tuple(call for call in index.calls_in(constructor)
                  if not call.guard and _self_field(call.callee) == method_name)
    if len(calls) != 1:
        return None
    installed = calls[0]
    if not _exact_positional_method_call(index, installed, installer):
        return None
    if any(item.span is not None and _span_key(item.span) < _span_key(installed.span)
           for item in index.return_observations_in(constructor)):
        return None
    builder_name = _self_field(builder_call.callee)
    builder = SymbolId(installer.source, buffer.owner.qualified_name + "." + builder_name) \
        if builder_name else None
    if builder is None or not _exact_positional_method_call(index, builder_call, builder, static=True):
        return None
    buffer_name = buffer.args[0].const_value
    if any(item.field in {method_name, builder_name, "register_buffer", buffer_name}
           for item in index.field_assigns_of(buffer.owner)):
        return None
    binding = producer.sinusoid_binding
    cosine, sine = producer.cosine_call, producer.sine_call
    cats = tuple(call for call in index.calls_in(binding.enclosing_callable)
                 if _protocol(index, binding.enclosing_callable, call) == "torch.cat"
                 and _expr_contains_span(binding.value, call.span))
    if len(cats) != 1:
        return None
    cat = cats[0]
    if cat.guard or cosine.guard or sine.guard or len(cat.args) != 1 \
            or cat.args[0].kind not in {"tuple", "list"} \
            or len(cat.args[0].children) != 2 \
            or set(dict(cat.kwargs)) != {"dim"} \
            or _int_literal(dict(cat.kwargs)["dim"]) != 1:
        return None
    if tuple(child.span for child in cat.args[0].children) != (cosine.span, sine.span):
        return None
    chain = _tensor_return_chain(index, binding.enclosing_callable, binding.value,
        lambda expr: expr is not None and expr.kind == "call" and expr.span == cat.span)
    if chain is None:
        return None
    binding_calls = {call.span for call in index.calls_in(binding.enclosing_callable)
                     if _expr_contains_span(binding.value, call.span)}
    if binding_calls != {cat.span, cosine.span, sine.span, *chain}:
        return None
    if not _fixed_return_lineage(index, producer, cat, installer):
        return None
    return installed, cat


def _fixed_producer(index, root, invocation):
    if not isinstance(invocation, AddressedInvocation):
        return None
    node = root.graph.node_for(invocation.callee_owner_occurrence)
    if node is None:
        return None
    forward = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    returns = tuple(item for item in index.return_observations_in(forward)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    selections = []
    for call in index.calls_in(forward):
        field = _self_field(call.callee.children[0]) \
            if call.callee.kind == "attribute" \
            and call.callee.name == "index_select" \
            and call.callee.children else None
        if field is None or len(call.args) != 2 or call.kwargs \
                or _int_literal(call.args[0]) != 0 \
                or not _expr_contains_span(returned.value, call.span):
            continue
        coordinate = coordinate_origin(
            index, forward, call.args[1], call.span)
        if coordinate is None:
            coordinate = _coordinate_from_exact_caller(
                index, root, node, invocation, forward,
                call.args[1], call.span)
        if coordinate is not None:
            selections.append((field, call, coordinate))
    if len(selections) != 1:
        return None
    field, forward_call, coordinate = selections[0]

    buffer_matches = []
    for method in index.callables_of(node.symbol):
        for call in index.calls_in(method.symbol):
            if _self_field(call.callee) != "register_buffer" \
                    or len(call.args) < 2 \
                    or call.args[0].kind != "constant" \
                    or call.args[0].const_value != field \
                    or dict(call.kwargs).get("persistent") is None \
                    or dict(call.kwargs)["persistent"].kind != "constant" \
                    or dict(call.kwargs)["persistent"].const_value is not False:
                continue
            source_call = _initial_local_call(
                index, method.symbol, call.args[1], call.span)
            if source_call is not None:
                buffer_matches.append((call, source_call))
    if len(buffer_matches) != 1:
        return None
    buffer_call, builder_call = buffer_matches[0]
    method_name = _self_field(builder_call.callee)
    builder = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.{method_name}") \
        if method_name else None
    if builder is None or index.callable_by_symbol(builder) is None:
        return None
    sinusoid = _sinusoidal_table(index, builder)
    if sinusoid is None:
        return None
    sinusoid_binding, cosine, sine, table_return = sinusoid
    spans = tuple(dict.fromkeys((
        forward_call.span, *coordinate.spans, buffer_call.span,
        builder_call.span, sinusoid_binding.span, cosine.span, sine.span,
        table_return.span)))
    return FixedSinusoidalTableProducer(
        invocation.callee_owner_occurrence, forward_call, coordinate.spans,
        buffer_call, builder_call, sinusoid_binding, cosine, sine,
        table_return, spans)


def _coordinate_from_exact_caller(
        index, root, node, invocation, forward, expression, before):
    """Transport an ordered coordinate through one addressed module call.

    Some fixed-table modules (XGLM) accept caller-produced position ids and add
    only an exact scalar offset before indexing the table.  The parameter name
    is never evidence: the exact call binding supplies the actual expression,
    and the callee side must be a transparent wrapper/offset of that formal.
    """
    bindings = bind_addressed_invocation(index, root, invocation)
    if bindings.status not in {"resolved", "partial"}:
        return None
    matches = []
    for binding in bindings.bindings:
        if not _transparent_coordinate_formal(
                index, node.symbol, forward, expression,
                binding.formal.name, before, frozenset()):
            continue
        origin = coordinate_origin(
            index, invocation.call.enclosing_callable,
            binding.actual, invocation.call.span)
        if origin is not None:
            matches.append(origin)
    return matches[0] if len(matches) == 1 else None


def _transparent_coordinate_formal(
        index, owner, callable_symbol, expression, formal, before, seen):
    if expression is None or expression.span is None:
        return False
    key = (expression.kind, expression.span)
    if key in seen:
        return False
    seen = seen | {key}
    if expression.kind == "name" and expression.name:
        earlier = tuple(
            item for item in index.bindings_in(callable_symbol)
            if item.span is not None and item.value is not None
            and _span_key(item.span) < _span_key(before)
            and _single_target(item) == expression.name)
        if earlier:
            latest_span = max((item.span for item in earlier), key=_span_key)
            latest = tuple(item for item in earlier if item.span == latest_span)
            return len(latest) == 1 and not latest[0].guard \
                and _transparent_coordinate_formal(
                    index, owner, callable_symbol, latest[0].value,
                    formal, latest[0].span, seen)
        return expression.name == formal
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in {
                "clone", "contiguous", "expand", "expand_as", "float",
                "long", "reshape", "to", "type_as", "unsqueeze", "view"} \
                and callee.children:
            return _transparent_coordinate_formal(
                index, owner, callable_symbol, callee.children[0], formal,
                before, seen)
        return False
    if expression.kind == "binop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 2:
        left, right = expression.children
        return (
            _transparent_coordinate_formal(
                index, owner, callable_symbol, left, formal,
                before, seen)
            and _exact_scalar_expression(index, owner, right)
        ) or (
            expression.operator == "+"
            and _transparent_coordinate_formal(
                index, owner, callable_symbol, right, formal,
                before, seen)
            and _exact_scalar_expression(index, owner, left)
        )
    return False


def _exact_scalar_expression(index, owner, expression):
    if expression.kind == "constant":
        return isinstance(expression.const_value, (int, float)) \
            and not isinstance(expression.const_value, bool)
    field = _self_field(expression)
    if field is None:
        return False
    assignments = tuple(
        item for item in index.field_assigns_of(owner)
        if item.field == field and not item.guard and item.span is not None)
    return len(assignments) == 1 \
        and assignments[0].value.kind == "constant" \
        and isinstance(assignments[0].value.const_value, (int, float)) \
        and not isinstance(assignments[0].value.const_value, bool)


def _sinusoidal_table(index, callable_symbol):
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    bindings = tuple(sorted(
        (item for item in index.bindings_in(callable_symbol)
         if item.span is not None and item.value is not None),
        key=lambda item: _span_key(item.span)))
    candidates = []
    for binding in bindings:
        target = _single_target(binding)
        if binding.guard or target is None \
                or not _transparent_from_name(returned.value, target):
            continue
        calls = tuple(call for call in index.calls_in(callable_symbol)
                      if call.span is not None
                      and _expr_contains_span(binding.value, call.span))
        cats = tuple(call for call in calls
                     if _protocol(index, callable_symbol, call) == "torch.cat")
        cosines = tuple(call for call in calls
                        if _protocol(index, callable_symbol, call) == "torch.cos")
        sines = tuple(call for call in calls
                      if _protocol(index, callable_symbol, call) == "torch.sin")
        if len(cats) != 1 or len(cosines) != 1 or len(sines) != 1 \
                or len(cosines[0].args) != 1 or len(sines[0].args) != 1:
            continue
        cosine_source = _reaching_binding(
            bindings, cosines[0].args[0], binding.span)
        sine_source = _reaching_binding(
            bindings, sines[0].args[0], binding.span)
        if cosine_source is None or cosine_source != sine_source \
                or cosine_source.value.kind != "binop" \
                or cosine_source.value.operator != "*":
            continue
        if not any(coordinate_origin(
                index, callable_symbol, child, cosine_source.span) is not None
                for child in cosine_source.value.children):
            continue
        later = tuple(item for item in bindings
                      if _single_target(item) == target
                      and _span_key(binding.span) < _span_key(item.span)
                      and _span_key(item.span) < _span_key(returned.span))
        if any(not item.guard or not _zero_pad_extension(
                index, callable_symbol, item.value, target)
               for item in later):
            continue
        candidates.append((binding, cosines[0], sines[0], returned))
    return candidates[0] if len(candidates) == 1 else None


def _initial_local_call(index, callable_symbol, expression, before):
    if expression.kind != "name":
        return None
    bindings = tuple(item for item in index.bindings_in(callable_symbol)
                     if item.span is not None and item.value is not None
                     and _span_key(item.span) < _span_key(before)
                     and _single_target(item) == expression.name)
    if not bindings or bindings[0].guard \
            or bindings[0].value.kind != "call":
        return None
    for item in bindings[1:]:
        if not item.guard or not _transparent_from_name(
                item.value, expression.name):
            return None
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.span == bindings[0].value.span)
    return matches[0] if len(matches) == 1 else None


def _zero_pad_extension(index, callable_symbol, expression, name):
    calls = tuple(call for call in index.calls_in(callable_symbol)
                  if call.span is not None
                  and _expr_contains_span(expression, call.span))
    return (_expr_contains_name(expression, name)
            and any(_protocol(index, callable_symbol, call) == "torch.cat"
                    for call in calls)
            and any(_protocol(index, callable_symbol, call) == "torch.zeros"
                    for call in calls))


def _reaching_binding(bindings, expression, before):
    if expression.kind != "name":
        return None
    matches = tuple(item for item in bindings
                    if _single_target(item) == expression.name
                    and not item.guard
                    and _span_key(item.span) < _span_key(before))
    return matches[-1] if matches else None


def _protocol(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    return proof.qualified_target if proof is not None else None


def _single_target(binding):
    names = tuple(name for target in binding.targets
                  for name in _target_names(target))
    return names[0] if len(names) == 1 else None


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     if child is not None for name in _target_names(child))
    return ()


def _self_field(expression):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _transparent_from_name(expression, name):
    if expression.kind == "name":
        return expression.name == name
    if expression.kind in {"call", "attribute"} and expression.children:
        receiver = expression.children[0]
        if expression.kind == "call" and receiver.kind == "attribute" \
                and receiver.children:
            receiver = receiver.children[0]
        return _transparent_from_name(receiver, name)
    return False


def _expr_contains_name(expression, name):
    if expression.kind == "name" and expression.name == name:
        return True
    return any(_expr_contains_name(child, name)
               for child in expression.children if child is not None) or any(
        _expr_contains_name(child, name)
        for _key, child in expression.keyword_children)


def _expr_contains_span(expression, span):
    if expression is None or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span)
               for child in expression.children if child is not None) or any(
        _expr_contains_span(child, span)
        for _key, child in expression.keyword_children)


def _int_literal(expression):
    if expression.kind == "constant" and isinstance(expression.const_value, int) \
            and not isinstance(expression.const_value, bool):
        return expression.const_value
    return None


def _span_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (span.source.component_key, span.source.canonical_path,
            span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


__all__ = [
    "FixedSinusoidalTableProducer",
    "FixedAbsolutePositionEvidence",
    "FixedAbsolutePositionClaimWitness",
    "decoder_fixed_absolute_position_for_path",
    "read_fixed_absolute_position",
]
