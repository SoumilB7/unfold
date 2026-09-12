"""Positive model-state width from an exact embedding-to-stack source route.

This is a presence-only width for the owned token-embedding route. A caller's
alternate external tensor is not measured. No attention-width convention or
config-key spelling selects the source operand.
"""
from __future__ import annotations

from dataclasses import dataclass

from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_path_for_config
from .execution_flow import resolve_addressed_invocations, resolve_execution_flow, _forward_symbol
from .expression_eval import ConfigExpressionEvaluator
from .program_index import CallSiteId, ProgramIndex, SourceSpan
from .reader_claims import ReaderFactProjection, reader_operand, retained_claim_reader
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult

_EMBEDDING = frozenset({'torch.nn.Embedding', 'torch.nn.modules.sparse.Embedding'})


def _position(span):
    return (span.line, span.col)


def _nodes(expression):
    if expression is not None:
        yield expression
        for child in expression.children:
            yield from _nodes(child)
        for _name, child in expression.keyword_children:
            yield from _nodes(child)


def _optional_embedding_guard(index, callable_symbol, binding, call):
    """Only the exact `if external is None: external = embedding(ids)` branch."""
    if not call.guard:
        return False
    if len(call.guard) != 1 or binding.guard != call.guard:
        return None
    step = call.guard[0]
    test = step.test
    if step.kind != 'if' or test is None or test.kind != 'compare' \
            or test.operator != 'is' or len(test.children) != 2:
        return None
    lhs, rhs = test.children
    if lhs.kind != 'name' or rhs.kind != 'constant' or rhs.const_value is not None:
        return None
    if len(binding.targets) != 1 or binding.targets[0].kind != 'name' \
            or binding.targets[0].name != lhs.name:
        return None
    forward = index.callable_by_symbol(callable_symbol)
    params = [param for param in forward.params if param.name == lhs.name]
    if len(params) != 1 or not params[0].has_default or params[0].default is None \
            or params[0].default.kind != 'constant' or params[0].default.const_value is not None:
        return None
    # The guarded branch contains this assignment alone. No sibling else or
    # hidden assignment is promoted from the graph's ambiguous merge.
    statements = [statement for module in index.modules
                  if module.source.source_id == callable_symbol.source
                  for statement in module.statements
                  if statement.enclosing_scope == callable_symbol
                  and statement.span != step.span
                  and statement.span.line >= step.span.line
                  and statement.span.end_line <= step.span.end_line]
    if len(statements) != 1 or statements[0].span != binding.span:
        return None
    return True


def _lineage(index, flow, target, source):
    """Exact direct-value aliases only; never promote a transformed graph edge."""
    if not target.args or target.args[0].kind != 'name':
        return None
    # Require an actual loop-carried assignment, not just any first argument.
    records = tuple(index.bindings_in(flow.callable_symbol))
    receivers = [row for row in records if row.value is not None and row.value.span == target.span]
    if len(receivers) != 1 or len(receivers[0].targets) != 1 or receivers[0].targets[0].kind != 'name' or receivers[0].targets[0].name != target.args[0].name:
        return None
    name, cutoff = target.args[0].name, target.span
    chain = []
    visited = set()
    external = False
    while name not in visited:
        visited.add(name)
        reaching = [record for record in records if record.span is not None
                    and _position(record.span) < _position(cutoff)
                    and not (record.value is not None and record.value.span == target.span)
                    and any(node.kind == 'name' and node.name == name
                            for item in record.targets for node in _nodes(item))]
        if not reaching:
            return None
        binding = max(reaching, key=lambda row: _position(row.span))
        if len(binding.targets) != 1 or binding.targets[0].kind != 'name' \
                or binding.assignment_kind not in {'assign', 'annassign'} or binding.value is None:
            return None
        chain.append(binding)
        if binding.value.kind == 'call' and binding.value.span == source.call.span:
            external = _optional_embedding_guard(index, flow.callable_symbol, binding, source.call)
            if external is None:
                return None
            break
        if binding.guard or binding.value.kind != 'name':
            return None
        name, cutoff = binding.value.name, binding.span
    else:
        return None
    # Writes through an alias/subscript and explicit receiver mutation are not
    # value-preserving aliases. Reads in other calls remain local-relation
    # evidence only, not a proof of arbitrary callee heap purity.
    start = source.call.span
    for module in index.modules:
        if module.source.source_id != flow.callable_symbol.source:
            continue
        for statement in module.statements:
            if statement.enclosing_scope != flow.callable_symbol \
                    or not (_position(start) < _position(statement.span) < _position(target.span)):
                continue
            if statement.kind in {'Assign', 'AnnAssign'} and statement.span not in {binding.span for binding in chain} and statement.value is not None and statement.value.kind in {'name', 'tuple', 'list', 'dict', 'set'} and any(node.kind == 'name' and node.name in visited for node in _nodes(statement.value)):
                if not all(target.kind == 'name' and target.name in visited for target in statement.targets):
                    return None
            for assigned in statement.targets:
                if assigned.kind != 'name' and any(node.kind == 'name' and node.name in visited
                                                  for node in _nodes(assigned)):
                    return None
            for node in _nodes(statement.value):
                if node.kind == 'call' and node.children:
                    callee = node.children[0]
                    if callee.kind == 'attribute' and callee.children \
                            and callee.children[0].kind == 'name' \
                            and callee.children[0].name in visited:
                        return None
    return tuple(chain), external


@dataclass(frozen=True)
class ModelStateWidth:
    value: int
    source_path: tuple[str, ...]
    source_kind: str
    checkpoint_path: tuple[str, ...] | None
    stage: object
    embedding_site: object
    repeated_sites: tuple
    external_input_unmeasured: bool
    spans: tuple[SourceSpan, ...]


def _derive(index, bundle, config_path, document, allow_root_stage):
    block = decoder_block_path_for_config(index, bundle, config_path,
                                         allow_root_stage=allow_root_stage)
    if block.status != 'resolved':
        return None
    chain = block.value
    root, owner = chain.component_root, chain.stage_occurrence
    inventory = resolve_container_inventory(index, root, owner)
    calls = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if flow.status != 'partial':
        return None
    node = root.graph.node_for(owner)
    if node is None:
        return None
    targets = tuple(proof.template.call for proof in chain.repeated_child.proofs)
    if not targets or any(call.enclosing_callable != flow.callable_symbol for call in targets):
        return None
    child = root.graph.node_for(chain.repeated_child.child_occurrence)
    child_forward = _forward_symbol(index, child.symbol) if child is not None else None
    record = index.callable_by_symbol(child_forward) if child_forward is not None else None
    params = tuple(param for param in record.params if param.name != 'self') if record else ()
    if not params or params[0].kind not in {'positional', 'posonly'} or any(
            name == params[0].name for target in targets for name, _value in target.kwargs):
        return None
    # Coroutine declarations do not synchronously produce/consume tensors.
    for symbol in (flow.callable_symbol, child_forward):
        declarations = [statement for module in index.modules if module.source.source_id == symbol.source
                        for statement in module.statements if statement.span == index.callable_by_symbol(symbol).span]
        if len(declarations) != 1 or declarations[0].kind != 'FunctionDef':
            return None
    candidates = []
    for invocation in calls.external_addressed:
        construction = invocation.construction
        if construction.external_reference.qualified_target not in _EMBEDDING \
                or construction.site.guard or len(invocation.call.args) != 1 or invocation.call.kwargs:
            continue
        site = construction.site
        kwargs = dict(site.kwargs)
        parameter_names = ('num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse')
        if set(kwargs) - set((*parameter_names, 'device', 'dtype')) or any(name in kwargs for name in parameter_names[:len(site.args)]):
            continue
        if len(kwargs) != len(site.kwargs) or any(not key for key in kwargs) \
                or len(site.args) > 7 or '_weight' in kwargs or '_freeze' in kwargs:
            continue
        dimension = site.args[1] if len(site.args) >= 2 else kwargs.get('embedding_dim')
        if dimension is None or (len(site.args) >= 2 and 'embedding_dim' in kwargs):
            continue
        # Direct source property only in this first bounded reader. General
        # dimension expressions/local aliases can be added with their own proof.
        relative = ConfigExpressionEvaluator(node.config_bindings, {}).config_path(dimension)
        if not relative:
            continue
        exact = (*config_path, *relative)
        try:
            operand = reader_operand(document, exact)
        except (ValueError, KeyError):
            continue
        if type(operand.value) is not int or operand.value <= 0:
            continue
        routes = tuple(_lineage(index, flow, target, invocation) for target in targets)
        if any(route is None for route in routes):
            continue
        spans = tuple(dict.fromkeys((*chain.address_spans, *invocation.provenance_spans,
                dimension.span, *(target.span for target in targets),
                *(binding.span for route in routes for binding in route[0]))))
        if any(not isinstance(span, SourceSpan) for span in spans):
            continue
        value = ModelStateWidth(operand.value, exact, operand.source_kind, operand.checkpoint_path,
                                owner, construction.occurrence, tuple(CallSiteId.of(t) for t in targets),
                                any(route[1] for route in routes), spans)
        candidates.append(value)
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True)
class ModelStateWidthClaimWitness:
    index: ProgramIndex
    bundle: object
    config_path: tuple[str, ...]
    document: object
    allow_root_stage: bool
    width: ModelStateWidth
    reader_symbol = 'model_unfolder.evidence.model_state_width.model_state_width_for_path'

    def _checked(self):
        actual = _derive(self.index, self.bundle, self.config_path, self.document, self.allow_root_stage)
        if actual is None or actual != self.width:
            raise ValueError('model-state width no longer matches exact source/default/connection proof')
        return actual

    def validate_result(self, result):
        width = self._checked()
        if result.status != 'resolved' or result.value is not self.width or result.owner != width.stage:
            raise ValueError('model-state width result differs from its actual retained derivation')
        if not set(width.spans) <= {span for origin in result.provenance for span in origin.spans}:
            raise ValueError('model-state width omitted its source/connection provenance')

    def project(self, owner, key, document):
        if (owner, key) != ('model', 'hidden_size') or document is not self.document:
            raise ValueError('model-state width projection belongs to one exact fact and document')
        width = self._checked()
        return ReaderFactProjection(owner, key, 'value', width.value,
            'class_default' if width.source_kind == 'class_default' else 'code_and_config',
            (width.checkpoint_path,) if width.checkpoint_path is not None else (),
            completeness='presence_only', required_spans=width.spans)


@retained_claim_reader
def model_state_width_for_path(index, bundle, config_path, *, allow_root_stage):
    from .config_access import current_prepared_document
    if not isinstance(index, ProgramIndex) or type(config_path) is not tuple \
            or any(type(item) is not str or not item for item in config_path) \
            or type(allow_root_stage) is not bool:
        raise TypeError('model-state width requires exact index/path and explicit root-stage authorization')
    document = current_prepared_document.get()
    width = _derive(index, bundle, config_path, document, allow_root_stage) if document is not None else None
    if width is None:
        return ReaderResult.failed(None, (ReaderFailure('unsupported_syntax',
            'no single direct owned embedding-to-repeated-state width route with exact source operand'),))
    return ReaderResult.resolved(width.stage, width,
        claim_witness=ModelStateWidthClaimWitness(index, bundle, config_path, document, allow_root_stage, width),
        provenance=(ReaderProvenance('source', spans=width.spans,
            config_paths=(width.source_path,), detail='positive token-embedding state route; external input width unmeasured'),))
