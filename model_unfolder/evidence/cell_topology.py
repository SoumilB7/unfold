"""Exact-owner decoder cell topology from positive residual equations.

This reader composes the U3 owner/invocation substrate with the canonical U7
attention, FFN and normalization evidence.  It interprets only exact
local assignments and returns.  A topology is emitted only when two complete
residual equations are positively present:

* sequential: attention merge -> FFN input -> FFN merge;
* parallel: attention and FFN consume the same input and one merge contains
  both exact contributions.

Residual addition may be written in the block or inside the exact addressed
attention/FFN child.  The latter is proven by evaluating that child's exact
forward and (at most) same-file module helper calls; familiar helper/class/field
names are never evidence.  Guarded rival calls, opaque transforms on a norm
boundary, disagreement across repeated-block candidates, dead equations and
unsupported expressions abstain.  Lexical order and absence of a flow edge are
never topology proof.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId, require_resolved_component_root
from .attention import decoder_gated_delta_geometry_for_path
from .attention_child import attention_child_evidence
from .config_guard import (
    ExactConfigGuardResolver,
    NormalizedConfigValue,
    constructor_normalized_config_selector,
)
from .container_inventory import resolve_container_inventory
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_candidates_for_config
from .models import SourceBundle
from .parallel_norm import (
    ExactBranchInvocation,
    exact_branch_census_at_block,
    exact_norm_sources_at_block,
)
from .execution_flow import (
    InvocationNodeId,
    resolve_addressed_invocations,
)
from .program_index import (
    CallSiteId,
    GuardStep,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_TRANSPARENT_FUNCTION_PROTOCOLS = frozenset({
    "torch.nn.functional.dropout",
})


@dataclass(frozen=True)
class CellBranchProof:
    mechanism: str                 # attention/mixer | ordinary/routed FFN
    invocation: ExactBranchInvocation
    pre_norm_site: CallSiteId | None
    post_norm_site: CallSiteId | None
    merge_kind: str                # block_add | child_integrated_add
    merge_span: SourceSpan
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.mechanism not in {
                "attention", "gated_delta_mixer",
                "ordinary_ffn", "routed_ffn"}:
            raise ValueError("a cell branch has a closed mechanism kind")
        if not isinstance(self.invocation, ExactBranchInvocation) \
                or self.invocation.mechanism != self.mechanism:
            raise ValueError("the branch retains its exact mechanism invocation")
        if self.merge_kind not in {"block_add", "child_integrated_add"}:
            raise ValueError("a branch carries an exact residual-merge proof kind")
        if not isinstance(self.merge_span, SourceSpan):
            raise TypeError("a residual merge carries an exact span")
        norm_sites = tuple(
            item for item in (self.pre_norm_site, self.post_norm_site)
            if item is not None)
        if any(not isinstance(item, CallSiteId) for item in norm_sites):
            raise TypeError("branch norm sites are exact CallSiteId values")
        if len(set(norm_sites)) != len(norm_sites):
            raise ValueError("pre/post norms are distinct exact calls")
        if any(item.enclosing_callable.source
               != self.invocation.node.call_site.enclosing_callable.source
               for item in norm_sites):
            raise ValueError(
                "branch norms belong to the exact block/component source")
        required = {
            self.invocation.call.span, self.merge_span,
            *(item.span for item in norm_sites),
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("branch provenance retains invocation + merge")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("branch spans are exact SourceSpan values")

    @property
    def pre_norm(self) -> bool:
        return self.pre_norm_site is not None

    @property
    def post_norm(self) -> bool:
        return self.post_norm_site is not None


@dataclass(frozen=True)
class DecoderCellTopologyEvidence:
    block_occurrence: OwnerOccurrenceId
    norm_placement: str            # pre | post | double
    residual_topology: str         # sequential | parallel
    parallel_input_norm_count: int | None
    mixers: tuple[CellBranchProof, ...]
    ffns: tuple[CellBranchProof, ...]
    final_return_span: SourceSpan
    norm_config_paths: tuple[tuple[str, ...], ...]
    residual_config_paths: tuple[tuple[str, ...], ...]
    config_source_kinds: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("cell topology names an exact decoder block")
        if self.norm_placement not in {"pre", "post", "double"}:
            raise ValueError("cell topology has a closed norm placement")
        if self.residual_topology not in {"sequential", "parallel"}:
            raise ValueError("cell topology has a closed residual topology")
        if self.parallel_input_norm_count is not None and (
                self.residual_topology != "parallel"
                or self.norm_placement not in {"pre", "double"}
                or self.parallel_input_norm_count not in {1, 2}):
            raise ValueError(
                "parallel input norm count requires one/two exact pre-norms")
        if self.residual_topology == "parallel" \
                and self.norm_placement in {"pre", "double"} \
                and self.parallel_input_norm_count is None:
            raise ValueError(
                "a pre-normalized parallel cell retains its exact norm count")
        if self.residual_topology != "parallel" \
                and self.parallel_input_norm_count is not None:
            raise ValueError(
                "sequential topology cannot carry a parallel norm count")
        if not self.mixers or not self.ffns \
                or any(not isinstance(item, CellBranchProof)
                       for item in (*self.mixers, *self.ffns)):
            raise TypeError("cell topology carries exact mixer and FFN proofs")
        if any(item.mechanism not in {"attention", "gated_delta_mixer"}
               for item in self.mixers) \
                or any(item.mechanism not in {"ordinary_ffn", "routed_ffn"}
                       for item in self.ffns):
            raise ValueError("cell topology has closed mixer/FFN mechanisms")
        branches = (*self.mixers, *self.ffns)
        if any(item.invocation.caller_occurrence != self.block_occurrence
               for item in branches):
            raise ValueError("both branches belong to the exact decoder block")
        expected = {
            (True, False): "pre",
            (False, True): "post",
            (True, True): "double",
        }
        states = {
            (item.pre_norm, item.post_norm) for item in branches
        }
        if len(states) != 1 or expected.get(next(iter(states))) \
                != self.norm_placement:
            raise ValueError("placement derives from both exact branch boundaries")
        exact_input_norm_count = len({
            item.pre_norm_site for item in branches
            if item.pre_norm_site is not None
        })
        if self.parallel_input_norm_count is not None \
                and self.parallel_input_norm_count != exact_input_norm_count:
            raise ValueError(
                "parallel norm count derives from the exact branch norm sites")
        if not isinstance(self.final_return_span, SourceSpan):
            raise TypeError("cell topology retains the exact final return")
        for paths in (self.norm_config_paths, self.residual_config_paths):
            if any(not isinstance(path, tuple) or not path or any(
                    not isinstance(part, str) or not part for part in path)
                    for path in paths):
                raise TypeError(
                    "cell topology config dependencies are exact paths")
            if tuple(dict.fromkeys(paths)) != paths:
                raise ValueError(
                    "cell topology config dependencies are occurrence-unique")
        all_paths = tuple(dict.fromkeys((
            *self.norm_config_paths, *self.residual_config_paths)))
        if tuple(path for path, _kind in self.config_source_kinds) != all_paths \
                or any(kind not in {"config_declared", "class_default"}
                       for _path, kind in self.config_source_kinds):
            raise ValueError(
                "every cell config dependency carries one exact source kind")
        required = {
            self.final_return_span,
            *(span for item in branches for span in item.spans),
        }
        if not required <= set(self.spans):
            raise ValueError("cell topology provenance is closed")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("cell topology spans are exact SourceSpan values")

    @property
    def attention(self) -> CellBranchProof:
        """The exact softmax-attention proof, when this cell has one."""
        values = tuple(
            item for item in self.mixers if item.mechanism == "attention")
        if len(values) != 1:
            raise AttributeError("cell does not have one exact attention proof")
        return values[0]

    @property
    def ffn(self) -> CellBranchProof:
        """Compatibility view when one FFN proof covers the selected paths."""
        if len(self.ffns) != 1:
            raise AttributeError("cell has more than one exact FFN path proof")
        return self.ffns[0]


@dataclass(frozen=True)
class _Actual:
    name: str
    value: "_Value"


@dataclass(frozen=True)
class _Value:
    kind: str
    children: tuple = ()
    call_site: CallSiteId | None = None
    span: SourceSpan | None = None
    label: str = ""
    actuals: tuple[_Actual, ...] = ()
    # A helper body may be inlined at more than one exact call site.  Its AST
    # spans are identical across those invocations, so span alone is not an
    # occurrence identity.  Keep the outer call site on values CREATED inside
    # the helper; caller-supplied values retain their original identity.
    inline_origin: CallSiteId | None = None


@dataclass(frozen=True)
class _Merge:
    value: _Value
    span: SourceSpan
    branch_sites: frozenset[CallSiteId]
    returned: bool
    kind: str = "block_add"


@dataclass(frozen=True)
class _EvaluatedCallable:
    values: tuple[_Value, ...]
    returns: tuple[tuple[_Value, SourceSpan, tuple], ...]
    role_calls: tuple[tuple[CallSiteId, _Value], ...]
    guard_config_paths: tuple[tuple[str, ...], ...] = ()
    guard_spans: tuple[SourceSpan, ...] = ()
    guard_source_kinds: tuple[tuple[tuple[str, ...], str], ...] = ()
    guard_complete: bool = True


@dataclass(frozen=True)
class _CompositeExpansion:
    callable_symbol: SymbolId
    mechanism: ExactBranchInvocation
    norms: tuple[tuple[CallSiteId, tuple], ...]
    transparent_sites: frozenset[CallSiteId]

    def __post_init__(self) -> None:
        if not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("a composite expansion names one exact callable")
        if not isinstance(self.mechanism, ExactBranchInvocation):
            raise TypeError("a composite expansion carries one exact mechanism")
        if any(not isinstance(site, CallSiteId) for site, _value in self.norms):
            raise TypeError("composite norms retain exact call sites")


@dataclass(frozen=True)
class _BranchCases:
    selected: tuple[
        tuple[tuple[GuardStep, ...], ExactBranchInvocation,
              ExactBranchInvocation], ...]
    all_cases: tuple[
        tuple[tuple[GuardStep, ...], ExactBranchInvocation,
              ExactBranchInvocation], ...]
    source_exhaustive: bool
    config_paths: tuple[tuple[str, ...], ...] = ()
    config_source_kinds: tuple[tuple[tuple[str, ...], str], ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not self.selected or not self.all_cases:
            raise ValueError("cell topology needs non-empty selected/source cases")
        if any(item not in self.all_cases for item in self.selected):
            raise ValueError("selected cell cases are an exact source-case subset")
        if self.config_paths and not self.spans:
            raise ValueError("config-selected cases retain source binding spans")
        if tuple(path for path, _kind in self.config_source_kinds) \
                != self.config_paths or any(
                    kind not in {"config_declared", "class_default"}
                    for _path, kind in self.config_source_kinds):
            raise ValueError(
                "selected cell cases retain every exact config source kind")


def _branch_cases(
        index, root, block_occurrence, mixers, ffns, *, config_selector):
    """Return exact executable mixer/FFN path pairs.

    Guarded alternatives are never unioned.  A source-exhaustive binary
    if/else may be evaluated on both paths.  Otherwise the exact block field
    feeding the guards must resolve to one config path whose current value(s)
    select every executable case.
    """
    if not mixers or not ffns:
        return ReaderFailure(
            "incomplete_graph", "cell paths require mixer and FFN calls")
    branches = (*mixers, *ffns)
    guarded = tuple(item for item in branches if item.call.guard)
    if not guarded:
        if len(mixers) != 1 or len(ffns) != 1:
            return ReaderFailure(
                "conflict", "unguarded cell has rival mixer or FFN calls")
        value = (((), mixers[0], ffns[0]),)
        return _BranchCases(value, value, True)

    decisions = {
        _span_key(item.call.guard[0].span) for item in guarded
        if item.call.guard and item.call.guard[0].span is not None
    }
    if None in decisions or len(decisions) != 1:
        return ReaderFailure(
            "incomplete_graph",
            "guarded cell calls do not share one exact decision")
    case_guards = tuple(dict.fromkeys(
        item.call.guard for item in guarded))
    unguarded_mixers = tuple(item for item in mixers if not item.call.guard)
    unguarded_ffns = tuple(item for item in ffns if not item.call.guard)
    if len(unguarded_mixers) > 1 or len(unguarded_ffns) > 1:
        return ReaderFailure(
            "conflict", "cell path has rival unguarded mechanism calls")

    cases = []
    for guard in case_guards:
        path_mixers = tuple(
            item for item in mixers if item.call.guard == guard)
        path_ffns = tuple(
            item for item in ffns if item.call.guard == guard)
        mixer = path_mixers or unguarded_mixers
        ffn = path_ffns or unguarded_ffns
        if len(mixer) != 1 or len(ffn) != 1:
            return ReaderFailure(
                "incomplete_graph",
                "each guarded path needs one exact mixer and one exact FFN")
        cases.append((guard, mixer[0], ffn[0]))

    # An exact top-level if/else with no nested predicate covers both runtime
    # paths by source alone.  Elif/nested-guard forms need their real operand.
    first_kinds = {guard[0].kind for guard in case_guards}
    source_exhaustive = (
        len(case_guards) == 2
        and first_kinds == {"if", "else"}
        and all(len(guard) == 1 for guard in case_guards)
    )
    path_info = _guard_selector_path(
        index, root, block_occurrence, case_guards)
    if path_info is None:
        if source_exhaustive:
            value = tuple(cases)
            return _BranchCases(value, value, True)
        return ReaderFailure(
            "incomplete_graph",
            "guarded cell alternatives lack an exact source-bound selector")
    selector_path, selector_field, assignment_span = path_info
    selected_value = None
    selected_dependencies = ()
    selected_spans = ()
    if config_selector is not None:
        selected = config_selector(selector_path)
        if isinstance(selected, NormalizedConfigValue):
            selected_value = selected.value
            selected_dependencies = selected.dependencies
            selected_spans = selected.spans
        elif isinstance(selected, tuple) and len(selected) == 3 \
                and isinstance(selected[0], bool):
            present, selected_value, selected_kind = selected
            if not present:
                selected_value = None
            else:
                selected_dependencies = ((selector_path, selected_kind),)
        else:
            selected_value = selected
            if selected is not None:
                selected_dependencies = (
                    (selector_path, "config_declared"),)
        if any(kind not in {"config_declared", "class_default"}
               for _path, kind in selected_dependencies):
            return ReaderFailure(
                "incomplete_graph",
                "the exact cell selector has untyped config provenance")
    selected_guards = _guards_for_selected_value(
        case_guards, selector_field, selected_value,
        source_exhaustive=source_exhaustive)
    if selected_guards is None:
        if source_exhaustive:
            value = tuple(cases)
            return _BranchCases(value, value, True)
        return ReaderFailure(
            "incomplete_graph",
            "the exact cell selector value does not close every guarded path")
    selected = tuple(item for item in cases if item[0] in selected_guards)
    if not selected:
        return ReaderFailure(
            "incomplete_graph", "the exact cell selector chose no path")
    spans = tuple(dict.fromkeys((
        assignment_span,
        *selected_spans,
        *(step.span for guard in selected_guards for step in guard),
    )))
    dependency_paths = tuple(
        path for path, _kind in selected_dependencies)
    return _BranchCases(
        selected, tuple(cases), source_exhaustive, dependency_paths,
        tuple(selected_dependencies), spans)


def _guard_selector_path(index, root, block_occurrence, guards):
    atoms = tuple(
        atom for guard in guards
        if (atom := _guard_atom(guard)) is not None)
    fields = {field for field, _value in atoms}
    if len(fields) != 1:
        return None
    field = next(iter(fields))
    block = root.graph.node_for(block_occurrence)
    if block is None:
        return None
    assignments = tuple(
        item for item in index.field_assigns_of(block.symbol)
        if item.field == field and not item.guard and item.span is not None)
    if len(assignments) != 1:
        return None
    assignment = assignments[0]
    init_record = index.callable_by_symbol(assignment.enclosing_callable)
    if init_record is None:
        return None
    parameters = frozenset(
        item.name for item in init_record.params if item.name != "self")
    access = _config_access_expression(assignment.value, parameters)
    if access is None:
        return None
    parameter, local_path = access
    bindings = tuple(
        item for item in block.config_bindings if item.parameter == parameter)
    if len(bindings) != 1:
        return None
    resolved = bindings[0].resolved_path(local_path)
    if resolved is None:
        return None
    prefix = tuple(getattr(root, "config_path", ()) or ())
    return (*prefix, *resolved), field, assignment.span


def _config_access_expression(value, constructor_parameters):
    current = value
    if current.kind == "subscript":
        if len(current.children) != 2:
            return None
        index_expr = current.children[1]
        if index_expr.kind != "name" \
                or index_expr.name not in constructor_parameters:
            return None
        current = current.children[0]
    segments = []
    while current.kind == "attribute" and len(current.children) == 1 \
            and current.name:
        segments.append(current.name)
        current = current.children[0]
    if current.kind != "name" or not current.name or not segments:
        return None
    return current.name, tuple(reversed(segments))


def _guard_atom(guard):
    values = []
    for step in guard:
        if step.kind not in {"if", "elif"} or step.test is None:
            continue
        test = step.test
        field = _self_attribute(test)
        if field is not None:
            values.append((field, True))
            continue
        if test.kind == "compare" and test.operator == "==" \
                and len(test.children) == 2:
            left, right = test.children
            field = _self_attribute(left)
            constant = right.const_value if right.kind == "constant" else None
            if field is None:
                field = _self_attribute(right)
                constant = left.const_value if left.kind == "constant" else None
            if field is not None and constant is not None:
                values.append((field, constant))
    return values[0] if len(values) == 1 else None


def _self_attribute(expr):
    if expr.kind != "attribute" or len(expr.children) != 1:
        return None
    root = expr.children[0]
    return expr.name if root.kind == "name" and root.name == "self" else None


def _guards_for_selected_value(
        guards, field, selected_value, *, source_exhaustive):
    atoms = {guard: _guard_atom(guard) for guard in guards}
    if isinstance(selected_value, bool):
        direct = tuple(
            guard for guard, atom in atoms.items()
            if atom == (field, selected_value))
        if direct:
            return direct
        if not selected_value and source_exhaustive:
            fallback = tuple(
                guard for guard in guards if guard[0].kind == "else")
            return fallback if len(fallback) == 1 else None
        return None
    values = (
        tuple(dict.fromkeys(selected_value))
        if isinstance(selected_value, (tuple, list)) else
        (selected_value,) if isinstance(selected_value, (str, int)) else ())
    if not values:
        return None
    selected = []
    for value in values:
        matched = tuple(
            guard for guard, atom in atoms.items()
            if atom == (field, value))
        if len(matched) != 1:
            return None
        selected.append(matched[0])
    return tuple(dict.fromkeys(selected))


def decoder_cell_topology_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
    guard_config_selector=None,
) -> ReaderResult[DecoderCellTopologyEvidence]:
    """Prove one unanimous exact cell topology for a selected config path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_cell_topology_for_path requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_cell_topology_for_path requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")

    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    gated_delta = decoder_gated_delta_geometry_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    gated_delta_by_block = (
        {gated_delta.value.block_occurrence: gated_delta.value}
        if gated_delta.status == "resolved" else {})
    results = tuple(
        _cell_topology_at_block(
            index, candidates.value.component_root, occurrence,
            gated_delta=gated_delta_by_block.get(occurrence),
            config_selector=config_selector,
            guard_config_selector=guard_config_selector)
        for occurrence in candidates.value.occurrences)
    ambiguous = tuple(item for item in results if item.status == "ambiguous")
    if ambiguous:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for item in ambiguous for span in item.ambiguity.sites))),
            provenance=candidates.provenance)
    if any(item.status != "resolved" for item in results):
        failures = tuple(
            failure for item in results for failure in item.failures)
        return ReaderResult.failed(
            candidates.value.stage_occurrence,
            failures or (ReaderFailure(
                "incomplete_graph",
                "not every exact block candidate proves a cell topology"),),
            provenance=candidates.provenance)
    signatures = {
        (item.value.norm_placement, item.value.residual_topology,
         item.value.parallel_input_norm_count)
        for item in results
    }
    if len(signatures) != 1:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                item.value.spans[0] for item in results))),
            provenance=candidates.provenance)
    value = results[0].value
    return ReaderResult.resolved(
        candidates.value.stage_occurrence, value,
        provenance=(
            *candidates.provenance,
            *(origin for item in results for origin in item.provenance),
            ReaderProvenance(
                "derived",
                detail=(
                    "every exact repeated-block candidate proves the same "
                    "norm placement and residual equations")),
        ))


def _cell_topology_at_block(
        index, root, block_occurrence, *, gated_delta=None,
        config_selector=None, guard_config_selector=None):
    root = require_resolved_component_root(
        root, caller="_cell_topology_at_block")
    census = exact_branch_census_at_block(index, root, block_occurrence)
    if census.status != "resolved":
        return census
    block = root.graph.node_for(block_occurrence)
    if block is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the decoder block is absent from its owner graph"),))
    base_selector = guard_config_selector or config_selector
    effective_selector = constructor_normalized_config_selector(
        index, block, base_selector,
        config_prefix=tuple(getattr(root, "config_path", ()) or ()))
    mixers = [census.value.attention]
    if gated_delta is not None:
        addressed = tuple(
            item for item in census.value.invocations.addressed
            if item.callee_owner_occurrence == gated_delta.mixer_occurrence)
        if len(addressed) != 1:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the proven recurrent mixer has no unique exact block call"),))
        invocation = addressed[0]
        mixer_node = root.graph.node_for(invocation.callee_owner_occurrence)
        if mixer_node is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "out_of_owner",
                "the proven recurrent mixer is absent from the owner graph"),))
        spans = tuple(dict.fromkeys((
            *invocation.provenance_spans, *gated_delta.spans)))
        mixers.append(ExactBranchInvocation(
            block_occurrence, invocation.call,
            InvocationNodeId(invocation.call_site, "addressed"),
            (mixer_node.symbol,),
            "gated_delta_mixer", "addressed_child", spans))

    cases = _branch_cases(
        index, root, block_occurrence, tuple(mixers), census.value.ffn,
        config_selector=effective_selector)
    if isinstance(cases, ReaderFailure):
        return ReaderResult.failed(block_occurrence, (cases,))

    norms = exact_norm_sources_at_block(
        index, root, block_occurrence, census.value.invocations,
        config_selector=effective_selector)
    transparent_sites = frozenset(
        item.call_site for item in census.value.invocations.external_addressed
        if item.construction.external_reference.qualified_target
        == "torch.nn.Dropout") | _functional_transparent_sites(
            index, census.value.flow.callable_symbol)
    expansions = {}
    attention_path = attention_child_evidence(
        index, root, block_occurrence)
    if attention_path.status == "resolved" \
            and len(attention_path.value.invocation_path) > 1:
        expansion = _composite_attention_expansion(
            index, root, census.value.attention, attention_path.value)
        if isinstance(expansion, ReaderFailure):
            return ReaderResult.failed(block_occurrence, (expansion,))
        expansions[census.value.attention.node.call_site] = expansion
        norms = {**norms, **dict(expansion.norms)}
    evaluated_cases = {}
    for active_guard, mixer, ffn in cases.all_cases:
        result = _evaluate_topology_case(
            index, census.value.flow.callable_symbol, mixer, ffn, norms,
            transparent_sites, active_guard, expansions)
        # Preserve source-only proofs as the first authority.  A concrete
        # config path is consulted only when unresolved guarded assignments
        # prevent those equations from closing (for example, unguarded branch
        # calls with guarded inputs followed by ``branch += other``).
        # This prevents an unrelated training/debug guard from weakening an
        # otherwise source-invariant topology.
        if isinstance(result, ReaderFailure) \
                and config_selector is not None and not active_guard:
            resolver = ExactConfigGuardResolver(
                index, block, effective_selector,
                config_prefix=tuple(getattr(root, "config_path", ()) or ()))
            result = _evaluate_topology_case(
                index, census.value.flow.callable_symbol, mixer, ffn, norms,
                transparent_sites, active_guard, expansions,
                guard_resolver=resolver)
        evaluated_cases[(active_guard, mixer, ffn)] = result
    values = tuple(evaluated_cases[item] for item in cases.selected)
    if any(isinstance(item, ReaderFailure) for item in values):
        return ReaderResult.failed(block_occurrence, tuple(
            item for item in values if isinstance(item, ReaderFailure)))
    signatures = {(item[0], item[1]) for item in values}
    if len(signatures) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "conflict",
            "selected cell paths prove different norm/residual topologies"),))
    placement, topology = next(iter(signatures))
    mixer_proofs = tuple(dict.fromkeys(item[2] for item in values))
    ffn_proofs = tuple(dict.fromkeys(item[3] for item in values))
    final_spans = tuple(dict.fromkeys(item[4] for item in values))
    if len(final_spans) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "selected cell paths have rival final returns"),))
    final_span = final_spans[0]
    all_values = tuple(evaluated_cases.values())
    all_source_proven = all(
        not isinstance(item, ReaderFailure) for item in all_values)
    evaluated_config_paths = tuple(dict.fromkeys(
        path for item in values for path in item[5]))
    evaluated_guard_spans = tuple(dict.fromkeys(
        span for item in values for span in item[6]))
    evaluated_source_kinds = tuple(dict.fromkeys((
        *cases.config_source_kinds,
        *(source for item in values for source in item[7]),
    )))
    if any(len({kind for candidate, kind in evaluated_source_kinds
                if candidate == path}) != 1
           for path in {candidate for candidate, _kind
                        in evaluated_source_kinds}):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "conflict",
            "one cell config dependency has rival evidence origins"),))
    norm_source_invariant = (
        cases.source_exhaustive and all_source_proven
        and len({item[0] for item in all_values}) == 1
        and not evaluated_config_paths)
    residual_source_invariant = (
        cases.source_exhaustive and all_source_proven
        and len({item[1] for item in all_values}) == 1
        and not evaluated_config_paths)
    all_config_paths = tuple(dict.fromkeys((
        *cases.config_paths, *evaluated_config_paths)))
    norm_config_paths = () if norm_source_invariant else all_config_paths
    residual_config_paths = (
        () if residual_source_invariant else all_config_paths)
    spans = tuple(dict.fromkeys((
        *census.value.spans,
        *(span for item in mixer_proofs for span in item.spans),
        *(span for item in ffn_proofs for span in item.spans),
        final_span,
        *cases.spans,
        *evaluated_guard_spans,
    )))
    input_norm_sites = {
        item.pre_norm_site for item in (*mixer_proofs, *ffn_proofs)
        if item.pre_norm_site is not None
    }
    parallel_input_norm_count = (
        len(input_norm_sites)
        if topology == "parallel" and placement in {"pre", "double"}
        else None)
    if parallel_input_norm_count not in {None, 1, 2}:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "conflict",
            "parallel branches use more than two distinct input norms"),))
    value = DecoderCellTopologyEvidence(
        block_occurrence, placement, topology, parallel_input_norm_count,
        mixer_proofs, ffn_proofs, final_span,
        norm_config_paths, residual_config_paths,
        tuple((path, dict(evaluated_source_kinds).get(
            path, "config_declared")) for path in dict.fromkeys((
                *norm_config_paths, *residual_config_paths))),
        spans)
    config_paths = tuple(dict.fromkeys((
        *norm_config_paths, *residual_config_paths)))
    provenance_kind = "code_and_config" if config_paths else "source"
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            provenance_kind, spans=spans, config_paths=config_paths,
            detail=(
                "every selected exact mixer/FFN path proves the same norm "
                "boundaries and residual equations; config dependence is "
                "retained separately for each derived fact")),))


def _composite_attention_expansion(index, root, outer_branch, evidence):
    """Describe one exact wrapper whose forward contains the attention call.

    Only a one-hop structural wrapper is expanded here.  Deeper paths remain
    explicit unsupported evidence; silently skipping a wrapper could erase a
    residual or normalization boundary.
    """
    if len(evidence.invocation_path) != 2 \
            or evidence.invocation_path[0].call_site \
            != outer_branch.node.call_site:
        return ReaderFailure(
            "incomplete_graph",
            "nested attention topology requires one exact wrapper hop")
    wrapper_occurrence = evidence.child_occurrence
    wrapper = root.graph.node_for(wrapper_occurrence)
    if wrapper is None:
        return ReaderFailure(
            "out_of_owner", "attention wrapper is absent from the owner graph")
    inner_invocation = evidence.invocation_path[1]
    inner = root.graph.node_for(inner_invocation.callee_owner_occurrence)
    if inner is None:
        return ReaderFailure(
            "out_of_owner", "attention compute child is absent from its graph")
    inventory = resolve_container_inventory(index, root, wrapper_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, wrapper_occurrence, inventory)
    if invocations.status != "resolved" \
            or inner_invocation not in invocations.addressed:
        return ReaderFailure(
            "incomplete_graph",
            "attention wrapper invocation census does not round-trip")
    norms = exact_norm_sources_at_block(
        index, root, wrapper_occurrence, invocations)
    transparent = frozenset(
        item.call_site for item in invocations.external_addressed
        if item.construction.external_reference.qualified_target
        == "torch.nn.Dropout") | _functional_transparent_sites(
            index, SymbolId(
                wrapper.symbol.source,
                f"{wrapper.symbol.qualified_name}.forward"))
    branch = ExactBranchInvocation(
        wrapper_occurrence, inner_invocation.call,
        InvocationNodeId(inner_invocation.call_site, "addressed"),
        (inner.symbol,), "attention", "addressed_child",
        tuple(dict.fromkeys((
            *inner_invocation.provenance_spans, *evidence.compute.spans))))
    return _CompositeExpansion(
        SymbolId(wrapper.symbol.source, f"{wrapper.symbol.qualified_name}.forward"),
        branch, tuple(norms.items()), transparent)


def _functional_transparent_sites(index, callable_symbol):
    sites = []
    for call in index.calls_in(callable_symbol):
        proof = resolve_import_reference(
            index, callable_symbol.source, callable_symbol, call.callee)
        if proof is not None \
                and proof.qualified_target in _TRANSPARENT_FUNCTION_PROTOCOLS:
            sites.append(CallSiteId.of(call))
    return frozenset(sites)


def _evaluate_topology_case(
        index, forward, mixer, ffn, norms, transparent_sites, active_guard,
        expansions, guard_resolver=None):
    role_by_site = {
        mixer.node.call_site: "mixer",
        ffn.node.call_site: "ffn",
        **{site: "norm" for site in norms},
        **{site: "transparent" for site in transparent_sites},
    }
    branch_by_site = {
        mixer.node.call_site: mixer,
        ffn.node.call_site: ffn,
    }
    evaluated = _evaluate_callable(
        index, forward, role_by_site, branch_by_site,
        active_guard=(active_guard or None), expansions=expansions,
        guard_resolver=guard_resolver)
    if not evaluated.guard_complete:
        return ReaderFailure(
            "incomplete_graph",
            "the exact cell control path contains an unresolved guard")
    role_values = dict(evaluated.role_calls)
    if set(branch_by_site) - set(role_values):
        return ReaderFailure(
            "incomplete_graph", "an exact branch call has no evaluated value")
    if not evaluated.returns or any(
            guard for _value, _span, guard in evaluated.returns):
        return ReaderFailure(
            "incomplete_graph",
            "cell topology requires an exact unguarded return path")
    if len(evaluated.returns) != 1:
        return ReaderFailure(
            "incomplete_graph", "rival return paths are not yet equivalent")
    final_value, final_span, _guard = evaluated.returns[0]
    merges = list(_block_merges(
        evaluated.values, final_value, frozenset(branch_by_site)))
    for branch in (mixer, ffn):
        if branch.node.call_site in expansions:
            continue
        proof = _child_integrated_merge(
            index, branch, role_values[branch.node.call_site], final_value)
        if proof is not None:
            merges.append(proof)
    result = _classify_equations(
        mixer, ffn, role_values, norms, tuple(merges), final_value,
        final_span)
    if isinstance(result, ReaderFailure):
        return result
    return (
        *result, final_span,
        evaluated.guard_config_paths, evaluated.guard_spans,
        evaluated.guard_source_kinds)


def _evaluate_callable(index, callable_symbol, role_by_site, branch_by_site,
                       initial_env=None, depth=0, active_guard=None,
                       expansions=None, guard_resolver=None):
    callable_record = index.callable_by_symbol(callable_symbol)
    if callable_record is None:
        return _EvaluatedCallable((), (), ())
    env = dict(initial_env or {})
    expansions = expansions or {}
    for param in callable_record.params:
        if param.kind == "positional" and param.name not in env:
            env[param.name] = _Value("input", label=param.name)
    calls = {
        _span_key(call.span): call for call in index.calls_in(callable_symbol)
        if call.span is not None
    }
    values = []
    role_calls = {}
    returns = []
    def event_is_enabled(item):
        if active_guard is not None:
            return _event_on_path(item.guard, active_guard)
        if guard_resolver is None:
            return True
        enabled = guard_resolver.enabled(item.guard, callable_symbol)
        return enabled is not False

    events = [
        (_position(item.span), 0, "binding", item)
        for item in index.bindings_in(callable_symbol)
        if item.span is not None
        and event_is_enabled(item)
    ] + [
        (_position(item.span), 1, "return", item)
        for item in index.return_observations_in(callable_symbol)
        if item.span is not None
        and event_is_enabled(item)
    ]
    events.sort(key=lambda item: (item[0], item[1]))

    def evaluate(expr):
        if expr is None:
            return _Value("unknown")
        if expr.kind == "name":
            if expr.name in env:
                return _Value(
                    "reference", (env[expr.name],), span=expr.span,
                    label=expr.name)
            return _Value("input", span=expr.span, label=expr.name)
        if expr.kind == "call":
            call = calls.get(_span_key(expr.span)) if expr.span else None
            if call is None:
                return _Value("unknown", span=expr.span)
            site = CallSiteId.of(call)
            formal_actuals = None
            branch = branch_by_site.get(site)
            if branch is not None:
                formal_actuals = _common_formal_actuals(
                    index, branch.candidate_symbols, call)
            if formal_actuals is None:
                receiver_exprs = ()
                if call.callee.kind == "attribute" \
                        and call.callee.children \
                        and _receiver_is_data(call.callee.children[0], env):
                    receiver_exprs = (call.callee.children[0],)
                actual_exprs = receiver_exprs + tuple(call.args) + tuple(
                    value for _name, value in call.kwargs)
                actuals = tuple(evaluate(item) for item in actual_exprs)
                named_actuals = ()
            else:
                named_actuals = tuple(
                    _Actual(name, evaluate(item))
                    for name, item in formal_actuals)
                actuals = tuple(item.value for item in named_actuals)
            role = role_by_site.get(site)
            if role is not None:
                if role == "transparent":
                    if not call.args:
                        return _Value("unknown", span=call.span)
                    actuals = (evaluate(call.args[0]),)
                    named_actuals = ()
                value = _Value(
                    role, actuals, site, call.span, actuals=named_actuals)
                role_calls[site] = value
                expansion = expansions.get(site)
                if expansion is not None:
                    if not named_actuals:
                        return _Value("unknown", span=call.span)
                    nested_roles = {
                        expansion.mechanism.node.call_site: "mixer",
                        **{norm_site: "norm"
                           for norm_site, _proof in expansion.norms},
                        **{transparent: "transparent"
                           for transparent in expansion.transparent_sites},
                    }
                    nested = _evaluate_callable(
                        index, expansion.callable_symbol, nested_roles,
                        {expansion.mechanism.node.call_site:
                         expansion.mechanism},
                        initial_env={item.name: item.value
                                     for item in named_actuals},
                        depth=depth + 1)
                    nested_calls = dict(nested.role_calls)
                    inner = nested_calls.get(
                        expansion.mechanism.node.call_site)
                    if len(nested.returns) != 1 or nested.returns[0][2] \
                            or inner is None:
                        return _Value("unknown", span=call.span)
                    rewritten_inner = _replace_call_site(
                        inner, expansion.mechanism.node.call_site, site)
                    role_calls[site] = rewritten_inner
                    return _replace_call_site(
                        nested.returns[0][0],
                        expansion.mechanism.node.call_site, site)
                return value
            inlined = _inline_module_function(
                index, callable_symbol, call, evaluate, depth)
            if inlined is not None:
                return inlined
            if call.callee.kind == "attribute" \
                    and call.callee.name == "to" and actuals \
                    and _direct_role_site(actuals[0], "norm") is not None:
                return _Value(
                    "transparent", (actuals[0],), site, call.span)
            return _Value("transform", actuals, site, call.span)
        if expr.kind == "binop" and expr.operator == "+" \
                and len(expr.children) == 2:
            return _Value(
                "add", tuple(evaluate(item) for item in expr.children),
                span=expr.span)
        if expr.kind in {"tuple", "list"}:
            return _Value(
                expr.kind, tuple(evaluate(item) for item in expr.children),
                span=expr.span)
        children = tuple(
            evaluate(item) for item in expr.children if item is not None)
        children += tuple(
            evaluate(item) for _name, item in expr.keyword_children
            if item is not None)
        return _Value("transform", children, span=expr.span, label=expr.kind)

    for _pos, _priority, kind, item in events:
        if kind == "binding":
            value = evaluate(item.value)
            if item.assignment_kind == "augassign":
                prior = _augmented_assignment_prior(
                    index, callable_symbol, item, env)
                operator = _augmented_assignment_operator(
                    index, callable_symbol, item)
                if prior is None or operator != "+":
                    value = _Value("unknown", span=item.span)
                else:
                    value = _Value(
                        "add", (prior, value), span=item.span)
            values.append(value)
            # Once a single exact path is selected, its guarded statements are
            # unconditional within that path.  Keeping the original guard here
            # would manufacture a rival old value and launder it into `choice`.
            _bind(
                env, item.targets, value,
                () if active_guard is not None or guard_resolver is not None
                else item.guard)
        else:
            value = evaluate(item.value)
            values.append(value)
            returns.append((value, item.span, item.guard))
    return _EvaluatedCallable(
        tuple(values), tuple(returns), tuple(role_calls.items()),
        tuple(dict.fromkeys(guard_resolver.paths)) if guard_resolver else (),
        tuple(dict.fromkeys(guard_resolver.spans)) if guard_resolver else (),
        tuple(dict.fromkeys(guard_resolver.source_kinds))
        if guard_resolver else (),
        guard_resolver.complete if guard_resolver else True)


def _augmented_assignment_prior(index, callable_symbol, binding, env):
    if len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    if target.kind != "name" or not target.name:
        return None
    return env.get(target.name)


def _augmented_assignment_operator(index, callable_symbol, binding):
    edges = tuple(
        item for item in index.dataflow
        if item.enclosing_callable == callable_symbol
        and item.span == binding.span and item.op.startswith("aug:"))
    if len(edges) != 1:
        return None
    return edges[0].op.removeprefix("aug:")


def _receiver_is_data(expression, env):
    if expression.kind == "call":
        return True
    names = {
        item.name for item in _expr_nodes(expression)
        if item.kind == "name" and item.name and item.name != "self"
    }
    return bool(names & set(env))


def _expr_nodes(expression):
    yield expression
    for child in expression.children:
        if child is not None:
            yield from _expr_nodes(child)
    for _name, child in expression.keyword_children:
        if child is not None:
            yield from _expr_nodes(child)


def _direct_role_site(value, kind):
    value = _unwrap_reference(value)
    if value.kind == kind:
        return value.call_site
    if value.kind == "transparent" and len(value.children) == 1:
        return _direct_role_site(value.children[0], kind)
    return None


def _event_on_path(event_guard, active_guard):
    if active_guard is None:
        return True
    if not event_guard:
        return True
    if not active_guard:
        return False
    return tuple(event_guard) == tuple(active_guard)


def _inline_module_function(index, caller_symbol, call, evaluate, depth):
    if depth >= 2 or call.callee.kind != "name" or not call.callee.name:
        return None
    symbol = SymbolId(caller_symbol.source, call.callee.name)
    target = index.callable_by_symbol(symbol)
    bindings = tuple(
        item for item in index.module_bindings_in(caller_symbol.source)
        if item.name == call.callee.name)
    if target is None or len(bindings) != 1 \
            or bindings[0].kind != "function":
        return None
    bindings = _bind_call_actuals(target, call, skip_receiver=False)
    if bindings is None:
        return None
    env = {name: evaluate(expr) for name, expr in bindings}
    transparent = _functional_transparent_sites(index, symbol)
    result = _evaluate_callable(
        index, symbol,
        {site: "transparent" for site in transparent}, {},
        initial_env=env, depth=depth + 1)
    if len(result.returns) != 1 or result.returns[0][2]:
        return None
    return _qualify_inline_value(
        result.returns[0][0], CallSiteId.of(call), target.span)


def _qualify_inline_value(value, origin, callable_span):
    """Namespace helper-created values by their exact invocation.

    Values injected as call arguments have spans outside ``callable_span`` and
    therefore remain the caller's values.  Values whose syntax belongs to the
    helper body receive the outer call-site occurrence.  Two calls to one
    residual helper can then never be collapsed merely because their internal
    ``+`` expression has the same source line.
    """
    children = tuple(
        _qualify_inline_value(item, origin, callable_span)
        for item in value.children)
    actuals = tuple(
        _Actual(
            item.name,
            _qualify_inline_value(item.value, origin, callable_span))
        for item in value.actuals)
    inline_origin = value.inline_origin
    if inline_origin is None and value.span is not None \
            and _span_within(value.span, callable_span):
        inline_origin = origin
    return _Value(
        value.kind, children, value.call_site, value.span, value.label,
        actuals, inline_origin)


def _replace_call_site(value, old_site, new_site):
    children = tuple(
        _replace_call_site(item, old_site, new_site)
        for item in value.children)
    actuals = tuple(
        _Actual(item.name, _replace_call_site(item.value, old_site, new_site))
        for item in value.actuals)
    site = new_site if value.call_site == old_site else value.call_site
    return _Value(
        value.kind, children, site, value.span, value.label, actuals,
        value.inline_origin)


def _bind(env, targets, value, guard):
    if len(targets) == 1 and targets[0].kind in {"tuple", "list"}:
        target_items = targets[0].children
        if value.kind in {"tuple", "list"} \
                and len(value.children) == len(target_items):
            values = value.children
        else:
            values = (value, *(
                _Value("unknown") for _ in range(max(0, len(target_items) - 1))))
        for target, item in zip(target_items, values):
            _bind_one(env, target, item, guard)
        return
    for target in targets:
        _bind_one(env, target, value, guard)


def _bind_one(env, target, value, guard):
    if target.kind != "name" or not target.name:
        return
    if guard:
        # A guarded write never proves the variable's value on the rival path.
        # Preserve the old value when known and an explicit unknown otherwise;
        # absence of a prior assignment is not evidence the guard always fires.
        prior = env.get(target.name, _Value("unknown", label=target.name))
        choices = _choice_items(prior) + _choice_items(value)
        env[target.name] = _Value(
            "choice", tuple(dict.fromkeys(choices)), span=value.span)
    else:
        env[target.name] = value


def _choice_items(value):
    return value.children if value.kind == "choice" else (value,)


def _common_formal_actuals(index, symbols, call):
    signatures = []
    records = []
    for symbol in symbols:
        forward = index.callable_by_symbol(SymbolId(
            symbol.source, f"{symbol.qualified_name}.forward"))
        if forward is None:
            return None
        positional = tuple(
            param.name for param in forward.params
            if param.kind == "positional")
        if forward.owner is not None:
            positional = positional[1:]
        signatures.append(positional)
        records.append(forward)
    if not signatures or len(set(signatures)) != 1:
        return None
    bindings = tuple(
        _bind_call_actuals(record, call, skip_receiver=record.owner is not None)
        for record in records)
    if any(item is None for item in bindings) or len(set(bindings)) != 1:
        return None
    return bindings[0]


def _bind_call_actuals(callable_record, call, *, skip_receiver):
    params = tuple(
        param for param in callable_record.params if param.kind == "positional")
    accepts_extra_kwargs = any(
        param.kind == "kwarg" for param in callable_record.params)
    if skip_receiver:
        params = params[1:]
    names = tuple(param.name for param in params)
    supplied = {}
    for position, value in enumerate(call.args):
        if position >= len(names):
            return None
        supplied[names[position]] = value
    for name, value in call.kwargs:
        if name not in names:
            if accepts_extra_kwargs:
                continue
            return None
        if name in supplied:
            return None
        supplied[name] = value
    return tuple((name, supplied[name]) for name in names if name in supplied)


def _block_merges(values, final_value, branch_sites):
    seen = set()
    for value in values:
        for item in _walk_values(value):
            if item.kind != "add" or item.span is None:
                continue
            # An inlined helper span is source identity, not invocation
            # identity.  Pair it with the exact outer call occurrence so two
            # calls to the same helper remain two residual equations.
            key = (_span_key(item.span), item.inline_origin)
            if key in seen:
                continue
            seen.add(key)
            sites = frozenset(
                site for site in branch_sites if _contains_site(item, site))
            if not sites:
                continue
            yield _Merge(
                item, item.span, sites, _contains_value(final_value, item))


def _child_integrated_merge(index, branch, block_value, final_value):
    actuals = block_value.actuals
    if not actuals:
        return None
    proofs = []
    for symbol in branch.candidate_symbols:
        forward = index.callable_by_symbol(SymbolId(
            symbol.source, f"{symbol.qualified_name}.forward"))
        if forward is None:
            return None
        positional = tuple(
            param.name for param in forward.params if param.kind == "positional")
        if forward.owner is not None:
            positional = positional[1:]
        if tuple(item.name for item in actuals) != tuple(
                name for name in positional
                if name in {item.name for item in actuals}):
            return None
        env = {item.name: item.value for item in actuals}
        evaluated = _evaluate_callable(
            index, forward.symbol, {}, {}, initial_env=env, depth=0)
        if len(evaluated.returns) != 1 or evaluated.returns[0][2]:
            return None
        returned, return_span, _guard = evaluated.returns[0]
        returned = _unwrap_reference(returned)
        if returned.kind in {"tuple", "list"} and returned.children:
            returned = _unwrap_reference(returned.children[0])
        top = _unwrap_reference(returned)
        if top.kind != "add" or top.span is None:
            return None
        terms = _flatten_add(top)
        direct_matches = tuple(
            (position, actual)
            for position, term in enumerate(terms)
            for actual in actuals
            if _unwrap_reference(term) == _unwrap_reference(actual.value))
        # A child-integrated residual is exactly one explicit non-input actual
        # plus exactly one computed contribution.  This rejects residual+
        # residual, output+residual+mask, and accidental matching of signal.
        if len(terms) != 2 or len(direct_matches) != 1:
            return None
        _position, residual = direct_matches[0]
        if residual.name == positional[0]:
            return None
        proofs.append((residual.value, top.span, return_span))
    if not proofs or len({item[0] for item in proofs}) != 1:
        return None
    residual = proofs[0][0]
    site = branch.node.call_site
    synthetic = _Value("add", (block_value, residual), span=proofs[0][1])
    branch_sites = frozenset((site, *_mechanism_sites(residual)))
    return _Merge(
        synthetic, proofs[0][1], branch_sites,
        _contains_value(final_value, block_value)
        or _contains_site(final_value, site),
        "child_integrated_add")


def _classify_equations(attention, ffn, role_values, norms, merges,
                        final_value, final_span):
    a_site = attention.node.call_site
    f_site = ffn.node.call_site
    a_value = role_values[a_site]
    f_value = role_values[f_site]
    a_input = a_value.actuals[0].value if a_value.actuals else None
    f_input = f_value.actuals[0].value if f_value.actuals else None
    if a_input is None or f_input is None:
        return ReaderFailure(
            "incomplete_graph", "branch first-input expressions are unresolved")

    parallel_shape = tuple(
        item for item in merges
        if item.returned and item.branch_sites == frozenset((a_site, f_site))
        and _has_residual_term(item.value, {a_site, f_site}))
    parallel = tuple(
        item for item in parallel_shape
        if _base_input(a_input, norms) == _base_input(f_input, norms))
    sequential = []
    for a_merge in merges:
        if a_merge.branch_sites != frozenset((a_site,)):
            continue
        if not _contains_value(f_input, a_merge.value) \
                and not _contains_site(f_input, a_site):
            continue
        for f_merge in merges:
            if f_merge.branch_sites != frozenset((a_site, f_site)) \
                    or not f_merge.returned:
                continue
            if not _residual_contains_merge(
                    f_merge.value, f_site, a_site, a_merge):
                continue
            sequential.append((a_merge, f_merge))

    if parallel and sequential:
        return ReaderFailure(
            "conflict", "parallel and sequential equations both reach return")
    if len(parallel) == 1:
        topology = "parallel"
        a_merge = f_merge = parallel[0]
    elif len(sequential) == 1:
        topology = "sequential"
        a_merge, f_merge = sequential[0]
    elif parallel_shape:
        return ReaderFailure(
            "incomplete_graph",
            "parallel branches do not prove one shared residual input")
    else:
        return ReaderFailure(
            "incomplete_graph",
            "one unique exact sequential/parallel residual equation is not proven")

    a_state = _norm_state(a_input, a_merge.value, a_site, norms)
    f_state = _norm_state(f_input, f_merge.value, f_site, norms)
    if a_state is None or f_state is None or a_state != f_state:
        return ReaderFailure(
            "incomplete_graph",
            "attention and FFN norm boundaries are unresolved or disagree")
    placement = {
        (True, False): "pre",
        (False, True): "post",
        (True, True): "double",
    }.get(a_state)
    if placement is None:
        return ReaderFailure(
            "incomplete_graph", "no exact normalization boundary is proven")

    a_proof = _branch_proof(
        attention, a_state, a_input, a_merge, norms)
    f_proof = _branch_proof(
        ffn, f_state, f_input, f_merge, norms)
    return placement, topology, a_proof, f_proof


def _norm_state(input_value, merge_value, site, norms):
    pre = _direct_norm_site(input_value, norms)
    if pre is False:
        return None
    contribution = _branch_term(merge_value, site)
    if contribution is None:
        return None
    post = _post_norm_state(contribution, site, norms)
    if post is None:
        return None
    return bool(pre), post


def _direct_norm_site(value, norms):
    value = _unwrap_reference(value)
    if value.kind == "choice":
        states = {_direct_norm_site(item, norms) for item in value.children}
        return states.pop() if len(states) == 1 else False
    if value.kind == "norm" and value.call_site in norms:
        return value.call_site
    if value.kind == "transparent" and len(value.children) == 1:
        return _direct_norm_site(value.children[0], norms)
    if value.kind in {"input", "add", "mixer", "ffn"}:
        return None
    return False


def _post_norm_state(value, branch_site, norms):
    value = _unwrap_reference(value)
    if value.kind == "norm" and value.call_site in norms \
            and _contains_site(value, branch_site):
        return True
    if value.kind == "transparent" and len(value.children) == 1:
        return _post_norm_state(value.children[0], branch_site, norms)
    if value.kind == "add":
        matched = tuple(
            child for child in value.children
            if _contains_site(child, branch_site))
        return (_post_norm_state(matched[0], branch_site, norms)
                if len(matched) == 1 else None)
    if value.kind in {"mixer", "ffn"} \
            and value.call_site == branch_site:
        return False
    if value.kind == "choice":
        states = {_post_norm_state(item, branch_site, norms)
                  for item in value.children}
        return states.pop() if len(states) == 1 else None
    # An opaque transform could hide a normalization.  Never turn its absence
    # from the block-local norm census into a negative proof.
    return None


def _branch_proof(branch, state, input_value, merge, norms):
    pre = _direct_norm_site(input_value, norms)
    pre_site = pre if isinstance(pre, CallSiteId) else None
    contribution = _branch_term(merge.value, branch.node.call_site)
    post_site = None
    if contribution is not None:
        post_site = _post_norm_site(
            contribution, branch.node.call_site, norms)
    norm_sites = tuple(
        item for item in (pre_site, post_site) if item is not None)
    spans = tuple(dict.fromkeys(
        span for span in (
            *branch.spans,
            merge.span,
            *(span for site in norm_sites for span in norms[site][2]),
        ) if isinstance(span, SourceSpan)))
    return CellBranchProof(
        branch.mechanism, branch, pre_site, post_site, merge.kind,
        merge.span, spans)


def _branch_term(value, site):
    terms = _flatten_add(value)
    matched = tuple(
        item for item in terms if _contains_site(item, site))
    return matched[0] if len(matched) == 1 else None


def _has_residual_term(value, branch_sites):
    return any(not any(_contains_site(term, site) for site in branch_sites)
               for term in _flatten_add(value))


def _residual_contains_merge(value, branch_site, prior_site, prior_merge):
    return any(
        not _contains_site(term, branch_site)
        and (_contains_value(term, prior_merge.value)
             or (prior_merge.kind == "child_integrated_add"
                 and _contains_site(term, prior_site)))
        for term in _residual_terms(value))


def _post_norm_site(value, branch_site, norms):
    value = _unwrap_reference(value)
    if value.kind == "norm" and value.call_site in norms \
            and _contains_site(value, branch_site):
        return value.call_site
    if value.kind == "choice":
        sites = {_post_norm_site(item, branch_site, norms)
                 for item in value.children}
        return sites.pop() if len(sites) == 1 else None
    return None


def _base_input(value, norms):
    value = _unwrap_reference(value)
    if value.kind == "norm" and value.call_site in norms and value.children:
        return _base_input(value.children[0], norms)
    if value.kind == "transparent" and len(value.children) == 1:
        return _base_input(value.children[0], norms)
    if value.kind == "choice":
        bases = {_base_input(item, norms) for item in value.children}
        return next(iter(bases)) if len(bases) == 1 else None
    return value


def _flatten_add(value):
    # Assignment references are transparent to the residual equation.  This
    # matters for augmented assignments (``branch += other``), whose exact
    # addition is necessarily reached through the later variable reference.
    # Unwrap only the single-source reference chain; choices and opaque
    # transforms remain boundaries and therefore cannot be laundered into an
    # additive proof.
    value = _unwrap_reference(value)
    if value.kind != "add":
        return (value,)
    result = []
    for child in value.children:
        result.extend(_flatten_add(child))
    return tuple(result)


def _residual_terms(value):
    """Return top-level residual terms without erasing assignment stages.

    A reference to a prior residual addition is one term of the later
    equation, not merely another flat summand.  Keeping that reference intact
    is what distinguishes sequential ``(x + attention) + ffn`` from parallel
    ``x + attention + ffn`` even though their fully flattened algebra is the
    same.
    """
    value = _unwrap_reference(value)
    if value.kind != "add":
        return (value,)
    result = []
    for child in value.children:
        if child.kind == "add":
            result.extend(_residual_terms(child))
        else:
            result.append(child)
    return tuple(result)


def _unwrap_reference(value):
    while value.kind == "reference" and len(value.children) == 1:
        value = value.children[0]
    return value


def _walk_values(value):
    yield value
    for child in value.children:
        yield from _walk_values(child)


def _contains_site(value, site):
    return any(item.call_site == site for item in _walk_values(value))


def _mechanism_sites(value):
    return tuple(dict.fromkeys(
        item.call_site for item in _walk_values(value)
        if item.kind in {"mixer", "ffn"}
        and item.call_site is not None))


def _contains_value(value, wanted):
    return any(item == wanted for item in _walk_values(value))


def _position(span):
    return (span.line, span.col, span.end_line, span.end_col)


def _span_within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (
        (inner.line, inner.col) >= (outer.line, outer.col)
        and (inner.end_line, inner.end_col)
        <= (outer.end_line, outer.end_col)
    )


def _span_key(span):
    if span is None:
        return None
    return (span.line, span.col, span.end_line, span.end_col)


__all__ = [
    "CellBranchProof",
    "DecoderCellTopologyEvidence",
    "decoder_cell_topology_for_path",
]
