"""Primary local-state regions, preserving source guards and repeat boundaries.

The regions establish call/operation ports, not opaque input-output dependence
or a particular module's invocation. Constructed stages remain containment.
"""
from dataclasses import dataclass, field
from functools import cached_property

from .claim_evidence import ClaimProofSummary
from .facts import EvidenceFact
from .local_port_routes import read_local_port_route, _slot
from .program_index import ExprNode, SourceSpan, SymbolId
from .runtime_source import RuntimeSourceBindings
from .unet_stage_execution import UNetStageExecutionGraph


def _ref(span):
    return f"sha256:{span.source.content_fingerprint}:{span.line}:{span.col}:{span.end_line}:{span.end_col}"


def _has_input(value):
    if isinstance(value, dict):
        return value.get("kind") == "region_input" or any(_has_input(x) for x in value.values())
    return isinstance(value, list) and any(_has_input(x) for x in value)


def read_primary_regions(index, forward, local):
    """Partition every indexed write to this local by its outer source region."""
    writes = [row for row in index.bindings_in(forward.symbol)
              if any(_slot(target, local) is not None for target in row.targets)]
    regions = {}
    for row in writes:
        span = row.guard[0].span if row.guard else row.span
        regions.setdefault(span, []).append(row)
    # A local may also be rebound by loop targets or unsupported statement
    # syntax. Keep those as explicit boundaries rather than skip their history.
    for loop in index.loops_in(forward.symbol):
        if _slot(loop.target, local) is not None:
            span = loop.guard[0].span if loop.guard else loop.span
            regions.setdefault(span, [])
    for unsupported in index.unsupported_execution_in(forward.symbol):
        if unsupported.construct_kind != "boolop":
            span = unsupported.guard[0].span if unsupported.guard else unsupported.span
            regions.setdefault(span, [])
    result, spans = [], set()
    for number, (span, rows) in enumerate(sorted(regions.items(), key=lambda item: (item[0].line, item[0].col))):
        end = SourceSpan(span.source, span.end_line, span.end_col + 1, span.end_line, span.end_col + 1)
        expression = ExprNode(kind="name", name=local, span=end)
        route = read_local_port_route(index, forward, expression, end,
                                      region_input=(local, span))
        kind = (rows[0].guard[0].kind if rows[0].guard else "assignment") if rows else "unresolved"
        result.append({"position": number, "kind": kind,
                       "route": route.value, "receives_previous_state": _has_input(route.value),
                       "source": _ref(span), "writes": [_ref(row.span) for row in rows]})
        spans.update((span, *route.spans))
    return result, tuple(sorted(spans, key=lambda span: (span.line, span.col, span.end_line, span.end_col)))


@dataclass(frozen=True)
class UNetPrimaryPortProof:
    graph: UNetStageExecutionGraph = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    fact_id = "root.denoiser.primary_state_ports"
    claim_kind = "connection"
    proof_kind = "guarded_primary_state_argument_result_ports"
    reader_symbols = ("evidence.unet_primary_ports.read_unet_primary_ports",)

    def __post_init__(self):
        if not isinstance(self.graph, UNetStageExecutionGraph) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("primary ports require exact source execution and runtime binding records")
        if self.bindings.symbol_at("") != self.graph.owner.root or len(self.graph.edges) != 1:
            raise ValueError("primary ports require the exact constructed root and one source carried state")
        if not self.bindings.forward_is_unmodified(""):
            raise ValueError("recorded root callable replacement defeats class-forward port authority")
        owner = self.graph.owner.root
        forward = self.bindings.index.callable_by_symbol(SymbolId(owner.source, owner.qualified_name + ".forward"))
        if forward is None:
            raise ValueError("primary ports require the indexed invoked root method")
        route = self.graph.edges[0].route
        calls = self.bindings.index.calls_in(forward.symbol)
        if route.producer not in calls or route.consumer not in calls \
                or route.producer_binding not in self.bindings.index.bindings_in(forward.symbol):
            raise ValueError("carried-state premises must be exact records in the root source index")

    @cached_property
    def investigation(self):
        owner = self.graph.owner.root
        forward = self.bindings.index.callable_by_symbol(SymbolId(owner.source, owner.qualified_name + ".forward"))
        local = self.graph.edges[0].route.carried_output
        rows, spans = read_primary_regions(self.bindings.index, forward, local)
        stages = (self.graph.edges[0].source, *self.graph.direct, self.graph.edges[0].target)
        for row in rows:
            # A region contains these source stage sites. This deliberately
            # does not bind an iteration to an individual constructed member.
            span = next(span for span in spans if _ref(span) == row["source"])
            fields = list(dict.fromkeys(stage.node_id.field for stage in stages
                if (span.line, span.col) <= stage.node_id.source_position <= (span.end_line, span.end_col)))
            row["stage_fields"] = fields
            row["constructed_stages"] = [path for name in fields for path in self.bindings.direct_members(
                name, repeated=any(stage.node_id.field == name and stage.node_id.kind == "repeated" for stage in stages))]
        return {"regions": rows, "input_is_formal": local in {p.name for p in forward.params},
                "target_binding": "unresolved; regions prove ports, not individual stage invocation"}, spans

    @property
    def value(self):
        return self.investigation[0]

    def summary(self):
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(_ref(span) for span in self.investigation[1]),
            document_fingerprints=(self.bindings.table.config_sha256,),
            index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_primary_ports(graph, bindings):
    if not bindings.forward_is_unmodified(""):
        return None
    proof = UNetPrimaryPortProof(graph, bindings)
    return EvidenceFact(key="primary_state_ports", owner="root.denoiser", value=proof.value,
                        status="code_proven", completeness="presence_only", claim_kind=proof.claim_kind,
                        claim_readers=proof.reader_symbols, claim_evidence=proof)
