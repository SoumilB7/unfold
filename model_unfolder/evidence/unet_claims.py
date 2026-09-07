"""UNet claim-kind proofs project existing reader evidence, never templates."""
from dataclasses import dataclass, field
from functools import cached_property

from .claim_evidence import ClaimProofSummary
from .facts import EvidenceFact, SourceSpan as FactSpan
from .runtime_source import RuntimeSourceBindings
from .unet_stage_execution import UNetStageExecutionGraph
from .unet_nested_mechanism import RuntimeNestedFFNAttempt
from .selected_composite_ffn import SelectedCompositeFFNMechanism
from .unet_cell_mechanism import StageJoinConnection
from .unet_stage_construction import RepeatedStageConstruction
from .unet_attention_source import UNetRuntimeAttentionSources
from .unet_nested_mechanism import AlternativeNestedOccurrenceId, runtime_nested_block_paths
from .unet_cell_mechanism import UNetCellMechanismInventory
from .unet_selected_spatial import UNetSelectedSpatialOperations


def _span_ref(span):
    return (f"sha256:{span.source.content_fingerprint}:"
            f"{span.line}:{span.col}:{span.end_line}:{span.end_col}")


@dataclass(frozen=True)
class UNetStageRelationClaimProof:
    fact_id: str
    graph: UNetStageExecutionGraph = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "relation"
    proof_kind = "constructed_stages_and_skip_route"
    reader_symbols = ("evidence.unet_claims.read_unet_stage_relations",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.constructed_stage_relations":
            raise ValueError("stage relation proof belongs to its exact fact")
        if not isinstance(self.graph, UNetStageExecutionGraph) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("stage relations require source execution evidence and reconciled instances")
        if self.bindings.symbol_at("") != self.graph.owner.root:
            raise ValueError("source stage graph does not belong to the constructed root")
        if len(self.graph.edges) != 1:
            raise ValueError("stage relation projection requires one exact skip route")

    @cached_property
    def value(self):
        edge = self.graph.edges[0]
        source = edge.source.node_id.field
        target = edge.target.node_id.field
        direct = tuple(dict.fromkeys(row.node_id.field for row in self.graph.direct))
        return {
            "producer_field": source,
            "consumer_field": target,
            "producer_stages": list(self.bindings.direct_members(source, repeated=True)),
            "consumer_stages": list(self.bindings.direct_members(target, repeated=True)),
            "intermediate_stages": [path for field in direct
                                    for path in self.bindings.direct_members(field, repeated=False)],
            "skip_route": {
                "producer": _span_ref(edge.route.producer.span),
                "accumulation": _span_ref(edge.route.accumulator_binding.span),
                "selection": _span_ref(edge.route.derived_binding.span),
                "consumer": _span_ref(edge.route.consumer.span),
            },
            # These are source-reader findings. No arrangement in the view
            # can discharge the remaining execution/order questions.
            "unresolved_relations": [{"kind": row.kind, "reason": row.detail,
                                       "reason_class": "investigation_missing"}
                                      for row in self.graph.unresolved],
        }

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        refs = tuple(sorted({_span_ref(span) for span in self.graph.edges[0].route.spans}))
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols, refs,
            document_fingerprints=(self.bindings.table.config_sha256,),
            index_fingerprints=(self.graph.index.fingerprint,))


def read_unet_stage_relations(graph, bindings):
    proof = UNetStageRelationClaimProof("root.denoiser.constructed_stage_relations", graph, bindings)
    owner = graph.owner.root
    return EvidenceFact(
        key="constructed_stage_relations", owner="root.denoiser", value=proof.value,
        status="code_proven", completeness="presence_only",
        source_spans=tuple(FactSpan(
            span.source.component_key, owner.qualified_name,
            owner.qualified_name + ".forward", span.source.canonical_path, span.line)
            for span in graph.edges[0].route.spans),
        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols, claim_evidence=proof)


@dataclass(frozen=True)
class UNetFFNClaimProof:
    """Selected implementation and returned computation, bound to instances.

    This proves the FFN's own computation. Whether its caller executes on an
    arbitrary input is a separate execution-axis question.
    """

    fact_id: str
    attempts: tuple[RuntimeNestedFFNAttempt, ...] = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "applied_function"
    proof_kind = "selected_composite_ffn_return_route"
    reader_symbols = ("evidence.unet_claims.read_unet_ffn_claims",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.ffn_mechanisms":
            raise ValueError("FFN proof belongs to its declared mechanism fact")
        if not isinstance(self.bindings, RuntimeSourceBindings) or not self.attempts:
            raise TypeError("FFN claims need exact runtime bindings and reader attempts")
        for attempt in self.attempts:
            if not isinstance(attempt, RuntimeNestedFFNAttempt):
                raise TypeError("FFN claims require the reader's typed attempt")
            proof = attempt.result.require_value()
            if not isinstance(proof, SelectedCompositeFFNMechanism):
                raise TypeError("FFN claim requires selected implementation and return-route evidence")
            position = proof.execution.append_calls.index(proof.execution.selected_append)
            for path in attempt.instance_paths:
                if self.bindings.symbol_at(path) != proof.frame.graph.root.symbol:
                    raise ValueError("FFN proof does not belong to the constructed callable")
                if self.bindings.symbol_at(f"{path}.{proof.execution.field}.{position}") != proof.input_transform.owner_symbol:
                    raise ValueError("selected transform is not the constructed implementation")
        # Different source alternatives may justify the same occurrence only
        # if the projected computation agrees. Never take the first rival.
        self.value

    @cached_property
    def value(self):
        rows = {}
        for attempt in self.attempts:
            proof = attempt.result.require_value()
            row = {"gated": proof.gated, "activation": proof.activation,
                   "projection_mode": proof.projection_mode}
            for path in attempt.instance_paths:
                position = proof.execution.append_calls.index(proof.execution.selected_append)
                output_positions = [number for number, call in enumerate(proof.execution.append_calls)
                                    if call.span.source == proof.execution.output_site.span.source
                                    and (call.span.line, call.span.col) <= (proof.execution.output_site.span.line, proof.execution.output_site.span.col)
                                    and (call.span.end_line, call.span.end_col) >= (proof.execution.output_site.span.end_line, proof.execution.output_site.span.end_col)]
                if len(output_positions) != 1:
                    raise ValueError("the output construction requires one exact container position")
                scoped = {**row,
                          "input_projection": f"{path}.{proof.execution.field}.{position}.{proof.input_transform.projection_resolution.selected.site.target}",
                          "output_projection": f"{path}.{proof.execution.field}.{output_positions[0]}"}
                required_primitives = {
                    scoped["input_projection"]: "linear",
                    scoped["output_projection"]: "linear",
                }
                for site in proof.execution.transparent_sites:
                    slots = [number for number, call in enumerate(proof.execution.append_calls)
                             if call.span.source == site.span.source
                             and (call.span.line, call.span.col) <= (site.span.line, site.span.col)
                             and (call.span.end_line, call.span.end_col) >= (site.span.end_line, site.span.end_col)]
                    if len(slots) != 1:
                        raise ValueError("transparent callable needs its exact container slot")
                    required_primitives[f"{path}.{proof.execution.field}.{slots[0]}"] = "dropout"
                if any(self.bindings.primitive_at(member) != primitive
                       for member, primitive in required_primitives.items()):
                    continue
                if path in rows and rows[path] != scoped:
                    raise ValueError("source alternatives disagree on the constructed FFN mechanism")
                rows[path] = scoped
        return dict(sorted(rows.items()))

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        refs = tuple(sorted({_span_ref(span) for attempt in self.attempts
                             for span in attempt.result.require_value().spans}))
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols, refs,
            document_fingerprints=(self.bindings.table.config_sha256,),
            index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_ffn_claims(attempts, bindings):
    positives = tuple(attempt for attempt in attempts if attempt.result.has_value)
    if not positives:
        return None
    proof = UNetFFNClaimProof("root.denoiser.ffn_mechanisms", positives, bindings)
    return EvidenceFact(
        key="ffn_mechanisms", owner="root.denoiser", value=proof.value,
        status="code_proven", completeness="presence_only",
        source_spans=tuple(dict.fromkeys(FactSpan(
            span.source.component_key,
            attempt.result.require_value().frame.graph.root.symbol.qualified_name,
            None, span.source.canonical_path, span.line)
            for attempt in positives for span in attempt.result.require_value().spans)),
        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols, claim_evidence=proof)


@dataclass(frozen=True)
class UNetJoinClaimProof:
    fact_id: str
    connections: tuple[StageJoinConnection, ...] = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "connection"
    proof_kind = "concat_output_to_constructed_child_input"
    reader_symbols = ("evidence.unet_claims.read_unet_join_claims",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.stage_join_connections" or not self.connections:
            raise ValueError("join claims need positive connections for their exact fact")
        if not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("join claims require runtime bindings")
        from .framework_operations import functional_operation_protocol_for_call
        from .unet_cell_connections import _direct_origin
        for row in self.connections:
            if not isinstance(row, StageJoinConnection):
                raise TypeError("join claims require exact reader connections")
            protocol = functional_operation_protocol_for_call(self.bindings.index, row.join)
            if protocol is None or protocol.kind != "concat":
                raise ValueError("the connected operation is not a proven framework concat")
            invocation = row.invocation.call
            forward = self.bindings.index.callable_by_symbol(invocation.enclosing_callable)
            if forward is None or row.join not in self.bindings.index.calls_in(forward.symbol) \
                    or invocation not in self.bindings.index.calls_in(forward.symbol):
                raise ValueError("join proof must cite exact indexed call observations")
            routes = [_direct_origin(self.bindings.index, forward, invocation, actual,
                                     {row.join.span: row.join})
                      for actual in (*invocation.args, *(value for key, value in invocation.kwargs if key != "**"))]
            if not any(route is not None and route[1] == row.bindings for route in routes):
                raise ValueError("join result does not reach the cited consumer through its declared bindings")

    @cached_property
    def value(self):
        stages = {}
        for row in self.connections:
            invocation = row.invocation
            stage = invocation.parent
            for path in self.bindings.matching_members(
                    stage.occurrence_id.parent_field, stage.occurrence_id.symbol,
                    repeated=isinstance(stage.construction, RepeatedStageConstruction)):
                if not self.bindings.forward_is_unmodified(path):
                    continue
                targets = tuple(dict.fromkeys(
                    target for construction in invocation.constructions
                    for candidate in construction.candidates
                    for target in self.bindings.matching_members(
                        f"{path}.{invocation.field}", candidate.symbol,
                        repeated=invocation.kind == "repeated")))
                if not targets:
                    continue
                operands = row.join.args[0] if row.join.args else dict(row.join.kwargs).get("tensors")
                if operands is None or operands.kind not in {"list", "tuple"}:
                    continue
                connection = {"operation": "concat", "targets": list(targets),
                              "operand_slots": list(range(len(operands.children))),
                              "input_lineage": "investigation_missing",
                              "input_lineage_reason": "caller formal to concat operand route remains open"}
                from .local_port_routes import read_local_port_route
                forward = self.bindings.index.callable_by_symbol(row.join.enclosing_callable)
                connection["operand_routes"] = [read_local_port_route(
                    self.bindings.index, forward, operand, row.join.span, row.join.guard).value
                    for operand in operands.children]
                if connection not in stages.setdefault(path, []):
                    stages[path].append(connection)
        return stages

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        refs_set = {_span_ref(span) for row in self.connections
                             for span in (row.join.span, row.invocation.call.span,
                                          *(binding.span for binding in row.bindings))}
        from .local_port_routes import read_local_port_route
        for row in self.connections:
            operands = row.join.args[0] if row.join.args else dict(row.join.kwargs).get("tensors")
            forward = self.bindings.index.callable_by_symbol(row.join.enclosing_callable)
            if operands is not None and operands.kind in {"list", "tuple"}:
                refs_set.update(_span_ref(span) for operand in operands.children for span in
                                read_local_port_route(self.bindings.index, forward, operand,
                                                      row.join.span, row.join.guard).spans)
        refs = tuple(sorted(refs_set))
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind,
                                 self.reader_symbols, refs,
                                 document_fingerprints=(self.bindings.table.config_sha256,),
                                 index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_join_claims(connections, bindings):
    if not connections:
        return None
    proof = UNetJoinClaimProof("root.denoiser.stage_join_connections", tuple(connections), bindings)
    return EvidenceFact(key="stage_join_connections", owner="root.denoiser",
                        value=proof.value, status="code_proven", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)


@dataclass(frozen=True)
class UNetContextConnectionClaimProof:
    fact_id: str
    sources: UNetRuntimeAttentionSources = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "connection"
    proof_kind = "required_root_formal_to_selected_context_argument"
    reader_symbols = ("evidence.unet_claims.read_unet_context_connections",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.context_connections":
            raise ValueError("context proof belongs to its exact connection fact")
        if not isinstance(self.sources, UNetRuntimeAttentionSources) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("context claims require the reader's typed source routes")
        if any(row.routes[0].route.owner.root != self.bindings.symbol_at("")
               for row in self.sources.sources):
            raise ValueError("context source route belongs to another constructed root")

    @cached_property
    def value(self):
        connections = {}
        for row in self.sources.sources:
            occurrence = row.nested.occurrence_id
            if not isinstance(occurrence, AlternativeNestedOccurrenceId):
                continue
            stage_path = row.stage.occurrence_id.parent_field
            if row.selected_stage.position is not None:
                stage_path += f".{row.selected_stage.position}"
            resolved = runtime_nested_block_paths(
                self.sources.nested_inventory, self.bindings, occurrence.alternative,
                selected_stage_path=stage_path)
            if resolved is None:
                continue
            route = row.routes[0].route
            for block_path in resolved[2]:
                target = f"{block_path}.{row.lane.construction.target}"
                if self.bindings.symbol_at(target) != row.lane.child_symbol:
                    continue
                if not self.bindings.route_forwards_unmodified(target):
                    continue
                value = {"source_formal": route.source_formal.name,
                         "target_formal": route.target_formal.name,
                         "stage": stage_path,
                         "role": "context_input",
                         "cross_attention": "investigation_missing",
                         "reason": "External context input is proven; distinct query lineage remains open."}
                if target in connections and connections[target] != value:
                    raise ValueError("source routes disagree on a constructed context input")
                connections[target] = value
        return dict(sorted(connections.items()))

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted({_span_ref(span) for row in self.sources.sources for span in row.spans})),
            document_fingerprints=(self.bindings.table.config_sha256,),
            index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_context_connections(sources, bindings):
    if sources is None or not sources.sources:
        return None
    proof = UNetContextConnectionClaimProof("root.denoiser.context_connections", sources, bindings)
    return EvidenceFact(key="context_connections", owner="root.denoiser", value=proof.value,
                        status="code_proven", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)


@dataclass(frozen=True)
class UNetCellArithmeticClaimProof:
    fact_id: str
    mechanisms: UNetCellMechanismInventory = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)
    execution: object = field(default=None, repr=False, compare=False)

    claim_kind = "applied_function"
    proof_kind = "exact_cell_return_and_conditional_arithmetic"
    reader_symbols = ("evidence.unet_claims.read_unet_cell_arithmetic",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.cell_arithmetic":
            raise ValueError("cell arithmetic belongs to its exact fact")
        if not isinstance(self.mechanisms, UNetCellMechanismInventory) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("cell arithmetic requires typed exact-class mechanism evidence")
        from .unet_selected_child_execution import UNetSelectedChildExecution
        if self.execution is not None and not isinstance(self.execution, UNetSelectedChildExecution):
            raise TypeError("conditional arithmetic uses typed constructor operands")

    @cached_property
    def value(self):
        from .unet_cell_connections import _instance_environments
        from .unet_selected_constructor import selected_instance_guard_evidence
        environments = _instance_environments(self.execution, self.bindings)
        result = {}
        for row in self.mechanisms.mechanisms:
            if row.residual_merge is None:
                continue
            for module in self.bindings.inventory.modules:
                if self.bindings.symbol_at(module.path) != row.occurrence_id.symbol:
                    continue
                conditioning = []
                for item in row.conditioning:
                    decisions = [selected_instance_guard_evidence(
                        env, item.binding.enclosing_callable, item.guard, item.binding.span)
                        for env in environments.get(module.path, ())]
                    if decisions and all(value is not None and value.value is False for value in decisions):
                        continue
                    conditional = not decisions or not all(value is not None and value.value is True for value in decisions)
                    # A syntactic origin set locates candidates; it does not
                    # prove dependence through an arbitrary helper return.
                    from .local_port_routes import read_local_port_route
                    forward = self.bindings.index.callable_by_symbol(item.binding.enclosing_callable)
                    expression = item.binding.value
                    operands = expression.children if expression is not None and expression.kind == "binop" else ()
                    conditioning.append({"operation": item.kind,
                                         "conditional": bool(item.guard) and conditional,
                                         "operand_routes": [read_local_port_route(
                                             self.bindings.index, forward, operand,
                                             item.binding.span, item.guard).value for operand in operands],
                                         "dependency": "call ports only; helper input-to-output dependence unresolved"})
                value = {"return_merge": "add", "return_scale": "divide" if row.residual_merge.scale_expression else None,
                         "branch_lineage": "investigation_missing", "conditioning": conditioning}
                if not self.bindings.forward_is_unmodified(module.path):
                    continue
                if module.path in result and result[module.path] != value:
                    raise ValueError("source alternatives disagree on exact return arithmetic")
                result[module.path] = value
        return dict(sorted(result.items()))

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        spans = {span for row in self.mechanisms.mechanisms if row.residual_merge is not None
                 for span in (row.residual_merge.span,
                              *(item.binding.span for item in row.conditioning),
                              *((row.residual_merge.scale_expression.span,)
                                if row.residual_merge.scale_expression is not None else ()))}
        from .local_port_routes import read_local_port_route
        from .unet_cell_connections import _instance_environments
        from .unet_selected_constructor import selected_instance_guard_evidence
        environments = _instance_environments(self.execution, self.bindings)
        for row in self.mechanisms.mechanisms:
            for item in row.conditioning:
                forward = self.bindings.index.callable_by_symbol(item.binding.enclosing_callable)
                expression = item.binding.value
                if expression is not None and expression.kind == "binop":
                    for operand in expression.children:
                        spans.update(read_local_port_route(self.bindings.index, forward,
                                     operand, item.binding.span, item.guard).spans)
                for module in self.bindings.inventory.modules:
                    if self.bindings.symbol_at(module.path) == row.occurrence_id.symbol:
                        for environment in environments.get(module.path, ()):
                            decision = selected_instance_guard_evidence(environment,
                                item.binding.enclosing_callable, item.guard, item.binding.span)
                            if decision is not None:
                                spans.update(decision.spans)
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
                                 tuple(sorted(_span_ref(span) for span in spans)),
                                 document_fingerprints=(self.bindings.table.config_sha256,),
                                 index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_cell_arithmetic(mechanisms, bindings, execution=None):
    if mechanisms is None:
        return None
    proof = UNetCellArithmeticClaimProof("root.denoiser.cell_arithmetic", mechanisms, bindings, execution)
    value = proof.value
    if not value:
        return None
    return EvidenceFact(key="cell_arithmetic", owner="root.denoiser", value=value,
                        status="code_proven", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)


@dataclass(frozen=True)
class UNetSpatialClaimProof:
    fact_id: str
    spatial: UNetSelectedSpatialOperations = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)

    claim_kind = "applied_function"
    proof_kind = "selected_spatial_primitive_return_route"
    reader_symbols = ("evidence.unet_claims.read_unet_spatial_claims",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.spatial_mechanisms":
            raise ValueError("spatial proof belongs to its exact fact")
        if not isinstance(self.spatial, UNetSelectedSpatialOperations) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("spatial claims require the selected execution and mechanism reader")

    @cached_property
    def value(self):
        rows = {}
        for operation in self.spatial.spatial_operations:
            population = operation.execution.population
            constructions = [item for item in population.present_constructions
                             if (item.site.span if item.site is not None else item.field_assign.value.span)
                             == operation.occurrence_id.construction_span]
            if len(constructions) != 1:
                continue
            for path in self.bindings.construction_members(population, constructions[0]):
                if self.bindings.symbol_at(path) != operation.occurrence_id.symbol:
                    continue
                if not self.bindings.forward_is_unmodified(path):
                    continue
                value = {"effect": operation.effect, "operand": operation.numeric_operand,
                         "primitive": operation.mechanism}
                if path in rows and rows[path] != value:
                    raise ValueError("selected spatial proofs disagree on an occurrence")
                rows[path] = value
        return dict(sorted(rows.items()))

    def summary(self):
        return self._summary

    @cached_property
    def _summary(self):
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
                                 tuple(sorted({_span_ref(span) for row in self.spatial.spatial_operations
                                               for span in row.operand_spans})),
                                 document_fingerprints=(self.bindings.table.config_sha256,),
                                 index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_spatial_claims(spatial, bindings):
    if spatial is None or not spatial.spatial_operations:
        return None
    proof = UNetSpatialClaimProof("root.denoiser.spatial_mechanisms", spatial, bindings)
    return EvidenceFact(key="spatial_mechanisms", owner="root.denoiser", value=proof.value,
                        status="code_proven", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)
