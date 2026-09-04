"""S7 shadow reconciliation of construction, execution and projection.

This module is deliberately a *join*, not a fifth architecture detector.  It
accepts the exact records authored by the four existing authorities and gives
every runtime occurrence one value on each orthogonal axis:

* meta-device instance inventory -> construction;
* recipe-qualified observation plus static reachability -> execution;
* existing typed facts / canonical projection receipts -> projection;
* instance/trace relations with source/config support -> relation rows.

Nothing here writes the IR or chooses a renderer.  The only fact objects it
carries are the existing :class:`EvidenceFact` rows supplied by the parser.
That is the S7 shadow contract: compare authorities now; author product facts
only in a separately reviewed family cutover.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .facts import EvidenceFact


CONSTRUCTION_KINDS = frozenset({
    "eager_constructed", "lazy_observed", "construction_conflict",
    "not_constructed",
})
EXECUTION_KINDS = frozenset({
    "observed", "statically_reachable", "proven_inactive",
    "execution_unresolved",
})
PROJECTION_KINDS = frozenset({
    "rendered", "grouped", "non_architectural", "projection_unresolved",
})
RELATION_KINDS = frozenset({
    "param_share", "activation_reuse", "multi_stream_residual",
    "per_layer_side_input", "intra_layer_shortcut", "layer_reuse",
    "conditional_skip", "side_head", "relation_unresolved",
})


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _closed_text(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")


@dataclass(frozen=True)
class AuthorityRule:
    """One executable row of the binding authority matrix (§1b)."""

    question: str
    primary: str
    supporting: tuple[str, ...] = ()
    conflict: str = "blocking"

    def __post_init__(self) -> None:
        _closed_text(self.question, field="authority question")
        _closed_text(self.primary, field="primary authority")
        if self.conflict not in {"blocking", "primary_wins"}:
            raise ValueError("authority conflict policy is closed")
        if any(not isinstance(item, str) or not item for item in self.supporting):
            raise ValueError("supporting authorities are non-empty strings")


# This is code, not prose interpreted by the reconciler.  A new authority
# question therefore requires a reviewed code change and a poison.
AUTHORITY_MATRIX: tuple[AuthorityRule, ...] = (
    AuthorityRule("runtime_class", "framework_factory_instance",
                  ("static_source_resolver", "library_revision")),
    AuthorityRule("constructed_modules", "meta_instance_inventory",
                  ("constructor_source", "checkpoint_metadata")),
    AuthorityRule("recipe_execution", "recipe_trace",
                  ("static_control_flow",)),
    AuthorityRule("alternative_reachability", "static_source_closure",
                  ("additional_traces",)),
    AuthorityRule("custom_mechanism", "exact_static_mechanism_reader",
                  ("trace", "export_conformance")),
    AuthorityRule("framework_primitive", "exact_runtime_type",
                  ("source_call_site",)),
    AuthorityRule("storage_layout", "instantiated_parameter_shapes",
                  ("source_assignment",)),
    AuthorityRule("checkpoint_value", "config_document"),
    AuthorityRule("class_default", "exact_class_source",
                  ("instance_confirmation",)),
    AuthorityRule("derived_runtime_value", "constructor_expression",
                  ("instance_confirmation",)),
    AuthorityRule("expected_parameter_shape", "instance_inventory"),
    AuthorityRule("stored_parameter_shape", "checkpoint_tensor_metadata"),
    AuthorityRule("parameter_sharing", "instance_parameter_identity",
                  ("config_declaration", "source_assignment",
                   "checkpoint_tensor_metadata")),
    AuthorityRule("projection", "canonical_fact_ir",
                  ("projection_receipt",)),
    AuthorityRule("renderer_decision", "presentation_only"),
)
_AUTHORITY_BY_QUESTION = {row.question: row for row in AUTHORITY_MATRIX}
if len(_AUTHORITY_BY_QUESTION) != len(AUTHORITY_MATRIX):
    raise RuntimeError("duplicate question in the reconciliation authority matrix")


def authority_for(question: str) -> AuthorityRule:
    """Return the exact authority row; unknown questions never gain precedence."""
    try:
        return _AUTHORITY_BY_QUESTION[question]
    except KeyError as exc:
        raise ValueError(f"no declared authority for {question!r}") from exc


@dataclass(frozen=True)
class RuntimeClassRef:
    module: str
    qualname: str

    def __post_init__(self) -> None:
        _closed_text(self.module, field="runtime class module")
        _closed_text(self.qualname, field="runtime class qualname")


@dataclass(frozen=True)
class StaticOccurrenceRef:
    """Serializable identity for one exact static owner occurrence."""

    root_source_fingerprint: str
    root_qualified_name: str
    construction_sites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (len(self.root_source_fingerprint) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in self.root_source_fingerprint)):
            raise ValueError("static occurrence needs its source fingerprint")
        _closed_text(self.root_qualified_name,
                     field="static root qualified name")
        if any(not isinstance(item, str) or not item
               for item in self.construction_sites):
            raise ValueError("construction-site identities are non-empty")


@dataclass(frozen=True)
class MeaningProvenance:
    """The second provenance key: source occurrence and exact config paths.

    ``instance_path`` lives on :class:`OccurrenceProvenance`; this object may
    never replace it.  Conversely a class name without a static occurrence is
    not meaning provenance and cannot be stored here.
    """

    static_occurrence: StaticOccurrenceRef | None = None
    framework_primitive: RuntimeClassRef | None = None
    config_paths: tuple[str, ...] = ()
    source_spans: tuple[str, ...] = ()
    fact_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.static_occurrence is not None and not isinstance(
                self.static_occurrence, StaticOccurrenceRef):
            raise TypeError("meaning provenance carries a typed static occurrence")
        if self.framework_primitive is not None and not isinstance(
                self.framework_primitive, RuntimeClassRef):
            raise TypeError("framework primitive provenance carries its exact type")
        for field_name in ("config_paths", "source_spans", "fact_keys"):
            values = getattr(self, field_name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must be unique and sorted")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field_name} entries are non-empty strings")


@dataclass(frozen=True)
class OccurrenceProvenance:
    """Two-key provenance: runtime instance address + meaning address."""

    instance_path: str
    runtime_class: RuntimeClassRef | None
    inventory_config_sha256: str
    meaning: MeaningProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.instance_path, str):
            raise TypeError("instance path is text; the root path may be empty")
        if self.runtime_class is not None and not isinstance(
                self.runtime_class, RuntimeClassRef):
            raise TypeError("runtime class must be a RuntimeClassRef")
        if (len(self.inventory_config_sha256) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in self.inventory_config_sha256)):
            raise ValueError("occurrence provenance needs the inventory config hash")
        if not isinstance(self.meaning, MeaningProvenance):
            raise TypeError("occurrence provenance needs its separate meaning key")


@dataclass(frozen=True)
class ConstructionAxis:
    kind: str
    guard: str = ""
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in CONSTRUCTION_KINDS:
            raise ValueError(f"unknown construction axis {self.kind!r}")
        if self.kind == "construction_conflict":
            if not self.conflicts or self.guard:
                raise ValueError("a construction conflict carries conflicts only")
        elif self.conflicts:
            raise ValueError("only construction_conflict carries conflicts")
        if self.kind == "not_constructed" and not self.guard:
            raise ValueError("not_constructed requires the exact source guard")
        if self.kind != "not_constructed" and self.guard:
            raise ValueError("only not_constructed carries a guard")


@dataclass(frozen=True)
class ExecutionAxis:
    kind: str
    recipe_ids: tuple[str, ...] = ()
    reason: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EXECUTION_KINDS:
            raise ValueError(f"unknown execution axis {self.kind!r}")
        if tuple(sorted(set(self.recipe_ids))) != self.recipe_ids:
            raise ValueError("recipe ids must be unique and sorted")
        if self.kind == "observed":
            if not self.recipe_ids or self.reason or self.detail:
                raise ValueError("observed execution carries recipe ids only")
        elif self.recipe_ids:
            raise ValueError("only observed execution carries recipe ids")
        if self.kind == "execution_unresolved":
            if self.reason not in {
                    "no_recipe_attempted", "unobserved_no_static_proof"}:
                raise ValueError(
                    "execution_unresolved requires a closed visible reason")
        elif self.reason or self.detail:
            raise ValueError("only execution_unresolved carries reason/detail")


@dataclass(frozen=True)
class ProjectionAxis:
    kind: str
    parent: str | None = None
    rule: str = ""
    reason: str = ""
    fact_keys: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PROJECTION_KINDS:
            raise ValueError(f"unknown projection axis {self.kind!r}")
        if tuple(sorted(set(self.fact_keys))) != self.fact_keys:
            raise ValueError("projection fact keys must be unique and sorted")
        if tuple(sorted(set(self.block_ids))) != self.block_ids:
            raise ValueError("projection block ids must be unique and sorted")
        if self.parent is not None and not isinstance(self.parent, str):
            raise TypeError("projection parent is an exact instance path")
        if self.kind == "rendered" and (
                not (self.fact_keys or self.block_ids)
                or self.parent is not None or self.reason or self.rule):
            raise ValueError(
                "rendered requires a product fact/block and carries no parent/rule/reason")
        if self.kind == "grouped" and (
                self.parent is None or not self.rule or self.reason):
            raise ValueError("grouped requires parent + rule and carries no reason")
        if self.kind == "non_architectural" and (
                not self.reason or self.parent is not None or self.rule
                or self.fact_keys or self.block_ids):
            raise ValueError("non_architectural requires a reason only")
        if self.kind == "projection_unresolved" and (
                not self.reason or self.parent is not None or self.rule
                or self.fact_keys or self.block_ids):
            raise ValueError("projection_unresolved requires a visible reason")


@dataclass(frozen=True)
class OccurrenceRow:
    provenance: OccurrenceProvenance
    construction: ConstructionAxis
    execution: ExecutionAxis
    projection: ProjectionAxis

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, OccurrenceProvenance):
            raise TypeError("an occurrence row needs two-key provenance")
        if not isinstance(self.construction, ConstructionAxis):
            raise TypeError("an occurrence row needs one construction value")
        if not isinstance(self.execution, ExecutionAxis):
            raise TypeError("an occurrence row needs one execution value")
        if not isinstance(self.projection, ProjectionAxis):
            raise TypeError("an occurrence row needs one projection value")


@dataclass(frozen=True)
class RelationRow:
    relation_id: str
    kind: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    detail: Mapping[str, Any]
    instance_evidence: tuple[str, ...]
    static_evidence: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    fact_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _closed_text(self.relation_id, field="relation id")
        if self.kind not in RELATION_KINDS:
            raise ValueError(f"unknown relation kind {self.kind!r}")
        if not self.sources or not self.targets or not self.instance_evidence:
            raise ValueError("a relation needs source, target and instance/trace evidence")
        for field_name in ("sources", "targets", "instance_evidence",
                           "static_evidence", "config_paths", "fact_keys"):
            values = getattr(self, field_name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"relation {field_name} must be unique and sorted")
        if self.kind == "relation_unresolved" and not self.detail.get("reason"):
            raise ValueError("relation_unresolved requires a visible reason")
        if self.kind != "relation_unresolved":
            # Parameter identity is itself the primary authority for sharing;
            # its declaration path is supporting evidence.  Every custom
            # execution mechanism still requires an exact source proof.
            if self.kind != "param_share" and not self.static_evidence:
                raise ValueError(
                    "a runtime-detected relation needs an exact source explanation; "
                    "otherwise use relation_unresolved")
        json.dumps(dict(self.detail), sort_keys=True)


@dataclass(frozen=True)
class ReconciliationTable:
    schema_version: int
    model: str
    config_sha256: str
    occurrences: tuple[OccurrenceRow, ...]
    relations: tuple[RelationRow, ...] = ()
    input_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unknown reconciliation schema")
        _closed_text(self.model, field="model")
        paths = tuple(row.provenance.instance_path for row in self.occurrences)
        if len(paths) != len(set(paths)):
            raise ValueError("the occurrence denominator contains duplicate paths")
        if paths != tuple(sorted(paths, key=lambda value: (value.count("."), value))):
            raise ValueError("occurrences must be in canonical parent-before-child order")
        relation_ids = tuple(row.relation_id for row in self.relations)
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation ids are unique")
        known = set(paths)
        for relation in self.relations:
            if not set(relation.sources + relation.targets) <= known:
                raise ValueError("a relation cites an occurrence outside the denominator")
        if tuple(sorted(set(self.input_failures))) != self.input_failures:
            raise ValueError("input failures must be unique and sorted")

    @property
    def unresolved_count(self) -> int:
        return sum(
            row.construction.kind == "construction_conflict"
            or row.execution.kind == "execution_unresolved"
            or row.projection.kind == "projection_unresolved"
            for row in self.occurrences
        ) + sum(row.kind == "relation_unresolved" for row in self.relations)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StaticOccurrenceClaim:
    """Exact runtime-pattern -> static occurrence bridge from OwnerGraph.

    ``path_pattern`` is an address pattern only: ``*`` is one repeated
    construction-site occurrence.  It carries no mechanism role.
    """

    path_pattern: tuple[str, ...]
    occurrence: StaticOccurrenceRef
    class_source_fingerprint: str
    class_qualname: str
    config_paths: tuple[str, ...] = ()
    source_spans: tuple[str, ...] = ()
    expected_count: int | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(part, str) or not part for part in self.path_pattern):
            raise ValueError("static path pattern has non-empty segments")
        if self.expected_count is not None and self.expected_count < 0:
            raise ValueError("static expected count is non-negative")
        if (len(self.class_source_fingerprint) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in self.class_source_fingerprint)):
            raise ValueError("static claim needs its class source fingerprint")
        _closed_text(self.class_qualname, field="static class qualified name")


@dataclass(frozen=True)
class ProjectionClaim:
    """A projection disposition supplied by the existing fact/IR consumer."""

    instance_path: str
    axis: ProjectionAxis
    facts: tuple[EvidenceFact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.instance_path, str):
            raise TypeError("projection claim path is text")
        if not isinstance(self.axis, ProjectionAxis):
            raise TypeError("projection claim carries a closed ProjectionAxis")
        if any(not isinstance(fact, EvidenceFact) for fact in self.facts):
            raise TypeError("projection claims cite existing EvidenceFact rows")
        keys = tuple(sorted(fact.ledger_key() for fact in self.facts))
        if self.axis.fact_keys != keys:
            raise ValueError("projection fact keys must equal the cited fact objects")
        if self.axis.kind == "grouped" and self.axis.parent == self.instance_path:
            raise ValueError("a grouped projection must name a distinct parent")


_TORCH_CONTAINER_TYPES = frozenset({
    ("torch.nn.modules.container", "ModuleList"),
    ("torch.nn.modules.container", "Sequential"),
    ("torch.nn.modules.container", "ModuleDict"),
})
_NO_PRODUCT_CITATION = "no product block or fact cites this occurrence"


def _is_framework_primitive(class_ref: Any) -> bool:
    """Whether the exact runtime type belongs to torch.nn's closed primitives.

    This is a framework ownership/address test, never a mechanism classifier.
    The exact module + qualname remain in :class:`MeaningProvenance`; no class
    spelling is translated into an architectural role.
    """
    return (isinstance(getattr(class_ref, "module", None), str)
            and class_ref.module.startswith("torch.nn.modules.")
            and isinstance(getattr(class_ref, "qualname", None), str)
            and bool(class_ref.qualname))


def _structural_block_ids(ir: Any) -> frozenset[str]:
    """Return exact ids from canonical block-shaped IR records.

    A record is block-shaped only when it has ``id`` plus a presentation
    discriminator.  Merely finding a string elsewhere in extras is not enough.
    This intentionally observes product output; it does not interpret the id.
    """
    from ..ir import ModelIR

    if not isinstance(ir, ModelIR):
        raise TypeError("product projection requires the canonical ModelIR")
    result: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            block_id = value.get("id")
            if (isinstance(block_id, str) and block_id
                    and any(key in value for key in (
                        "kind", "role", "view", "children", "label"))):
                result.add(block_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(ir.to_dict())
    return frozenset(result)


def _fact_static_claims(
    index: Any,
    fact: EvidenceFact,
    static_claims: Sequence[StaticOccurrenceClaim],
) -> tuple[StaticOccurrenceClaim, ...]:
    """Join fact source spans to exact indexed class spans.

    The join is content/source/line based.  Owner names and ledger-key words do
    not participate, so a renamed class or fact key cannot change the result.
    """
    source_by_path: dict[str, Any] = {}
    for node in getattr(index, "source_nodes", ()):
        canonical = str(Path(node.source_id.canonical_path).resolve())
        source_by_path[canonical] = node.source_id
    classes_by_key: dict[tuple[str, str], list[Any]] = {}
    for record in getattr(index, "classes", ()):
        span = record.span
        if span is None:
            continue
        key = (record.symbol.source.content_fingerprint,
               record.symbol.qualified_name)
        classes_by_key.setdefault(key, []).append(record)
    matched: set[StaticOccurrenceClaim] = set()
    for span in fact.source_spans:
        if not span.file or not span.line:
            continue
        source = source_by_path.get(str(Path(span.file).resolve()))
        if source is None:
            continue
        for claim in static_claims:
            if claim.class_source_fingerprint != source.content_fingerprint:
                continue
            records = classes_by_key.get((claim.class_source_fingerprint,
                                          claim.class_qualname), ())
            if any(record.span.line <= span.line <= record.span.end_line
                   for record in records):
                matched.add(claim)
    return tuple(sorted(matched, key=lambda row: (
        row.path_pattern, row.class_source_fingerprint, row.class_qualname)))


def projection_claims_from_product(
    *,
    index: Any,
    inventory: Any,
    static_claims: Sequence[StaticOccurrenceClaim],
    ir: Any,
    facts: Mapping[str, EvidenceFact],
    render_events: Sequence[Any],
) -> tuple[ProjectionClaim, ...]:
    """Build occurrence projection claims from the existing product only.

    Custom occurrences enter through an exact fact-source -> static-owner ->
    runtime-path join.  Closed torch.nn primitives may additionally join by the
    exact attribute spelling under a proven parent to an id the product really
    drew.  Pure containers receive only their definitional non-architectural
    disposition.  No runtime/class/model name supplies mechanism meaning.
    """
    if any(not isinstance(value, EvidenceFact) for value in facts.values()):
        raise TypeError("product projection consumes typed EvidenceFact rows")
    if any(key != value.ledger_key() for key, value in facts.items()):
        raise ValueError("product fact mapping keys must equal typed ledger keys")
    event_type = ("model_unfolder.renderers.html.render_context", "RenderEvent")
    for event in render_events:
        if (type(event).__module__, type(event).__qualname__) != event_type:
            raise TypeError("product projection consumes exact RenderEvent rows")
    modules = tuple(getattr(inventory, "modules", ()))
    paths = tuple(row.path for row in modules)
    if len(paths) != len(set(paths)):
        raise ValueError("product projection needs an exact inventory denominator")
    block_ids = _structural_block_ids(ir)

    rendered_facts: dict[str, dict[str, EvidenceFact]] = {}
    grouped_facts: dict[str, dict[str, dict[str, EvidenceFact]]] = {}
    nodes_by_parent: dict[str, set[str]] = {}
    fact_claim_cache: dict[str, tuple[StaticOccurrenceClaim, ...]] = {}
    for event in render_events:
        event_facts = tuple(sorted(event.facts_projected))
        missing = tuple(key for key in event_facts if key not in facts)
        if missing:
            raise ValueError(
                f"render event cites facts absent from the typed ledger: {missing}")
        candidates: dict[str, dict[str, EvidenceFact]] = {}
        for key in event_facts:
            fact = facts[key]
            claims = fact_claim_cache.setdefault(
                key, _fact_static_claims(index, fact, static_claims))
            for claim in claims:
                for path in paths:
                    if not _path_matches(path, claim.path_pattern):
                        continue
                    candidates.setdefault(path, {})[key] = fact

        # A drill's own custom occurrence is identified mechanically by the
        # exact runtime child attributes that its event drew.  Other candidate
        # occurrences below that anchor are grouped within its drill.  For an
        # architecture-level event (empty block path), the outermost evidenced
        # occurrence is the own block and evidenced descendants are grouped.
        anchors = {
            path for path in candidates
            if any(
                child.rpartition(".")[2] in event.node_ids
                for child in paths
                if child.rpartition(".")[0] == path)
        }
        if not anchors and not event.block_path:
            anchors = {
                path for path in candidates
                if not any(path.startswith(other + ".")
                           for other in candidates if other != path)
            }
        for path, path_facts in candidates.items():
            if path in anchors:
                rendered_facts.setdefault(path, {}).update(path_facts)
                nodes_by_parent.setdefault(path, set()).update(event.node_ids)
                continue
            parents = tuple(sorted(
                (anchor for anchor in anchors
                 if path.startswith(anchor + ".")),
                key=lambda value: (-value.count("."), value)))
            if parents:
                grouped_facts.setdefault(path, {}).setdefault(
                    parents[0], {}).update(path_facts)

    claims_by_path: dict[str, ProjectionClaim] = {}
    # A fact projected by the product and joined to this exact occurrence is a
    # direct rendered claim.  Block ids are supplementary, never the proof.
    for path, path_facts in rendered_facts.items():
        ordered_facts = tuple(path_facts[key] for key in sorted(path_facts))
        claims_by_path[path] = ProjectionClaim(
            path,
            ProjectionAxis(
                "rendered",
                fact_keys=tuple(fact.ledger_key() for fact in ordered_facts)),
            ordered_facts,
        )
    for path, parent_rows in grouped_facts.items():
        if path in claims_by_path or len(parent_rows) != 1:
            continue
        parent, path_facts = next(iter(parent_rows.items()))
        ordered_facts = tuple(path_facts[key] for key in sorted(path_facts))
        claims_by_path[path] = ProjectionClaim(
            path,
            ProjectionAxis(
                "grouped", parent=parent,
                rule="fact source occurrence is drawn inside its proven parent",
                fact_keys=tuple(fact.ledger_key() for fact in ordered_facts)),
            ordered_facts,
        )

    module_by_path = {row.path: row for row in modules}
    for path, module in module_by_path.items():
        exact_type = (module.class_ref.module, module.class_ref.qualname)
        if exact_type in _TORCH_CONTAINER_TYPES:
            claims_by_path[path] = ProjectionClaim(
                path, ProjectionAxis("non_architectural", reason="container"))
            continue
        if not _is_framework_primitive(module.class_ref) or not path:
            continue
        parent, _, attribute = path.rpartition(".")
        if attribute in block_ids:
            claims_by_path[path] = ProjectionClaim(
                path, ProjectionAxis("rendered", block_ids=(attribute,)))
            continue
        if attribute in nodes_by_parent.get(parent, ()):
            claims_by_path[path] = ProjectionClaim(
                path, ProjectionAxis(
                    "grouped", parent=parent,
                    rule="exact parent attribute equals a drawn child id",
                    block_ids=(attribute,)))

    return tuple(claims_by_path[path] for path in sorted(
        claims_by_path, key=lambda value: (value.count("."), value)))


def _path_matches(path: str, pattern: tuple[str, ...]) -> bool:
    parts = tuple(path.split(".")) if path else ()
    return len(parts) == len(pattern) and all(
        expected == "*" or expected == actual
        for actual, expected in zip(parts, pattern))


def _call_paths(observations: Sequence[Any]) -> tuple[dict[str, set[str]], set[str]]:
    exact: dict[str, set[str]] = {}
    ambiguous: set[str] = set()
    for result in observations:
        observation = getattr(result, "observation", None)
        if getattr(result, "status", None) != "ok" or observation is None:
            continue
        recipe_id = observation.recipe.recipe_id
        for call in getattr(observation, "module_calls", ()):
            aliases = tuple(part.strip() for part in call.path.split("|") if part.strip())
            if len(aliases) != 1:
                ambiguous.update(aliases)
                continue
            exact.setdefault(aliases[0], set()).add(recipe_id)
        # The S7 relation observer is another recipe-qualified positive trace.
        # It records exact layer/sibling addresses rather than ModuleCall DTOs.
        for call in (*getattr(observation, "boundaries", ()),
                     *getattr(observation, "sibling_calls", ())):
            exact.setdefault(call.path, set()).add(recipe_id)
    return exact, ambiguous


def _attempted_recipe_ids(observations: Sequence[Any]) -> tuple[str, ...]:
    """All recipe ids that reached the typed execution boundary, success or not."""
    ids = {
        result.recipe.recipe_id for result in observations
        if getattr(result, "recipe", None) is not None
    }
    return tuple(sorted(ids))


def reconcile(
    *,
    model: str,
    inventory: Any,
    observations: Sequence[Any],
    config_document: Any,
    static_claims: Sequence[StaticOccurrenceClaim] = (),
    projection_claims: Sequence[ProjectionClaim] = (),
    relation_rows: Sequence[RelationRow] = (),
) -> ReconciliationTable:
    """Join already-authored evidence without adding an architecture opinion.

    ``inventory`` is an S6 ``InstanceInventory`` and ``config_document`` a
    ``PreparedDocument``.  The production package must not import the isolated
    physics package (S6 isolation law), so the external DTO boundary is checked
    by its exact concrete type identity plus the record's own closure.  A
    duck-typed lookalike cannot pass, while physics remains dependency-free.
    """
    from .document import PreparedDocument

    inventory_type = type(inventory)
    if (inventory_type.__module__, inventory_type.__qualname__) != (
            "physics.instance_inventory", "InstanceInventory"):
        raise TypeError("reconcile requires an InstanceInventory")
    if not isinstance(config_document, PreparedDocument):
        raise TypeError("reconcile requires a PreparedDocument")
    modules = tuple(getattr(inventory, "modules", ()))
    provenance = getattr(inventory, "provenance", None)
    config_hash = getattr(provenance, "config_sha256", "")
    checkpoint = getattr(config_document, "checkpoint", None)
    if not modules or provenance is None or not isinstance(checkpoint, dict):
        raise ValueError("reconciliation inputs have no occurrence/config denominator")
    if _sha256(checkpoint) != config_hash:
        raise ValueError("config document and instance inventory hashes disagree")
    accepted_results = {
        ("physics.execution_observation", "ObservationResult"),
        ("physics.relation_observation", "RelationObservationResult"),
    }
    for result in observations:
        if (type(result).__module__, type(result).__qualname__) not in accepted_results:
            raise TypeError("reconciliation accepts only typed physics results")
        if getattr(result, "status", None) == "ok" \
                and getattr(result, "provenance", None) != provenance:
            raise ValueError("successful observation provenance disagrees with inventory")

    claims_by_path: dict[str, ProjectionClaim] = {}
    for claim in projection_claims:
        if claim.instance_path in claims_by_path:
            # This is the C-7 kill shot: two projection values cannot be voted
            # or merged into one plausible disposition.
            raise ValueError(
                f"two projection values for occurrence {claim.instance_path!r}")
        claims_by_path[claim.instance_path] = claim

    observed, alias_ambiguous = _call_paths(observations)
    attempted_recipe_ids = _attempted_recipe_ids(observations)
    lazy: dict[str, Any] = {}
    for result in observations:
        obs = getattr(result, "observation", None)
        if getattr(result, "status", None) == "ok" and obs is not None:
            for row in getattr(obs, "lazy_observed", ()):
                lazy[row.path] = row.class_ref

    module_by_path = {row.path: row for row in modules}
    if len(module_by_path) != len(modules):
        raise ValueError("inventory module paths are not a denominator")
    for path, class_ref in lazy.items():
        if path in module_by_path:
            continue
        module_by_path[path] = _LazyModule(path, class_ref)

    rows: list[OccurrenceRow] = []
    for path in sorted(module_by_path, key=lambda value: (value.count("."), value)):
        module = module_by_path[path]
        runtime_class = RuntimeClassRef(
            module.class_ref.module, module.class_ref.qualname)
        matches = tuple(claim for claim in static_claims
                        if _path_matches(path, claim.path_pattern))
        conflicts: list[str] = []
        static = matches[0] if len(matches) == 1 else None
        if len(matches) > 1:
            conflicts.append("multiple static owner occurrences match this instance path")
        if static is not None and (
                static.class_qualname != runtime_class.qualname
                or static.class_source_fingerprint not in {
                    item.sha256 for item in provenance.source_files
                    if item.module == runtime_class.module
                }):
            conflicts.append(
                "instance runtime class disagrees with the exact static owner")
        if conflicts:
            construction = ConstructionAxis(
                "construction_conflict", conflicts=tuple(sorted(conflicts)))
        else:
            construction = ConstructionAxis(
                "lazy_observed" if path in lazy else "eager_constructed")

        recipe_ids = tuple(sorted(observed.get(path, ())))
        if recipe_ids:
            execution = ExecutionAxis("observed", recipe_ids=recipe_ids)
        elif path in alias_ambiguous:
            execution = ExecutionAxis(
                "execution_unresolved",
                reason="unobserved_no_static_proof",
                detail="trace observed an aliased object but not this exact address")
        elif not attempted_recipe_ids:
            execution = ExecutionAxis(
                "execution_unresolved", reason="no_recipe_attempted")
        else:
            execution = ExecutionAxis(
                "execution_unresolved",
                reason="unobserved_no_static_proof")

        projection_claim = claims_by_path.get(path)
        projection = (projection_claim.axis if projection_claim else ProjectionAxis(
            "projection_unresolved",
            reason=_NO_PRODUCT_CITATION))
        facts = projection_claim.facts if projection_claim else ()
        fact_paths = tuple(sorted({item for fact in facts
                                  for item in fact.config_paths}))
        source_spans = tuple(sorted({
            f"{span.file or ''}:{span.line or 0}"
            for fact in facts for span in fact.source_spans
        }))
        static_meaning = static.occurrence if static is not None else None
        primitive_meaning = (runtime_class
                             if _is_framework_primitive(module.class_ref)
                             else None)
        meaning = MeaningProvenance(
            static_occurrence=static_meaning,
            framework_primitive=primitive_meaning,
            config_paths=tuple(sorted(set(
                fact_paths + (static.config_paths if static else ())))),
            source_spans=tuple(sorted(set(
                source_spans + (static.source_spans if static else ())))),
            fact_keys=projection.fact_keys,
        )
        rows.append(OccurrenceRow(
            OccurrenceProvenance(path, runtime_class, config_hash, meaning),
            construction, execution, projection))

    # Source-declared guarded None children are real construction negatives,
    # not named_modules() rows.  Add them to the denominator with no runtime
    # class and never pretend they executed or were projected.
    for parent in modules:
        for guarded in parent.guarded_none_children:
            relative = str(guarded.get("path", ""))
            # S6 inventories persist guarded paths from the model root.  A
            # synthetic/older producer may provide a parent-relative path;
            # accept it only when prefixing is mechanically unambiguous.
            if parent.path and (relative == parent.path
                                or relative.startswith(parent.path + ".")):
                path = relative
            else:
                path = f"{parent.path}.{relative}".strip(".")
            if not relative or path in module_by_path:
                continue
            guards = tuple(guarded.get("guards") or ())
            guard_text = " and ".join(
                f"{item.get('predicate')} is {item.get('branch')}"
                for item in guards if isinstance(item, Mapping))
            if not guard_text:
                raise ValueError("a not-constructed child lacks its exact guard")
            source_file = str(guarded.get("source_file", ""))
            line = guarded.get("line")
            if not source_file or not isinstance(line, int) or line <= 0:
                raise ValueError(
                    "a not-constructed child lacks exact source provenance")
            rows.append(OccurrenceRow(
                OccurrenceProvenance(
                    path, None, config_hash, MeaningProvenance(
                        source_spans=(
                            f"{source_file}:{line}",
                        ))),
                ConstructionAxis("not_constructed", guard=guard_text),
                ExecutionAxis("proven_inactive"),
                ProjectionAxis(
                    "projection_unresolved",
                    reason=_NO_PRODUCT_CITATION),
            ))

    rows.sort(key=lambda row: (
        row.provenance.instance_path.count("."), row.provenance.instance_path))

    expected_paths = set(module_by_path)
    expected_paths.update(
        row.provenance.instance_path for row in rows
        if row.construction.kind == "not_constructed")
    actual_paths = {row.provenance.instance_path for row in rows}
    if actual_paths != expected_paths:
        raise ValueError("reconciliation silently dropped an inventory occurrence")

    # Count conflicts are evaluated by the construction authority.  They can
    # only turn matching eager rows into blocking conflict rows; they can never
    # merge the two counts or rewrite the denominator.
    mutable = {row.provenance.instance_path: row for row in rows}
    for claim in static_claims:
        if claim.expected_count is None:
            continue
        matching = [path for path in module_by_path
                    if _path_matches(path, claim.path_pattern)]
        if len(matching) == claim.expected_count:
            continue
        detail = (f"instance count {len(matching)} disagrees with static count "
                  f"{claim.expected_count} for {'.'.join(claim.path_pattern)}")
        authority_for("constructed_modules")
        for path in matching:
            row = mutable[path]
            mutable[path] = dataclasses.replace(
                row, construction=ConstructionAxis(
                    "construction_conflict", conflicts=(detail,)))
    rows = sorted(mutable.values(), key=lambda row: (
        row.provenance.instance_path.count("."), row.provenance.instance_path))

    input_failures = tuple(sorted({
        f"{getattr(getattr(result, 'recipe', None), 'recipe_id', '<request>')}:"
        f"{getattr(getattr(result, 'failure', None), 'kind', 'failed')}:"
        f"{getattr(getattr(result, 'failure', None), 'detail', '')}"
        for result in observations if getattr(result, "status", None) != "ok"
    }))
    return ReconciliationTable(
        1, model, config_hash, tuple(rows),
        tuple(sorted(relation_rows, key=lambda row: row.relation_id)),
        input_failures)


@dataclass(frozen=True)
class _LazyModule:
    path: str
    class_ref: Any
    guarded_none_children: tuple = ()


def _site_key(site: Any) -> str:
    span = site.span
    return (
        f"{span.source.content_fingerprint}:"
        f"{site.enclosing_callable.qualified_name}:"
        f"{span.line}:{span.col}:{span.end_line}:{span.end_col}:{site.ordinal}"
    )


def _relation_id(kind: str, sources: tuple[str, ...], targets: tuple[str, ...],
                 detail: Mapping[str, Any]) -> str:
    digest = _sha256({
        "kind": kind, "sources": sources, "targets": targets,
        "detail": dict(detail),
    })[:16]
    return f"{kind}:{digest}"


def _fact_source_keys(fact: EvidenceFact, inventory: Any) -> tuple[str, ...]:
    """Normalize a fact's source spans to content addresses.

    Fact spans predate S7 and may contain absolute local paths.  The inventory
    already carries the authoritative source bytes, so persisted S7 artifacts
    use the matching source hash and line rather than a machine-specific path.
    """
    files = tuple(inventory.provenance.source_files)
    rows: set[str] = set()
    for span in fact.source_spans:
        if not span.file or not span.line:
            continue
        name = Path(span.file).name
        matches = tuple(item for item in files if Path(item.path).name == name)
        if len(matches) == 1:
            rows.add(f"sha256:{matches[0].sha256}:{span.line}")
    return tuple(sorted(rows))


def _module_for_parameter(name: str, paths: frozenset[str]) -> str | None:
    matches = tuple(path for path in paths
                    if (path and name.startswith(path + "."))
                    or (not path and "." not in name))
    return max(matches, key=len) if matches else None


def _relation_observations(results: Sequence[Any], inventory: Any) -> tuple[Any, ...]:
    rows = []
    for result in results:
        result_type = type(result)
        if (result_type.__module__, result_type.__qualname__) != (
                "physics.relation_observation", "RelationObservationResult"):
            raise TypeError("relation evidence requires RelationObservationResult rows")
        if result.status == "ok" and result.observation is not None:
            if result.provenance != inventory.provenance:
                raise ValueError("relation observation provenance disagrees with inventory")
            rows.append(result.observation)
    return tuple(rows)


def _primary_shape(boundary: Any, *, output: bool = False) -> tuple[int, ...] | None:
    values = boundary.outputs if output else boundary.inputs
    return tuple(values[0].shape) if values else None


def _stream_count(observation: Any) -> tuple[int, int] | None:
    """Return (axis, count) when the recipe leaves one extra rank-4 axis.

    The recipe supplies the input batch/sequence dimensions.  We remove those
    exact dimensions once from the layer-boundary prefix; the sole remainder
    is an observed stream axis.  Ambiguity stays unresolved.
    """
    if not observation.boundaries or not observation.recipe.tensor_arguments:
        return None
    shape = _primary_shape(observation.boundaries[0])
    output = _primary_shape(observation.boundaries[0], output=True)
    recipe_shape = tuple(observation.recipe.tensor_arguments[0].shape)
    if (shape is None or output is None or len(shape) != 4 or shape != output
            or len(recipe_shape) < 2):
        return None
    remaining = list(enumerate(shape[:-1]))
    for dimension in recipe_shape[:2]:
        match = next((index for index, (_axis, value) in enumerate(remaining)
                      if value == dimension), None)
        if match is None:
            return None
        remaining.pop(match)
    if len(remaining) != 1 or remaining[0][1] <= 1:
        return None
    if any(_primary_shape(row) != shape
           or _primary_shape(row, output=True) != shape
           for row in observation.boundaries):
        return None
    return remaining[0]


def _proof_matches(proof: Any, class_refs: Sequence[Any], inventory: Any) -> bool:
    if proof is None or not class_refs:
        return False
    classes = {(item.module, item.qualname) for item in class_refs}
    allowed_hashes = {item.sha256 for item in inventory.provenance.source_files}
    cited_hashes = {
        value.split(":", 2)[1] for value in proof.spans
        if value.startswith("sha256:") and len(value.split(":", 2)) == 3
    }
    return ((proof.class_module, proof.class_qualname) in classes
            and proof.callable == f"{proof.class_qualname}.forward"
            and proof.source_fingerprint in cited_hashes
            and proof.source_fingerprint in allowed_hashes)


def relation_rows_from_evidence(
    *,
    inventory: Any,
    relation_observations: Sequence[Any],
    facts: Mapping[str, EvidenceFact],
    static_proofs: Sequence[Any] = (),
) -> tuple[RelationRow, ...]:
    """Reconcile existing relation evidence without authoring a new fact.

    Instance/trace records determine positive occurrence and exact endpoints.
    Existing facts and source proofs determine custom mechanism meaning.  Any
    positive candidate missing that second half becomes ``relation_unresolved``
    rather than acquiring a name from a runtime class or field spelling.
    """
    inventory_type = type(inventory)
    if (inventory_type.__module__, inventory_type.__qualname__) != (
            "physics.instance_inventory", "InstanceInventory"):
        raise TypeError("relation reconciliation requires InstanceInventory")
    if any(not isinstance(value, EvidenceFact) for value in facts.values()):
        raise TypeError("relation reconciliation consumes existing EvidenceFact rows")
    observations = _relation_observations(relation_observations, inventory)
    paths = frozenset(row.path for row in inventory.modules)
    proofs: dict[str, Any] = {}
    for proof in static_proofs:
        kind = getattr(proof, "kind", None)
        if kind in proofs:
            raise ValueError(f"rival static relation proofs for {kind}")
        proofs[kind] = proof
    rows: list[RelationRow] = []

    # Parameter sharing is definitionally proven by object identity.  The
    # existing tie fact supplies its config/source declaration when present.
    tie_fact = facts.get("model.tie_word_embeddings")
    for alias in inventory.parameter_aliases:
        endpoints = tuple(sorted({
            path for name in alias.names
            if (path := _module_for_parameter(name, paths)) is not None
        }))
        if len(endpoints) < 2:
            continue
        source_support = (_fact_source_keys(tie_fact, inventory)
                          if tie_fact is not None else ())
        config_support = tuple(sorted(tie_fact.config_paths)) \
            if tie_fact is not None and tie_fact.value is True else ()
        detail = {"parameter_paths": list(alias.names)}
        # Parameter-object identity is the primary authority and is complete
        # for the sharing question.  Source/config rows are supporting only;
        # their absence does not weaken the positive identity observation.
        kind = "param_share"
        rows.append(RelationRow(
            _relation_id(kind, (endpoints[0],), endpoints[1:], detail), kind,
            (endpoints[0],), endpoints[1:], detail,
            ("instance:parameter-object-identity",), source_support,
            config_support,
            (tie_fact.ledger_key(),) if tie_fact is not None else ()))

    kv_fact = facts.get("decoder.attention.kv_sharing_schedule")
    side_fact = facts.get("decoder.per_layer_embedding_pathway")
    for observation in observations:
        recipe = observation.recipe.recipe_id
        boundary_by_index = {row.index: row for row in observation.boundaries}
        cross_by_pair = {(row.producer_index, row.consumer_index): row
                         for row in observation.cross_layer_uses}

        if kv_fact is not None and isinstance(kv_fact.value, (tuple, list)):
            for target_index, source_index in enumerate(kv_fact.value):
                if source_index is None:
                    continue
                source = boundary_by_index.get(source_index)
                target = boundary_by_index.get(target_index)
                traced = cross_by_pair.get((source_index, target_index))
                if source is None or target is None or traced is None:
                    continue
                detail = {"what": "key_value", "from_layer": source_index,
                          "to_layer": target_index}
                spans = _fact_source_keys(kv_fact, inventory)
                kind = "activation_reuse" if spans else "relation_unresolved"
                if not spans:
                    detail["reason"] = "cross-layer tensor use lacks source proof"
                rows.append(RelationRow(
                    _relation_id(kind, (source.path,), (target.path,), detail), kind,
                    (source.path,), (target.path,), detail,
                    (f"trace:{recipe}:{traced.consumer_argument}",), spans,
                    tuple(sorted(kv_fact.config_paths)),
                    (kv_fact.ledger_key(),)))

        stream = _stream_count(observation)
        if stream is not None:
            proof = proofs.get("recurrent_state_mix")
            module_by_path = {row.path: row for row in inventory.modules}
            layer_classes = tuple({
                (module_by_path[row.path].class_ref.module,
                 module_by_path[row.path].class_ref.qualname):
                module_by_path[row.path].class_ref
                for row in observation.boundaries
                if row.path in module_by_path
            }.values())
            source_paths = tuple(sorted(row.path for row in observation.boundaries))
            detail = {"stream_axis": stream[0], "stream_count": stream[1]}
            if not _proof_matches(proof, layer_classes, inventory):
                kind = "relation_unresolved"
                detail["reason"] = "rank-4 layer state lacks an exact mixing proof"
                spans = ()
            else:
                kind = "multi_stream_residual"
                spans = tuple(proof.spans)
            rows.append(RelationRow(
                _relation_id(kind, source_paths, source_paths, detail), kind,
                source_paths, source_paths, detail,
                (f"trace:{recipe}:rank4-shape-preserving-boundaries",), spans))

        if side_fact is not None and observation.boundaries:
            first_order = min(row.call_order for row in observation.boundaries)
            extra_shapes = {
                tuple(item.shape)
                for boundary in observation.boundaries
                for item in boundary.inputs[1:]
            }
            layer_count = len(observation.boundaries)

            def is_layer_bank(shape: tuple[int, ...]) -> bool:
                """A pre-stack output contains one exact per-layer axis.

                Exact-shape siblings such as rotary position tensors are not
                candidates.  Removing one axis whose extent equals the exact
                observed layer count must yield a boundary side-input shape.
                """
                if len(shape) < 2:
                    return False
                return any(
                    value == layer_count
                    and (*shape[:axis], *shape[axis + 1:]) in extra_shapes
                    for axis, value in enumerate(shape)
                )

            candidates = tuple(sibling for sibling in observation.sibling_calls
                               if sibling.call_order < first_order
                               and any(is_layer_bank(tuple(out.shape))
                                       for out in sibling.outputs))
            if len(candidates) == 1:
                sibling = candidates[0]
                bank_shapes = tuple(tuple(out.shape) for out in sibling.outputs
                                    if is_layer_bank(tuple(out.shape)))
                side_shapes = {
                    (*shape[:axis], *shape[axis + 1:])
                    for shape in bank_shapes
                    for axis, value in enumerate(shape)
                    if value == layer_count
                }
                targets = tuple(sorted(
                    boundary.path for boundary in observation.boundaries
                    if any(tuple(item.shape) in side_shapes
                           for item in boundary.inputs[1:])))
                spans = _fact_source_keys(side_fact, inventory)
                detail = {"input_shapes": [list(shape)
                                           for shape in sorted(side_shapes)]}
                kind = "per_layer_side_input" if spans and targets \
                    else "relation_unresolved"
                if kind == "relation_unresolved":
                    detail["reason"] = "side-input candidate lacks exact source/targets"
                rows.append(RelationRow(
                    _relation_id(kind, (sibling.path,), targets or (sibling.path,),
                                 detail), kind, (sibling.path,),
                    targets or (sibling.path,), detail,
                    (f"trace:{recipe}:pre-stack-shape-lineage",), spans,
                    tuple(sorted(side_fact.config_paths)),
                    (side_fact.ledger_key(),)))

        if observation.boundaries:
            last = max(row.call_order for row in observation.boundaries)
            final_shape = _primary_shape(observation.boundaries[-1], output=True)
            candidates = tuple(sibling for sibling in observation.sibling_calls
                               if sibling.call_order > last and sibling.inputs
                               and tuple(sibling.inputs[0].shape) == final_shape
                               and sibling.outputs
                               and len(sibling.outputs[0].shape) < len(final_shape or ()))
            if len(candidates) == 1:
                sibling = candidates[0]
                proof = proofs.get("post_stack_collapse")
                detail = {"input_shape": list(sibling.inputs[0].shape),
                          "output_shape": list(sibling.outputs[0].shape)}
                parent_path = observation.stack_path.rpartition(".")[0]
                parent_module = next((row for row in inventory.modules
                                      if row.path == parent_path), None)
                parent_classes = (parent_module.class_ref,) if parent_module else ()
                if not _proof_matches(proof, parent_classes, inventory):
                    kind = "relation_unresolved"
                    detail["reason"] = "post-stack rank collapse lacks source proof"
                    spans = ()
                else:
                    kind = "side_head"
                    spans = tuple(proof.spans)
                source_paths = tuple(sorted(row.path for row in observation.boundaries))
                rows.append(RelationRow(
                    _relation_id(kind, source_paths, (sibling.path,), detail), kind,
                    source_paths, (sibling.path,), detail,
                    (f"trace:{recipe}:post-stack-rank-collapse",), spans))

    ordered = tuple(sorted(rows, key=lambda row: row.relation_id))
    if len({row.relation_id for row in ordered}) != len(ordered):
        raise ValueError("relation evidence produced duplicate identities")
    return ordered


def _occurrence_ref(occurrence: Any) -> StaticOccurrenceRef:
    return StaticOccurrenceRef(
        occurrence.root.source.content_fingerprint,
        occurrence.root.qualified_name,
        tuple(_site_key(site) for site in occurrence.sites),
    )


def static_claims_from_owner_graph(
    graph: Any,
    *,
    expected_counts: Mapping[tuple[str, ...], int] | None = None,
) -> tuple[StaticOccurrenceClaim, ...]:
    """Project an OwnerGraph into neutral runtime-address patterns.

    This does not select an owner or a role.  It preserves the graph's exact
    construction edges; an ``element`` edge contributes one ``*`` segment
    because the static graph represents a symbolic repeated construction, not
    fabricated concrete indices.
    """
    root = getattr(graph, "root", None)
    if root is None or not hasattr(graph, "walk"):
        raise TypeError("static claim projection requires an OwnerGraph")
    expected_counts = dict(expected_counts or {})
    rows: list[StaticOccurrenceClaim] = []

    def visit(node: Any, pattern: tuple[str, ...]) -> None:
        bindings = tuple(
            ".".join(binding.resolved_prefix)
            for binding in node.config_bindings
            if binding.resolved_prefix
        )
        spans = tuple(
            "sha256:"
            f"{site.span.source.content_fingerprint}:"
            f"{site.span.line}:{site.span.col}:"
            f"{site.span.end_line}:{site.span.end_col}"
            for site in node.occurrence.sites
        )
        rows.append(StaticOccurrenceClaim(
            pattern, _occurrence_ref(node.occurrence),
            node.symbol.source.content_fingerprint,
            node.symbol.qualified_name,
            tuple(sorted(set(bindings))), tuple(sorted(set(spans))),
            expected_counts.get(pattern),
        ))
        for child in node.children:
            segment = (child.via_field, "*") if child.via_kind == "element" \
                else (child.via_field,)
            visit(child, (*pattern, *segment))

    visit(root, ())
    return tuple(rows)


def unresolved_axis_findings(table: ReconciliationTable | Mapping[str, Any]) -> list[str]:
    """Blocking Sable-net payload for every unresolved/conflicting axis.

    Mapping input supports the persisted shadow artifact.  Missing/empty rows
    are findings, so an empty register cannot make the net vacuously green.
    """
    data = table.to_dict() if isinstance(table, ReconciliationTable) else dict(table)
    occurrences = data.get("occurrences")
    if not isinstance(occurrences, list) and not isinstance(occurrences, tuple):
        return ["reconciliation table has no occurrence denominator"]
    if not occurrences:
        return ["reconciliation occurrence denominator is empty"]
    findings: list[str] = []
    for row in occurrences:
        provenance = row.get("provenance", {}) if isinstance(row, Mapping) else {}
        path = provenance.get("instance_path", "<missing>")
        for axis in ("construction", "execution", "projection"):
            value = row.get(axis, {}) if isinstance(row, Mapping) else {}
            kind = value.get("kind") if isinstance(value, Mapping) else None
            if kind in {"construction_conflict", "execution_unresolved",
                        "projection_unresolved"} or kind is None:
                findings.append(f"{path}: {axis}={kind or 'missing'}")
    for relation in data.get("relations") or ():
        if relation.get("kind") == "relation_unresolved":
            findings.append(
                f"relation {relation.get('relation_id', '<missing>')}: unresolved")
    return sorted(findings)


__all__ = [
    "AUTHORITY_MATRIX", "CONSTRUCTION_KINDS", "EXECUTION_KINDS",
    "PROJECTION_KINDS", "RELATION_KINDS", "AuthorityRule",
    "ConstructionAxis", "ExecutionAxis", "MeaningProvenance",
    "OccurrenceProvenance", "OccurrenceRow", "ProjectionAxis",
    "ProjectionClaim", "ReconciliationTable", "RelationRow",
    "RuntimeClassRef", "StaticOccurrenceClaim", "StaticOccurrenceRef",
    "authority_for", "projection_claims_from_product", "reconcile",
    "relation_rows_from_evidence",
    "static_claims_from_owner_graph",
    "unresolved_axis_findings",
]
