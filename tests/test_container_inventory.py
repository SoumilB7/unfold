"""U3-B2 — neutral repeated-container address inventory
(resolve_container_inventory).

B2 consumes a RESOLVED D0 ComponentRootResolution + an EXPLICIT owner occurrence
(normally B1's model-stage occurrence).  It never selects an owner and never falls
back to the root.  It preserves the authoritative ContainerElementsRecord + every
original ConstructionSite/ConstructionSiteId, emits observed syntactic kind and
source order only (never execution order), groups same-owner+field records as
typed rivals, and cites a count's config path only when a recorded observation's
span lies INSIDE the count expression (never a sibling occurrence).
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    ComponentRootResolution,
    OwnerOccurrenceId,
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import (
    ContainerAddress,
    ContainerRival,
    ContainerInventory,
    resolve_container_inventory,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ChildCandidate,
    ConfigPathObservation,
    ConfigSegment,
    ConstructionSite,
    ConstructionSiteId,
    ContainerElementsRecord,
    ExprNode,
    SourceId,
    SourceSpan,
    SymbolId,
)


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(files, arch="Wrapper"):
    flat = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(
        source="local", files=tuple(flat),
        component_files={k: tuple(v) for k, v in files.items()},
        component_architectures={"root": arch})


def _index(tmp_path, files, arch="Wrapper"):
    return pi.build_program_index(_bundle(files, arch))


def _cr(idx, files, arch="Wrapper"):
    return resolve_component_root(idx, _bundle(files, arch), "root")


def _model_stage(tmp_path, src, arch="Wrapper"):
    """D0 -> B1 (must resolve) -> B2 at B1's model-stage occurrence.  No fallback."""
    files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = _index(tmp_path, files, arch)
    cr = _cr(idx, files, arch)
    b1 = resolve_declared_model_stage(idx, cr)
    assert b1.status == "resolved", b1.status
    return idx, cr, b1, resolve_container_inventory(idx, cr, b1.occurrence)


def _child(site):
    return (site.candidates[0].symbol
            if len(site.candidates) == 1 and site.candidates[0].symbol else None)


_STACK_SRC = """
    class Block:
        def __init__(self, config): pass
    class BaseModel:
        def __init__(self, config):
            self.num_hidden_layers = config.num_hidden_layers
            self.layers = ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = BaseModel(config)
"""


# --------------------------------------------------------------------------- #
# Consumes B1's model-stage occurrence; symbolic repetition; site preservation
# --------------------------------------------------------------------------- #

def test_consumes_b1_model_stage_occurrence(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, _STACK_SRC)
    assert b1.status == "resolved" and b1.attribute == "model"
    assert inv.status == "resolved"
    assert inv.owner_symbol.qualified_name == "BaseModel"          # the model stage, not Wrapper
    assert inv.owner_occurrence == b1.occurrence
    assert [c.field for c in inv.containers] == ["layers"]


def test_symbolic_repetition_is_one_element_site_plus_count(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, _STACK_SRC)
    (container,) = inv.containers
    assert container.syntactic_kind == "modulelist"
    assert len(container.element_sites) == 1                       # ONE element site, never N
    assert _child(container.element_sites[0]).qualified_name == "Block"
    assert container.count_expression is not None


def test_authoritative_record_and_site_ids_are_preserved(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, _STACK_SRC)
    (container,) = inv.containers
    authoritative = next(r for r in idx.containers
                         if r.owner == inv.owner_symbol and r.field == "layers")
    assert container.record is authoritative                      # exact record object
    assert [s.site_id for s in container.element_sites] == \
           [s.site_id for s in authoritative.elements]            # exact ConstructionSiteIds


# --------------------------------------------------------------------------- #
# Literal list; observed-syntax kind (Sequential element coverage is incomplete)
# --------------------------------------------------------------------------- #

def test_literal_list_emits_each_element_site(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class A:
            def __init__(self, config): pass
        class B:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.pair = ModuleList([A(config), B(config)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    (container,) = inv.containers
    assert {_child(s).qualified_name for s in container.element_sites} == {"A", "B"}
    assert container.count_expression is None                      # a literal list has no count


def test_kind_is_observed_syntax_and_sequential_coverage_is_honest(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class A:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.seq = Sequential(A(config), A(config))
                self.experts = ModuleDict({})
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    kinds = {c.field: c.syntactic_kind for c in inv.containers}
    assert kinds == {"seq": "sequential", "experts": "moduledict"}
    seq = next(c for c in inv.containers if c.field == "seq")
    # ProgramIndex does not emit direct positional args as elements -> honest 0.
    assert seq.element_sites == ()


# --------------------------------------------------------------------------- #
# Source order only — NEVER execution order
# --------------------------------------------------------------------------- #

def test_source_order_is_source_order_not_execution(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class A:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.first = ModuleList([A(config)])
                self.second = ModuleList([A(config), A(config)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert [(c.source_order, c.field) for c in inv.containers] == [(0, "first"), (1, "second")]
    assert not hasattr(inv.containers[0], "happens_before")
    assert not hasattr(inv.containers[0], "execution_order")


# --------------------------------------------------------------------------- #
# Rival / dynamic elements preserved; same-field guarded records -> ContainerRival
# --------------------------------------------------------------------------- #

def test_dynamic_element_construction_is_unresolved(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class BaseModel:
            def __init__(self, config):
                self.layers = ModuleList([make_block(config) for _ in range(config.n)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    (container,) = inv.containers
    (site,) = container.element_sites
    assert _child(site) is None and site in container.unresolved_sites


def test_same_field_guarded_records_are_typed_rivals(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class A:
            def __init__(self, config): pass
        class B:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                if config.flag:
                    self.blocks = ModuleList([A(config)])
                else:
                    self.blocks = ModuleList([B(config)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert inv.containers == ()                       # no single container for a rival field
    (rival,) = inv.rivals
    assert rival.field == "blocks" and len(rival.records) == 2
    # both authoritative records preserved with their element guards + spans
    assert all(r.span is not None for r in rival.records)
    assert all(any(elem.guard for elem in r.elements) for r in rival.records)


# --------------------------------------------------------------------------- #
# Owner scoping + explicit diffusion-root selection (no B1-failure fallback)
# --------------------------------------------------------------------------- #

def test_only_the_owners_own_containers_are_enumerated(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class Block:
            def __init__(self, config): pass
        class Other:
            def __init__(self, config):
                self.decoys = ModuleList([Block(config)])
        class BaseModel:
            def __init__(self, config):
                self.layers = ModuleList([Block(config) for _ in range(config.n)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert [c.field for c in inv.containers] == ["layers"]         # Other.decoys excluded


def test_explicit_diffusion_root_selection_no_fallback(tmp_path):
    """A root with no base_model_prefix: B1 does NOT resolve a model stage.  To
    inventory it the caller must EXPLICITLY pass the D0 root occurrence — B2 never
    falls back from a failed/absent B1 on its own."""
    files = {"root": (_write(tmp_path, "m.py", """
        class Block:
            def __init__(self, config): pass
        class DiffusionModel:
            def __init__(self, config):
                self.blocks = ModuleList([Block(config) for _ in range(config.depth)])
    """),)}
    idx = _index(tmp_path, files, "DiffusionModel")
    cr = resolve_component_root(idx, _bundle(files, "DiffusionModel"), "root")
    b1 = resolve_declared_model_stage(idx, cr)
    assert b1.status == "absent"                                   # no base_model_prefix
    # explicit root selection by the caller (never an automatic fallback):
    inv = resolve_container_inventory(idx, cr, cr.graph.root.occurrence)
    assert inv.status == "resolved" and inv.owner_symbol.qualified_name == "DiffusionModel"
    assert [c.field for c in inv.containers] == ["blocks"]


# --------------------------------------------------------------------------- #
# Count config-path: sibling-path laundering is refused; strict span containment
# --------------------------------------------------------------------------- #

def test_sibling_path_is_never_laundered_into_a_count_citation(tmp_path):
    """config.num_hidden_layers is read directly (a sibling) AND inside the count;
    the sibling read's span is OUTSIDE the count expression, and the count's own
    read is not separately observed -> None, never the sibling."""
    idx, cr, b1, inv = _model_stage(tmp_path, _STACK_SRC)
    (container,) = inv.containers
    assert container.count_expression is not None
    assert container.count_config_path is None                    # sibling never laundered


# --------------------------------------------------------------------------- #
# Absent / failed / D0 refusal
# --------------------------------------------------------------------------- #

def test_absent_when_owner_has_no_containers(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, """
        class Norm:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.norm = Norm(config)
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert inv.status == "absent" and inv.containers == () and inv.rivals == ()
    assert inv.owner_symbol.qualified_name == "BaseModel"


def test_owner_not_in_graph_fails(tmp_path):
    idx, cr, b1, inv = _model_stage(tmp_path, _STACK_SRC)
    ghost = OwnerOccurrenceId(SymbolId(SourceId("/x.py", "fp", component_key="root"), "Ghost"))
    res = resolve_container_inventory(idx, cr, ghost)
    assert res.status == "failed" and res.failure_kind == "owner_not_in_graph"


def test_non_resolved_d0_and_bad_types_are_rejected(tmp_path):
    idx, cr, b1, _ = _model_stage(tmp_path, _STACK_SRC)
    # a non-resolved D0 (e.g. a broken component file) may never be inventoried
    absent_root = ComponentRootResolution(status="absent", component_key="root")
    with pytest.raises(ValueError):
        resolve_container_inventory(idx, absent_root, cr.graph.root.occurrence)
    with pytest.raises(TypeError):
        resolve_container_inventory(idx, cr.graph, cr.graph.root.occurrence)   # not a D0
    with pytest.raises(TypeError):
        resolve_container_inventory(idx, cr, object())                          # not an occurrence


def test_broken_component_file_makes_b2_refuse(tmp_path):
    """A valid root plus a broken sibling file: D0's hidden-rival law returns
    failed, so B2 (requiring a resolved D0) refuses -- it cannot bypass D0."""
    good = _write(tmp_path, "good.py",
                  "class Block:\n    def __init__(self, config): pass\n"
                  "class Wrapper:\n    base_model_prefix = 'model'\n"
                  "    def __init__(self, config):\n"
                  "        self.layers = ModuleList([Block(config)])\n")
    broken = _write(tmp_path, "broken.py", "class Other(:\n")
    files = {"root": (good, broken)}
    idx = _index(tmp_path, files)
    cr = resolve_component_root(idx, _bundle(files), "root")
    assert cr.status == "failed"
    with pytest.raises(ValueError):
        resolve_container_inventory(idx, cr, OwnerOccurrenceId(
            SymbolId(SourceId("/good.py", "fp", component_key="root"), "Wrapper")))


# --------------------------------------------------------------------------- #
# Closure / forgery poisons (synthetic records)
# --------------------------------------------------------------------------- #

_SRC = SourceId("/m.py", "fp", component_key="root")


def _sym(name):
    return SymbolId(_SRC, name)


def _span(line, col=0, end_line=0, end_col=0):
    return SourceSpan(_SRC, line, col, end_line or line, end_col or col)


def _site(owner, field, span, symbol_name="Block"):
    init = _sym(owner.qualified_name + ".__init__")
    sid = ConstructionSiteId(owner, init, span, 0)
    cand = (ChildCandidate(ExprNode(kind="name", name=symbol_name), _sym(symbol_name), "direct_name"),)
    return ConstructionSite(sid, owner, init, "element", field,
                            ExprNode(kind="name", name=symbol_name), (), (), (), cand, "modulelist", span)


def _record(owner, field, elements=(), count=None, span=None):
    init = _sym(owner.qualified_name + ".__init__")
    return ContainerElementsRecord(owner, init, field, "modulelist", elements, count,
                                   span or _span(2))


def test_container_address_element_owner_must_match(tmp_path):
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    owner = _sym("BaseModel")
    foreign_site = _site(_sym("OtherOwner"), "layers", _span(3))   # element of a DIFFERENT owner
    rec = _record(owner, "layers", (foreign_site,))
    with pytest.raises(ValueError):
        ContainerAddress(occ, rec, 0)


def _count_range_config_n():
    """A structural count expression `range(config.n)` with spans."""
    config_name = ExprNode(kind="name", name="config", span=_span(5, 26, 5, 32))
    n_attr = ExprNode(kind="attribute", name="n", children=(config_name,), span=_span(5, 26, 5, 40))
    range_name = ExprNode(kind="name", name="range", span=_span(5, 20, 5, 25))
    return ExprNode(kind="call", children=(range_name, n_attr), span=_span(5, 20, 5, 41))


def test_count_citation_invariants():
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    owner = _sym("BaseModel")
    init = _sym("BaseModel.__init__")
    count = _count_range_config_n()
    rec = _record(owner, "layers", (), count, _span(5))
    cfg = ExprNode(kind="name", name="config")
    inside = ConfigPathObservation(owner, init, cfg, (ConfigSegment("n"),), "attr", _span(5, 26, 5, 40))
    outside = ConfigPathObservation(owner, init, cfg, (ConfigSegment("n"),), "attr", _span(6, 8, 6, 20))
    wrong_path = ConfigPathObservation(owner, init, cfg, (ConfigSegment("other"),), "attr", _span(5, 26, 5, 40))
    dynamic = ConfigPathObservation(owner, init, cfg, (ConfigSegment("", True),), "attr", _span(5, 26, 5, 40))
    # exact path + span inside the count -> allowed
    ContainerAddress(occ, rec, 0, inside)
    # a sibling (span outside the count) -> rejected
    with pytest.raises(ValueError):
        ContainerAddress(occ, rec, 0, outside)
    # a WRONG path whose span is inside the count -> rejected (structural membership)
    with pytest.raises(ValueError):
        ContainerAddress(occ, rec, 0, wrong_path)
    # a dynamic segment -> rejected
    with pytest.raises(ValueError):
        ContainerAddress(occ, rec, 0, dynamic)
    # a citation with no count expression -> rejected
    with pytest.raises(ValueError):
        ContainerAddress(occ, _record(owner, "layers", (), None), 0, inside)


def test_cross_owner_same_source_container_is_rejected():
    """A container record whose owner is a DIFFERENT symbol in the SAME source may
    never be laundered into an inventory bound to another owner symbol."""
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    foreign = ContainerAddress(occ, _record(_sym("OtherModel"), "layers", (), None, _span(3)), 0)
    with pytest.raises(ValueError):
        ContainerInventory("resolved", occ, _sym("BaseModel"), (foreign,))


def test_cross_owner_same_source_rival_is_rejected():
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    r1 = _record(_sym("OtherModel"), "blocks", (), None, _span(3))
    r2 = _record(_sym("OtherModel"), "blocks", (), None, _span(5))
    rival = ContainerRival(occ, "blocks", (r1, r2), 0)
    with pytest.raises(ValueError):
        ContainerInventory("resolved", occ, _sym("BaseModel"), (), (rival,))


def test_d0_from_a_different_program_index_is_index_mismatch(tmp_path):
    """A D0 resolution built from a DIFFERENT ProgramIndex is rejected rather than
    silently enumerating mismatched records."""
    files_a = {"root": (_write(tmp_path, "a.py", _STACK_SRC),)}
    idx_a = _index(tmp_path, files_a)
    files_b = {"root": (_write(tmp_path, "b.py", """
        class Other:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config):
                self.parts = ModuleList([Other(config)])
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """),)}
    idx_b = _index(tmp_path, files_b)
    cr_b = _cr(idx_b, files_b)
    # pass index A but a D0 resolved from index B:
    res = resolve_container_inventory(idx_a, cr_b, cr_b.graph.root.occurrence)
    assert res.status == "failed" and res.failure_kind == "index_mismatch"


def test_owner_class_absent_from_index_is_mismatch_not_absent(tmp_path):
    """Index A has the wrapper + child files (model stage = the child); Index B has
    the IDENTICAL wrapper file but OMITS the child.  The graph-root check passes
    (same wrapper), but the OWNER class is absent from B -> index_mismatch, never a
    masquerading `absent`."""
    wrapper = _write(tmp_path, "wrapper.py",
                     "from child import BaseModel\n"
                     "class Wrapper:\n    base_model_prefix = 'model'\n"
                     "    def __init__(self, config):\n"
                     "        self.model = BaseModel(config)\n")
    child = _write(tmp_path, "child.py",
                   "class Block:\n    def __init__(self, config): pass\n"
                   "class BaseModel:\n    def __init__(self, config):\n"
                   "        self.layers = ModuleList([Block(config) for _ in range(config.n)])\n")
    files_a = {"root": (wrapper, child)}
    idx_a = _index(tmp_path, files_a)
    cr_a = _cr(idx_a, files_a)
    b1 = resolve_declared_model_stage(idx_a, cr_a)
    assert b1.status == "resolved"                                  # model stage = child BaseModel
    idx_b = _index(tmp_path, {"root": (wrapper,)})                  # identical wrapper, no child
    assert idx_b.class_by_symbol(cr_a.graph.root.symbol) is not None  # graph-root check passes
    res = resolve_container_inventory(idx_b, cr_a, b1.occurrence)
    assert res.status == "failed" and res.failure_kind == "index_mismatch"


def test_container_rival_closure():
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    owner = _sym("BaseModel")
    r1 = _record(owner, "blocks", (), None, _span(3))
    r2 = _record(owner, "blocks", (), None, _span(5))
    other = _record(owner, "different", (), None, _span(7))
    ContainerRival(occ, "blocks", (r1, r2), 0)                     # valid
    with pytest.raises(ValueError):                                # <2 records
        ContainerRival(occ, "blocks", (r1,), 0)
    with pytest.raises(ValueError):                                # mixed fields
        ContainerRival(occ, "blocks", (r1, other), 0)


def test_inventory_status_closure():
    occ = OwnerOccurrenceId(_sym("BaseModel"))
    with pytest.raises(ValueError):
        ContainerInventory("maybe", occ)
    with pytest.raises(ValueError):                                # resolved needs owner + payload
        ContainerInventory("resolved", occ, _sym("BaseModel"))
    with pytest.raises(ValueError):                                # failed needs a kind
        ContainerInventory("failed", occ)
    with pytest.raises(ValueError):                                # detail without kind
        ContainerInventory("failed", occ, failure_detail="x")
