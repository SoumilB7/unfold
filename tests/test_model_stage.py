"""U3-B1 — the declared model-stage address boundary (resolve_declared_model_stage).

The resolver consumes the ALREADY-RESOLVED root OwnerGraph and resolves the
model-stage occurrence a root DECLARES via a closed framework address protocol
(base_model_prefix), matched against the graph's exact child/unresolved
occurrences. It uses only the code-declared literal + exact reference binding
through inheritance + the graph — never class names, model types, role
vocabulary, embedding/layer/norm evidence, call ordering, return flow, a
most-plausible-child heuristic, or a family table. A resolved result carries the
exact existing graph child occurrence (graph.node_for returns it).
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    ComponentRootResolution,
    DeclaredModelStageResolution,
    FrameworkAddressProtocol,
    ModelStageDeclaration,
    OwnerGraph,
    OwnerNode,
    OwnerOccurrenceId,
    OwnerRival,
    resolve_component_root,
    resolve_declared_model_stage,
    resolve_owner_graph,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ConstructionSiteId,
    SourceId,
    SourceSpan,
    SymbolId,
)

CFG = "    def __init__(self, config): pass\n"


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


def _cr(idx, files, arch="Wrapper", component_key="root"):
    """The resolved (or otherwise typed) D0 ComponentRootResolution B1 consumes."""
    return resolve_component_root(idx, _bundle(files, arch), component_key)


def _graph(idx, name):
    sym = next(c.symbol for c in idx.classes if c.symbol.qualified_name == name)
    return resolve_owner_graph(idx, sym)


def _resolve(tmp_path, src, root="Wrapper", files=None):
    if files is None:
        files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = _index(tmp_path, files, root)
    return idx, resolve_declared_model_stage(idx, _cr(idx, files, root))


def _assert_round_trip(idx, root_name, res):
    """A resolved address is valid only if it round-trips through the AUTHORITATIVE
    OwnerGraph — the exact node exists and is identical (seam assertion)."""
    assert res.status == "resolved"
    graph = _graph(idx, root_name)
    node = graph.node_for(res.occurrence)
    assert node is not None, "resolved occurrence is absent from the authoritative graph"
    assert node.occurrence == res.occurrence
    if res.self_stage:
        assert node is graph.root
    return node


# --------------------------------------------------------------------------- #
# Resolved: direct + inherited (single clean base) + occurrence is the graph child
# --------------------------------------------------------------------------- #

def test_direct_declaration_resolves_to_graph_child(tmp_path):
    idx, res = _resolve(tmp_path, """
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
                self.lm_head = Linear(config.h, config.v)
    """)
    assert res.status == "resolved" and res.attribute == "model" and not res.self_stage
    node = _assert_round_trip(idx, "Wrapper", res)
    assert node.symbol.qualified_name == "BaseModel"


def test_inherited_single_clean_base_resolves(tmp_path):
    idx, res = _resolve(tmp_path, """
        class PreTrained:
            base_model_prefix = "transformer"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(PreTrained):
            def __init__(self, config):
                self.transformer = BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "transformer"
    assert res.declaration.inherited and res.declaration.declaring_class.qualified_name == "PreTrained"
    node = _assert_round_trip(idx, "Wrapper", res)
    assert node.symbol.qualified_name == "BaseModel"


# --------------------------------------------------------------------------- #
# Poison 1 + 2 — helper occurrence is the exact two-site graph child; node_for
# --------------------------------------------------------------------------- #

def test_helper_occurrence_is_exact_two_site_graph_child(tmp_path):
    idx, res = _resolve(tmp_path, """
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = self._build(config)
            def _build(self, config):
                return BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    node = _assert_round_trip(idx, "Wrapper", res)
    assert node.symbol.qualified_name == "BaseModel"
    assert len(res.occurrence.sites) == 2          # helper-call site + return site


def test_every_resolved_occurrence_is_in_the_graph(tmp_path):
    idx, res = _resolve(tmp_path, """
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "resolved" and not res.self_stage
    assert _graph(idx, "Wrapper").node_for(res.occurrence) is not None


# --------------------------------------------------------------------------- #
# Poison 3 — one dispatch site with two class candidates -> not resolved
# --------------------------------------------------------------------------- #

def test_dispatch_site_two_candidates_carries_owner_rivals(tmp_path):
    """One site, multiple class candidates: ambiguous carrying the AUTHORITATIVE
    OwnerRival records from graph.conflicts — never a fabricated occurrence."""
    idx, res = _resolve(tmp_path, """
        class BaseA:
            def __init__(self, config): pass
        class BaseB:
            def __init__(self, config): pass
        MODELS = {"a": BaseA, "b": BaseB}
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = MODELS[config.kind](config)
    """)
    assert res.status == "ambiguous"
    assert res.rival_occurrences == ()                        # NO fabricated occurrence
    assert len(res.rival_owners) == 2
    graph = _graph(idx, "Wrapper")
    assert all(isinstance(r, OwnerRival) and r.parent == graph.root.occurrence
               for r in res.rival_owners)
    assert {r.candidate.qualified_name for r in res.rival_owners} == {"BaseA", "BaseB"}


# --------------------------------------------------------------------------- #
# Poison 4 — candidate with no resolved child symbol -> not resolved
# --------------------------------------------------------------------------- #

def test_symbol_less_candidate_does_not_resolve(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = SomeExternalModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "external_model_stage"


# --------------------------------------------------------------------------- #
# Poison 5 + 6 — same base spelling in two files / across components -> distinct
# --------------------------------------------------------------------------- #

def test_same_base_spelling_two_files_reversed_binds_locally(tmp_path):
    f1 = _write(tmp_path, "a.py",
                "class PreTrained:\n    base_model_prefix = \"model\"\n"
                "class BaseModel:\n" + CFG +
                "class Wrapper(PreTrained):\n    def __init__(self, config):\n        self.model = BaseModel(config)\n")
    f2 = _write(tmp_path, "b.py", "class PreTrained:\n    base_model_prefix = \"transformer\"\n")
    for order in ((f1, f2), (f2, f1)):
        files = {"root": order}
        idx = _index(tmp_path, files)
        res = resolve_declared_model_stage(idx, _cr(idx, files))
        assert res.status == "resolved" and res.attribute == "model"   # local base wins, stable
        assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


def test_same_base_spelling_across_components_is_distinct(tmp_path):
    root_f = _write(tmp_path, "root.py",
                    "class PreTrained:\n    base_model_prefix = \"model\"\n"
                    "class BaseModel:\n" + CFG +
                    "class Wrapper(PreTrained):\n    def __init__(self, config):\n        self.model = BaseModel(config)\n")
    vis_f = _write(tmp_path, "vis.py", "class PreTrained:\n    base_model_prefix = \"transformer\"\n")
    files = {"root": (root_f,), "vision": (vis_f,)}
    idx = _index(tmp_path, files)
    res = resolve_declared_model_stage(idx, _cr(idx, files))
    assert res.status == "resolved" and res.attribute == "model"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


# --------------------------------------------------------------------------- #
# Poison 7 — imported alias binding selects only its exact base
# --------------------------------------------------------------------------- #

def test_imported_alias_binding_selects_exact_base(tmp_path):
    base_f = _write(tmp_path, "base_mod.py", "class PreTrained:\n    base_model_prefix = \"model\"\n")
    other_f = _write(tmp_path, "other_mod.py", "class PreTrained:\n    base_model_prefix = \"WRONG\"\n")
    wrap_f = _write(tmp_path, "wrap.py",
                    "from base_mod import PreTrained\n"
                    "class BaseModel:\n" + CFG +
                    "class Wrapper(PreTrained):\n    def __init__(self, config):\n        self.model = BaseModel(config)\n")
    files = {"root": (base_f, other_f, wrap_f)}
    idx = _index(tmp_path, files)
    res = resolve_declared_model_stage(idx, _cr(idx, files))
    assert res.status == "resolved" and res.attribute == "model"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


# --------------------------------------------------------------------------- #
# Poison 8 — duplicate class-body declarations -> last (source order) wins
# --------------------------------------------------------------------------- #

def test_duplicate_class_body_declarations_last_wins(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Encoder:
            def __init__(self, config): pass
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "encoder"
            base_model_prefix = "model"
            def __init__(self, config):
                self.encoder = Encoder(config)
                self.model = BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


# --------------------------------------------------------------------------- #
# Poison 9 — direct literal shadows an inherited dynamic declaration
# --------------------------------------------------------------------------- #

def test_direct_literal_shadows_inherited_dynamic(tmp_path):
    idx, res = _resolve(tmp_path, """
        class PreTrained:
            base_model_prefix = SOME_DYNAMIC
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(PreTrained):
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


# --------------------------------------------------------------------------- #
# Poison 10 — an unresolved EARLIER base plus a known later declaration -> failed
# (never skip the unresolved earlier base to use the later declaration)
# --------------------------------------------------------------------------- #

def test_unresolved_earlier_base_with_known_later_declaration_is_mro_incomplete(tmp_path):
    idx, res = _resolve(tmp_path, """
        class KnownPreTrained:
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(ExternalMixin, KnownPreTrained):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "mro_incomplete"


# --------------------------------------------------------------------------- #
# Poison 11 — B1 requires a RESOLVED D0 ComponentRootResolution (D0 isolation)
# --------------------------------------------------------------------------- #

def _sym(name="Wrapper"):
    return SymbolId(SourceId("/m.py", "fp", component_key="root"), name)


def _root_node(occurrence, symbol):
    return OwnerNode(occurrence=occurrence, symbol=symbol, config_bindings=(),
                     config_prefix_candidates=((),), via_site=None,
                     via_field="", via_kind="root")


def test_b1_requires_a_component_root_resolution(tmp_path):
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", "class Wrapper:\n" + CFG),)})
    sym = _sym()
    # a bare OwnerGraph is no longer accepted — B1 consumes a D0 resolution.
    graph = OwnerGraph(root=_root_node(OwnerOccurrenceId(sym), sym))
    with pytest.raises(TypeError):
        resolve_declared_model_stage(idx, graph)
    with pytest.raises(TypeError):
        resolve_declared_model_stage(idx, OwnerOccurrenceId(sym))
    # a NON-resolved component root may never be resolved (inherits D0).
    absent_root = ComponentRootResolution(status="absent", component_key="root")
    with pytest.raises(ValueError):
        resolve_declared_model_stage(idx, absent_root)


# --------------------------------------------------------------------------- #
# Poison 12 — forged attribute/declaration/occurrence combinations rejected
# --------------------------------------------------------------------------- #

def test_forged_resolution_combinations_are_rejected():
    root = _sym()
    span = SourceSpan(root.source, 1)
    decl = ModelStageDeclaration(root, "model", span, False,
                                 proof_trace=(root,), precedence_basis="root-direct")
    site = ConstructionSiteId(root, SymbolId(root.source, "Wrapper.__init__"), span, 0)
    child_occ = OwnerOccurrenceId(root).child(site)
    # attribute must equal declaration.attribute
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("resolved", root, attribute="other",
                                     declaration=decl, occurrence=child_occ)
    # self_stage requires empty prefix
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("resolved", root, attribute="model",
                                     declaration=decl, occurrence=OwnerOccurrenceId(root),
                                     self_stage=True)
    # non-self resolved requires a non-empty child occurrence
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("resolved", root, attribute="model",
                                     declaration=decl, occurrence=OwnerOccurrenceId(root))
    # rival occurrences must be complete OwnerOccurrenceId values, not site IDs
    with pytest.raises(TypeError):
        DeclaredModelStageResolution("ambiguous", root,
                                     rival_occurrences=(site, site))
    # rival owners must be authoritative OwnerRival records, not site IDs
    with pytest.raises(TypeError):
        DeclaredModelStageResolution("ambiguous", root,
                                     rival_owners=(site, site))
    # a resolved declaration's proof trace must begin at the root
    other = _sym("Other")
    foreign = ModelStageDeclaration(other, "model", SourceSpan(other.source, 1), False,
                                    proof_trace=(other,), precedence_basis="root-direct")
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("resolved", root, attribute="model",
                                     declaration=foreign, occurrence=child_occ)


# --------------------------------------------------------------------------- #
# Declaration-side outcomes (conflict / dynamic / no field / no declaration)
# --------------------------------------------------------------------------- #

def test_base_order_follows_python_precedence(tmp_path):
    """Two directly-declaring bases: the FIRST base wins (exact Python MRO), and
    reversing the ACTUAL base order flips the result accordingly."""
    body = """
        class A:
            base_model_prefix = "model"
        class B:
            base_model_prefix = "transformer"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper({order}):
            def __init__(self, config):
                self.model = BaseModel(config)
                self.transformer = BaseModel(config)
    """
    idx, res = _resolve(tmp_path, body.format(order="A, B"))
    assert res.status == "resolved" and res.attribute == "model"       # A first
    assert res.declaration.declaring_class.qualified_name == "A"
    idx2, res2 = _resolve(tmp_path, body.format(order="B, A"))
    assert res2.status == "resolved" and res2.attribute == "transformer"  # B first
    assert res2.declaration.declaring_class.qualified_name == "B"


def test_direct_dynamic_declaration_is_failed(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = SOME_CONSTANT
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "dynamic_declaration"


def test_declaration_to_no_constructed_field_is_absent(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.lm_head = Linear(config.h, config.v)
    """)
    assert res.status == "absent" and res.attribute == "model"


def test_no_declaration_no_bases_is_absent(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "absent"


def test_declared_but_no_field_never_becomes_root(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.embed = Embedding(config.v, config.h)
                self.layers = ModuleList([Block(config) for i in range(config.n)])
    """)
    assert res.status == "absent" and not res.self_stage


def test_explicit_empty_prefix_is_self_stage(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = ""
            def __init__(self, config):
                self.embed = Embedding(config.v, config.h)
    """)
    assert res.status == "resolved" and res.self_stage and res.attribute == ""
    assert res.occurrence == OwnerOccurrenceId(
        next(c.symbol for c in idx.classes if c.symbol.qualified_name == "Wrapper"))
    _assert_round_trip(idx, "Wrapper", res)   # helper asserts node is graph.root for self-stage


def test_same_field_rival_sites_carry_owner_rivals(tmp_path):
    """Two sites write the same field: ambiguous carrying the authoritative
    OwnerRival records, with NO fabricated OwnerOccurrenceId."""
    idx, res = _resolve(tmp_path, """
        class BaseA:
            def __init__(self, config): pass
        class BaseB:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                if config.flag:
                    self.model = BaseA(config)
                else:
                    self.model = BaseB(config)
    """)
    assert res.status == "ambiguous"
    assert res.rival_occurrences == ()                        # NO fabricated occurrence
    assert len(res.rival_owners) == 2
    graph = _graph(idx, "Wrapper")
    assert all(isinstance(r, OwnerRival) and r.parent == graph.root.occurrence
               for r in res.rival_owners)


def test_unsupported_construction_of_declared_field_is_failed(tmp_path):
    idx, res = _resolve(tmp_path, """
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = make_model_cls(config.kind)(config)
    """)
    assert res.status == "failed" and res.failure_kind == "unsupported_construction"


def test_factory_construction_resolves(tmp_path):
    idx, res = _resolve(tmp_path, """
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = BaseModel._from_config(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


def test_valid_root_with_broken_component_file_cannot_bypass_d0(tmp_path):
    """A fully valid Wrapper plus a broken sibling file in the SAME component:
    D0's hidden-rival law returns failed (the broken file could hide another exact
    root class), so B1 inherits that verdict and refuses — it cannot bypass D0."""
    good = _write(tmp_path, "wrap.py",
                  "class BaseModel:\n" + CFG +
                  "class Wrapper:\n    base_model_prefix = 'model'\n"
                  "    def __init__(self, config):\n        self.model = BaseModel(config)\n")
    broken = _write(tmp_path, "broken.py", "class Other(:\n")
    files = {"root": (good, broken)}
    idx = _index(tmp_path, files)
    cr = _cr(idx, files)
    assert cr.status == "failed"                        # D0 hidden-rival / parse-failure law
    with pytest.raises(ValueError):
        resolve_declared_model_stage(idx, cr)           # B1 cannot bypass D0


# --------------------------------------------------------------------------- #
# Lazy EXACT precedence — the Codex V2 poison set
# --------------------------------------------------------------------------- #

def test_declared_first_base_then_unindexed_mixin_resolves(tmp_path):
    """Wrapper(DeclaredBase, UnindexedMixin): the first exactly-bound base declares
    directly -> decisive over the later unindexed mixin (C3 position 1).  This is
    the real transformers shape (<Model>PreTrainedModel, GenerationMixin)."""
    idx, res = _resolve(tmp_path, """
        class DeclaredBase:
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(DeclaredBase, UnindexedMixin):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    assert res.declaration.precedence_basis == "first-base-direct"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


def test_unindexed_mixin_first_then_declared_base_is_mro_incomplete(tmp_path):
    """Wrapper(UnindexedMixin, DeclaredBase): the unresolved FIRST base can affect
    precedence and must never be skipped to reach the later declaration."""
    idx, res = _resolve(tmp_path, """
        class DeclaredBase:
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(UnindexedMixin, DeclaredBase):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "mro_incomplete"


def test_declared_base_with_unindexed_grandbase_resolves(tmp_path):
    """A directly declaring class is decisive BEFORE its ancestors are inspected,
    so an unindexed grand-base beyond it does not block the lookup."""
    idx, res = _resolve(tmp_path, """
        class DeclaredBase(UnindexedGrand):
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(DeclaredBase):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "resolved" and res.attribute == "model"
    assert res.declaration.precedence_basis == "first-base-direct"
    assert _assert_round_trip(idx, "Wrapper", res).symbol.qualified_name == "BaseModel"


def test_first_base_no_decl_incomplete_ancestry_second_declaring_is_mro_incomplete(tmp_path):
    """First base does not declare directly and its ancestry is incomplete; a
    second base declares.  The first base's linearization could precede and
    declare, so precedence is unprovable -> mro_incomplete (never jump to base 2)."""
    idx, res = _resolve(tmp_path, """
        class FirstBase(UnindexedGrand):
            pass
        class SecondBase:
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(FirstBase, SecondBase):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "mro_incomplete"


def test_decisive_dynamic_declaration_before_later_literal_fails(tmp_path):
    """The decisive (first-base) declaration is dynamic; a later base declares a
    literal.  The decisive dynamic fails -> we do NOT skip to the later literal."""
    idx, res = _resolve(tmp_path, """
        class DynBase:
            base_model_prefix = SOME_DYNAMIC
        class LitBase:
            base_model_prefix = "model"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(DynBase, LitBase):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "dynamic_declaration"


def test_fully_indexed_diamond_uses_exact_c3(tmp_path):
    """Diamond D(B, C), B(A), C(A): C3 MRO is [D, B, C, A].  Only C and the shared
    ancestor A declare; C precedes A in C3, so C wins.  A naive DFS (D->B->A)
    would wrongly pick A — this proves exact C3, not DFS."""
    idx, res = _resolve(tmp_path, """
        class A:
            base_model_prefix = "from_a"
        class B(A):
            pass
        class C(A):
            base_model_prefix = "from_c"
        class BaseC:
            def __init__(self, config): pass
        class Wrapper(B, C):
            def __init__(self, config):
                self.from_c = BaseC(config)
    """)
    assert res.status == "resolved" and res.attribute == "from_c"
    assert res.declaration.declaring_class.qualified_name == "C"
    assert res.declaration.precedence_basis == "c3"


def test_reversing_file_order_changes_nothing(tmp_path):
    """Exact binding is by declaring SOURCE, so the order files are indexed in
    never changes the resolution.  (Two same-named PreTrained in two files; the
    local one wins in both orders.)"""
    f1 = _write(tmp_path, "a.py",
                "class PreTrained:\n    base_model_prefix = \"model\"\n"
                "class BaseModel:\n" + CFG +
                "class Wrapper(PreTrained):\n    def __init__(self, config):\n        self.model = BaseModel(config)\n")
    f2 = _write(tmp_path, "b.py", "class PreTrained:\n    base_model_prefix = \"other\"\n")
    outcomes = set()
    for order in ((f1, f2), (f2, f1)):
        files = {"root": order}
        idx = _index(tmp_path, files)
        res = resolve_declared_model_stage(idx, _cr(idx, files))
        outcomes.add((res.status, res.attribute))
    assert outcomes == {("resolved", "model")}


def test_resolved_declaration_carries_a_proof_trace(tmp_path):
    """Every resolved declaration proves itself: an exact class chain root ->
    declaring_class, a basis, and the declaration span."""
    idx, res = _resolve(tmp_path, """
        class PreTrained:
            base_model_prefix = "transformer"
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(PreTrained):
            def __init__(self, config):
                self.transformer = BaseModel(config)
    """)
    assert res.status == "resolved"
    d = res.declaration
    assert d.precedence_basis == "first-base-direct"
    assert d.proof_trace[0].qualified_name == "Wrapper"
    assert d.proof_trace[-1] == d.declaring_class == d.proof_trace[-1]
    assert d.proof_trace[-1].qualified_name == "PreTrained"
    assert d.span is not None


def test_proof_trace_must_end_at_declaring_class():
    root = _sym("Wrapper")
    other = _sym("Other")
    span = SourceSpan(other.source, 1)
    with pytest.raises(ValueError):
        ModelStageDeclaration(root, "model", span, True, proof_trace=(other,),
                              precedence_basis="c3")


# --------------------------------------------------------------------------- #
# V3 — exhaustive unresolved-kind rule (a matching unresolved/conflict is never
# absent; no fabricated occurrences; unknown kinds default to typed failure)
# --------------------------------------------------------------------------- #

def test_duplicate_import_is_never_absent(tmp_path):
    """The duplicate-import example: `SharedModel` bound by two imports ->
    ambiguous_import unresolved with no preserved rivals -> typed failure, NEVER
    absent, NEVER a fabricated occurrence."""
    ba = _write(tmp_path, "ba.py", "class SharedModel:\n    def __init__(self, config): pass\n")
    bb = _write(tmp_path, "bb.py", "class SharedModel:\n    def __init__(self, config): pass\n")
    w = _write(tmp_path, "w.py",
               "from ba import SharedModel\n"
               "from bb import SharedModel\n"
               "class Wrapper:\n"
               "    base_model_prefix = 'model'\n"
               "    def __init__(self, config):\n"
               "        self.model = SharedModel(config)\n")
    files = {"root": (ba, bb, w)}
    idx = _index(tmp_path, files)
    res = resolve_declared_model_stage(idx, _cr(idx, files))
    assert res.status != "absent"
    assert res.status == "failed" and res.failure_kind == "unresolved_construction"
    assert res.rival_occurrences == () and res.rival_owners == ()


def test_rival_helper_returns_emit_no_fabricated_occurrence(tmp_path):
    """A helper with two return constructions preserves OwnerRivals at the root
    occurrence, but they are not attributable to an exact root field occurrence ->
    typed failure with NO fabricated occurrence, never absent."""
    idx, res = _resolve(tmp_path, """
        class BaseA:
            def __init__(self, config): pass
        class BaseB:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                self.model = self._build(config)
            def _build(self, config):
                if config.flag:
                    return BaseA(config)
                return BaseB(config)
    """)
    assert res.status != "absent"
    assert res.rival_occurrences == ()                  # NO fabricated occurrence
    assert res.status == "failed" and res.failure_kind == "unresolved_construction"


def test_rival_factory_sites_emit_no_fabricated_occurrence(tmp_path):
    """Two factory constructions of one field: ambiguous or typed failure, but
    never a fabricated OwnerOccurrenceId."""
    idx, res = _resolve(tmp_path, """
        class BaseA:
            def __init__(self, config): pass
        class BaseB:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                if config.flag:
                    self.model = BaseA._from_config(config)
                else:
                    self.model = BaseB._from_config(config)
    """)
    assert res.status in {"ambiguous", "failed"} and res.status != "absent"
    assert res.rival_occurrences == ()                  # NO fabricated occurrence
    if res.status == "ambiguous":
        assert all(isinstance(r, OwnerRival) for r in res.rival_owners)


def test_two_resolved_children_carry_only_graph_occurrences(tmp_path):
    """When a field genuinely has >=2 RESOLVED children, the rivals are real graph
    occurrences (each round-trips via node_for) — the only case that carries
    OwnerOccurrenceIds."""
    idx, res = _resolve(tmp_path, """
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "layers"
            def __init__(self, config):
                self.layers = [BaseModel(config), BaseModel(config)]
    """)
    # A list of two elements is not a single field occurrence; whatever the graph
    # decides, no rival occurrence may be fabricated outside the graph.
    graph = _graph(idx, "Wrapper")
    for occ in res.rival_occurrences:
        assert graph.node_for(occ) is not None


# --------------------------------------------------------------------------- #
# V3 — invalid fully-indexed hierarchy fails (C3 validated before the shortcut)
# --------------------------------------------------------------------------- #

def test_invalid_fully_indexed_hierarchy_fails_not_resolves(tmp_path):
    """Wrapper(B1, B2) with B2(B1): Python itself rejects this MRO.  B1 declares
    directly and is the first base, but because the closure is FULLY indexed the
    resolver must validate C3 first and FAIL the invalid hierarchy rather than
    resolve via the first-base shortcut."""
    idx, res = _resolve(tmp_path, """
        class B1:
            base_model_prefix = "model"
        class B2(B1):
            pass
        class BaseModel:
            def __init__(self, config): pass
        class Wrapper(B1, B2):
            def __init__(self, config):
                self.model = BaseModel(config)
    """)
    assert res.status == "failed" and res.failure_kind == "mro_incomplete"


# --------------------------------------------------------------------------- #
# V3 — ModelStageDeclaration provenance-closure forgeries
# --------------------------------------------------------------------------- #

def _sym_in(path, name):
    return SymbolId(SourceId(path, "fp-" + path, component_key="root"), name)


def test_declaration_span_must_live_in_declaring_source():
    decl_cls = _sym_in("/decl.py", "PreTrained")
    root = _sym_in("/decl.py", "Wrapper")
    foreign_span = SourceSpan(SourceId("/elsewhere.py", "fp2", component_key="root"), 1)
    with pytest.raises(ValueError):
        ModelStageDeclaration(decl_cls, "model", foreign_span, True,
                              proof_trace=(root, decl_cls), precedence_basis="first-base-direct")


def test_unknown_precedence_basis_is_rejected():
    root = _sym()
    with pytest.raises(ValueError):
        ModelStageDeclaration(root, "model", SourceSpan(root.source, 1), False,
                              proof_trace=(root,), precedence_basis="dfs")


def test_empty_proof_trace_is_rejected():
    root = _sym()
    with pytest.raises(ValueError):
        ModelStageDeclaration(root, "model", SourceSpan(root.source, 1), False,
                              proof_trace=(), precedence_basis="root-direct")


def test_root_direct_requires_single_uninherited_trace():
    root = _sym("Wrapper")
    base = _sym("Base")
    # a 2-element trace ending at the declaring root is not root-direct
    with pytest.raises(ValueError):
        ModelStageDeclaration(root, "model", SourceSpan(root.source, 1), False,
                              proof_trace=(base, root), precedence_basis="root-direct")
    # root-direct is never inherited
    with pytest.raises(ValueError):
        ModelStageDeclaration(root, "model", SourceSpan(root.source, 1), True,
                              proof_trace=(root,), precedence_basis="root-direct")


def test_inherited_basis_requires_inherited_flag():
    root = _sym("Wrapper")
    base = _sym("Base")
    with pytest.raises(ValueError):
        ModelStageDeclaration(base, "model", SourceSpan(base.source, 1), False,
                              proof_trace=(root, base), precedence_basis="first-base-direct")


# --------------------------------------------------------------------------- #
# V4 — cross-root / cross-field forgeries rejected on EVERY rival channel
# --------------------------------------------------------------------------- #

def _decl(root, basis="root-direct", inherited=False, trace=None):
    trace = trace if trace is not None else (root,)
    return ModelStageDeclaration(trace[-1], "model", SourceSpan(trace[-1].source, 1),
                                 inherited, proof_trace=trace, precedence_basis=basis)


def _site(sym):
    return ConstructionSiteId(sym, SymbolId(sym.source, f"{sym.qualified_name}.__init__"),
                              SourceSpan(sym.source, 1), 0)


def test_cross_root_rival_declaration_is_rejected():
    root, other = _sym("Wrapper"), _sym("Other")
    foreign = _decl(other)                       # proof_trace begins at Other, not root
    native = _decl(root)
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("ambiguous", root, rival_declarations=(foreign, native))


def test_cross_root_and_empty_rival_occurrence_is_rejected():
    root, other = _sym("Wrapper"), _sym("Other")
    native = OwnerOccurrenceId(root).child(_site(root))
    foreign = OwnerOccurrenceId(other).child(_site(other))
    decl = _decl(root)
    with pytest.raises(ValueError):          # cross-root occurrence
        DeclaredModelStageResolution("ambiguous", root, attribute="model", declaration=decl,
                                     rival_occurrences=(foreign, native))
    with pytest.raises(ValueError):          # empty-site occurrence
        DeclaredModelStageResolution("ambiguous", root, attribute="model", declaration=decl,
                                     rival_occurrences=(OwnerOccurrenceId(root), native))


def test_cross_root_rival_owner_is_rejected():
    root, other = _sym("Wrapper"), _sym("Other")
    native = OwnerRival(OwnerOccurrenceId(root), _site(root), None, "A", "direct_name")
    foreign = OwnerRival(OwnerOccurrenceId(other), _site(other), None, "B", "direct_name")
    decl = _decl(root)
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("ambiguous", root, attribute="model", declaration=decl,
                                     rival_owners=(foreign, native))


def test_occurrence_side_ambiguity_requires_applicable_declaration():
    root = _sym("Wrapper")
    r1 = OwnerRival(OwnerOccurrenceId(root), _site(root), None, "A", "direct_name")
    r2 = OwnerRival(OwnerOccurrenceId(root), _site(root), None, "B", "direct_name")
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("ambiguous", root, rival_owners=(r1, r2))   # no declaration


def test_self_stage_is_legal_only_for_resolved():
    root = _sym("Wrapper")
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("absent", root, self_stage=True)
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("failed", root, failure_kind="x", self_stage=True)


def test_failure_detail_requires_failure_kind():
    root = _sym("Wrapper")
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("absent", root, failure_detail="oops")


def test_failed_carries_no_rivals_or_occurrence():
    root = _sym("Wrapper")
    r = OwnerRival(OwnerOccurrenceId(root), _site(root), None, "A", "direct_name")
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("failed", root, failure_kind="x", rival_owners=(r, r))
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("failed", root, failure_kind="x",
                                     occurrence=OwnerOccurrenceId(root).child(_site(root)))


def test_resolver_carries_only_the_declared_fields_rivals(tmp_path):
    """Cross-FIELD isolation: two rival fields exist; resolving 'model' carries
    ONLY 'model' rivals — never 'encoder' rivals leaked from graph.conflicts."""
    idx, res = _resolve(tmp_path, """
        class BaseA:
            def __init__(self, config): pass
        class BaseB:
            def __init__(self, config): pass
        class Wrapper:
            base_model_prefix = "model"
            def __init__(self, config):
                if config.f:
                    self.model = BaseA(config)
                else:
                    self.model = BaseB(config)
                if config.g:
                    self.encoder = BaseA(config)
                else:
                    self.encoder = BaseB(config)
    """)
    assert res.status == "ambiguous" and len(res.rival_owners) == 2   # not 4
    graph = _graph(idx, "Wrapper")
    model_sites = {s.site_id for s in idx.construction_sites_of(graph.root.symbol)
                   if s.target == "model" and s.target_kind == "field"}
    assert all(r.site in model_sites for r in res.rival_owners)       # only 'model'


def test_framework_protocol_requires_nonempty_declaration_attr():
    with pytest.raises(ValueError):
        FrameworkAddressProtocol("")


# --------------------------------------------------------------------------- #
# Closed protocol registry + resolution shape laws
# --------------------------------------------------------------------------- #

def test_framework_protocol_registry_is_a_closed_code_type():
    from model_unfolder.evidence import component_owner as co
    assert all(isinstance(p, FrameworkAddressProtocol)
               for p in co._FRAMEWORK_ADDRESS_PROTOCOLS)
    assert any(p.declaration_attr == "base_model_prefix"
               for p in co._FRAMEWORK_ADDRESS_PROTOCOLS)


def test_status_vocabulary_is_closed():
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("maybe", _sym())


def test_resolved_requires_occurrence_and_declaration():
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("resolved", _sym())


def test_ambiguous_requires_two_rivals():
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("ambiguous", _sym())


def test_failed_requires_failure_kind():
    with pytest.raises(ValueError):
        DeclaredModelStageResolution("failed", _sym())
