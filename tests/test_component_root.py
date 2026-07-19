"""U3-D0 — the component-root address boundary (resolve_component_root).

Bridges a SourceBundle component address (component key + declared architecture)
to an exact root OwnerOccurrenceId using EXACT-identity addressing only: no
model_type, suffix, substring, shortest-name, field-count, file-order,
import-order or role-marker selection.

The poisons pin, per the V1 re-vet:
  * exact addressing, rival preservation, honest absence (poisons 1-8);
  * the HIDDEN-RIVAL LAW — any parse/read failure in a component makes
    uniqueness unprovable, so it fails before counting visible candidates, and
    failures are isolated per component;
  * ComponentRootCandidate is self-verifying (component/spelling/span-source
    forgery cannot be built; span is required);
  * ComponentRootResolution is closed (foreign failures/candidates,
    occurrence/graph and root component/spelling forgery cannot be built);
  * canonical, file-order-independent rival ordering;
  * a resolved status is an ADDRESS claim only — an unresolved constructor/config
    binding stays explicit inside graph.root.unresolved.
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    ComponentRootCandidate,
    ComponentRootResolution,
    OwnerGraph,
    OwnerNode,
    OwnerOccurrenceId,
    resolve_component_root,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ConstructionSiteId,
    SourceId,
    SourceSpan,
    SymbolId,
)


def _root_node(occurrence, symbol):
    """A minimal root OwnerNode for forging inconsistent graphs in poisons."""
    return OwnerNode(occurrence=occurrence, symbol=symbol, config_bindings=(),
                     config_prefix_candidates=((),), via_site=None,
                     via_field="", via_kind="root")


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(component_files, architectures, *, architecture=None):
    flat = []
    for group in component_files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(
        source="local", files=tuple(flat),
        component_files={k: tuple(v) for k, v in component_files.items()},
        component_architectures=dict(architectures),
        architecture=architecture)


def _idx(bundle):
    return pi.build_program_index(bundle)


CFG = "    def __init__(self, config): pass\n"


def _resolved(tmp_path, *, component="root", name="C"):
    f = _write(tmp_path, "m.py", f"class {name}:\n{CFG}")
    bundle = _bundle({component: (f,)}, {component: name})
    idx = _idx(bundle)
    return idx, bundle, resolve_component_root(idx, bundle, component)


def _class_span(idx, symbol):
    return next(r.span for r in idx.classes if r.symbol == symbol)


def _root_sym(name="C", path="/m.py", fp="fp", component="root"):
    return SymbolId(SourceId(path, fp, component_key=component), name)


# --------------------------------------------------------------------------- #
# Poison 1 — same class spelling in two files of one component -> ambiguous
# --------------------------------------------------------------------------- #

def test_same_class_two_files_one_component_is_ambiguous(tmp_path):
    a = _write(tmp_path, "modeling_a.py", f"class Denoiser:\n{CFG}")
    b = _write(tmp_path, "modeling_b.py", f"class Denoiser:\n{CFG}")
    bundle = _bundle({"root": (a, b)}, {"root": "Denoiser"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    assert res.status == "ambiguous"
    assert len(res.candidates) == 2
    assert {c.symbol.source.canonical_path for c in res.candidates} == {
        pi._canonical_path(a), pi._canonical_path(b)}
    assert res.occurrence is None and res.graph is None


# --------------------------------------------------------------------------- #
# Poison 2 — same class under root and vision -> root selects root only
# --------------------------------------------------------------------------- #

def test_same_class_root_and_vision_root_selects_root(tmp_path):
    root_f = _write(tmp_path, "modeling_root.py", f"class Tower:\n{CFG}")
    vis_f = _write(tmp_path, "modeling_vis.py", f"class Tower:\n{CFG}")
    bundle = _bundle({"root": (root_f,), "vision": (vis_f,)},
                     {"root": "Tower", "vision": "Tower"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    assert res.status == "resolved"
    assert res.occurrence.root.source.component_key == "root"
    assert res.occurrence.root.source.canonical_path == pi._canonical_path(root_f)


# --------------------------------------------------------------------------- #
# Poison 3 — exact class missing but similar suffix present -> absent
# --------------------------------------------------------------------------- #

def test_exact_missing_similar_suffix_is_absent(tmp_path):
    f = _write(tmp_path, "modeling.py", f"class DenoiserModel:\n{CFG}")
    bundle = _bundle({"root": (f,)}, {"root": "Denoiser"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    assert res.status == "absent"
    assert res.candidates == () and res.occurrence is None


# --------------------------------------------------------------------------- #
# Poison 4 — a component file fails parsing, another looks similar -> failed
# --------------------------------------------------------------------------- #

def test_parse_failure_with_similar_class_is_failed_not_selected(tmp_path):
    bad = _write(tmp_path, "modeling_bad.py", "class Denoiser(:\n    def __init__(self):\n")
    good = _write(tmp_path, "modeling_good.py", f"class DenoiserBlock:\n{CFG}")
    bundle = _bundle({"root": (bad, good)}, {"root": "Denoiser"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    assert res.status == "failed"
    assert res.parse_failures
    assert all(pf.source.component_key == "root" for pf in res.parse_failures)
    assert res.occurrence is None and res.graph is None


# --------------------------------------------------------------------------- #
# Poison 5 — bundle.architecture cannot address a non-root component
# --------------------------------------------------------------------------- #

def test_bundle_architecture_cannot_address_non_root(tmp_path):
    vis_f = _write(tmp_path, "modeling_vis.py", f"class Tower:\n{CFG}")
    bundle = _bundle({"vision": (vis_f,)}, {}, architecture="Tower")
    res = resolve_component_root(_idx(bundle), bundle, "vision")
    assert res.status == "absent"                       # root-only compat address
    root_f = _write(tmp_path, "modeling_root.py", f"class Tower:\n{CFG}")
    root_bundle = _bundle({"root": (root_f,)}, {}, architecture="Tower")
    assert resolve_component_root(_idx(root_bundle), root_bundle, "root").status == "resolved"


# --------------------------------------------------------------------------- #
# Poison 6 — same source indexed for two components -> two addresses
# --------------------------------------------------------------------------- #

def test_same_source_two_components_two_addresses(tmp_path):
    f = _write(tmp_path, "modeling_shared.py", f"class Encoder:\n{CFG}")
    bundle = _bundle({"root": (f,), "text_encoder": (f,)},
                     {"root": "Encoder", "text_encoder": "Encoder"})
    idx = _idx(bundle)
    root_res = resolve_component_root(idx, bundle, "root")
    te_res = resolve_component_root(idx, bundle, "text_encoder")
    assert root_res.status == "resolved" and te_res.status == "resolved"
    assert root_res.occurrence.root.source.component_key == "root"
    assert te_res.occurrence.root.source.component_key == "text_encoder"
    assert root_res.occurrence.root != te_res.occurrence.root


# --------------------------------------------------------------------------- #
# Poison 7 — resolved root preserves supplied constructor-prefix bindings
# --------------------------------------------------------------------------- #

def test_resolved_root_preserves_supplied_prefix_bindings(tmp_path):
    f = _write(tmp_path, "modeling.py", "class Wrapper:\n    def __init__(self, config, extra): pass\n")
    bundle = _bundle({"root": (f,)}, {"root": "Wrapper"})
    res = resolve_component_root(_idx(bundle), bundle, "root",
                                 root_param_prefixes={"config": ("text_config",)})
    assert res.status == "resolved"
    bindings = {b.parameter: b for b in res.graph.root.config_bindings}
    assert bindings["config"].prefixes == (("text_config",),)
    assert bindings["config"].origin == "root_argument"


# --------------------------------------------------------------------------- #
# Poison 8 — empty architecture stays absent, never "pick the only class"
# --------------------------------------------------------------------------- #

def test_empty_architecture_is_absent_never_picks_only_class(tmp_path):
    f = _write(tmp_path, "modeling.py", f"class OnlyClass:\n{CFG}")
    no_arch = _bundle({"root": (f,)}, {})
    res = resolve_component_root(_idx(no_arch), no_arch, "root")
    assert res.status == "absent" and res.candidates == ()
    empty_arch = _bundle({"root": (f,)}, {"root": ""})
    assert resolve_component_root(_idx(empty_arch), empty_arch, "root").status == "absent"


# --------------------------------------------------------------------------- #
# V1 correction 1 — hidden-rival parse-failure law
# --------------------------------------------------------------------------- #

def test_healthy_exact_plus_broken_sibling_is_failed_not_resolved(tmp_path):
    good = _write(tmp_path, "good.py", f"class Denoiser:\n{CFG}")
    broken = _write(tmp_path, "broken.py", "class Other(:\n")
    bundle = _bundle({"root": (good, broken)}, {"root": "Denoiser"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    # a broken sibling could redefine Denoiser -> uniqueness unproven -> failed
    assert res.status == "failed"
    assert res.occurrence is None and res.candidates == ()
    assert any(pf.source.component_key == "root" for pf in res.parse_failures)


def test_two_candidates_plus_broken_sibling_is_failed_all_failures(tmp_path):
    a = _write(tmp_path, "a.py", f"class Denoiser:\n{CFG}")
    b = _write(tmp_path, "b.py", f"class Denoiser:\n{CFG}")
    broken = _write(tmp_path, "broken.py", "class X(:\n")
    bundle = _bundle({"root": (a, b, broken)}, {"root": "Denoiser"})
    res = resolve_component_root(_idx(bundle), bundle, "root")
    assert res.status == "failed"
    assert res.candidates == ()                          # failed carries failures only
    assert res.parse_failures
    assert all(pf.source.component_key == "root" for pf in res.parse_failures)


def test_component_failures_are_isolated(tmp_path):
    root_good = _write(tmp_path, "rg.py", f"class Root:\n{CFG}")
    vision_bad = _write(tmp_path, "vb.py", "class V(:\n")
    bundle = _bundle({"root": (root_good,), "vision": (vision_bad,)},
                     {"root": "Root", "vision": "V"})
    idx = _idx(bundle)
    assert resolve_component_root(idx, bundle, "root").status == "resolved"
    assert resolve_component_root(idx, bundle, "vision").status == "failed"
    # symmetric: a root failure must not block a healthy vision
    root_bad = _write(tmp_path, "rb.py", "class R(:\n")
    vision_good = _write(tmp_path, "vg.py", f"class Vis:\n{CFG}")
    bundle2 = _bundle({"root": (root_bad,), "vision": (vision_good,)},
                      {"root": "R", "vision": "Vis"})
    idx2 = _idx(bundle2)
    assert resolve_component_root(idx2, bundle2, "root").status == "failed"
    assert resolve_component_root(idx2, bundle2, "vision").status == "resolved"


# --------------------------------------------------------------------------- #
# V1 correction 2 — ComponentRootCandidate is self-verifying
# --------------------------------------------------------------------------- #

def test_candidate_rejects_mismatched_component():
    sym = _root_sym("C")
    with pytest.raises(ValueError):
        ComponentRootCandidate("vision", "C", sym, SourceSpan(sym.source, 1))


def test_candidate_rejects_mismatched_spelling():
    sym = _root_sym("C")
    with pytest.raises(ValueError):
        ComponentRootCandidate("root", "D", sym, SourceSpan(sym.source, 1))


def test_candidate_rejects_mismatched_span_source():
    sym = _root_sym("C")
    foreign = SourceSpan(SourceId("/other.py", "fp2", component_key="root"), 1)
    with pytest.raises(ValueError):
        ComponentRootCandidate("root", "C", sym, foreign)


def test_candidate_requires_exact_span_and_nonempty_fields():
    sym = _root_sym("C")
    with pytest.raises(TypeError):
        ComponentRootCandidate("root", "C", sym, None)
    with pytest.raises(ValueError):
        ComponentRootCandidate("", "C", sym, SourceSpan(sym.source, 1))
    with pytest.raises(ValueError):
        ComponentRootCandidate("root", "", sym, SourceSpan(sym.source, 1))


# --------------------------------------------------------------------------- #
# V1 correction 3 — ComponentRootResolution is closed
# --------------------------------------------------------------------------- #

def test_resolution_rejects_non_parse_failure_entry():
    with pytest.raises(TypeError):
        ComponentRootResolution("failed", "root", "C", parse_failures=("not a failure",))


def test_resolution_rejects_foreign_parse_failure(tmp_path):
    bad = _write(tmp_path, "vbad.py", "class Q(:\n")
    bundle = _bundle({"vision": (bad,)}, {"vision": "Q"})
    idx = _idx(bundle)
    vision_failures = tuple(pf for pf in idx.parse_failures
                            if pf.source.component_key == "vision")
    assert vision_failures
    with pytest.raises(ValueError):
        ComponentRootResolution("failed", "root", "C", parse_failures=vision_failures)


def test_ambiguous_rejects_candidate_from_other_component():
    root_sym = _root_sym("C", "/r.py", "fr", "root")
    root_cand = ComponentRootCandidate("root", "C", root_sym, SourceSpan(root_sym.source, 1))
    vis_sym = _root_sym("C", "/v.py", "fv", "vision")
    vis_cand = ComponentRootCandidate("vision", "C", vis_sym, SourceSpan(vis_sym.source, 1))
    with pytest.raises(ValueError):
        ComponentRootResolution("ambiguous", "root", "C", candidates=(root_cand, vis_cand))


def test_ambiguous_rejects_candidate_wrong_declaration():
    s1 = _root_sym("C", "/a.py", "f1", "root")
    c1 = ComponentRootCandidate("root", "C", s1, SourceSpan(s1.source, 1))
    s2 = _root_sym("D", "/b.py", "f2", "root")
    c2 = ComponentRootCandidate("root", "D", s2, SourceSpan(s2.source, 1))
    with pytest.raises(ValueError):
        ComponentRootResolution("ambiguous", "root", "C", candidates=(c1, c2))


def test_resolved_rejects_occurrence_graph_mismatch(tmp_path):
    _, _, res = _resolved(tmp_path)
    fake = OwnerOccurrenceId(_root_sym("Other", "/x.py", "fx", "root"))
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "root", "C", occurrence=fake, graph=res.graph)


def test_resolved_rejects_root_component_mismatch(tmp_path):
    _, _, res = _resolved(tmp_path)
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "vision", "C",
                                occurrence=res.graph.root.occurrence, graph=res.graph)


def test_resolved_rejects_root_spelling_mismatch(tmp_path):
    _, _, res = _resolved(tmp_path)
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "root", "D",
                                occurrence=res.graph.root.occurrence, graph=res.graph)


def test_non_absent_requires_declared_architecture(tmp_path):
    _, _, res = _resolved(tmp_path)
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "root", None,
                                occurrence=res.graph.root.occurrence, graph=res.graph)


# --------------------------------------------------------------------------- #
# V1 re-vet correction 1 — resolved-root identity fully closed
# --------------------------------------------------------------------------- #

def test_resolved_rejects_graph_root_symbol_occurrence_mismatch():
    # a manually inconsistent graph: root symbol C but occurrence root D
    sym_c = _root_sym("C", "/c.py", "fc", "root")
    sym_d = _root_sym("D", "/d.py", "fd", "root")
    occ = OwnerOccurrenceId(sym_d)
    graph = OwnerGraph(root=_root_node(occ, sym_c))
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "root", "C", occurrence=occ, graph=graph)


def test_resolved_rejects_component_root_carrying_a_child_site():
    sym = _root_sym("C", "/c.py", "fc", "root")
    site = ConstructionSiteId(sym, SymbolId(sym.source, "C.__init__"),
                              SourceSpan(sym.source, 5), 0)
    occ = OwnerOccurrenceId(sym, (site,))          # a root with a construction chain
    graph = OwnerGraph(root=_root_node(occ, sym))
    with pytest.raises(ValueError):
        ComponentRootResolution("resolved", "root", "C", occurrence=occ, graph=graph)


# --------------------------------------------------------------------------- #
# V1 correction 4 — determinism (canonical, file-order-independent rivals)
# --------------------------------------------------------------------------- #

def test_reversed_file_order_yields_identical_candidates(tmp_path):
    a = _write(tmp_path, "a.py", f"class Denoiser:\n{CFG}")
    b = _write(tmp_path, "b.py", f"class Denoiser:\n{CFG}")
    fwd = _bundle({"root": (a, b)}, {"root": "Denoiser"})
    rev = _bundle({"root": (b, a)}, {"root": "Denoiser"})
    r1 = resolve_component_root(_idx(fwd), fwd, "root")
    r2 = resolve_component_root(_idx(rev), rev, "root")
    assert r1.status == r2.status == "ambiguous"

    def key(c):
        return (c.symbol.source.canonical_path, c.symbol.qualified_name, c.span.line)

    assert [key(c) for c in r1.candidates] == [key(c) for c in r2.candidates]


def test_reversed_broken_file_order_yields_identical_failures(tmp_path):
    a = _write(tmp_path, "a.py", "class A(:\n")
    b = _write(tmp_path, "b.py", "class B(:\n")
    fwd = _bundle({"root": (a, b)}, {"root": "Denoiser"})
    rev = _bundle({"root": (b, a)}, {"root": "Denoiser"})
    r1 = resolve_component_root(_idx(fwd), fwd, "root")
    r2 = resolve_component_root(_idx(rev), rev, "root")
    assert r1.status == r2.status == "failed"
    assert len(r1.parse_failures) == 2

    def key(pf):
        return (pf.source.canonical_path, pf.kind, pf.detail)

    assert [key(pf) for pf in r1.parse_failures] == [key(pf) for pf in r2.parse_failures]


# --------------------------------------------------------------------------- #
# V1 correction 5 — address resolution vs graph completeness
# --------------------------------------------------------------------------- #

def test_multiparam_root_address_resolves_binding_unresolved(tmp_path):
    f = _write(tmp_path, "m.py", "class Wrapper:\n    def __init__(self, config, extra): pass\n")
    bundle = _bundle({"root": (f,)}, {"root": "Wrapper"})
    res = resolve_component_root(_idx(bundle), bundle, "root")  # no supplied prefixes
    assert res.status == "resolved" and res.address_resolved
    root = res.graph.root
    # the unresolved constructor/config binding is explicit inside the graph
    assert any(u.kind == "root_config_binding" for u in root.unresolved)
    # no config prefix is fabricated: no constructor parameter was bound to a
    # prefix (the root's own document prefix () is definitional, not fabricated).
    assert root.config_bindings == ()


def test_address_resolved_property_replaces_resolved(tmp_path):
    _, _, res = _resolved(tmp_path)
    assert res.address_resolved is True
    assert not hasattr(res, "resolved")   # the misleading name is gone


# --------------------------------------------------------------------------- #
# Resolution status/shape laws
# --------------------------------------------------------------------------- #

def test_resolution_status_vocabulary_is_closed():
    with pytest.raises(ValueError):
        ComponentRootResolution(status="maybe", component_key="root")


def test_resolved_requires_occurrence_and_graph():
    with pytest.raises(ValueError):
        ComponentRootResolution(status="resolved", component_key="root",
                                declared_architecture="C")


def test_ambiguous_requires_two_rivals():
    with pytest.raises(ValueError):
        ComponentRootResolution(status="ambiguous", component_key="root",
                                declared_architecture="C", candidates=())


def test_failed_requires_parse_failures():
    with pytest.raises(ValueError):
        ComponentRootResolution(status="failed", component_key="root",
                                declared_architecture="C")


def test_absent_carries_nothing_further(tmp_path):
    idx, bundle, res = _resolved(tmp_path)
    assert res.address_resolved and isinstance(res.occurrence, OwnerOccurrenceId)
    cand = ComponentRootCandidate("root", "C", res.graph.root.symbol,
                                  _class_span(idx, res.graph.root.symbol))
    with pytest.raises(ValueError):
        ComponentRootResolution(status="absent", component_key="root",
                                declared_architecture="C", candidates=(cand,))
