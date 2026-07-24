"""U3-A — the ONE raw ProgramIndex: schema + walker + immutable assembly.

These fixtures are the acceptance surface for the observation-only index
(docs/U3_RUNBOOK.md, master plan §20.6).  They prove the walker OBSERVES and
never RESOLVES: candidates are retained (0 / 1 / several), no winner is chosen,
no ConflictRecord is emitted by the walker, identity is qualified + content-
fingerprinted, and the aggregate bundle fingerprint canonically covers identity/
ownership/provenance/content (never raw file bytes or iteration order).

The required families (Soumil, 2026-07-19): the nine spec fixtures (alias
imports, helper methods, inherited methods, factory functions, comprehensions,
conditional construction, equivalent candidates, rival candidates, unsupported
dynamic dispatch) PLUS content-changed-mtime-preserved, same class at two sites/
roles, conflicting owner/config-prefix candidates retained as rivals, malformed+
healthy partial usability, two modules defining one class name, literal-None vs
no default, two constructions on one line, dynamic factory with no candidate,
multiple factory candidates without a winner, branch calls proving lexical order
is not runtime order, external node without provenance raising, and expression-
source renaming leaving structural observations equivalent.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _write(tmp_path, name: str, src: str) -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(files: dict) -> SourceBundle:
    """files: {component -> (path, ...)}; the flat list is the union."""
    flat: list = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(source="local", files=tuple(flat),
                        component_files={k: tuple(v) for k, v in files.items()})


def _index(tmp_path, name: str, src: str, component: str = "root") -> pi.ProgramIndex:
    path = _write(tmp_path, name, src)
    return pi.build_program_index(_bundle({component: (path,)}))


def _class(idx, qual):
    return next((c for c in idx.classes if c.symbol.qualified_name == qual), None)


def _callable(idx, qual):
    return next((c for c in idx.callables if c.symbol.qualified_name == qual), None)


def _sites_for(idx, field: str):
    out = [s for s in idx.construction_sites if s.target == field]
    for cont in idx.containers:
        out.extend(e for e in cont.elements if cont.field == field or field == "*")
    return out


def _identifiers(idx, qual):
    callable_record = _callable(idx, qual)
    assert callable_record is not None
    return idx.identifiers_in(callable_record.symbol)


def _shape(expr):
    """A structural signature of an ExprNode: kinds + operators + arity, with
    identifier names and the diagnostic source segment REMOVED — so a consistent
    rename leaves it invariant."""
    if expr is None:
        return None
    return (expr.kind, expr.operator,
            tuple(_shape(c) for c in expr.children),
            tuple((str(i), _shape(v)) for i, (_, v) in enumerate(expr.keyword_children)))


def _count_source_walks(monkeypatch):
    """Instrument the real walker without replacing any observation logic."""
    real_run = pi._SourceWalker.run
    count = [0]

    def counted_run(self):
        count[0] += 1
        return real_run(self)

    monkeypatch.setattr(pi._SourceWalker, "run", counted_run)
    return lambda: count[0]


# --------------------------------------------------------------------------- #
# Exact callable-scoped identifier observations (U3-A1)
# --------------------------------------------------------------------------- #

def test_identifier_census_covers_parameters_and_name_binding_positions(tmp_path):
    idx = _index(tmp_path, "modeling_identifiers.py", """
        @decorate(flag)
        def forward(self, num_frames: FrameCount = default_value):
            bare_name
            target = num_frames
            for loop_name in iterable:
                with manager as bound_name:
                    values = [item for comp_name in source]
            del target
            return target
    """)
    rows = {(item.name, item.context) for item in _identifiers(idx, "forward")}
    assert {
        ("self", "parameter"), ("num_frames", "parameter"),
        ("FrameCount", "annotation"), ("default_value", "default"),
        ("decorate", "decorator"), ("flag", "decorator"),
        ("bare_name", "load"), ("target", "store"),
        ("num_frames", "load"), ("loop_name", "store"),
        ("iterable", "load"), ("manager", "load"),
        ("bound_name", "store"), ("values", "store"),
        ("item", "load"), ("comp_name", "store"),
        ("source", "load"), ("target", "del"), ("target", "load"),
    } <= rows


def test_identifier_observation_carries_exact_parameter_span(tmp_path):
    idx = _index(tmp_path, "modeling_identifier_span.py", """
        class Root:
            def forward(self, num_frames, *, mask=None):
                return num_frames
    """)
    row = next(item for item in _identifiers(idx, "Root.forward")
               if item.name == "num_frames" and item.context == "parameter")
    assert row.span.source == row.enclosing_callable.source
    assert row.span.line == 3
    assert row.span.end_line == 3
    assert row.span.end_col > row.span.col


def test_nested_callable_identifiers_do_not_contaminate_parent(tmp_path):
    idx = _index(tmp_path, "modeling_nested_identifier.py", """
        class Root:
            def forward(self, hidden_states):
                def helper(num_frames):
                    return num_frames
                return hidden_states
    """)
    outer = {(item.name, item.context)
             for item in _identifiers(idx, "Root.forward")}
    nested = {(item.name, item.context)
              for item in _identifiers(idx, "Root.forward.helper")}
    assert not any(name == "num_frames" for name, _ in outer)
    assert ("num_frames", "parameter") in nested
    assert ("num_frames", "load") in nested
    outer_symbol = _callable(idx, "Root.forward").symbol
    assert any(item.enclosing_callable == outer_symbol
               and item.syntax_kind == "nested_callable"
               for item in idx.unsupported_syntax)


def test_lambda_identifiers_are_not_attributed_to_parent_callable(tmp_path):
    idx = _index(tmp_path, "modeling_lambda_identifier.py", """
        def forward(hidden_states):
            fn = lambda num_frames: num_frames
            return hidden_states
    """)
    rows = {(item.name, item.context) for item in _identifiers(idx, "forward")}
    assert not any(name == "num_frames" for name, _ in rows)
    assert ("fn", "store") in rows
    symbol = _callable(idx, "forward").symbol
    assert any(item.enclosing_callable == symbol
               and item.syntax_kind == "nested_lambda"
               for item in idx.unsupported_syntax)


def test_bare_unsupported_expression_is_callable_scoped(tmp_path):
    idx = _index(tmp_path, "modeling_bare_unsupported.py", """
        def forward(hidden_states):
            (ready := hidden_states)
            return hidden_states
    """)
    symbol = _callable(idx, "forward").symbol
    rows = [item for item in idx.unsupported_syntax
            if item.enclosing_callable == symbol]
    assert any(item.syntax_kind == "NamedExpr" for item in rows)
    assert all(item.span.source == symbol.source for item in rows)


def test_supported_fstring_does_not_create_false_unsupported_record(tmp_path):
    idx = _index(tmp_path, "modeling_fstring.py", """
        def forward(hidden_states):
            label = f"shape={hidden_states.shape}"
            return hidden_states
    """)
    symbol = _callable(idx, "forward").symbol
    assert not [item for item in idx.unsupported_syntax
                if item.enclosing_callable == symbol]


def test_identifier_query_is_component_and_source_qualified(tmp_path):
    shared = _write(tmp_path, "modeling_shared_identifier.py", """
        class Root:
            def forward(self, num_frames):
                return num_frames
    """)
    idx = pi.build_program_index(_bundle({"root": (shared,), "vision": (shared,)}))
    roots = [item for item in idx.callables
             if item.symbol.qualified_name == "Root.forward"]
    assert len(roots) == 2
    root_rows = idx.identifiers_in(next(
        item.symbol for item in roots if item.symbol.source.component_key == "root"))
    vision_rows = idx.identifiers_in(next(
        item.symbol for item in roots if item.symbol.source.component_key == "vision"))
    assert root_rows and vision_rows
    assert {item.enclosing_callable.source.component_key for item in root_rows} == {"root"}
    assert {item.enclosing_callable.source.component_key for item in vision_rows} == {"vision"}


def test_identifier_observation_contract_rejects_forgery(tmp_path):
    idx = _index(tmp_path, "modeling_identifier_contract.py", """
        def forward(x):
            return x
    """)
    row = _identifiers(idx, "forward")[0]
    with pytest.raises(ValueError):
        pi.IdentifierObservation(row.owner, row.enclosing_callable, row.name,
                                 "mechanism", row.span)
    foreign = pi.SourceId("/foreign.py", "f", component_key="root")
    with pytest.raises(ValueError):
        pi.IdentifierObservation(row.owner, row.enclosing_callable, row.name,
                                 row.context, pi.SourceSpan(foreign, 1))
    with pytest.raises(TypeError):
        idx.identifiers_in("forward")


# --------------------------------------------------------------------------- #
# Spec family 1 — alias imports
# --------------------------------------------------------------------------- #

def test_alias_import_construction_records_alias_provenance(tmp_path):
    idx = _index(tmp_path, "modeling_alias.py", """
        from pkg.attn import FlashAttention as FA
        class Blk:
            def __init__(self, config):
                self.attn = FA(config)
    """)
    aliases = {i.alias: i.target for i in idx.imports}
    assert aliases["FA"] == "pkg.attn.FlashAttention"
    site = next(s for s in idx.construction_sites if s.target == "attn")
    (cand,) = site.candidates
    assert cand.provenance == "alias_import:FA"
    assert cand.symbol is None  # external — resolution is U3-B's job
    assert cand.reference.kind == "name" and cand.reference.name == "FA"


def test_module_conditional_import_is_observed_with_its_exact_guard(tmp_path):
    idx = _index(tmp_path, "modeling_guarded_import.py", """
        if capability_available():
            from pkg.fast import fast_kernel

        class Block:
            def forward(self, x):
                return fast_kernel(x)
    """)
    record = next(item for item in idx.imports
                  if item.alias == "fast_kernel")
    assert record.target == "pkg.fast.fast_kernel"
    assert len(record.guard) == 1
    assert record.guard[0].kind == "if"
    assert record.guard[0].test.source_segment == "capability_available()"
    assert record.guard[0].span.source == record.source


# --------------------------------------------------------------------------- #
# Spec family 2 — helper methods (self-call fold edges)
# --------------------------------------------------------------------------- #

def test_helper_method_call_is_recorded_as_a_fold_edge(tmp_path):
    idx = _index(tmp_path, "modeling_helper.py", """
        class MLP:
            pass
        class Blk:
            def __init__(self, config):
                self.mlp = self._build_mlp(config)
            def _build_mlp(self, config):
                return MLP(config)
    """)
    init = _callable(idx, "Blk.__init__")
    assert "_build_mlp" in init.self_method_calls
    # the helper's OWN construction is observed under the helper's owner+callable
    helper = _callable(idx, "Blk._build_mlp")
    assert helper is not None
    built = [s for s in idx.construction_sites
             if s.enclosing_callable.qualified_name == "Blk._build_mlp"]
    assert any(c.symbol and c.symbol.qualified_name == "MLP"
               for s in built for c in s.candidates)


# --------------------------------------------------------------------------- #
# Spec family 3 — inherited methods (per-class records; resolution is U3-B)
# --------------------------------------------------------------------------- #

def test_inherited_methods_are_recorded_per_defining_class(tmp_path):
    idx = _index(tmp_path, "modeling_inherit.py", """
        class Base:
            def forward(self, x):
                return self.norm(x)
        class Child(Base):
            def __init__(self, config):
                self.norm = LayerNorm(config.dim)
    """)
    child = _class(idx, "Child")
    assert [b.name for b in child.bases] == ["Base"]
    # forward is owned by Base; __init__ by Child — the index does not flatten
    # the MRO (that is the resolver's job).
    assert _callable(idx, "Base.forward").owner.qualified_name == "Base"
    assert _callable(idx, "Child.__init__").owner.qualified_name == "Child"
    assert _callable(idx, "Child.forward") is None


# --------------------------------------------------------------------------- #
# Spec family 4 — factory functions (proof-bearing candidate edge)
# --------------------------------------------------------------------------- #

def test_factory_classmethod_resolves_to_the_base_as_a_proof_edge(tmp_path):
    idx = _index(tmp_path, "modeling_factory.py", """
        class TextTower:
            pass
        class Wrapper:
            def __init__(self, config):
                self.text = TextTower._from_config(config)
    """)
    site = next(s for s in idx.construction_sites if s.target == "text")
    (cand,) = site.candidates
    assert cand.provenance == "factory:_from_config"
    assert cand.symbol.qualified_name == "TextTower"  # base is LOCAL and proven
    assert site.via == "factory:_from_config"


def test_local_factory_construction_is_observed_without_claiming_field_ownership(
        tmp_path):
    idx = _index(tmp_path, "modeling_local_factory.py", """
        class Child:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                built = Child._from_config(config.child)
                self.slot = built
    """)
    site = next(s for s in idx.construction_sites
                if s.target_kind == "local" and s.target == "built")
    assert site.via == "factory:_from_config"
    assert len(site.candidates) == 1
    assert site.candidates[0].symbol.qualified_name == "Child"
    # Observation stays neutral: the later alias is a separate binding; the
    # ProgramIndex does not silently rewrite the local site as a field site.
    assert not any(s.target_kind == "field" and s.target == "slot"
                   for s in idx.construction_sites)
    alias = next(b for b in idx.bindings_in(site.enclosing_callable)
                 if any(t.source_segment == "self.slot" for t in b.targets))
    assert alias.value.source_segment == "built"


# --------------------------------------------------------------------------- #
# Spec family 5 — comprehensions (ModuleList + count expression)
# --------------------------------------------------------------------------- #

def test_modulelist_comprehension_records_elements_and_count(tmp_path):
    idx = _index(tmp_path, "modeling_comp.py", """
        import torch.nn as nn
        class Block:
            pass
        class Stack:
            def __init__(self, config):
                self.layers = nn.ModuleList([Block(config) for i in range(config.n)])
    """)
    (cont,) = idx.containers
    assert cont.kind == "modulelist" and cont.field == "layers"
    assert cont.count is not None and cont.count.kind == "call"
    (elem,) = cont.elements
    assert elem.target_kind == "element" and elem.via == "modulelist"
    (cand,) = elem.candidates
    assert cand.symbol.qualified_name == "Block"


# --------------------------------------------------------------------------- #
# Spec family 6 — conditional construction (distinct guards)
# --------------------------------------------------------------------------- #

def test_conditional_construction_keeps_both_branches_with_guards(tmp_path):
    idx = _index(tmp_path, "modeling_cond.py", """
        class Blk:
            def __init__(self, config):
                if config.use_a:
                    self.head = A(config)
                else:
                    self.head = B(config)
    """)
    sites = [s for s in idx.construction_sites if s.target == "head"]
    assert len(sites) == 2
    guards = {s.candidates[0].reference.name: s.guard[-1].kind for s in sites}
    assert guards == {"A": "if", "B": "else"}
    # both field assigns are retained with their branch guard
    assigns = [f for f in idx.field_assigns if f.field == "head"]
    assert {f.guard[-1].kind for f in assigns} == {"if", "else"}


# --------------------------------------------------------------------------- #
# Spec family 7 — equivalent candidates (same proven symbol at two sites)
# --------------------------------------------------------------------------- #

def test_equivalent_candidates_share_one_proven_symbol(tmp_path):
    idx = _index(tmp_path, "modeling_equiv.py", """
        class Foo:
            pass
        class Blk:
            def __init__(self, config):
                self.a = Foo(config)
                self.b = Foo(config)
    """)
    syms = {s.target: s.candidates[0].symbol.qualified_name
            for s in idx.construction_sites if s.target in ("a", "b")}
    assert syms == {"a": "Foo", "b": "Foo"}
    # two DISTINCT occurrences, one shared child identity
    a = next(s for s in idx.construction_sites if s.target == "a")
    b = next(s for s in idx.construction_sites if s.target == "b")
    assert a.site_id != b.site_id
    assert a.candidates[0].symbol == b.candidates[0].symbol


# --------------------------------------------------------------------------- #
# Spec family 8 / 12 / 18 — rival candidates retained, no winner, no conflict
# --------------------------------------------------------------------------- #

def test_registry_dispatch_retains_rivals_without_choosing(tmp_path):
    idx = _index(tmp_path, "modeling_rivals.py", """
        class EagerAttn: pass
        class FlashAttn: pass
        ATTENTION_CLASSES = {"eager": EagerAttn, "flash": FlashAttn}
        class Blk:
            def __init__(self, config):
                self.attn = ATTENTION_CLASSES[config._attn_implementation](config)
    """)
    site = next(s for s in idx.construction_sites if s.target == "attn")
    names = sorted(c.reference.name for c in site.candidates)
    assert names == ["EagerAttn", "FlashAttn"]  # BOTH retained; no winner
    assert all(c.provenance == "registry_subscript:ATTENTION_CLASSES"
               for c in site.candidates)
    assert all(c.symbol is not None for c in site.candidates)  # both proven local
    # the WALKER never emits a conflict — that is the U3-B resolver's job
    assert idx.__class__.__name__ == "ProgramIndex"
    assert not hasattr(idx, "conflicts")


# --------------------------------------------------------------------------- #
# Spec family 9 / 17 — unsupported dynamic dispatch → no proven candidate
# --------------------------------------------------------------------------- #

def test_dynamic_factory_yields_no_candidate_and_marks_unsupported(tmp_path):
    idx = _index(tmp_path, "modeling_dynamic.py", """
        class Blk:
            def __init__(self, config):
                self.layer = make_layer_cls(config.kind)(config)
    """)
    site = next(s for s in idx.construction_sites if s.target == "layer")
    assert site.candidates == ()  # zero candidates — dynamic construction
    kinds = {u.syntax_kind for u in idx.unsupported_syntax}
    assert "dynamic_construction" in kinds


def test_unknown_registry_yields_no_candidate(tmp_path):
    idx = _index(tmp_path, "modeling_unknown_reg.py", """
        class Blk:
            def __init__(self, config):
                self.attn = SOME_EXTERNAL_REGISTRY[config.kind](config)
    """)
    site = next(s for s in idx.construction_sites if s.target == "attn")
    assert site.candidates == ()
    assert site.via == "registry_subscript:SOME_EXTERNAL_REGISTRY"
    assert "unresolved_registry" in {u.syntax_kind for u in idx.unsupported_syntax}


# --------------------------------------------------------------------------- #
# Soumil — content changes while mtime is preserved (stale index impossible)
# --------------------------------------------------------------------------- #

def test_content_change_with_preserved_mtime_changes_the_fingerprint(tmp_path):
    path = _write(tmp_path, "modeling_mtime.py", """
        class Blk:
            def __init__(self, config):
                self.a = A(config)
    """)
    st = os.stat(path)
    idx1 = pi.build_program_index(_bundle({"root": (path,)}))
    # rewrite with DIFFERENT content but restore the original mtime/atime
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("class Blk:\n    def __init__(self, config):\n        self.a = B(config)\n")
    os.utime(path, (st.st_atime, st.st_mtime))
    assert os.stat(path).st_mtime == st.st_mtime  # mtime truly preserved
    idx2 = pi.build_program_index(_bundle({"root": (path,)}))
    assert idx1.fingerprint != idx2.fingerprint  # content, not mtime, is identity
    n1 = idx1.source_nodes[0].source_id.content_fingerprint
    n2 = idx2.source_nodes[0].source_id.content_fingerprint
    assert n1 != n2


# --------------------------------------------------------------------------- #
# Content-addressed immutable source-observation cache
# --------------------------------------------------------------------------- #

def test_repeated_build_reuses_one_exact_source_walk(
        tmp_path, monkeypatch):
    path = _write(tmp_path, "modeling_cached.py", """
        class Blk:
            def __init__(self, config):
                self.a = A(config)
    """)
    bundle = _bundle({"root": (path,)})
    pi.clear_program_index_source_cache()
    walk_count = _count_source_walks(monkeypatch)
    first = pi.build_program_index(bundle)
    second = pi.build_program_index(bundle)
    assert walk_count() == 1
    assert first == second


def test_cache_misses_when_content_changes_with_mtime_preserved(
        tmp_path, monkeypatch):
    path = _write(tmp_path, "modeling_cached_mtime.py", """
        class Blk:
            def __init__(self, config):
                self.a = A(config)
    """)
    bundle = _bundle({"root": (path,)})
    pi.clear_program_index_source_cache()
    walk_count = _count_source_walks(monkeypatch)
    first = pi.build_program_index(bundle)
    stat = os.stat(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "class Blk:\n"
            "    def __init__(self, config):\n"
            "        self.a = B(config)\n")
    os.utime(path, (stat.st_atime, stat.st_mtime))
    second = pi.build_program_index(bundle)
    assert walk_count() == 2
    assert first.fingerprint != second.fingerprint


def test_cache_keeps_identical_bytes_separate_across_component_addresses(
        tmp_path, monkeypatch):
    path = _write(tmp_path, "modeling_shared_cache.py", "class Shared: pass\n")
    bundle = _bundle({
        "root": (path,),
        "text_encoder": (path,),
    })
    pi.clear_program_index_source_cache()
    walk_count = _count_source_walks(monkeypatch)
    index = pi.build_program_index(bundle)
    assert walk_count() == 2
    assert {node.source_id.component_key for node in index.source_nodes} == {
        "root", "text_encoder",
    }


def test_cache_key_includes_the_complete_walker_vocabulary(
        tmp_path, monkeypatch):
    from model_unfolder import everchanging

    path = _write(tmp_path, "modeling_vocab_cache.py", """
        class Blk:
            def __init__(self, config):
                self.a = A(config)
    """)
    bundle = _bundle({"root": (path,)})
    pi.clear_program_index_source_cache()
    real_vocab = everchanging.load_program_index_vocab()
    real_factories = everchanging.load_constructor_classmethods()
    walk_count = _count_source_walks(monkeypatch)
    monkeypatch.setattr(
        everchanging, "load_program_index_vocab",
        lambda: real_vocab)
    monkeypatch.setattr(
        everchanging, "load_constructor_classmethods",
        lambda: real_factories)
    pi.build_program_index(bundle)
    changed = dict(real_vocab)
    changed["config_roots"] = frozenset(
        (*real_vocab["config_roots"], "alternate_config"))
    monkeypatch.setattr(
        everchanging, "load_program_index_vocab",
        lambda: changed)
    pi.build_program_index(bundle)
    monkeypatch.setattr(
        everchanging, "load_constructor_classmethods",
        lambda: frozenset((*real_factories, "alternate_factory")))
    pi.build_program_index(bundle)
    assert walk_count() == 3


def test_explicit_cache_clear_forces_a_fresh_walk(tmp_path, monkeypatch):
    path = _write(tmp_path, "modeling_cache_clear.py", "class Root: pass\n")
    bundle = _bundle({"root": (path,)})
    pi.clear_program_index_source_cache()
    walk_count = _count_source_walks(monkeypatch)
    pi.build_program_index(bundle)
    pi.clear_program_index_source_cache()
    pi.build_program_index(bundle)
    assert walk_count() == 2
    assert pi._observe_source.cache_info().maxsize == 128


# --------------------------------------------------------------------------- #
# Soumil — same qualified class at two distinct sites/roles
# --------------------------------------------------------------------------- #

def test_same_class_two_sites_two_occurrences_one_identity(tmp_path):
    idx = _index(tmp_path, "modeling_two_roles.py", """
        class Norm: pass
        class Blk:
            def __init__(self, config):
                self.pre = Norm(config)
                self.post = Norm(config)
    """)
    pre = next(s for s in idx.construction_sites if s.target == "pre")
    post = next(s for s in idx.construction_sites if s.target == "post")
    assert pre.site_id != post.site_id            # two occurrences
    assert pre.candidates[0].symbol == post.candidates[0].symbol  # one class id
    assert pre.target != post.target              # two roles/slots


# --------------------------------------------------------------------------- #
# Soumil — conflicting owner/config-prefix candidates retained as rivals
# --------------------------------------------------------------------------- #

def test_conflicting_config_prefix_candidates_are_retained_as_rivals(tmp_path):
    idx = _index(tmp_path, "modeling_prefix.py", """
        class TextTower: pass
        class VisionTower: pass
        TOWER_CLASSES = {"text": TextTower, "vision": VisionTower}
        class Wrapper:
            def __init__(self, config):
                self.tower = TOWER_CLASSES[config.modality](
                    config.text_config if config.modality == "text" else config.vision_config)
        """)
    site = next(s for s in idx.construction_sites if s.target == "tower")
    assert len(site.candidates) == 2  # rivals retained
    # the rival config prefixes both appear as config observations, unresolved —
    # the resolver (U3-B) decides uniqueness, not the walker.
    prefixes = {tuple(s.name for s in c.segments)[0] for c in idx.config_paths
                if c.segments}
    assert {"modality"} <= prefixes
    assert "text_config" in prefixes and "vision_config" in prefixes


# --------------------------------------------------------------------------- #
# Soumil — one malformed file + one healthy file → partial usability
# --------------------------------------------------------------------------- #

def test_malformed_file_becomes_parse_failure_healthy_stays_usable(tmp_path):
    good = _write(tmp_path, "modeling_good.py", """
        class Good:
            def __init__(self, config):
                self.a = A(config)
    """)
    bad = _write(tmp_path, "modeling_bad.py", "class Bad(:\n    def __init__(self):\n")
    idx = pi.build_program_index(_bundle({"root": (good, bad)}))
    assert _class(idx, "Good") is not None            # healthy fully indexed
    assert any(f.kind == "syntax_error" and f.source.canonical_path == pi._canonical_path(bad)
               for f in idx.parse_failures)
    # the bad file is NOT a source node, but its identity is in the aggregate
    assert all(n.source_id.canonical_path != pi._canonical_path(bad)
               for n in idx.source_nodes)


# --------------------------------------------------------------------------- #
# Soumil — two modules defining the same class name (name is not identity)
# --------------------------------------------------------------------------- #

def test_two_modules_same_class_name_are_distinct_symbols(tmp_path):
    a = _write(tmp_path, "modeling_a.py", "class Attention:\n    pass\n")
    b = _write(tmp_path, "modeling_b.py", "class Attention:\n    pass\n")
    idx = pi.build_program_index(_bundle({"root": (a, b)}))
    attns = [c for c in idx.classes if c.symbol.qualified_name == "Attention"]
    assert len(attns) == 2
    assert attns[0].symbol != attns[1].symbol           # SourceId differs
    assert attns[0].symbol.source != attns[1].symbol.source


# --------------------------------------------------------------------------- #
# Soumil — literal None default vs no default
# --------------------------------------------------------------------------- #

def test_literal_none_default_is_distinct_from_no_default(tmp_path):
    idx = _index(tmp_path, "modeling_defaults.py", """
        class Blk:
            def forward(self, x, mask=None, scale=1.0, cache=compute()):
                return x
    """)
    params = {p.name: p for p in _callable(idx, "Blk.forward").params}
    assert params["x"].has_default is False               # no default
    assert params["mask"].has_default is True
    assert params["mask"].default.kind == "constant" and params["mask"].default.const_value is None
    assert params["scale"].default.kind == "constant" and params["scale"].default.const_value == 1.0
    assert params["cache"].has_default is True
    assert params["cache"].default.kind == "call"          # dynamic/computed default


# --------------------------------------------------------------------------- #
# Soumil — two construction calls on the SAME source line
# --------------------------------------------------------------------------- #

def test_two_constructions_on_one_line_get_distinct_site_ids(tmp_path):
    idx = _index(tmp_path, "modeling_oneline.py", """
        class Blk:
            def __init__(self, config):
                self.a = A(config); self.b = B(config)
    """)
    a = next(s for s in idx.construction_sites if s.target == "a")
    b = next(s for s in idx.construction_sites if s.target == "b")
    assert a.span.line == b.span.line          # genuinely on one line
    assert a.site_id.ordinal != b.site_id.ordinal
    assert a.site_id != b.site_id


# --------------------------------------------------------------------------- #
# Soumil — branch calls prove lexical order is NOT runtime order
# --------------------------------------------------------------------------- #

def test_branch_calls_lexical_order_is_not_runtime_order(tmp_path):
    idx = _index(tmp_path, "modeling_branch.py", """
        class Blk:
            def forward(self, x):
                if x.flag:
                    a = self.left(x)
                else:
                    a = self.right(x)
                return a
    """)
    calls = [c for c in idx.calls if c.enclosing_callable.qualified_name == "Blk.forward"]
    left = next(c for c in calls if c.callee.name == "left")
    right = next(c for c in calls if c.callee.name == "right")
    # lexical order is monotonic textual position ...
    assert left.lexical_order < right.lexical_order
    # ... but the guards are MUTUALLY EXCLUSIVE branches of one `if`, so no
    # reader may read runtime order out of lexical order.
    assert left.guard[-1].kind == "if" and right.guard[-1].kind == "else"


# --------------------------------------------------------------------------- #
# Soumil — external node without provenance must raise
# --------------------------------------------------------------------------- #

def test_external_node_without_provenance_raises():
    fp = pi.content_fingerprint("x")
    with pytest.raises(ValueError):
        pi.SourceFileNode(pi.SourceId("/lib/attn.py", fp, external=True))
    # a well-formed external node is accepted
    ok = pi.SourceFileNode(pi.SourceId(
        "/lib/attn.py", fp, external=True,
        external_provenance="diffusers.models.attention"))
    assert ok.source_id.external


def test_build_rejects_non_external_external_nodes(tmp_path):
    good = _write(tmp_path, "modeling_ok.py", "class C:\n    pass\n")
    fp = pi.content_fingerprint("x")
    internal_masquerading = pi.SourceFileNode(pi.SourceId("/x.py", fp, component_key="root"))
    with pytest.raises(ValueError):
        pi.build_program_index(_bundle({"root": (good,)}),
                               external_nodes=(internal_masquerading,))


def test_external_node_is_folded_into_the_aggregate_fingerprint(tmp_path):
    good = _write(tmp_path, "modeling_ext.py", "class C:\n    pass\n")
    base = pi.build_program_index(_bundle({"root": (good,)}))
    ext = pi.SourceFileNode(pi.SourceId(
        "/lib/attn.py", pi.content_fingerprint("y"), external=True,
        external_provenance="diffusers.models.attention"))
    with_ext = pi.build_program_index(_bundle({"root": (good,)}), external_nodes=(ext,))
    assert base.fingerprint != with_ext.fingerprint
    assert any(n.source_id.external for n in with_ext.source_nodes)


# --------------------------------------------------------------------------- #
# Soumil — expression-source rename leaves structural observations equivalent
# --------------------------------------------------------------------------- #

def test_consistent_rename_leaves_structural_shape_equivalent(tmp_path):
    original = """
        class Blk:
            def __init__(self, config):
                self.a = (config.hidden_size + 1) * self.factor(config)
    """
    renamed = """
        class Blk:
            def __init__(self, config):
                self.a = (config.hidden_size + 1) * self.helper(config)
    """
    idx1 = _index(tmp_path, "modeling_r1.py", original)
    idx2 = _index(tmp_path, "modeling_r2.py", renamed)
    f1 = next(f for f in idx1.field_assigns if f.field == "a")
    f2 = next(f for f in idx2.field_assigns if f.field == "a")
    # source segments differ (factor vs helper) ...
    assert f1.value.source_segment != f2.value.source_segment
    # ... yet the STRUCTURAL shape (kinds/operators/arity) is identical
    assert _shape(f1.value) == _shape(f2.value)


# --------------------------------------------------------------------------- #
# Aggregate fingerprint laws
# --------------------------------------------------------------------------- #

def test_aggregate_fingerprint_is_order_independent(tmp_path):
    a = _write(tmp_path, "m_a.py", "class A:\n    pass\n")
    b = _write(tmp_path, "m_b.py", "class B:\n    pass\n")
    f1 = pi.build_program_index(_bundle({"root": (a, b)})).fingerprint
    f2 = pi.build_program_index(_bundle({"root": (b, a)})).fingerprint
    assert f1 == f2  # canonical over identity, not iteration order


def test_aggregate_fingerprint_includes_component_ownership(tmp_path):
    shared = _write(tmp_path, "m_shared.py", "class S:\n    pass\n")
    one = pi.build_program_index(_bundle({"root": (shared,)})).fingerprint
    # same file owned by TWO components -> different bundle identity + two nodes
    two_idx = pi.build_program_index(
        _bundle({"root": (shared,), "text_encoder": (shared,)}))
    assert one != two_idx.fingerprint
    comps = sorted(n.source_id.component_key for n in two_idx.source_nodes)
    assert comps == ["root", "text_encoder"]


def test_aggregate_fingerprint_is_not_raw_file_contents(tmp_path):
    a = _write(tmp_path, "m_c.py", "class A:\n    pass\n")
    idx = pi.build_program_index(_bundle({"root": (a,)}))
    with open(a, encoding="utf-8") as fh:
        raw = fh.read()
    assert idx.fingerprint != raw and raw not in idx.fingerprint


# --------------------------------------------------------------------------- #
# Module lexical bindings — builtin protocols cannot be selected by spelling
# --------------------------------------------------------------------------- #

def test_module_binding_census_respects_python_scope_boundaries(tmp_path):
    idx = _index(tmp_path, "bindings.py", """
        import os as operating_system
        from somewhere import thing as imported_thing
        assigned = 1
        if flag:
            conditional = 2
        for loop_target in values:
            pass
        try:
            pass
        except Error as exception_target:
            recovered = True
        match subject:
            case {"value": captured}:
                matched = captured
        class DeclaredClass:
            class_local = 1
        def declared_function(function_parameter):
            function_local = function_parameter
        comprehension = [item for item in values]
    """)
    source = idx.source_nodes[0].source_id
    names = {binding.name for binding in idx.module_bindings_in(source)}
    assert {
        "operating_system", "imported_thing", "assigned", "conditional",
        "loop_target", "exception_target", "recovered", "captured", "matched",
        "DeclaredClass", "declared_function", "comprehension",
    } <= names
    assert "class_local" not in names
    assert "function_parameter" not in names
    assert "function_local" not in names
    assert "item" not in names              # comprehension-local target
    assert all(binding.span.source == source
               for binding in idx.module_bindings_in(source))


def test_module_binding_query_is_typed(tmp_path):
    idx = _index(tmp_path, "bindings.py", "value = 1")
    with pytest.raises(TypeError):
        idx.module_bindings_in(object())


def test_module_binding_observation_closes_its_identity():
    source = pi.SourceId("/module.py", "fp", component_key="root")
    span = pi.SourceSpan(source, 1)
    assert pi.ModuleBindingObservation(source, "name", "store", span).name == "name"
    with pytest.raises(ValueError):
        pi.ModuleBindingObservation(source, "", "store", span)
    with pytest.raises(ValueError):
        pi.ModuleBindingObservation(source, "name", "guessed", span)
    other = pi.SourceId("/other.py", "fp2", component_key="root")
    with pytest.raises(ValueError):
        pi.ModuleBindingObservation(source, "name", "store", pi.SourceSpan(other, 1))


# --------------------------------------------------------------------------- #
# ParseContext ownership — exactly ONE immutable index per context
# --------------------------------------------------------------------------- #

def test_parse_context_program_index_is_call_local_and_memoized(tmp_path):
    a = _write(tmp_path, "m_ctx.py", "class A:\n    pass\n")
    from model_unfolder.evidence.context import ParseContext
    ctx = ParseContext(source_bundle=_bundle({"root": (a,)}))
    first = ctx.program_index()
    second = ctx.program_index()
    assert first is second                      # exactly one per context
    other = ParseContext(source_bundle=_bundle({"root": (a,)}))
    assert other.program_index() is not first   # call-local: a new context builds its own


def test_parse_context_reader_results_are_exact_path_scoped():
    from model_unfolder.evidence.context import ParseContext

    ctx = ParseContext(source_bundle=_bundle({}))
    first = object()
    assert ctx.cached_reader_result(
        "reader", ("decoder",), lambda: first) is first
    assert ctx.cached_reader_result(
        "reader", ("decoder",),
        lambda: pytest.fail("the exact reader/path result was recomputed")) \
        is first

    second = object()
    assert ctx.cached_reader_result(
        "reader", ("vision",), lambda: second) is second
    assert ctx.reader_results == {
        ("reader", ("decoder",)): first,
        ("reader", ("vision",)): second,
    }
    with pytest.raises(ValueError):
        ctx.cached_reader_result("", (), lambda: None)
    with pytest.raises(TypeError):
        ctx.cached_reader_result("reader", ("",), lambda: None)


# --------------------------------------------------------------------------- #
# All record families are populated by a single representative source
# --------------------------------------------------------------------------- #

def test_every_record_family_is_populated(tmp_path):
    idx = _index(tmp_path, "modeling_all.py", """
        import torch.nn as nn
        from pkg import Helper as H
        ATTENTION_CLASSES = {"eager": Eager, "flash": Flash}
        class Eager: pass
        class Flash: pass
        class Blk(nn.Module):
            variant = "x"
            def __init__(self, config):
                super().__init__()
                self.dim = config.hidden_size
                self.attn = ATTENTION_CLASSES[config.impl](config)
                self.layers = nn.ModuleList([Sub(config) for i in range(config.n)])
                if config.bias:
                    self.proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.act = ACT2FN[config.hidden_act]
                self.h = H(config)
                self.dyn = pick_cls(config.kind)(config)
            def forward(self, x):
                y = self.attn(x)
                z = y + 1
                return self.proj(z)
    """)
    assert idx.source_nodes and idx.modules and idx.imports and idx.module_bindings
    assert idx.classes
    assert idx.callables and idx.field_assigns and idx.construction_sites
    assert idx.identifiers
    assert idx.containers and idx.dispatch_registries and idx.calls
    assert idx.attribute_accesses and idx.config_paths and idx.controls
    assert idx.dataflow and idx.unsupported_syntax  # unresolved registry marks this
    assert idx.fingerprint


# --------------------------------------------------------------------------- #
# The query surface selects by ADDRESS only (no name/substring role selection)
# --------------------------------------------------------------------------- #

def test_query_surface_is_address_only(tmp_path):
    idx = _index(tmp_path, "m_q.py", """
        class Foo:
            def __init__(self, config):
                self.a = Bar(config)
    """)
    query_methods = [n for n in dir(idx)
                     if not n.startswith("_") and callable(getattr(idx, n))]
    # no query method advertises an architectural ROLE or a name-search verb
    forbidden = ("find", "named", "ffn", "attention", "vision", "audio",
                 "best", "guess", "role", "search")
    for name in query_methods:
        assert not any(tok in name.lower() for tok in forbidden), name
    # address-based queries work
    foo = _class(idx, "Foo")
    assert idx.class_by_symbol(foo.symbol) is foo
    assert idx.classes_in(foo.symbol.source)
