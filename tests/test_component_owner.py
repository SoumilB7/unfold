"""U3-B — the ComponentOwner resolver: construction-graph resolution over the
raw ProgramIndex.

Proves the resolver turns the index's OBSERVED candidates into a resolved owner
graph — parent owner -> construction site -> field/slot -> child class — while
propagating the exact config-path prefix down each edge, and emits TYPED
conflicts (never a silent drop, never a guess) where uniqueness cannot be proven:
rival_owner_chain (>=2 candidate classes for one slot) and rival_config_prefix
(a config argument that resolves to >=2 distinct prefixes).
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    ConfigPrefixRival,
    OwnerGraph,
    OwnerRival,
    resolve_owner_graph,
)
from model_unfolder.evidence.models import SourceBundle


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _index(tmp_path, files: dict):
    flat = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return pi.build_program_index(SourceBundle(
        source="local", files=tuple(flat),
        component_files={k: tuple(v) for k, v in files.items()}))


def _root(idx, name):
    return next(c.symbol for c in idx.classes if c.symbol.qualified_name == name)


def _child(node, field):
    return next((c for c in node.children if c.via_field == field), None)


# --------------------------------------------------------------------------- #
# Linear chain + config-prefix propagation
# --------------------------------------------------------------------------- #

def test_owner_chain_and_config_prefix_propagate(tmp_path):
    src = """
        import torch.nn as nn
        class RMSNorm:
            def __init__(self, dim): pass
        class Attn:
            def __init__(self, config): pass
        class Block:
            def __init__(self, config):
                self.attn = Attn(config)
                self.norm = RMSNorm(config.hidden_size)
        class TextModel:
            def __init__(self, config):
                self.layers = nn.ModuleList([Block(config) for i in range(config.n)])
        class Wrapper:
            def __init__(self, config):
                self.text = TextModel(config.text_config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    assert g.conflicts == ()
    text = _child(g.root, "text")
    assert text.symbol.qualified_name == "TextModel"
    assert text.config_prefix == ("text_config",)          # config.text_config
    block = _child(text, "layers")
    assert block.symbol.qualified_name == "Block" and block.via_kind == "element"
    assert block.config_prefix == ("text_config",)
    norm = _child(block, "norm")
    assert norm.symbol.qualified_name == "RMSNorm"
    assert norm.config_prefix == ("text_config", "hidden_size")  # nested prefix


# --------------------------------------------------------------------------- #
# Helper fold: self.x = self._build(...) -> the helper's return construction
# --------------------------------------------------------------------------- #

def test_helper_fold_resolves_through_the_return_site(tmp_path):
    src = """
        class MLP:
            def __init__(self, config): pass
        class Block:
            def __init__(self, config):
                self.mlp = self._build_mlp(config)
            def _build_mlp(self, config):
                return MLP(config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Block"))
    mlp = _child(g.root, "mlp")
    assert mlp is not None and mlp.symbol.qualified_name == "MLP"
    assert mlp.via_kind == "return"
    # The helper-call site AND return site participate in occurrence identity.
    assert len(mlp.occurrence.sites) == 2


# --------------------------------------------------------------------------- #
# Factory: X._from_config(config.text_config) -> base X, prefix from args[0]
# --------------------------------------------------------------------------- #

def test_factory_resolves_base_and_propagates_prefix(tmp_path):
    src = """
        class TextTower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.text = TextTower._from_config(config.text_config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    text = _child(g.root, "text")
    assert text.symbol.qualified_name == "TextTower"
    assert text.config_prefix == ("text_config",)


# --------------------------------------------------------------------------- #
# rival_owner_chain: a registry dispatch with >=2 candidates -> typed conflict
# --------------------------------------------------------------------------- #

def test_registry_rivals_emit_rival_owner_chain_conflict(tmp_path):
    src = """
        class EagerAttn:
            def __init__(self, config): pass
        class FlashAttn:
            def __init__(self, config): pass
        ATTN = {"eager": EagerAttn, "flash": FlashAttn}
        class Block:
            def __init__(self, config):
                self.attn = ATTN[config._attn_implementation](config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Block"))
    # the resolver does NOT pick a winner: no attn child owner is fabricated
    assert _child(g.root, "attn") is None
    assert any(u.field == "attn" and u.kind == "rival_owner"
               for u in g.root.unresolved)
    kinds = [c.kind for c in g.conflicts]
    assert "rival_owner_chain" in kinds
    rivals = next(c for c in g.conflicts if c.kind == "rival_owner_chain").rivals
    assert all(isinstance(r, OwnerRival) for r in rivals)
    assert {r.reference for r in rivals} == {"EagerAttn", "FlashAttn"}
    assert {r.parent for r in rivals} == {g.root.occurrence}
    assert {r.site for r in rivals} == {g.root.unresolved[0].site}


# --------------------------------------------------------------------------- #
# rival_config_prefix: one construction, two possible config prefixes
# --------------------------------------------------------------------------- #

def test_ternary_config_arg_emits_rival_config_prefix_conflict(tmp_path):
    src = """
        class Tower:
            def __init__(self, config): pass
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(
                    config.text_config if config.is_text else config.vision_config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    kinds = [c.kind for c in g.conflicts]
    assert "rival_config_prefix" in kinds
    rc = next(c for c in g.conflicts if c.kind == "rival_config_prefix")
    assert all(isinstance(r, ConfigPrefixRival) for r in rc.rivals)
    assert {r.prefix for r in rc.rivals} == {
        ("text_config",), ("vision_config",),
    }
    # The child identity resolves, but its config location does not. Crucially,
    # it does NOT fall back to the root prefix ().
    tower = _child(g.root, "tower")
    assert tower.symbol.qualified_name == "Tower"
    assert tower.config_prefix is None
    assert set(tower.config_prefix_candidates) == {
        ("text_config",), ("vision_config",),
    }


# --------------------------------------------------------------------------- #
# Cross-file resolution: a child class defined in another indexed source
# --------------------------------------------------------------------------- #

def test_cross_file_unique_class_resolves(tmp_path):
    tower = _write(tmp_path, "modeling_tower.py", "class Tower:\n    def __init__(self, config): pass\n")
    wrap = _write(tmp_path, "modeling_wrap.py", """
        from modeling_tower import Tower
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config.text_config)
    """)
    idx = _index(tmp_path, {"root": (wrap, tower)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    tnode = _child(g.root, "tower")
    assert tnode is not None and tnode.symbol.qualified_name == "Tower"
    assert tnode.symbol.source.canonical_path.endswith("modeling_tower.py")
    assert tnode.config_prefix == ("text_config",)


def test_cross_file_module_alias_resolves_only_through_exact_import(tmp_path):
    tower = _write(tmp_path, "modeling_tower.py",
                   "class Tower:\n    def __init__(self, config): pass\n")
    wrap = _write(tmp_path, "modeling_wrap.py", """
        import modeling_tower as towers
        class Wrapper:
            def __init__(self, config):
                self.tower = towers.Tower(config.vision_config)
    """)
    idx = _index(tmp_path, {"root": (wrap, tower)})
    graph = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    node = _child(graph.root, "tower")
    assert node is not None and node.symbol.source.canonical_path == tower
    assert node.config_prefix == ("vision_config",)


def test_cross_file_bare_name_without_import_is_not_guessed(tmp_path):
    a = _write(tmp_path, "modeling_a.py", "class Tower:\n    def __init__(self, config): pass\n")
    b = _write(tmp_path, "modeling_b.py", "class Tower:\n    def __init__(self, config): pass\n")
    wrap = _write(tmp_path, "modeling_wrap.py", """
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config)
    """)
    idx = _index(tmp_path, {"root": (wrap, a, b)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    # A unique or ambiguous same-name class elsewhere is irrelevant without an
    # import binding. Bare-name uniqueness across the bundle is not proof.
    assert _child(g.root, "tower") is None
    assert any(u.field == "tower" and u.kind == "external"
               for u in g.root.unresolved)


# --------------------------------------------------------------------------- #
# Dynamic construction: no candidate -> unresolved dynamic, never invented
# --------------------------------------------------------------------------- #

def test_dynamic_construction_is_unresolved_not_invented(tmp_path):
    src = """
        class Block:
            def __init__(self, config):
                self.layer = make_layer_cls(config.kind)(config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Block"))
    assert g.root.children == ()
    assert any(u.field == "layer" and u.kind == "dynamic" for u in g.root.unresolved)


# --------------------------------------------------------------------------- #
# Cycle guard: a class constructing itself does not loop forever
# --------------------------------------------------------------------------- #

def test_self_referential_construction_terminates(tmp_path):
    src = """
        class Node:
            def __init__(self, config):
                self.child = Node(config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Node"))
    # Node -> child Node -> (cycle stops); finite tree, no exception
    first = _child(g.root, "child")
    assert first is not None and first.symbol.qualified_name == "Node"
    # The recursion terminates with an explicit cycle record, not silent empty
    # children that could be mistaken for a proven leaf.
    assert any(u.kind == "cycle" for u in first.unresolved)
    assert isinstance(g, OwnerGraph)


# --------------------------------------------------------------------------- #
# Graph API
# --------------------------------------------------------------------------- #

def test_owner_graph_walk_and_lookup(tmp_path):
    src = """
        class Attn:
            def __init__(self, config): pass
        class Block:
            def __init__(self, config):
                self.attn = Attn(config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    g = resolve_owner_graph(idx, _root(idx, "Block"))
    names = [n.symbol.qualified_name for n in g.walk()]
    assert names == ["Block", "Attn"]
    attn = _child(g.root, "attn")
    assert g.node_for(attn.occurrence) is attn
    assert g.nodes_for_symbol(_root(idx, "Attn")) == (attn,)
    with pytest.raises(TypeError):
        g.node_for(_root(idx, "Attn"))


def test_same_class_at_two_sites_remains_two_owner_occurrences(tmp_path):
    src = """
        class Norm:
            def __init__(self, dim): pass
        class Block:
            def __init__(self, config):
                self.pre = Norm(config.hidden_size)
                self.post = Norm(config.hidden_size)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    norm_symbol = _root(idx, "Norm")
    graph = resolve_owner_graph(idx, _root(idx, "Block"))
    pre, post = _child(graph.root, "pre"), _child(graph.root, "post")
    assert pre.symbol == post.symbol == norm_symbol
    assert pre.occurrence != post.occurrence
    assert graph.nodes_for_symbol(norm_symbol) == (pre, post)
    assert graph.node_for(pre.occurrence) is pre
    assert graph.node_for(post.occurrence) is post


def test_two_fields_calling_same_helper_do_not_collapse(tmp_path):
    src = """
        class Norm:
            def __init__(self, config): pass
        class Block:
            def __init__(self, config):
                self.pre = self._make(config)
                self.post = self._make(config)
            def _make(self, config):
                return Norm(config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    graph = resolve_owner_graph(idx, _root(idx, "Block"))
    pre, post = _child(graph.root, "pre"), _child(graph.root, "post")
    assert pre.occurrence != post.occurrence
    assert pre.occurrence.sites[-1] == post.occurrence.sites[-1]  # shared return
    assert pre.occurrence.sites[0] != post.occurrence.sites[0]   # distinct fields


def test_ambiguous_prefix_propagates_as_ambiguity_not_parent_fallback(tmp_path):
    src = """
        class Leaf:
            def __init__(self, config): pass
        class Tower:
            def __init__(self, config):
                self.leaf = Leaf(config)
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(
                    config.text_config if config.is_text else config.vision_config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    graph = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    tower = _child(graph.root, "tower")
    leaf = _child(tower, "leaf")
    expected = {("text_config",), ("vision_config",)}
    assert tower.config_prefix is None and set(tower.config_prefix_candidates) == expected
    assert leaf.config_prefix is None and set(leaf.config_prefix_candidates) == expected
    assert sum(c.kind == "rival_config_prefix" for c in graph.conflicts) == 2


def test_multiple_root_parameters_require_explicit_binding(tmp_path):
    src = """
        class Root:
            def __init__(self, config, adapter): pass
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    root = _root(idx, "Root")
    unresolved = resolve_owner_graph(idx, root)
    assert any(u.kind == "root_config_binding" for u in unresolved.root.unresolved)
    assert unresolved.root.config_bindings == ()
    explicit = resolve_owner_graph(idx, root,
                                   root_param_prefixes={"config": (), "adapter": ("adapter",)})
    assert {b.parameter: b.resolved_prefix for b in explicit.root.config_bindings} == {
        "config": (), "adapter": ("adapter",),
    }


def test_factory_input_is_not_falsely_bound_to_init_parameter(tmp_path):
    src = """
        class Tower:
            def __init__(self, unrelated): pass
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower.from_pretrained(config.vision_config)
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    graph = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    tower = _child(graph.root, "tower")
    assert tower.config_prefix == ("vision_config",)
    assert tower.config_bindings[0].parameter == "@factory_input"
    assert all(binding.parameter != "unrelated" for binding in tower.config_bindings)


def test_guarded_constructors_for_one_field_are_rivals_not_two_children(tmp_path):
    src = """
        class Dense: pass
        class Gated: pass
        class Block:
            def __init__(self, config):
                if config.gated:
                    self.ffn = Gated()
                else:
                    self.ffn = Dense()
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    graph = resolve_owner_graph(idx, _root(idx, "Block"))
    assert _child(graph.root, "ffn") is None
    assert any(u.field == "ffn" and u.kind == "rival_owner"
               for u in graph.root.unresolved)
    conflict = next(c for c in graph.conflicts if c.kind == "rival_owner_chain")
    assert {r.reference for r in conflict.rivals} == {"Dense", "Gated"}
    assert len({r.site for r in conflict.rivals}) == 2


def test_helper_with_rival_return_constructions_is_not_selected(tmp_path):
    src = """
        class Dense: pass
        class Gated: pass
        class Block:
            def __init__(self, config):
                self.ffn = self._make(config)
            def _make(self, config):
                if config.gated:
                    return Gated()
                return Dense()
    """
    idx = _index(tmp_path, {"root": (_write(tmp_path, "m.py", src),)})
    graph = resolve_owner_graph(idx, _root(idx, "Block"))
    assert _child(graph.root, "ffn") is None
    assert any(u.field == "ffn" and u.kind == "rival_owner"
               for u in graph.root.unresolved)
    conflict = next(c for c in graph.conflicts if c.kind == "rival_owner_chain")
    assert len(conflict.rivals) == 2
