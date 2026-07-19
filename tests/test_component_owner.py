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

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    OwnerGraph,
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
    assert text.owner.qualified_name == "TextModel"
    assert text.config_prefix == ("text_config",)          # config.text_config
    block = _child(text, "layers")
    assert block.owner.qualified_name == "Block" and block.via_kind == "element"
    assert block.config_prefix == ("text_config",)
    norm = _child(block, "norm")
    assert norm.owner.qualified_name == "RMSNorm"
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
    assert mlp is not None and mlp.owner.qualified_name == "MLP"
    assert mlp.via_kind == "return"


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
    assert text.owner.qualified_name == "TextTower"
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
    assert {r[0] for r in rivals} == {"EagerAttn", "FlashAttn"}


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
    assert ("text_config",) in rc.rivals and ("vision_config",) in rc.rivals
    # the child owner still resolves (Tower); only its prefix is ambiguous
    assert _child(g.root, "tower").owner.qualified_name == "Tower"


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
    assert tnode is not None and tnode.owner.qualified_name == "Tower"
    assert tnode.owner.source.canonical_path.endswith("modeling_tower.py")
    assert tnode.config_prefix == ("text_config",)


def test_cross_file_ambiguous_class_name_is_not_guessed(tmp_path):
    a = _write(tmp_path, "modeling_a.py", "class Tower:\n    def __init__(self, config): pass\n")
    b = _write(tmp_path, "modeling_b.py", "class Tower:\n    def __init__(self, config): pass\n")
    wrap = _write(tmp_path, "modeling_wrap.py", """
        class Wrapper:
            def __init__(self, config):
                self.tower = Tower(config)
    """)
    idx = _index(tmp_path, {"root": (wrap, a, b)})
    g = resolve_owner_graph(idx, _root(idx, "Wrapper"))
    # two classes named Tower -> the resolver refuses to guess which owns the slot
    assert _child(g.root, "tower") is None
    assert any(u.field == "tower" and u.kind == "ambiguous_crossfile"
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
    assert first is not None and first.owner.qualified_name == "Node"
    # the recursion terminates: the deepest Node has no further resolved child
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
    names = [n.owner.qualified_name for n in g.walk()]
    assert names == ["Block", "Attn"]
    assert g.node_for(_root(idx, "Attn")).via_field == "attn"
