"""Exact-output and operation-count controls for call-local owner indexes."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from model_unfolder.evidence import component_owner as owners
from model_unfolder.evidence.constructor_values import canonical_construction_target
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    ConstructionSite, ProgramIndex, SourceId, build_program_index,
)


SOURCE = """
class Leaf:
    def __init__(self, config):
        self.value = config.value

class Root:
    def __init__(self, config, other):
        self.left = Leaf(config.left)
        self.right = Leaf(other.right)
"""


def _index(tmp_path, source=SOURCE):
    path = tmp_path / "model.py"
    path.write_text(source)
    return build_program_index(SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)}))


def _root(index):
    return next(item.symbol for item in index.classes
                if item.symbol.qualified_name == "Root")


def _graph(index, **kwargs):
    return owners.resolve_owner_graph(
        index, _root(index),
        root_param_prefixes={"config": (), "other": ("other",)}, **kwargs)


def test_occurrence_index_walks_once_and_keeps_same_class_instances(tmp_path, monkeypatch):
    graph = _graph(_index(tmp_path))
    before, before_hash = asdict(graph), hash(graph)
    left, right = graph.root.children
    assert left.symbol == right.symbol and left.occurrence != right.occurrence
    assert left.config_prefix == ("left",)
    assert right.config_prefix == ("other", "right")
    original = owners.OwnerGraph.walk
    visits = []

    def counted(self):
        for node in original(self):
            visits.append(node.occurrence)
            yield node

    monkeypatch.setattr(owners.OwnerGraph, "walk", counted)
    for _ in range(100):
        assert graph.node_for(left.occurrence) is left
        assert graph.node_for(right.occurrence) is right
        assert graph.nodes_for_symbol(left.symbol) == (left, right)
    missing = replace(left.occurrence, root=replace(left.symbol, qualified_name="Absent"))
    assert graph.node_for(missing) is None
    assert len(visits) == 3
    assert asdict(graph) == before and hash(graph) == before_hash


def test_duplicate_occurrence_keeps_original_first_match_and_symbol_order(tmp_path):
    original = _graph(_index(tmp_path))
    left, right = original.root.children
    later = replace(left, via_field="later_record")
    graph = replace(original, root=replace(
        original.root, children=(left, right, later)))
    assert graph.node_for(left.occurrence) is left
    assert graph.nodes_for_symbol(left.symbol) == (left, right, later)


def test_site_membership_does_not_hash_full_records_or_accept_same_id_forgery(
        tmp_path, monkeypatch):
    index = _index(tmp_path)
    site = next(item for item in index.construction_sites if item.target == "left")
    symbol = site.candidates[0].symbol

    def forbidden_hash(_self):
        raise AssertionError("full construction census was rehashed")

    monkeypatch.setattr(ConstructionSite, "__hash__", forbidden_hash)
    for _ in range(100):
        assert index.contains_construction_site(site)
        assert index.contains_construction_site(replace(site))
        assert canonical_construction_target(index, site, symbol) is not None
    forged = replace(site, target="unrecorded")
    assert forged.site_id == site.site_id
    assert not index.contains_construction_site(forged)
    assert canonical_construction_target(index, forged, symbol) is None
    # The address bucket retains both records if the index actually has both.
    extended = replace(index, construction_sites=(*index.construction_sites, forged))
    assert extended.contains_construction_site(forged)
    assert not index.contains_construction_site(forged)


def test_owner_graph_memo_preserves_prefix_order_depth_and_new_index(tmp_path, monkeypatch):
    index = _index(tmp_path)
    original_init = owners._Resolver.__init__
    builds = []

    def counted(self, *args, **kwargs):
        builds.append(args[0])
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(owners._Resolver, "__init__", counted)
    graph = _graph(index)
    for _ in range(100):
        assert _graph(index) is graph
    assert len(builds) == 1
    reordered = owners.resolve_owner_graph(index, _root(index), root_param_prefixes={
        "other": ("other",), "config": ()})
    assert [item.parameter for item in reordered.root.config_bindings] == ["other", "config"]
    assert reordered != graph
    assert _graph(index, max_depth=2) is not graph
    changed = owners.resolve_owner_graph(index, _root(index), root_param_prefixes={
        "config": ("changed",), "other": ("other",)})
    assert changed.root.children[0].config_prefix == ("changed", "left")
    extended = replace(index, construction_sites=tuple(
        site for site in index.construction_sites if site.target != "left"))
    changed_graph = _graph(extended)
    assert [child.via_field for child in changed_graph.root.children] == ["right"]
    assert _graph(index) is graph
    assert len(builds) == 5


def test_source_content_change_with_same_path_and_mtime_cannot_reuse_owner_graph(tmp_path):
    first = _index(tmp_path)
    first_graph = _graph(first)
    path = tmp_path / "model.py"
    before = path.stat()
    path.write_text(SOURCE.replace("config.left", "config.changed"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = build_program_index(SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)}))
    second_graph = _graph(second)
    assert first.fingerprint != second.fingerprint
    assert first_graph.root.children[0].config_prefix == ("left",)
    assert second_graph.root.children[0].config_prefix == ("changed",)
    assert _graph(first) is first_graph


def test_address_queries_equal_original_ordered_scans(tmp_path):
    source = (
        "from external import Leaf as Imported\n"
        "from rival import Leaf as Imported\n"
        "import torch.nn as nn\n" + SOURCE.replace(
            "self.right = Leaf(other.right)",
            "self.right = Leaf(other.right)\n"
            "        self.layers = nn.ModuleList([Leaf(config.layer)])")
        + "\nclass Leaf:\n    def __init__(self, different): pass\n")
    index = _index(tmp_path, source)
    assert len(index.classes_at_qualified_name("Leaf")) == 2
    assert len(index.imports_aliased(index.imports[0].source, "Imported")) == 2
    assert index.containers
    for record in index.classes:
        assert index.classes_at_qualified_name(record.symbol.qualified_name) == tuple(
            item for item in index.classes
            if item.symbol.qualified_name == record.symbol.qualified_name)
    for record in index.imports:
        assert index.imports_aliased(record.source, record.alias) == tuple(
            item for item in index.imports
            if item.source == record.source and item.alias == record.alias)
    for record in index.classes:
        assert index.containers_of(record.symbol) == tuple(
            item for item in index.containers if item.owner == record.symbol)


def _declared_hash(index):
    return hash(tuple(getattr(index, item.name) for item in fields(ProgramIndex)
                      if (item.compare if item.hash is None else item.hash)))


def test_index_hash_matches_declared_fields_and_hashes_children_once(tmp_path, monkeypatch):
    index = _index(tmp_path)
    expected = _declared_hash(index)
    original = SourceId.__hash__
    calls = []

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(SourceId, "__hash__", counted)
    assert hash(index) == expected
    first_count = len(calls)
    assert first_count > 0
    for _ in range(100):
        assert hash(index) == expected
    assert len(calls) == first_count


def test_index_copy_and_pickle_drop_all_derived_state(tmp_path):
    index = _index(tmp_path)
    _graph(index)
    original_hash = hash(index)
    derived = {"_structural_hash", "_address_index", "_call_memo"}
    assert derived <= index.__dict__.keys()
    for restored in (deepcopy(index), pickle.loads(pickle.dumps(index))):
        assert not derived & restored.__dict__.keys()
        assert restored == index and hash(restored) == original_hash
    changed = replace(index, classes=())
    assert not derived & changed.__dict__.keys()
    assert changed.fingerprint == index.fingerprint and changed != index


def test_pickle_recomputes_structural_hash_under_each_process_salt(tmp_path):
    index = _index(tmp_path)
    hash(index)
    path = tmp_path / "index.pickle"
    path.write_bytes(pickle.dumps(index))
    code = """
from dataclasses import fields
import json, pickle, sys
with open(sys.argv[1], 'rb') as stream:
    index = pickle.load(stream)
assert '_structural_hash' not in index.__dict__
expected = hash(tuple(getattr(index, field.name) for field in fields(type(index))
                      if (field.compare if field.hash is None else field.hash)))
assert hash(index) == expected
print(json.dumps({'hash_matches_own_process': True}))
"""
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": seed,
                 "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == {"hash_matches_own_process": True}


def test_hash_collisions_do_not_replace_full_equality(tmp_path, monkeypatch):
    index = _index(tmp_path)
    changed = replace(index, classes=())
    assert changed.fingerprint == index.fingerprint
    monkeypatch.setattr(ProgramIndex, "__hash__", lambda self: 0)
    values = {index: "original", changed: "different content"}
    assert len(values) == 2
    assert values[index] == "original" and values[changed] == "different content"


def test_unhashable_malformed_content_retains_type_error(tmp_path):
    index = replace(_index(tmp_path), source_nodes=[])
    for _ in range(2):
        with pytest.raises(TypeError):
            hash(index)
        assert "_structural_hash" not in index.__dict__


def test_callable_buckets_preserve_exact_records_and_reject_orphan_calls(tmp_path, monkeypatch):
    from model_unfolder.evidence.program_index import CallObservation
    index = _index(tmp_path)
    assert index.calls
    expected = {call for record in index.callables for call in index.calls_in(record.symbol)}
    original = index.calls[0]
    assert original in expected
    for record in index.callables:
        assert index.attribute_accesses_in(record.symbol) == tuple(
            item for item in index.attribute_accesses if item.enclosing_callable == record.symbol)
    # Membership uses only its callable's records, never a rehashed whole census.
    monkeypatch.setattr(CallObservation, '__hash__', lambda _self: (_ for _ in ()).throw(
        AssertionError('whole call census hash returned')))
    for _ in range(100):
        assert index.contains_callable_call(original)
        assert index.contains_callable_call(replace(original))
    forged = replace(original, args=())
    assert forged != original
    assert not index.contains_callable_call(forged)
    orphaned = replace(index, callables=tuple(
        row for row in index.callables if row.symbol != original.enclosing_callable))
    assert original in orphaned.calls_in(original.enclosing_callable)
    assert not orphaned.contains_callable_call(original)
    assert index.contains_callable_call(original)
