"""U11-E2c occurrence-qualified attention input-role controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.attention_invocation_role import (
    framework_attention_invocation_role,
)
from model_unfolder.evidence.attention_lane import (
    framework_attention_lane_positive_proof_in_graph,
)
from model_unfolder.evidence.component_owner import resolve_owner_graph
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.container_inventory import (
    resolve_container_inventory_in_graph,
)
from model_unfolder.evidence.execution_flow import (
    resolve_addressed_invocations_in_graph,
)
from model_unfolder.evidence.import_source import (
    canonical_called_import_target,
    resolve_called_import_source,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index


ATTENTION = """
import torch
from torch.nn import functional as F

class Fast:
    def __call__(self, module: Attention, primary, context=None):
        q = module.a(primary)
        if context is None: context = primary
        k = module.b(context)
        v = module.c(context)
        return F.scaled_dot_product_attention(q, k, v)

class Slow:
    def __call__(self, module: Attention, primary, context=None):
        q = module.x(primary)
        if context is None: context = primary
        k = module.y(context)
        v = module.z(context)
        probs = module.score(q, k)
        return torch.bmm(probs, v)

class Attention:
    def __init__(self, strategy=None):
        if strategy is None:
            strategy = Fast() if available() else Slow()
        self.install(strategy)
    def install(self, value): self.worker = value
    def score(self, q, k):
        scratch = torch.empty(q.shape[0], q.shape[1], k.shape[1])
        scores = torch.baddbmm(scratch, q, k.transpose(-1, -2))
        probs = scores.softmax(dim=-1)
        return probs
    def forward(self, hidden, external=None, **options):
        return self.worker(self, hidden, context=external, **options)
"""


BLOCK = """
from .attention_processor import Attention

class Block:
    def __init__(self, choose=False):
        self.saved = choose
        self.unit = Attention()
    def forward(self, value, side):
        return self.unit(
            value, external=side if self.saved else None)

class Root:
    def __init__(self): self.block = Block(False)
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _read(tmp_path, *, choose=False, block=BLOCK, attention=ATTENTION):
    package = Path(tmp_path) / "diffusers"
    _write(package / "__init__.py", "")
    _write(package / "models" / "__init__.py", "")
    _write(package / "models" / "attention_processor.py", attention)
    block = block.replace("Block(False)", f"Block({choose!r})")
    root_file = _write(package / "models" / "block.py", block)
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_file,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    attention_sites = tuple(
        item for item in index.construction_sites
        if any(candidate.reference.kind == "name"
               and candidate.reference.name == "Attention"
               for candidate in item.candidates))
    assert len(attention_sites) == 1
    attention_site = attention_sites[0]
    constructor = next(item for item in index.calls_in(
        attention_site.enclosing_callable) if item.span == attention_site.span)
    imported = resolve_called_import_source(
        index, bundle, "root", constructor)
    assert imported.status == "resolved"
    index = imported.index
    root_sites = tuple(
        item for item in index.construction_sites
        if item.owner.qualified_name == "Root"
        and len(item.candidates) == 1
        and item.candidates[0].symbol == attention_site.owner)
    assert len(root_sites) == 1
    root_site = root_sites[0]
    block_symbol = root_site.candidates[0].symbol
    block_frame = constructor_frame(index, canonical_construction_target(
        index, root_site, block_symbol))
    graph = resolve_owner_graph(index, block_symbol)
    invocations = resolve_addressed_invocations_in_graph(
        index, graph, graph.root.occurrence,
        resolve_container_inventory_in_graph(
            index, graph, graph.root.occurrence))
    invocation = next(item for item in invocations.addressed
                      if item.callee_owner_occurrence.sites[-1]
                      == attention_site.site_id)
    target = canonical_called_import_target(bundle, imported)
    lane = framework_attention_lane_positive_proof_in_graph(
        index, graph, invocation, canonical_import=target)
    assert lane is not None
    return framework_attention_invocation_role(index, block_frame, lane)


def test_false_constructor_branch_proves_self_input(tmp_path):
    value = _read(tmp_path).require_value()
    assert value.kind == "self"
    assert value.primary_roots == ("value",)
    assert value.context_roots == ()
    assert value.context_selection.decisions[0].decision is False


def test_true_constructor_branch_proves_distinct_context_slot(tmp_path):
    value = _read(tmp_path, choose=True).require_value()
    assert value.kind == "context_slot"
    assert value.primary_roots == ("value",)
    assert value.context_roots == ("side",)
    assert value.context_selection.decisions[0].decision is True


def test_same_formal_for_primary_and_context_is_self_not_context(tmp_path):
    block = BLOCK.replace(
        "external=side if self.saved else None", "external=value")
    value = _read(tmp_path, block=block).require_value()
    assert value.kind == "self"
    assert value.context_roots == value.primary_roots == ("value",)


def test_complete_symbol_field_formal_and_flag_renaming_is_powerless(tmp_path):
    block = (BLOCK.replace("Block", "Unit")
             .replace("Root", "Top")
             .replace("choose", "decision")
             .replace("saved", "held")
             .replace("unit", "child")
             .replace("value", "main")
             .replace("side", "other")
             .replace("block", "entry"))
    block = block.replace("Unit(False)", "Unit(True)")
    # Keep the helper's exact architecture parameter in sync.
    block = block.replace("class Top", "class Root")
    block = block.replace("self.entry = Unit(True)",
                          "self.block = Unit(True)")
    value = _read(tmp_path, choose=True, block=block).require_value()
    assert value.kind == "context_slot"
    assert value.primary_roots == ("main",)
    assert value.context_roots == ("other",)


@pytest.mark.parametrize("old,new", [
    ("side if self.saved else None", "side if runtime() else None"),
    ("self.saved = choose", "self.saved = runtime()"),
])
def test_unknown_selector_preserves_both_roles_without_choosing(
        tmp_path, old, new):
    result = _read(tmp_path, block=BLOCK.replace(old, new))
    assert result.status == "incomplete"
    value = result.require_value()
    assert value.kind == "conditional"
    assert [item.kind for item in value.alternatives] == [
        "context_slot", "self"]
    assert value.context_roots == ("side",)


def test_unknown_primary_lineage_still_fails(tmp_path):
    block = BLOCK.replace("value, external=side", "opaque, external=side")
    assert _read(tmp_path, block=block).status == "failed"


@pytest.mark.parametrize("expression,kind", [
    ("side if runtime() else side", "context_slot"),
    ("value if runtime() else None", "self"),
])
def test_unknown_selector_is_harmless_when_both_roles_are_equivalent(
        tmp_path, expression, kind):
    block = BLOCK.replace("side if self.saved else None", expression)
    result = _read(tmp_path, block=block)
    assert result.status == "resolved"
    assert result.require_value().kind == kind


def test_expanded_mapping_cannot_supply_an_omitted_context(tmp_path):
    block = BLOCK.replace(
        "value, external=side if self.saved else None",
        "value, **options")
    assert _read(tmp_path, block=block).status == "failed"


def test_dto_rejects_role_and_cross_lane_forgery(tmp_path):
    value = _read(tmp_path, choose=True).require_value()
    with pytest.raises(ValueError, match="self role"):
        replace(value, kind="self")
    with pytest.raises(ValueError, match="provenance"):
        replace(value, spans=())


def test_conditional_dto_cannot_be_laundered_as_a_decided_role(tmp_path):
    block = BLOCK.replace(
        "self.saved = choose", "self.saved = runtime()")
    value = _read(tmp_path, block=block).require_value()
    with pytest.raises(ValueError, match="determine the exact role state"):
        replace(value, kind="self")
    with pytest.raises(ValueError, match="both exact if branches"):
        replace(value, alternatives=value.alternatives[:1])


def test_constructed_lane_presence_clears_only_its_exact_outer_guard(tmp_path):
    block = BLOCK.replace(
        "return self.unit(\n            value, external=side if self.saved else None)",
        """if self.unit is not None:
            if self.saved:
                routed = side
            else:
                routed = value
            return self.unit(
                routed, external=side if self.saved else None)""")
    result = _read(tmp_path, block=block)
    value = result.require_value()
    assert value.kind == "self"
    assert value.primary_roots == ("value",)
    assert len(value.presence_decisions) == 1
    assert value.presence_decisions[0].step.test.children[0].name == "unit"


def test_sibling_presence_guard_cannot_certify_the_lane(tmp_path):
    block = BLOCK.replace(
        "self.unit = Attention()",
        "self.unit = Attention()\n        self.other = object()").replace(
        "return self.unit(\n            value, external=side if self.saved else None)",
        """if self.other is not None:
            if self.saved:
                routed = side
            else:
                routed = value
            return self.unit(
                routed, external=side if self.saved else None)""")
    assert _read(tmp_path, block=block).status == "failed"
