"""U3-F5d — candidate-equivalent dispatch storage controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.dispatch_attention_storage import (
    dispatch_attention_projection_storage_evidence,
)
from model_unfolder.evidence.execution_flow import resolve_addressed_invocations
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    from torch import nn
    from torch.nn import functional as F

    class Projection(nn.Linear):
        pass

    class Eager(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.packed = Projection(config.hidden, config.hidden * 3)
            self.out = Projection(config.hidden, config.hidden)
            self.mode = config.mode
        def split(self, packed):
            if self.mode:
                return packed, packed, packed
            else:
                return packed, packed, packed
        def forward(self, x):
            packed = self.packed(x)
            q, k, v = self.split(packed)
            mixed = F.scaled_dot_product_attention(q, k, v)
            return self.out(mixed)

    class Flash(Eager):
        def __init__(self, config):
            super().__init__(config)
            self.extra = config.extra
        def forward(self, x):
            packed = self.packed(x)
            q, k, v = self.split(packed)
            mixed = F.scaled_dot_product_attention(q, k, v)
            return self.out(mixed)

    CHOICES = {
        "eager": Eager,
        "flash": Flash,
    }

    class Cell(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.branch = CHOICES[config.implementation](config)
        def forward(self, x):
            return self.branch(x)

    class Core(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.items = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for item in self.items:
                x = item(x)
            return x

    class Wrapper(nn.Module):
        base_model_prefix = "core"
        def __init__(self, config):
            super().__init__()
            self.core = Core(config)
"""


def _pipeline(tmp_path, source=_SOURCE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    child_inventory = resolve_container_inventory(
        index, root, repeated.child_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, repeated.child_occurrence, child_inventory)
    call = next(
        item.call for item in invocations.unresolved
        if item.call.callee.name == "branch")
    return index, root, repeated.child_occurrence, call


def test_every_dispatch_candidate_independently_proves_fused_storage(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "resolved", result.failures
    assert result.value.mode == "fused_qkv"
    assert [item.candidate.candidate.symbol.qualified_name
            for item in result.value.proofs] == ["Eager", "Flash"]
    assert all(len(item.projections) == 1 for item in result.value.proofs)
    assert not hasattr(result.value, "child_occurrence")


def test_inherited_projection_requires_exact_super_constructor(tmp_path):
    source = _SOURCE.replace(
        "            super().__init__(config)\n            self.extra = config.extra",
        "            self.extra = config.extra",
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "failed"
    assert any("Flash" in item.detail for item in result.failures)


def test_non_exhaustive_split_helper_cannot_certify_three_lanes(tmp_path):
    source = _SOURCE.replace(
        """            if self.mode:
                return packed, packed, packed
            else:
                return packed, packed, packed""",
        """            if self.mode:
                return packed, packed, packed""",
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "failed"


def test_linear_subclass_that_skips_base_constructor_is_not_storage_proof(
        tmp_path):
    source = _SOURCE.replace(
        """    class Projection(nn.Linear):
        pass""",
        """    class Projection(nn.Linear):
        def __init__(self, *args, **kwargs):
            self.not_a_linear_weight = 1""",
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "failed"


def test_candidate_with_different_storage_makes_equivalence_ambiguous(tmp_path):
    source = _SOURCE.replace(
        """        def forward(self, x):
            packed = self.packed(x)
            q, k, v = self.split(packed)
            mixed = F.scaled_dot_product_attention(q, k, v)
            return self.out(mixed)

    CHOICES""",
        """        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            mixed = F.scaled_dot_product_attention(q, k, v)
            return self.out(mixed)

    CHOICES""",
        1,
    ).replace(
        """            super().__init__(config)
            self.extra = config.extra""",
        """            super().__init__(config)
            self.q = Projection(config.hidden, config.hidden)
            self.k = Projection(config.hidden, config.hidden)
            self.v = Projection(config.hidden, config.hidden)
            self.extra = config.extra""",
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_equivalent_storage_dto_rejects_missing_candidate_proof(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = dispatch_attention_projection_storage_evidence(
        index, root, parent, call)
    assert result.status == "resolved"
    with pytest.raises(ValueError):
        replace(result.value, proofs=result.value.proofs[:1])


def test_real_falcon_registry_candidates_are_unanimously_fused():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce
    from model_unfolder.evidence.repeated_child import (
        resolve_repeated_child_at_owner,
    )

    context = ParseContext.build(
        _coerce(AutoConfig.for_model("falcon").to_dict()))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    owner = root.graph.root.occurrence
    inventory = resolve_container_inventory(index, root, owner)
    repeated = resolve_repeated_child_at_owner(
        index, root, owner, inventory)
    child_inventory = resolve_container_inventory(
        index, root, repeated.child_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, repeated.child_occurrence, child_inventory)
    call = next(
        item.call for item in invocations.unresolved
        if item.call.callee.name == "self_attention")
    result = dispatch_attention_projection_storage_evidence(
        index, root, repeated.child_occurrence, call)
    assert result.status == "resolved", result.failures
    assert result.value.mode == "fused_qkv"
    assert len(result.value.proofs) == 2
