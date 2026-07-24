"""U3-F5a — exact attention-child mechanism controls."""
from __future__ import annotations

import json
import pathlib
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention_child import attention_child_evidence
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class DefinitelyAnMLP:
        def __init__(self, config):
            self.up = nn.Linear(config.hidden, config.wide)
            self.down = nn.Linear(config.wide, config.hidden)
        def forward(self, x):
            return self.down(F.gelu(self.up(x)))

    class DefinitelyAnFFN:
        def __init__(self, config):
            self.q = nn.Linear(config.hidden, config.hidden)
            self.k = nn.Linear(config.hidden, config.hidden)
            self.v = nn.Linear(config.hidden, config.hidden)
        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)

    class Unit:
        def __init__(self, config):
            self.first = DefinitelyAnFFN(config)
            self.second = DefinitelyAnMLP(config)
        def forward(self, x):
            x = self.first(x)
            return self.second(x)

    class Core:
        def __init__(self, config):
            self.units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])
        def forward(self, x):
            for unit in self.units:
                x = unit(x)
            return x

    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _pipeline(tmp_path, source=_SOURCE, *, architecture="Shell"):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": architecture},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    assert repeated.status == "resolved"
    return bundle, index, root, repeated


def test_attention_math_wins_over_misleading_class_and_field_names(tmp_path):
    _, index, root, repeated = _pipeline(tmp_path)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.child_occurrence.sites[-1].owner.qualified_name == "Unit"
    assert result.value.compute.protocol == "dot_softmax"
    assert result.value.compute.child_symbol.qualified_name == "DefinitelyAnFFN"
    assert result.provenance[0].spans


def test_complete_class_field_and_local_rename_changes_no_mechanism(tmp_path):
    renamed = (_SOURCE
        .replace("DefinitelyAnMLP", "Mixer")
        .replace("DefinitelyAnFFN", "Projection")
        .replace("Unit", "Cell")
        .replace("Core", "Engine")
        .replace("Shell", "Wrapper")
        .replace('base_model_prefix = "core"', 'base_model_prefix = "engine"')
        .replace("self.core =", "self.engine =")
        .replace("self.units", "self.items")
        .replace("for unit in self.items", "for item in self.items")
        .replace("x = unit(x)", "x = item(x)"))
    _, index, root, repeated = _pipeline(
        tmp_path, renamed, architecture="Wrapper")
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.compute.child_symbol.qualified_name == "Projection"


@pytest.mark.parametrize("body", [
    "return F.softmax(x, dim=-1)",
    "return torch.matmul(x, x)",
])
def test_half_of_the_protocol_is_not_attention_proof(tmp_path, body):
    source = _SOURCE.replace(
        """            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        f"            {body}",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_exact_sdpa_protocol_is_positive_without_name_markers(tmp_path):
    source = _SOURCE.replace(
        """            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        """            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            return F.scaled_dot_product_attention(q, k, v)""",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.compute.protocol == "scaled_dot_product_attention"


def test_exact_bound_fallback_function_is_followed(tmp_path):
    source = _SOURCE.replace(
        "    class DefinitelyAnMLP:",
        """    def select(fallback):
        return fallback

    def local_kernel(module, q, k, v):
        weights = F.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(weights, v)

    class DefinitelyAnMLP:""",
    ).replace(
        """            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        """            operation = select(local_kernel)
            return operation(self, q, k, v)""",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.compute.callable_symbol.qualified_name == "local_kernel"


def test_direct_exact_free_helper_is_followed_without_name_semantics(tmp_path):
    source = _SOURCE.replace(
        "    class DefinitelyAnMLP:",
        """    def calculate(a, b, c):
        weights = F.softmax(torch.matmul(a, b), dim=-1)
        return torch.matmul(weights, c)

    class DefinitelyAnMLP:""",
    ).replace(
        """            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        "            return calculate(q, k, v)",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert result.value.compute.callable_symbol.qualified_name == "calculate"


def test_arbitrary_resolver_argument_does_not_prove_a_fallback(tmp_path):
    source = _SOURCE.replace(
        "    class DefinitelyAnMLP:",
        """    def choose(candidate):
        return lambda *args: args[1]

    def hidden_kernel(module, q, k, v):
        weights = F.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(weights, v)

    class DefinitelyAnMLP:""",
    ).replace(
        """            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        """            operation = choose(hidden_kernel)
            return operation(self, q, k, v)""",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_guarded_compute_is_not_promoted_to_unconditional_attention(tmp_path):
    source = _SOURCE.replace(
        """            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        """            if self.training:
                weights = F.softmax(torch.matmul(q, k), dim=-1)
                return torch.matmul(weights, v)
            return q""",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_unreferenced_attention_helper_cannot_launder_an_mlp(tmp_path):
    source = _SOURCE.replace(
        "    class DefinitelyAnMLP:",
        """    def unused_kernel(q, k, v):
        weights = F.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(weights, v)

    class DefinitelyAnMLP:""",
    ).replace(
        """            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        "            return self.q(x)",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_two_code_proven_attention_children_are_ambiguity(tmp_path):
    source = _SOURCE.replace(
        "self.second = DefinitelyAnMLP(config)",
        "self.second = DefinitelyAnFFN(config)",
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_sibling_owner_attention_cannot_clear_requested_block(tmp_path):
    source = _SOURCE.replace(
        "    class Core:",
        """    class SiblingAttention:
        def __init__(self, config):
            pass
        def forward(self, x):
            weights = F.softmax(torch.matmul(x, x), dim=-1)
            return torch.matmul(weights, x)

    class Sibling:
        def __init__(self, config):
            self.attention = SiblingAttention(config)
        def forward(self, x):
            return self.attention(x)

    class Core:""",
    ).replace(
        "            self.second = DefinitelyAnMLP(config)",
        """            self.second = DefinitelyAnMLP(config)
            self.sibling = Sibling(config)""",
    )
    # Remove the requested block's compute while leaving a sibling with it.
    source = source.replace(
        """            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            weights = F.softmax(torch.matmul(q, k), dim=-1)
            return torch.matmul(weights, v)""",
        "            return self.q(x)",
        1,
    )
    _, index, root, repeated = _pipeline(tmp_path, source)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "failed"


def test_cross_owner_or_cross_index_occurrence_is_rejected(tmp_path):
    _, index, root, repeated = _pipeline(tmp_path)
    result = attention_child_evidence(
        index, root, root.graph.root.occurrence)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"
    with pytest.raises(TypeError):
        attention_child_evidence(index, root, object())


@pytest.mark.parametrize("slug", [
    "bloom",
    "deepseek-v3",
    "gemma-2-2b-it",
    "gpt-oss-20b",
    "llama-7b",
    "olmo-2-1124-7b",
    "qwen3-8b",
    "stablelm-2-1-6b",
])
def test_real_decoder_block_attention_controls(slug):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads((corpus / f"{slug}.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    result = attention_child_evidence(
        index, root, repeated.child_occurrence)
    assert result.status == "resolved"
    assert root.graph.node_for(result.value.child_occurrence) is not None
    assert result.value.compute.protocol in {
        "dot_softmax", "scaled_dot_product_attention"}


def test_nested_qwen2_vl_text_stack_remains_outside_this_boundary():
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    data = json.loads(
        (corpus / "qwen2-vl-7b-instruct.json").read_text())
    context = ParseContext.build(_coerce(data["config"]))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    assert repeated.status == "incomplete"
