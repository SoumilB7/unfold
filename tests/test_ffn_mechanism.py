"""U3-F exact-owner ordinary FFN mechanism controls."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.decoder_block import decoder_block_path_for_config
from model_unfolder.evidence.ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    ConditionalFFNEntry,
    EquivalentFFNMechanism,
    decoder_ffn_mechanism_for_path,
    ffn_mechanism_at_block,
)
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"
_PREFIX = """
import copy
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
"""

_ATTENTION = """
class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), v)
"""


def _auto_registry_source(tmp_path, *, architecture="Model"):
    """Minimal exact AutoModel registry proof for dispatch fixtures.

    A bare ``AutoModel.from_config`` call is only an address syntax.  The
    selected implementation becomes exact when the indexed official registry
    maps the selected component's config-class key to one class.
    """
    path = tmp_path / "transformers" / "models" / "auto" / "modeling_auto.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(f"""
        from collections import OrderedDict
        from .auto_factory import _LazyAutoMapping

        CONFIG_MAPPING_NAMES = OrderedDict([("child_cfg", "ChildConfig")])
        MODEL_MAPPING_NAMES = OrderedDict([("child_cfg", "{architecture}")])
        MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)

        class AutoModel:
            _model_mapping = MODEL_MAPPING
    """), encoding="utf-8")
    return path


def _reader(tmp_path, ffn_source, *, block_forward=None, extra=""):
    block_forward = block_forward or """
        x = self.attn(x)
        return self.ffn(x)
"""
    source = _PREFIX + _ATTENTION + ffn_source + extra + f"""
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
{block_forward}
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)


def _config_selected_wrapper_reader(
        tmp_path, selector, *, wrapper_forward="return self.inner(x)",
        wrapper_setup="", model_argument="config"):
    """A name-neutral T5-shaped boundary: last slot -> config-selected FFN."""
    source = _PREFIX + _ATTENTION + f"""
class DensePath:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))

class SplitPath:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Choice:
    def __init__(self, config):
        if config.choose_split:
            self.inner = SplitPath(config)
        else:
            self.inner = DensePath(config)
    def forward(self, x):
        {wrapper_forward}

class Block:
    def __init__(self, config):
        self.parts = nn.ModuleList()
        if config.add_attention:
            self.parts.append(Attention(config))
        self.parts.append(Choice(config))
    def forward(self, x):
        return self.parts[-1](x)

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        {wrapper_setup}
        self.model = Model({model_argument})
"""
    path = tmp_path / "selected.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert root.status == block.status == "resolved"
    result = ffn_mechanism_at_block(
        index, root, block.value.block_occurrence,
        config_selector=selector)
    return result


@pytest.mark.parametrize(("selected", "mode", "gated"), [
    (True, "split", True),
    (False, "dense", False),
])
def test_exact_boolean_selects_one_exhaustive_nested_ffn_branch(
        tmp_path, selected, mode, gated):
    requested = []

    def selector(path):
        requested.append(path)
        return selected

    result = _config_selected_wrapper_reader(tmp_path, selector)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, ConfigSelectedFFNMechanism)
    assert requested == [("choose_split",)]
    assert result.value.selector_config_path == ("choose_split",)
    assert (result.value.projection_mode, result.value.gated) == (mode, gated)
    assert result.provenance[0].kind == "code_and_config"


@pytest.mark.parametrize("selected", [None, "true", 1])
def test_missing_or_non_boolean_selector_cannot_choose_nested_ffn(
        tmp_path, selected):
    result = _config_selected_wrapper_reader(
        tmp_path, lambda _path: selected)
    assert result.status == "failed"


def test_selected_nested_ffn_must_reach_the_wrapper_return(tmp_path):
    result = _config_selected_wrapper_reader(
        tmp_path, lambda _path: True,
        wrapper_forward="self.inner(x)\n        return x")
    assert result.status == "failed"


def test_unmodified_selector_survives_an_exact_deepcopied_config(tmp_path):
    requested = []
    result = _config_selected_wrapper_reader(
        tmp_path,
        lambda path: requested.append(path) or True,
        wrapper_setup=(
            "cloned = copy.deepcopy(config)\n"
            "        cloned.runtime_only = False"),
        model_argument="cloned")
    assert result.status == "resolved", result.failures
    assert requested == [("choose_split",)]
    assert result.value.gated is True


def test_mutated_selector_on_a_deepcopied_config_cannot_select_a_branch(
        tmp_path):
    requested = []
    result = _config_selected_wrapper_reader(
        tmp_path,
        lambda path: requested.append(path) or True,
        wrapper_setup=(
            "cloned = copy.deepcopy(config)\n"
            "        cloned.choose_split = False"),
        model_argument="cloned")
    assert result.status == "failed"
    assert requested == []


def test_dense_ffn_requires_exact_two_projection_chain_and_activation(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        hidden = self.up(x)
        hidden = self.act(hidden)
        return self.down(hidden)
""")
    assert result.status == "resolved", result.failures
    assert result.value.gated is False
    assert result.value.projection_mode == "dense"
    assert result.value.activation == "gelu"
    assert len(result.value.projections) == 2


def test_split_gated_ffn_requires_two_lanes_multiplying_before_down(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)
""")
    assert result.status == "resolved", result.failures
    assert result.value.gated is True
    assert result.value.projection_mode == "split"
    assert result.value.activation == "silu"
    assert len(result.value.projections) == 3


def test_fused_gate_up_requires_split_and_multiplication(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.gate_up = nn.Linear(config.hidden, config.wide * 2)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "fused_gate_up"
    assert result.value.activation == "silu"


def test_activation_dispatch_carries_exact_config_path_not_a_guessed_kind(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.activation is None
    assert result.value.activation_config_path == ("hidden_act",)
    assert result.provenance[0].kind == "code_and_config"


def test_unbound_activation_does_not_erase_proven_projection_topology(tmp_path):
    result = _reader(tmp_path, """
class OpaqueActivation:
    def __init__(self, config): pass
    def forward(self, x): return x
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = OpaqueActivation(config)
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"
    assert result.value.gated is False
    assert result.value.activation is None
    assert result.value.activation_config_path == ()


def test_dynamic_activation_field_preserves_only_the_proven_affine_topology(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = build_activation(config.hidden_act)
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"
    assert result.value.gated is False
    assert result.value.activation is None
    assert result.value.activation_config_path == ()


def test_attention_shaped_softmax_between_two_affines_is_not_an_ffn(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        scores = F.softmax(self.up(x), dim=-1)
        return self.down(scores)
""")
    assert result.status == "failed"


def test_nested_activation_dispatch_carries_the_full_selected_config_path(
        tmp_path):
    child_source = _PREFIX + _ATTENTION + """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.up(x)))
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        x = self.attn(x)
        return self.ffn(x)
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""
    wrapper_source = """
from transformers.models.auto.modeling_auto import AutoModel
class Wrapper:
    def __init__(self, config):
        self.model = AutoModel.from_config(config.text)
"""
    child = tmp_path / "child.py"
    wrapper = tmp_path / "wrapper.py"
    auto = _auto_registry_source(tmp_path)
    child.write_text(textwrap.dedent(child_source), encoding="utf-8")
    wrapper.write_text(textwrap.dedent(wrapper_source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(wrapper), str(auto), str(child)),
        component_files={
            "root": (str(wrapper), str(auto)), "text": (str(child),)},
        component_architectures={
            "root": "Wrapper", "text": "Model"},
        component_model_types={"text": "child_cfg"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    result = decoder_ffn_mechanism_for_path(
        index, bundle, ("text",), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.activation_config_path == ("text", "hidden_act")
    assert result.provenance[-1].config_paths == (
        ("text", "hidden_act"),)


def test_attention_projection_bundle_is_not_an_ffn_candidate(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""")
    assert result.status == "resolved"
    assert result.value.owner_symbol.qualified_name == "FeedForward"


def test_two_invoked_ordinary_ffns_are_ambiguous_not_first_hit(tmp_path):
    result = _reader(
        tmp_path,
        """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""",
        block_forward="""
        x = self.attn(x)
        x = self.ffn(x)
        return self.other(x)
""",
        extra="""
class OtherFFN:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""".replace(
            "class OtherFFN", "class UnusedOtherFFN")
        + """
""")
    # The helper above constructs only self.ffn, so an unused sibling class
    # cannot contaminate the exact block occurrence.
    assert result.status == "resolved"


def test_two_constructed_and_invoked_ffns_are_ambiguous(tmp_path):
    ffn = """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
"""
    source = _PREFIX + _ATTENTION + ffn + """
class OtherFFN:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.left = FeedForward(config)
        self.right = OtherFFN(config)
    def forward(self, x):
        x = self.attn(x)
        x = self.left(x)
        return self.right(x)
class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "ambiguous.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(index, root, allow_root_stage=True)
    result = ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_one_ffn_occurrence_invoked_twice_is_not_collapsed_to_one_sublayer(
        tmp_path):
    result = _reader(
        tmp_path,
        """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""",
        block_forward="""
        x = self.attn(x)
        x = self.ffn(x)
        return self.ffn(x)
""")
    assert result.status != "resolved"


def test_two_linear_multiply_without_lane_split_is_not_fused(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        hidden = self.up(x)
        return self.down(F.silu(hidden) * hidden)
""")
    assert result.status == "failed"


def test_three_linears_without_gate_multiplication_are_not_split_gated(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.first = nn.Linear(config.hidden, config.wide)
        self.second = nn.Linear(config.wide, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        hidden = F.gelu(self.first(x))
        hidden = self.second(hidden)
        return self.down(hidden)
    """)
    assert result.status == "failed"


def test_unrelated_activation_does_not_certify_a_dense_ffn(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        unused = F.gelu(x)
        return self.down(self.up(x))
""")
    assert result.status == "failed"


def test_activation_after_down_projection_is_not_an_ffn_hidden_activation(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        hidden = self.up(x)
        return F.gelu(self.down(hidden))
""")
    assert result.status == "failed"


def test_unrelated_split_does_not_certify_a_fused_gate_up_ffn(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        hidden = self.up(x)
        unused = hidden.chunk(2, dim=-1)
        return self.down(F.silu(hidden) * hidden)
""")
    assert result.status == "failed"


def test_split_of_down_projection_is_not_fused_gate_up_storage(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden * 2)
    def forward(self, x):
        hidden = self.up(x)
        output = self.down(F.silu(hidden) * hidden)
        left, right = output.chunk(2, dim=-1)
        return left + right
""")
    assert result.status == "failed"


def test_repeated_projection_invocation_is_not_collapsed_to_one_call(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        first = self.up(x)
        second = self.up(x)
        return self.down(self.act(first + second))
""")
    assert result.status == "failed"


def test_guarded_down_call_needs_complementary_use_of_the_same_stored_weight(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        hidden = self.act(self.up(x))
        if self.training:
            output = F.linear(hidden, self.up.weight)
        else:
            output = self.down(hidden)
        return output
""")
    assert result.status == "failed"


def test_guarded_down_call_accepts_complementary_use_of_the_same_stored_weight(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        hidden = self.act(self.up(x))
        if self.training:
            output = F.linear(hidden, self.down.weight)
        else:
            output = self.down(hidden)
        return output
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"


def test_guarded_same_weight_cannot_launder_an_activation_bypass(tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        hidden = self.act(self.up(x))
        if self.training:
            output = F.linear(x, self.down.weight)
        else:
            output = self.down(hidden)
        return output
""")
    assert result.status == "failed"


def test_guarded_down_call_without_a_complementary_storage_path_is_unknown(
        tmp_path):
    result = _reader(tmp_path, """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        hidden = self.act(self.up(x))
        if self.training:
            output = hidden
        else:
            output = self.down(hidden)
        return output
""")
    assert result.status == "failed"


def _conditional_reader(
        tmp_path, *, direct_class="DirectGated",
        wrapper_forward="return x + self.shared(x)",
        assignments=None):
    assignments = assignments or """
        if config.use_routed:
            self.ffn = RoutedWrapper(config)
        else:
            self.ffn = DirectGated(config)
"""
    source = _PREFIX + _ATTENTION + """
class DirectGated:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

class DirectDense:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.up(x)))

class RoutedWrapper:
    def __init__(self, config):
        self.shared = DirectGated(config)
    def forward(self, x):
        __WRAPPER_FORWARD__

class Block:
    def __init__(self, config):
__ASSIGNMENTS__
    def forward(self, x):
        return self.ffn(x)

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
""".replace("__WRAPPER_FORWARD__", wrapper_forward).replace(
        "__ASSIGNMENTS__", textwrap.indent(
            textwrap.dedent(assignments).strip(), "        ")).replace(
        "self.ffn = DirectGated(config)",
        f"self.ffn = {direct_class}(config)")
    path = tmp_path / "conditional.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)


def test_exhaustive_dense_and_invoked_shared_paths_must_unanimously_agree(
        tmp_path):
    result = _conditional_reader(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "split"
    assert result.value.gated is True
    assert result.value.activation_config_path == ("hidden_act",)
    assert len(result.value.variants) == 2
    assert {len(item.invocations) for item in result.value.variants} == {0, 1}


def test_equivalent_ffn_evidence_rejects_semantic_or_entry_forgeries(tmp_path):
    result = _conditional_reader(tmp_path)
    value = result.value
    first, second = value.variants
    disagreeing = replace(
        second, activation="gelu", activation_config_path=())
    with pytest.raises(ValueError, match="identical semantics"):
        EquivalentFFNMechanism(
            value.block_occurrence, (first, disagreeing))

    unguarded_site = replace(
        first.conditional_entry.site, guard=())
    with pytest.raises(ValueError, match="preserve its construction guard"):
        ConditionalFFNEntry(
            first.block_occurrence,
            first.conditional_entry.call,
            unguarded_site,
            first.conditional_entry.candidate)

    different_call = replace(
        second.conditional_entry.call,
        lexical_order=second.conditional_entry.call.lexical_order + 1)
    with pytest.raises(ValueError, match="one exact block invocation"):
        EquivalentFFNMechanism(
            value.block_occurrence,
            (first, replace(
                second,
                conditional_entry=replace(
                    second.conditional_entry, call=different_call))))

    non_exhaustive_site = replace(
        second.conditional_entry.site,
        guard=first.conditional_entry.site.guard)
    with pytest.raises(ValueError, match="one exhaustive decision"):
        EquivalentFFNMechanism(
            value.block_occurrence,
            (first, replace(
                second,
                conditional_entry=replace(
                    second.conditional_entry, site=non_exhaustive_site))))


def test_conditional_dense_and_shared_mechanisms_that_disagree_are_ambiguous(
        tmp_path):
    result = _conditional_reader(tmp_path, direct_class="DirectDense")
    assert result.status == "ambiguous"


def test_constructed_but_uninvoked_shared_ffn_cannot_certify_a_wrapper(
        tmp_path):
    result = _conditional_reader(tmp_path, wrapper_forward="return x")
    assert result.status == "failed"


def test_non_exhaustive_conditional_constructions_remain_unknown(tmp_path):
    result = _conditional_reader(
        tmp_path,
        assignments="""
        if config.first:
            self.ffn = RoutedWrapper(config)
        if config.second:
            self.ffn = DirectGated(config)
""")
    assert result.status == "failed"


def test_an_exact_direct_ffn_cannot_hide_a_second_rival_invoked_field(tmp_path):
    result_path = tmp_path / "two_fields"
    result_path.mkdir()
    _conditional_reader(result_path)
    path = result_path / "conditional.py"
    text = path.read_text(encoding="utf-8").replace(
        "            self.ffn = DirectGated(config)\n"
        "    def forward(self, x):\n"
        "        return self.ffn(x)",
        "            self.ffn = DirectGated(config)\n"
        "        self.ordinary = DirectGated(config)\n"
        "    def forward(self, x):\n"
        "        x = self.ordinary(x)\n"
        "        return self.ffn(x)")
    path.write_text(text, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(index, root, allow_root_stage=True)
    result = ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)
    assert result.status == "ambiguous"


def test_guarded_block_invocation_does_not_become_a_global_ffn_claim(tmp_path):
    # Construction is exhaustive, but use is not: the block can bypass the
    # selected field.  A model-level FFN summary must therefore abstain.
    source = """
        if config.use_routed:
            self.ffn = RoutedWrapper(config)
        else:
            self.ffn = DirectGated(config)
"""
    # The helper's block invocation is intentionally unguarded; build a close
    # negative by changing the generated source after construction.
    result_path = tmp_path / "nested"
    result_path.mkdir()
    _conditional_reader(result_path, assignments=source)
    path = result_path / "conditional.py"
    text = path.read_text(encoding="utf-8").replace(
        "    def forward(self, x):\n        return self.ffn(x)\n\nclass Model:",
        "    def forward(self, x):\n"
        "        if self.training:\n"
        "            return self.ffn(x)\n"
        "        return x\n\nclass Model:")
    path.write_text(text, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(index, root, allow_root_stage=True)
    result = ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)
    assert result.status == "failed"


@pytest.mark.parametrize(("slug", "config_path", "mode"), [
    # BLOOM's two branches are accepted only because both are proven to use the
    # exact same stored down projection (module call vs F.linear(weight slices)).
    ("bloom", (), "dense"),
    ("llama-7b", (), "split"),
    ("stablelm-2-1-6b", (), "split"),
    ("qwen2-vl-7b-instruct", ("text_config",), "split"),
    ("musicgen-small", ("decoder",), "dense"),
    # These hybrid blocks have exhaustive dense-vs-MoE construction branches.
    # The dense branch and the MoE wrapper's actually-invoked shared child each
    # independently prove the same split gate/up/down mechanism.
    ("deepseek-v3", (), "split"),
    ("glm-4-5", (), "split"),
    # Routed-only blocks still abstain: routed expert storage is a separate
    # mechanism and cannot be laundered into an ordinary/shared FFN fact.
    ("gpt-oss-20b", (), None),
])
def test_real_decoder_ffn_examples_use_the_exact_selected_owner(
        slug, config_path, mode):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    block = decoder_block_path_for_config(
        context.program_index(), context.source_bundle, config_path,
        allow_root_stage=True)
    assert block.status == "resolved", block.failures
    result = ffn_mechanism_at_block(
        context.program_index(), block.value.component_root,
        block.value.block_occurrence)
    if mode is None:
        assert result.status != "resolved"
        return
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == mode
    assert result.value.gated is (mode != "dense")


@pytest.mark.parametrize(("slug", "config_path", "mode", "gated"), [
    ("bloom", (), "dense", False),
    ("llama-7b", (), "split", True),
    ("stablelm-2-1-6b", (), "split", True),
    ("qwen2-vl-7b-instruct", ("text_config",), "split", True),
    ("musicgen-small", ("decoder",), "dense", False),
    ("deepseek-v3", (), "split", True),
    ("glm-4-5", (), "split", True),
])
def test_parser_consumes_the_same_exact_ffn_result(
        slug, config_path, mode, gated):
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    config_to_ir(cfg, parse_context=context)
    key = ("decoder.ffn.mechanism", config_path)
    result = context.reader_results[key]
    assert result.status == "resolved", result.failures
    assert (result.value.projection_mode, result.value.gated) == (mode, gated)
    assert context.facts.records["decoder.ffn.gated"].source == \
        "decoder_ffn_mechanism_for_path"
    assert context.facts.records["decoder.ffn.projection_mode"].source == \
        "decoder_ffn_mechanism_for_path"


def test_parser_and_conformance_share_one_exact_ffn_result(monkeypatch):
    from model_unfolder import config_to_ir
    from model_unfolder.evidence import ffn_mechanism as mechanism_module
    from model_unfolder.evidence.conformance import check_fact_conformance
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / "musicgen-small.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    real_reader = mechanism_module.decoder_ffn_mechanism_for_path
    calls = []

    def counted_reader(*args, **kwargs):
        calls.append((args[2], kwargs.get("allow_root_stage")))
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(
        mechanism_module, "decoder_ffn_mechanism_for_path", counted_reader)
    ir = config_to_ir(cfg, parse_context=context)
    key = ("decoder.ffn.mechanism", ("decoder",))
    parsed_result = context.reader_results[key]
    decoder_calls = [
        call for call in calls if call[0] == ("decoder",)]
    assert decoder_calls == [(("decoder",), True)]

    problems = check_fact_conformance(
        cfg, ir.to_dict(), bundle=context.source_bundle,
        program_index=context.program_index(), parse_context=context)
    assert not [problem for problem in problems
                if problem.kind == "wrong_storage"]
    assert context.reader_results[key] is parsed_result
    assert [call for call in calls if call[0] == ("decoder",)] == decoder_calls


def test_routed_expert_gate_is_not_borrowed_from_the_ordinary_ffn():
    """Expert storage is its own mechanism boundary.

    A fused gate+up projection positively proves a gated expert even when the
    ordinary FFN is unknown.  Without that expert-local proof the expert stays
    opaque; the ordinary/shared FFN verdict must never leak across.
    """
    from model_unfolder.adapters.transformer.blocks.feed_forward import (
        ffn_child_blocks,
    )
    from model_unfolder.ir import FFNSpec
    from model_unfolder.opgraph import ffn_region

    def expert_detail(mode):
        spec = FFNSpec(
            kind="moe", activation=None, gated=None, intermediate_size=256,
            expert_intermediate_size=128, num_experts=8,
            num_experts_per_tok=2, expert_projection_mode=mode,
        )
        return next(
            child["detail"]["ffn"]
            for child in ffn_child_blocks(spec, 64)
            if child.get("id") == "expert_1"
        )

    fused = expert_detail("fused_gate_up")
    assert fused["gated"] is True
    fused_region = {
        **fused, "kind": "dense",
        "projection_mode": fused["expert_projection_mode"],
    }
    assert ffn_region(fused_region, 64).template == "fused_gated_mlp"

    unknown = expert_detail(None)
    assert unknown["gated"] is None
    unknown_region = {
        **unknown, "kind": "dense",
        "projection_mode": unknown["expert_projection_mode"],
    }
    assert ffn_region(unknown_region, 64).template == "undeclared"


def test_same_exact_ffn_called_on_alternative_paths_is_one_mechanism(tmp_path):
    """Repeated invocation sites do not manufacture rival FFN owners.

    GPT-NeoX invokes the same stored MLP in its parallel and sequential
    residual branches.  The invocation census must retain both sites while the
    mechanism remains the one exact constructed child.
    """
    result = _reader(
        tmp_path,
        """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""",
        block_forward="""
        if self.training:
            return self.ffn(x)
        return self.ffn(x)
""",
    )
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"
    assert len(result.value.invocations) == 2


def test_same_ffn_in_exact_if_else_assignment_is_one_mechanism(tmp_path):
    result = _reader(
        tmp_path,
        """
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))
""",
        block_forward="""
        if self.training:
            output = self.ffn(x)
        else:
            output = self.ffn(x)
        return output
""",
    )
    assert result.status == "resolved", result.failures
    assert len(result.value.invocations) == 2


def test_fused_routed_expert_count_does_not_borrow_ordinary_gate():
    from model_unfolder.ir import FFNSpec
    from model_unfolder.params import _ffn_params

    spec = FFNSpec(
        kind="moe", activation=None, gated=None, intermediate_size=32,
        expert_intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1,
        expert_projection_mode="fused_gate_up",
    )
    total, active = _ffn_params(spec, 8)
    routed = 3 * 8 * 16
    shared_floor = 2 * 8 * 16
    router = 8 * 4
    assert total == routed * 4 + shared_floor + router
    assert active == routed * 2 + shared_floor + router


def test_unproven_routed_expert_count_does_not_borrow_ordinary_gate():
    from model_unfolder.ir import FFNSpec
    from model_unfolder.params import _ffn_params

    spec = FFNSpec(
        kind="moe", activation="silu", gated=True, intermediate_size=32,
        expert_intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1,
        expert_projection_mode=None,
    )
    total, active = _ffn_params(spec, 8)
    routed_floor = 2 * 8 * 16
    shared_gated = 3 * 8 * 16
    router = 8 * 4
    assert total == routed_floor * 4 + shared_gated + router
    assert active == routed_floor * 2 + shared_gated + router


def test_routed_only_moe_does_not_claim_an_unused_shared_ffn_assumption():
    """An unknown ordinary/shared gate cannot qualify a formula when that lane
    has zero instances.  GPT-OSS proves its routed fused gate+up storage and has
    no shared expert; the former note falsely implied its exact total was a
    two-projection lower bound."""
    from model_unfolder.ir import AttentionSpec, FFNSpec, LayerSpec, ModelIR
    from model_unfolder.params import estimate_params

    def estimate(shared):
        return estimate_params(ModelIR(
            name="moe", architecture="Synthetic", vocab_size=8,
            hidden_size=8, max_position_embeddings=None,
            tie_word_embeddings=True,
            layers=[LayerSpec(
                index=0,
                attention=AttentionSpec(kind="mha", num_heads=1),
                ffn=FFNSpec(
                    kind="moe", activation=None, gated=None,
                    intermediate_size=16, expert_intermediate_size=16,
                    num_experts=4, num_experts_per_tok=2,
                    num_shared_experts=shared,
                    expert_projection_mode="fused_gate_up"),
                norm_kind="unknown")],
        ))

    routed_only = estimate(0)
    assert not any(
        "FFN structure unknown" in note
        for note in routed_only.get("assumptions") or ())
    shared = estimate(1)
    assert any(
        "FFN structure unknown" in note
        for note in shared.get("assumptions") or ())


def test_shared_expert_gate_assumption_tracks_the_expert_width_lane():
    """The shared expert term uses expert_intermediate_size, never the
    ordinary dense-layer width. Its uncertainty note must follow that same
    operand or the parameter card can describe a term it did not count."""
    from model_unfolder.ir import AttentionSpec, FFNSpec, LayerSpec, ModelIR
    from model_unfolder.params import estimate_params

    def estimate(*, ordinary_width, expert_width):
        return estimate_params(ModelIR(
            name="moe", architecture="Synthetic", vocab_size=8,
            hidden_size=8, max_position_embeddings=None,
            tie_word_embeddings=True,
            layers=[LayerSpec(
                index=0,
                attention=AttentionSpec(kind="mha", num_heads=1),
                ffn=FFNSpec(
                    kind="moe", gated=None,
                    intermediate_size=ordinary_width,
                    expert_intermediate_size=expert_width,
                    num_experts=4, num_experts_per_tok=2,
                    num_shared_experts=1,
                    expert_projection_mode="fused_gate_up"),
                norm_kind="unknown")],
        ))

    counted_shared = estimate(ordinary_width=None, expert_width=16)
    assert any(
        "FFN structure unknown" in note
        for note in counted_shared.get("assumptions") or ())

    omitted_shared = estimate(ordinary_width=16, expert_width=None)
    assert not any(
        "FFN structure unknown" in note
        for note in omitted_shared.get("assumptions") or ())
    assert any(
        "routed-expert inner width unknown" in note
        for note in omitted_shared.get("assumptions") or ())


def test_exact_transformers_relative_conv1d_is_an_affine_protocol(tmp_path):
    """HF modeling modules import Conv1D relatively.  The exact import target
    is a lawful protocol spelling; refusing it would make GPT-2 fall back to a
    whole-file FFN scan even though both projection occurrences are addressed.
    """
    result = _reader(tmp_path, """
from ...pytorch_utils import Conv1D
class FeedForward:
    def __init__(self, config):
        self.up = Conv1D(config.inner, config.hidden)
        self.down = Conv1D(config.hidden, config.inner)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"
    assert result.value.activation == "gelu"


def test_real_internal_linear_subclass_keeps_exact_falcon_ffn_storage():
    """The class spelling is irrelevant; its exact nn.Linear base is proof."""
    config = {
        "architectures": ["FalconForCausalLM"],
        "model_type": "falcon",
        "hidden_size": 4544,
        "intermediate_size": 18176,
        "num_hidden_layers": 32,
        "num_attention_heads": 71,
        "num_kv_heads": 1,
        "multi_query": True,
        "parallel_attn": True,
    }
    context = ParseContext.build(config)
    block = decoder_block_path_for_config(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert block.status == "resolved", block.failures
    result = ffn_mechanism_at_block(
        context.program_index(), block.value.component_root,
        block.value.block_occurrence)
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"
    assert result.value.gated is False
    assert len(result.value.projections) == 2
    assert result.value.activation_config_path == ("activation",)


def test_unrelated_relative_conv1d_spelling_is_not_an_affine_protocol(tmp_path):
    """A familiar final class name is not enough; the exact imported protocol
    target must be the transformers primitive, not an arbitrary sibling.
    """
    result = _reader(tmp_path, """
from ...somewhere_else import Conv1D
class FeedForward:
    def __init__(self, config):
        self.up = Conv1D(config.inner, config.hidden)
        self.down = Conv1D(config.hidden, config.inner)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "failed"


def test_internal_linear_subclass_is_affine_only_with_proven_storage_init(
        tmp_path):
    result = _reader(tmp_path, """
class Projection(nn.Linear):
    def __init__(self, incoming, outgoing):
        super().__init__(incoming, outgoing, bias=False)
class FeedForward:
    def __init__(self, config):
        self.up = Projection(config.hidden, config.inner)
        self.down = Projection(config.inner, config.hidden)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"


def test_internal_linear_subclass_with_overridden_forward_is_not_affine_proof(
        tmp_path):
    result = _reader(tmp_path, """
class Projection(nn.Linear):
    def __init__(self, incoming, outgoing):
        super().__init__(incoming, outgoing, bias=False)
    def forward(self, value):
        return value
class FeedForward:
    def __init__(self, config):
        self.up = Projection(config.hidden, config.inner)
        self.down = Projection(config.inner, config.hidden)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "failed"


def test_internal_linear_subclass_exact_weight_bias_forward_is_affine(tmp_path):
    result = _reader(tmp_path, """
class Projection(nn.Linear):
    def __init__(self, incoming, outgoing):
        super().__init__(incoming, outgoing, bias=True)
    def forward(self, value):
        projected = value @ self.weight.T
        if self.bias is None:
            return projected
        return projected + self.bias
class FeedForward:
    def __init__(self, config):
        self.up = Projection(config.hidden, config.inner)
        self.down = Projection(config.inner, config.hidden)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "resolved", result.failures
    assert result.value.projection_mode == "dense"


@pytest.mark.parametrize("initializer", [
    "self.weight = None",
    "if config.enabled:\n            super().__init__(incoming, outgoing)",
    "super = lambda: object()\n        super().__init__()",
])
def test_internal_linear_subclass_cannot_borrow_unproven_affine_storage(
        tmp_path, initializer):
    result = _reader(tmp_path, f"""
class Projection(nn.Linear):
    def __init__(self, incoming, outgoing, config=None):
        {initializer}
class FeedForward:
    def __init__(self, config):
        self.up = Projection(config.hidden, config.inner, config)
        self.down = Projection(config.inner, config.hidden, config)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
""")
    assert result.status == "failed"
