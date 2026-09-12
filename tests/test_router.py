"""U7 exact-owner router policy controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.router import (
    RouterOwnerAddress,
    decoder_router_selection_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


_DEEPSEEK = """
scores = logits.sigmoid()
choice = scores + self.bias
group_scores = choice.view(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group).reshape(-1, self.num_experts)
choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
idx = torch.topk(choice, k=self.top_k, dim=-1)[1]
weights = scores.gather(1, idx)
if self.normalize:
    weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
weights = weights * self.scale
return idx, weights
"""

_GPT_OSS = """
values, idx = torch.topk(logits, self.top_k)
weights = F.softmax(values, dim=-1)
return idx, weights
"""

_DBRX = """
scores = F.softmax(logits, dim=-1)
weights, idx = torch.topk(scores, self.top_k)
if self.p is not None:
    weights = weights / torch.norm(weights, p=self.p, dim=-1, keepdim=True)
return idx, weights
"""

_MIXTRAL = """
scores = F.softmax(logits, dim=-1)
weights, idx = torch.topk(scores, self.top_k)
weights = weights / weights.sum(dim=-1, keepdim=True)
return idx, weights
"""

_SPARSE_MIXER = """
threshold, selected = logits.max(dim=-1, keepdim=True)
masked = logits.masked_fill(logits < threshold, float("-inf"))
weights = torch.softmax(masked, dim=-1)
weights = weights.gather(dim=-1, index=selected)
remainder = torch.scatter(logits, -1, selected, float("-inf"))
threshold_2, selected_2 = remainder.max(dim=-1, keepdim=True)
masked_2 = remainder.masked_fill(remainder < threshold_2, float("-inf"))
weights_2 = torch.softmax(masked_2, dim=-1)
weights_2 = weights_2.gather(dim=-1, index=selected_2)
return (
    torch.concat((selected, selected_2), dim=-1),
    torch.concat((weights, weights_2), dim=-1),
)
"""

_BRANCHED_ROUTER = """
scores = F.softmax(logits, dim=-1)
if self.policy == "grouped":
    group_scores = scores.view(-1, self.n_group, self.num_experts // self.n_group).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group).reshape(-1, self.num_experts)
    selected_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    weights, idx = torch.topk(selected_scores, self.top_k)
else:
    weights, idx = torch.topk(scores, self.top_k)
return idx, weights
"""


def _source(policy, *, sibling="", route_field="route"):
    policy = textwrap.indent(textwrap.dedent(policy).strip(), " " * 8)
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

class Experts(nn.Module):
    def __init__(self, config):
        self.count = config.num_experts
        self.width = config.intermediate
        self.hidden = config.hidden
        self.fused = nn.Parameter(torch.empty(self.count, 2 * self.width, self.hidden))
        self.down = nn.Parameter(torch.empty(self.count, self.hidden, self.width))
    def forward(self, hidden, routes, weights):
        out = torch.zeros_like(hidden)
        for expert in routes:
            gate, up = F.linear(hidden, self.fused[expert]).chunk(2, dim=-1)
            mixed = F.silu(gate) * up
            value = F.linear(mixed, self.down[expert]) * weights
            out.index_add_(0, expert, value)
        return out

class Route(nn.Module):
    def __init__(self, config):
        self.experts = Experts(config)
        self.gate = nn.Linear(config.hidden, config.num_experts)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.normalize = config.normalize
        self.p = config.p
        self.scale = config.scale
        self.policy = config.policy
        self.bias = nn.Parameter(torch.zeros(config.num_experts))
    def choose(self, logits):
{policy}
    def forward(self, x):
        idx, weights = self.choose(self.gate(x))
        return self.experts(x, idx, weights)

{sibling}

class Attention(nn.Module):
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        return self.q(x)

class Block(nn.Module):
    def __init__(self, config):
        self.attn = Attention(config)
        self.{route_field} = Route(config)
    def forward(self, x):
        x = self.attn(x)
        return self.{route_field}(x)

class Model(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper(nn.Module):
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""


def _read(tmp_path, policy, *, config_selector=None, **kwargs):
    return _read_source(
        tmp_path, _source(policy, **kwargs), config_selector=config_selector)


def _read_source(tmp_path, source, *, config_selector=None):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    return decoder_router_selection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=config_selector)


def test_grouped_bias_router_carries_only_code_bound_operands(tmp_path):
    result = _read(tmp_path, _DEEPSEEK)
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.selection_kind == "topk"
    assert (value.scoring_fn, value.scoring_before_topk) == ("sigmoid", True)
    assert value.score_source_kind == "affine"
    assert len(value.score_source_calls) == 1
    assert value.expert_count_path == ("num_experts",)
    assert value.expert_count_spans
    assert value.selection_count_path == ("top_k",)
    assert value.selection_count_literal is None
    assert value.bias_correction is True
    assert value.group_score_kind == "top2_sum"
    assert value.group_count_path == ("n_group",)
    assert value.topk_group_path == ("topk_group",)
    assert (value.normalization_kind, value.normalization_path) == (
        "sum", ("normalize",))
    assert value.scale_path == ("scale",)


def test_adjusted_score_gather_is_not_a_selection_only_bias(tmp_path):
    policy = _DEEPSEEK.replace(
        "weights = scores.gather(1, idx)",
        "weights = choice.gather(1, idx)")
    result = _read(tmp_path, policy)
    assert result.status == "resolved", result.failures
    # The bias now changes the mixing weights too.  It is not the
    # selection-only stored adjustment represented by the bias card.
    assert result.value.bias_correction is False


@pytest.mark.parametrize(("policy", "before", "normalization"), (
    (_GPT_OSS, False, None),
    (_DBRX, True, "p_norm"),
    (_MIXTRAL, True, "sum"),
))
def test_score_order_and_normalization_are_enacted_not_defaulted(
        tmp_path, policy, before, normalization):
    result = _read(tmp_path, policy)
    assert result.status == "resolved", result.failures
    assert result.value.scoring_fn == "softmax"
    assert result.value.scoring_before_topk is before
    assert result.value.normalization_kind == normalization
    if normalization == "p_norm":
        assert result.value.normalization_path == ("p",)
    else:
        assert result.value.normalization_path == ()


def test_post_selection_score_transform_stays_after_topk_in_the_view(
        monkeypatch):
    from model_unfolder.renderers.html.block_views import moe_router

    captured = {}

    def _capture(graph, *_args, **_kwargs):
        captured["flow"] = tuple(graph.flow)
        captured["facts_projected"] = _kwargs.get("facts_projected")
        return ""

    monkeypatch.setattr(moe_router, "render_graph", _capture)
    moe_router.build_moe_router_view(
        {"name": "fixture", "extras": {"fact_provenance": {
            "decoder.ffn.routing_policy": {
                "status": "code_proven", "value": {},
            },
        }}}, {}, "mount",
        {"detail": {"ffn": {"routing": {
            "selection_kind": "topk",
            "scoring_func": "softmax",
            "scoring_before_topk": False,
            "score_source_kind": "affine",
        }}}})
    assert captured["flow"] == (
        "g_in", "g_gate", "g_topk", "g_score", "g_out")
    assert captured["facts_projected"] == frozenset({
        "decoder.ffn.routing_policy",
    })

    from model_unfolder.adapters.transformer.blocks.feed_forward import (
        _moe_router_step_cards,
    )
    from model_unfolder.ir import FFNSpec

    cards = {item["id"]: item for item in _moe_router_step_cards(
        FFNSpec(kind="moe", routing={
            "selection_kind": "topk",
            "scoring_func": "softmax",
            "scoring_before_topk": False,
        }), "128", "8", 2)}
    assert "raw selected logits" in cards["g_topk"]["description"]
    assert "already-selected experts" in cards["g_score"]["description"]


def test_unknown_router_policy_is_one_opaque_node_not_default_topk(
        monkeypatch):
    from model_unfolder.renderers.html.block_views import moe_router
    from model_unfolder.adapters.transformer.blocks.feed_forward import (
        _moe_router_step_cards,
    )
    from model_unfolder.ir import FFNSpec

    captured = {}

    def _capture(graph, *_args, **_kwargs):
        captured["nodes"] = tuple((node.id, node.kind) for node in graph.nodes)
        captured["flow"] = tuple(graph.flow)
        return ""

    monkeypatch.setattr(moe_router, "render_graph", _capture)
    routing = {"evidence": {"status": "ambiguous"}}
    moe_router.build_moe_router_view(
        {"name": "fixture"}, {}, "mount",
        {"detail": {"ffn": {"routing": routing}}})
    assert captured["nodes"] == (
        ("g_in", "port"), ("g_unknown", "opaque"), ("g_out", "port"))
    assert captured["flow"] == ("g_in", "g_unknown", "g_out")

    cards = _moe_router_step_cards(
        FFNSpec(kind="moe", routing=routing), "128", "8", 2)
    assert [card["id"] for card in cards] == ["g_unknown"]
    assert "no gate, top-k" in cards[0]["description"].lower()
    assert "is assumed" in cards[0]["description"].lower()


def test_exact_policy_does_not_invent_an_unproved_linear_gate(
        tmp_path, monkeypatch):
    source = _source(_GPT_OSS).replace(
        "self.gate = nn.Linear(config.hidden, config.num_experts)",
        "self.gate = nn.ReLU()")
    result = _read_source(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.score_source_kind is None
    assert result.value.score_source_calls == ()

    from model_unfolder.renderers.html.block_views import moe_router
    captured = {}

    def _capture(graph, *_args, **_kwargs):
        captured["nodes"] = tuple((node.id, node.kind) for node in graph.nodes)
        captured["flow"] = tuple(graph.flow)
        return ""

    monkeypatch.setattr(moe_router, "render_graph", _capture)
    moe_router.build_moe_router_view(
        {"name": "fixture"}, {}, "mount",
        {"detail": {"ffn": {"routing": {
            "selection_kind": "topk",
            "scoring_func": "softmax",
            "scoring_before_topk": False,
            "score_source_kind": None,
        }}}})
    assert ("g_gate", "linear") not in captured["nodes"]
    assert captured["flow"] == ("g_in", "g_topk", "g_score", "g_out")

    # The summary card is a second projection of the same fact.  It must not
    # retain the conventional hidden->experts Linear claim after the drill has
    # honestly omitted that producer.
    from model_unfolder.adapters.transformer.blocks.feed_forward import (
        _moe_child_blocks,
    )
    from model_unfolder.ir import FFNSpec
    blocks = _moe_child_blocks(
        FFNSpec(
            kind="moe", routing={
                "selection_kind": "topk",
                "scoring_func": "softmax",
                "scoring_before_topk": False,
                "score_source_kind": None,
            },
            num_experts=8, num_experts_per_tok=2,
        ),
        "128", "256",
    )
    router = next(item for item in blocks if item["id"] == "router")
    assert "unresolved producer" in router["description"]
    assert "128 \u2192 8" not in router["facts"]


def test_unrelated_functional_linear_cannot_certify_router_logits(tmp_path):
    source = _source(_GPT_OSS).replace(
        "self.gate = nn.Linear(config.hidden, config.num_experts)",
        "self.gate = nn.ReLU()",
    ).replace(
        "idx, weights = self.choose(self.gate(x))",
        "dead = F.linear(x, self.bias.unsqueeze(0))\n"
        "        idx, weights = self.choose(self.gate(x))",
    )
    result = _read_source(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.score_source_kind is None
    assert result.value.score_source_calls == ()


def test_nonlinear_child_cannot_borrow_its_inner_affine_identity(tmp_path):
    nonlinear = """
class NonlinearGate(nn.Module):
    def __init__(self, config):
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden))
    def forward(self, x):
        logits = F.linear(x, self.weight)
        return torch.tanh(logits)
"""
    source = _source(_GPT_OSS).replace(
        "class Route(nn.Module):", nonlinear + "\nclass Route(nn.Module):",
    ).replace(
        "self.gate = nn.Linear(config.hidden, config.num_experts)",
        "self.gate = NonlinearGate(config)",
    )
    result = _read_source(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.score_source_kind is None


def test_router_score_source_dto_rejects_an_unrelated_indexed_call(tmp_path):
    result = _read(tmp_path, _GPT_OSS)
    assert result.status == "resolved", result.failures
    with pytest.raises(ValueError, match="re-prove"):
        replace(
            result.value,
            score_source_calls=(result.value.selection_calls[0],),
            spans=tuple(dict.fromkeys((
                *result.value.spans,
                result.value.selection_calls[0].span,
            ))),
        )


def test_uninvoked_sibling_router_cannot_leak_sigmoid_or_bias(tmp_path):
    sibling = """
class Decoy(nn.Module):
    def forward(self, x):
        score = x.sigmoid() + self.bias
        return torch.topk(score, 2)
"""
    result = _read(tmp_path, _GPT_OSS, sibling=sibling)
    assert result.status == "resolved"
    assert result.value.scoring_fn == "softmax"
    assert result.value.bias_correction is False


def test_dead_score_transform_cannot_certify_the_live_selection(tmp_path):
    result = _read(tmp_path, """
dead = F.softmax(logits, dim=-1)
values, idx = torch.topk(logits, self.top_k)
return idx, values
""")
    assert result.status == "failed"


def test_import_alias_cannot_change_function_style_topk_operands(tmp_path):
    source = _source(_GPT_OSS).replace(
        "import torch\n", "import torch\nfrom torch import topk as choose\n",
        1).replace("torch.topk(logits, self.top_k)",
                   "choose(logits, self.top_k)")
    result = _read_source(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.scoring_before_topk is False


def test_returned_but_unrelated_normalization_and_scale_do_not_leak(tmp_path):
    result = _read(tmp_path, """
scores = F.softmax(logits, dim=-1)
weights, idx = torch.topk(scores, self.top_k)
unrelated_norm = logits / logits.sum(dim=-1, keepdim=True)
unrelated_scale = logits * self.scale
return idx, weights, unrelated_norm, unrelated_scale
""")
    assert result.status == "resolved", result.failures
    assert result.value.normalization_kind is None
    assert result.value.scale_path == ()


def test_sparse_mixer_requires_the_full_enacted_protocol(tmp_path):
    positive = _read(tmp_path, _SPARSE_MIXER)
    assert positive.status == "resolved", positive.failures
    assert positive.value.selection_kind == "sparse_mixer"
    assert positive.value.selection_count_literal == 2

    partial = _read(tmp_path, _SPARSE_MIXER.replace(
        "remainder = torch.scatter(logits, -1, selected, float(\"-inf\"))",
        "remainder = logits"))
    assert partial.status == "failed"


def test_one_router_class_can_select_two_source_proven_policy_shapes(tmp_path):
    grouped = _read(
        tmp_path, _BRANCHED_ROUTER,
        config_selector=lambda path: "grouped" if path == ("policy",) else None)
    assert grouped.status == "resolved", grouped.failures
    assert grouped.value.group_score_kind == "top1_max"
    assert grouped.value.branch_config_paths == (("policy",),)

    plain_path = tmp_path / "plain"
    plain_path.mkdir()
    plain = _read(
        plain_path, _BRANCHED_ROUTER,
        config_selector=lambda path: "plain" if path == ("policy",) else None)
    assert plain.status == "resolved", plain.failures
    assert plain.value.group_score_kind is None
    assert plain.value.group_count_path == ()
    assert plain.value.branch_config_paths == (("policy",),)

    unresolved_path = tmp_path / "unresolved"
    unresolved_path.mkdir()
    unresolved = _read(unresolved_path, _BRANCHED_ROUTER)
    assert unresolved.status == "failed"


def test_group_mask_without_known_aggregation_does_not_invent_group_policy(
        tmp_path):
    policy = _BRANCHED_ROUTER.replace(
        ".max(dim=-1).values", ".mean(dim=-1)")
    result = _read(
        tmp_path, policy,
        config_selector=lambda path: "grouped" if path == ("policy",) else None)
    assert result.status == "resolved", result.failures
    assert result.value.group_score_kind is None
    assert result.value.group_count_path == ()
    assert result.value.topk_group_path == ()
    assert result.value.group_spans == ()


def test_same_router_class_on_an_uninvoked_field_is_not_a_rival(tmp_path):
    source = _source(_GPT_OSS).replace(
        "self.route = Route(config)",
        "self.route = Route(config)\n        self.unused = Route(config)")
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    result = decoder_router_selection_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved"
    address = result.value.owner_address
    assert address.owner_graph.node_for(address.owner_occurrence) is not None
    assert address.bridge_site is not None
    assert address.bridge_site.owner.qualified_name == "Block"


def test_router_dto_rejects_cross_block_occurrence(tmp_path):
    value = _read(tmp_path, _GPT_OSS).value
    with pytest.raises(ValueError):
        replace(
            value,
            owner_address=replace(
                value.owner_address,
                block_occurrence=value.owner_occurrence),
        )


def test_router_dto_closes_owner_and_mechanism_provenance(tmp_path):
    value = _read(tmp_path, _DEEPSEEK).value
    with pytest.raises(ValueError):
        replace(
            value,
            owner_address=replace(
                value.owner_address,
                owner_occurrence=value.block_occurrence),
        )
    with pytest.raises(ValueError):
        replace(value, bias_spans=())
    with pytest.raises(ValueError):
        replace(value, group_spans=())
    with pytest.raises(ValueError):
        replace(value, normalization_spans=())
    with pytest.raises(ValueError):
        replace(value, scale_spans=())
    with pytest.raises(ValueError):
        replace(value, spans=tuple(
            span for span in value.spans if span not in value.scale_spans))


def test_router_address_rejects_a_bridge_not_owned_by_the_index(tmp_path):
    value = _read(tmp_path, _DEEPSEEK).value
    address = value.owner_address
    assert isinstance(address, RouterOwnerAddress)
    assert address.bridge_site is not None
    forged = replace(
        address.bridge_site,
        site_id=replace(
            address.bridge_site.site_id,
            ordinal=address.bridge_site.site_id.ordinal + 1000,
        ),
    )
    with pytest.raises(ValueError):
        replace(address, bridge_site=forged)


def test_router_evidence_rejects_a_call_not_owned_by_the_index(tmp_path):
    value = _read(tmp_path, _DEEPSEEK).value
    forged_call = replace(
        value.selection_calls[0],
        lexical_order=value.selection_calls[0].lexical_order + 1000,
    )
    with pytest.raises(ValueError):
        replace(value, selection_calls=(forged_call,))


@pytest.mark.parametrize(("slug", "expected"), (
    ("deepseek-v3", ("sigmoid", True, True, "top2_sum", "sum")),
    ("glm-4-5", ("sigmoid", True, True, "top2_sum", "sum")),
    ("gpt-oss-20b", ("softmax", False, False, None, None)),
    ("dbrx-base", ("softmax", True, False, None, "p_norm")),
))
def test_real_router_controls(slug, expected):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_router_selection_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    value = result.value
    assert (
        value.scoring_fn,
        value.scoring_before_topk,
        value.bias_correction,
        value.group_score_kind,
        value.normalization_kind,
    ) == expected


@pytest.mark.parametrize(("slug", "path"), (
    ("deepseek-v3", ("n_routed_experts",)),
    ("glm-4-5", ("n_routed_experts",)),
    ("gpt-oss-20b", ("num_local_experts",)),
    ("dbrx-base", ("ffn_config", "moe_num_experts")),
))
def test_real_router_expert_count_is_the_exact_score_output_dimension(
        slug, path):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_router_selection_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.expert_count_path == path
    assert result.value.expert_count_spans


def test_router_dto_cannot_claim_a_different_expert_count_path(tmp_path):
    value = _read(tmp_path, _DEEPSEEK).value
    with pytest.raises(ValueError):
        replace(value, expert_count_path=("hidden",))
    with pytest.raises(ValueError):
        replace(value, expert_count_spans=())


def test_real_dense_control_has_no_router_policy():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_router_selection_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"


def test_attention_local_topk_cannot_be_relabelled_as_an_moe_router(tmp_path):
    """A top-k dataflow is not routing unless it belongs to the expert path."""
    result = _read_source(tmp_path, """
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.top_k = config.top_k
    def forward(self, x):
        scores = F.softmax(self.q(x), dim=-1)
        weights, indices = torch.topk(scores, self.top_k)
        return weights

class FeedForward(nn.Module):
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.intermediate)
        self.down = nn.Linear(config.intermediate, config.hidden)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
    def forward(self, x):
        x = self.attn(x)
        return self.ffn(x)

class Model(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper(nn.Module):
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
""")
    assert result.status == "failed"


def test_descriptive_router_strings_cannot_override_enacted_mixtral_code():
    from transformers import AutoConfig

    from model_unfolder.adapters.transformer.parser import _moe_routing

    honest = AutoConfig.for_model("mixtral").to_dict()
    honest["architectures"] = ["MixtralForCausalLM"]
    spoofed = dict(honest)
    spoofed.update(scoring_func="sigmoid", topk_method="noaux_tc")

    honest_routing = _moe_routing(honest, ParseContext.build(honest))
    spoofed_routing = _moe_routing(spoofed, ParseContext.build(spoofed))
    assert honest_routing == spoofed_routing
    assert honest_routing["scoring_func"] == "softmax"
    assert honest_routing["bias_correction"] is False
