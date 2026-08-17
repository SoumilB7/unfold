"""U9 separate-call Q/K rotary application controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.attention_child import attention_child_evidence
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_candidates_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.separate_rotary import read_separate_qk_rotary


def _read(tmp_path, key_call="apply_one(key, cos, sin)", *, helper="apply_one",
          transpose=False):
    post = "\n                query = query.transpose(1, 2)\n                key = key.transpose(1, 2)" if transpose else ""
    source = tmp_path / "modeling_separate_rotary.py"
    source.write_text(textwrap.dedent(f"""
        import torch
        import torch.nn.functional as F
        from torch.nn import Linear, ModuleList
        def rotate_one(x):
            first = x[..., : x.shape[-1] // 2]
            second = x[..., x.shape[-1] // 2 :]
            return torch.cat((-second, first), dim=-1)
        def apply_one(x, cos, sin):
            return (x * cos) + (rotate_one(x) * sin)
        def apply_many(x, cos, sin):
            pieces = [apply_one(x, cos, sin) for _ in range(2)]
            return torch.cat(pieces, dim=-1)
        class Attention:
            def __init__(self):
                self.q = Linear(4, 4)
                self.k = Linear(4, 4)
                self.v = Linear(4, 4)
            def forward(self, x, cos, sin):
                query = self.q(x)
                key = self.k(x)
                value = self.v(x)
                query = {helper}(query, cos, sin)
                key = {key_call}
                {post}
                return F.scaled_dot_product_attention(query, key, value)
        class Block:
            def __init__(self): self.attn = Attention()
            def forward(self, x, cos, sin): return self.attn(x, cos, sin)
        class Root:
            def __init__(self):
                self.layers = ModuleList([Block() for _ in range(2)])
            def forward(self, x, cos, sin):
                for layer in self.layers:
                    x = layer(x, cos, sin)
                return x
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    candidates = decoder_block_candidates_at_root(
        index, root, allow_root_stage=True)
    assert candidates.status == "resolved", candidates.failures
    block = candidates.value.occurrences[0]
    attention = attention_child_evidence(index, root, block)
    assert attention.status == "resolved", attention.failures
    return read_separate_qk_rotary(index, root, attention.value)


def test_identical_separate_qk_calls_reaching_half_turn_are_rope(tmp_path):
    result = _read(tmp_path)
    assert result.status == "resolved", result.failures
    assert (result.value.kind, result.value.application) == (
        "rope", "qk_rotation")


def test_different_position_factors_do_not_complete_the_pair(tmp_path):
    result = _read(tmp_path, "apply_one(key, sin, cos)")
    assert result.status == "absent"


def test_unrotated_key_does_not_become_qk_rope(tmp_path):
    result = _read(tmp_path, "key")
    assert result.status == "absent"


def test_exact_nested_rotation_helper_closure_is_proven(tmp_path):
    result = _read(
        tmp_path, "apply_many(key, cos, sin)", helper="apply_many")
    assert result.status == "resolved", result.failures


def test_transparent_post_rotation_calls_retain_the_rotary_producer(tmp_path):
    result = _read(tmp_path, transpose=True)
    assert result.status == "resolved", result.failures
