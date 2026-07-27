"""U3-F exact attention/FFN input-normalization occurrence controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.parallel_norm import (
    ParallelNormEvidence,
    decoder_parallel_norm_count_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _source(*, block_forward, norm_fields=None, extra=""):
    norm_fields = norm_fields or """
        self.before = nn.LayerNorm(config.hidden)
        self.after = nn.LayerNorm(config.hidden)
"""
    norm_fields = textwrap.indent(
        textwrap.dedent(norm_fields).strip(), " " * 16)
    block_forward = textwrap.indent(
        textwrap.dedent(block_forward).strip(), " " * 16)
    extra = textwrap.indent(
        textwrap.dedent(extra).strip(), " " * 8) if extra.strip() else ""
    return f"""
        import torch
        from torch import nn
        from torch.nn import functional as F

        class Compute:
            def __init__(self, config):
                self.q = nn.Linear(config.hidden, config.hidden)
                self.k = nn.Linear(config.hidden, config.hidden)
                self.v = nn.Linear(config.hidden, config.hidden)
            def forward(self, signal):
                q = self.q(signal)
                k = self.k(signal)
                v = self.v(signal)
                scores = torch.matmul(q, k.transpose(-1, -2))
                return torch.matmul(F.softmax(scores, dim=-1), v)

        class Transform:
            def __init__(self, config):
                self.up = nn.Linear(config.hidden, config.wide)
                self.down = nn.Linear(config.wide, config.hidden)
                self.act = nn.GELU()
            def forward(self, signal):
                return self.down(self.act(self.up(signal)))

        class Cell:
            def __init__(self, config):
                self.compute = Compute(config)
                self.transform = Transform(config)
{norm_fields}
            def forward(self, signal):
{block_forward}

        class Body:
            def __init__(self, config):
                self.stack = nn.ModuleList(
                    [Cell(config) for _ in range(config.layers)])
            def forward(self, signal):
                for cell in self.stack:
                    signal = cell(signal)
                return signal

        class Outer:
            base_model_prefix = "body"
            def __init__(self, config):
                self.body = Body(config)

{extra}
    """


def _read(tmp_path, *, block_forward, norm_fields=None, extra=""):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_source(
        block_forward=block_forward,
        norm_fields=norm_fields,
        extra=extra,
    )), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Outer"},
        architecture="Outer",
    )
    return decoder_parallel_norm_count_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_one_exact_shared_norm_occurrence_feeds_both_branches(tmp_path):
    result = _read(
        tmp_path,
        norm_fields="""
        self.shared = nn.LayerNorm(config.hidden)
""",
        block_forward="""
                normalized = self.shared(signal)
                attention = self.compute(normalized)
                transformed = self.transform(normalized)
                return signal + attention + transformed
""")
    assert result.status == "resolved", result.failures
    assert result.value.norm_count == 1
    assert result.value.attention.norm_occurrence \
        == result.value.ffn_inputs[0].norm_occurrence


def test_two_exact_norm_occurrences_feed_the_two_branches(tmp_path):
    result = _read(
        tmp_path,
        block_forward="""
                attention = self.compute(self.before(signal))
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
""")
    assert result.status == "resolved", result.failures
    assert result.value.norm_count == 2
    assert result.value.attention.norm_occurrence \
        != result.value.ffn_inputs[0].norm_occurrence


def test_complete_class_field_and_local_rename_preserves_count(tmp_path):
    result = _read(
        tmp_path,
        norm_fields="""
        self.alpha = nn.RMSNorm(config.hidden)
        self.beta = nn.RMSNorm(config.hidden)
""",
        block_forward="""
                left = self.compute(self.alpha(signal))
                right = self.transform(self.beta(signal))
                return signal + left + right
""")
    assert result.status == "resolved"
    assert result.value.norm_count == 2
    assert result.value.attention.primitive == "rmsnorm"


def test_unrelated_norm_and_sibling_class_cannot_vote(tmp_path):
    result = _read(
        tmp_path,
        norm_fields="""
        self.before = nn.LayerNorm(config.hidden)
        self.after = nn.LayerNorm(config.hidden)
        self.unused = nn.RMSNorm(config.hidden)
""",
        block_forward="""
                attention = self.compute(self.before(signal))
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
""",
        extra="""
        class Sibling:
            def __init__(self, config):
                self.norm = nn.RMSNorm(config.hidden)
            def forward(self, signal):
                return self.norm(signal)
""")
    assert result.status == "resolved"
    assert result.value.norm_count == 2
    assert {result.value.attention.primitive,
            result.value.ffn_inputs[0].primitive} == {"layernorm"}


def test_raw_branch_input_is_not_silently_called_one_norm(tmp_path):
    result = _read(
        tmp_path,
        block_forward="""
                attention = self.compute(self.before(signal))
                transformed = self.transform(signal)
                return signal + attention + transformed
""")
    assert result.status == "failed"


def test_transformed_norm_output_is_not_an_exact_direct_input_edge(tmp_path):
    result = _read(
        tmp_path,
        block_forward="""
                normalized = self.before(signal)
                shifted = normalized + 1
                attention = self.compute(shifted)
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
""")
    assert result.status == "failed"


def test_branch_rival_norm_producers_are_never_first_picked(tmp_path):
    result = _read(
        tmp_path,
        block_forward="""
                if self.training:
                    normalized = self.before(signal)
                else:
                    normalized = self.after(signal)
                attention = self.compute(normalized)
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
    """)
    assert result.status == "failed"
    assert any("does not uniquely feed" in item.detail
               for item in result.failures)


def test_similar_custom_callable_is_not_a_norm_protocol(tmp_path):
    result = _read(
        tmp_path,
        norm_fields="""
        self.before = Identity(config)
        self.after = nn.LayerNorm(config.hidden)
""",
        block_forward="""
                attention = self.compute(self.before(signal))
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
""",
        extra="""
        class Identity:
            def __init__(self, config):
                self.hidden = config.hidden
            def forward(self, signal):
                return signal
""")
    assert result.status == "failed"


@pytest.mark.parametrize(("slug", "path", "status", "count"), [
    ("bloom", (), "resolved", 2),
    ("llama-7b", (), "resolved", 2),
    ("qwen2-vl-7b-instruct", ("text_config",), "resolved", 2),
])
def test_real_corpus_sources_have_exact_or_typed_unknown_results(
        slug, path, status, count):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_parallel_norm_count_for_path(
        context.program_index(), context.source_bundle, path,
        allow_root_stage=True)
    assert result.status == status
    assert getattr(result.value, "norm_count", None) == count
    if count is not None:
        assert result.value.spans


@pytest.mark.parametrize(("family", "relative", "architecture", "status", "count"), [
    ("gptj", "gptj/modeling_gptj.py", "GPTJForCausalLM", "failed", None),
    ("gpt_neox", "gpt_neox/modeling_gpt_neox.py",
     "GPTNeoXForCausalLM", "resolved", 2),
    ("falcon", "falcon/modeling_falcon.py",
     "FalconForCausalLM", "failed", None),
])
def test_equivalent_transformers_sources_preserve_honest_boundaries(
        family, relative, architecture, status, count):
    import transformers

    source = Path(transformers.__file__).parent / "models" / relative
    if not source.exists():
        pytest.skip(f"{family} modeling source is unavailable")
    bundle = SourceBundle(
        source="local",
        files=(str(source),),
        component_files={"root": (str(source),)},
        component_architectures={"root": architecture},
        architecture=architecture,
    )
    result = decoder_parallel_norm_count_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == status
    assert getattr(result.value, "norm_count", None) == count


def test_result_closure_rejects_a_fabricated_count(tmp_path):
    value = _read(
        tmp_path,
        norm_fields="""
        self.shared = nn.LayerNorm(config.hidden)
""",
        block_forward="""
                normalized = self.shared(signal)
                attention = self.compute(normalized)
                transformed = self.transform(normalized)
                return signal + attention + transformed
""").value
    with pytest.raises(ValueError):
        replace(value, norm_count=2)
    with pytest.raises(ValueError):
        ParallelNormEvidence(
            value.block_occurrence,
            value.attention,
            value.ffn_inputs,
            value.norm_count,
            ())
