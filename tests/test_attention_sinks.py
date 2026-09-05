"""U3-F exact learned attention-sink mechanism controls."""
from __future__ import annotations

import json
import pathlib
import textwrap

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention_sinks import (
    decoder_attention_sinks_for_path,
)
from model_unfolder.evidence.models import SourceBundle


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Projection:
        def __init__(self, config):
            self.q = nn.Linear(config.hidden, config.hidden)
            self.k = nn.Linear(config.hidden, config.hidden)
            self.v = nn.Linear(config.hidden, config.hidden)
            self.learned = nn.Parameter(torch.empty(config.heads))

        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            scores = torch.matmul(q, k)
            lane = self.learned.reshape(1, 1, -1)
            joined = torch.cat([scores, lane], dim=-1)
            probs = F.softmax(joined, dim=-1)
            return torch.matmul(probs[..., :-1], v)

    class Other:
        def __init__(self, config):
            self.learned = nn.Parameter(torch.empty(config.heads))
        def forward(self, x):
            return x

    class Cell:
        def __init__(self, config):
            self.first = Projection(config)
            self.second = Other(config)
        def forward(self, x):
            x = self.first(x)
            return self.second(x)

    class Core:
        def __init__(self, config):
            self.cells = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for cell in self.cells:
                x = cell(x)
            return x

    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _reader(tmp_path, source=_SOURCE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Shell"},
    )
    return decoder_attention_sinks_for_path(
        pi.build_program_index(bundle), bundle, (),
        allow_root_stage=True)


def test_exact_parameter_score_concat_softmax_chain_is_positive(tmp_path):
    result = _reader(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.parameter.site.owner.qualified_name == "Projection"
    assert result.value.parameter.site.span.line
    assert result.value.join_call.callee.source_segment == "torch.cat"
    assert result.value.softmax_call.callee.source_segment == "F.softmax"
    assert result.provenance[-1].spans


def test_complete_class_field_and_local_rename_preserves_evidence(tmp_path):
    source = (_SOURCE
              .replace("Projection", "Mixer")
              .replace("learned", "offset")
              .replace("scores", "matrix")
              .replace("lane", "extra")
              .replace("joined", "combined")
              .replace("Core", "Engine")
              .replace('base_model_prefix = "core"',
                       'base_model_prefix = "engine"')
              .replace("self.core =", "self.engine =")
              .replace("self.cells", "self.items")
              .replace("for cell in self.items", "for item in self.items")
              .replace("x = cell(x)", "x = item(x)"))
    result = _reader(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.attention_symbol.qualified_name == "Mixer"


def test_parameter_name_without_parameter_protocol_is_not_evidence(tmp_path):
    result = _reader(
        tmp_path,
        _SOURCE.replace(
            "self.learned = nn.Parameter(torch.empty(config.heads))",
            "self.learned = nn.Linear(config.hidden, config.hidden)",
            1))
    assert result.status == "failed"


def test_unused_parameter_does_not_vote(tmp_path):
    result = _reader(
        tmp_path,
        _SOURCE.replace(
            "lane = self.learned.reshape(1, 1, -1)",
            "lane = torch.empty(1, 1, 1)",
            1))
    assert result.status == "failed"


def test_parameter_appended_after_softmax_does_not_vote(tmp_path):
    source = _SOURCE.replace(
        """            joined = torch.cat([scores, lane], dim=-1)
            probs = F.softmax(joined, dim=-1)
            return torch.matmul(probs[..., :-1], v)""",
        """            probs = F.softmax(scores, dim=-1)
            joined = torch.cat([probs, lane], dim=-1)
            return torch.matmul(joined[..., :-1], v)""")
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_unrelated_sibling_parameter_cannot_vote(tmp_path):
    source = _SOURCE.replace(
        "self.learned = nn.Parameter(torch.empty(config.heads))",
        "self.learned = torch.empty(config.heads)",
        1)
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_two_independent_parameters_reaching_the_join_are_ambiguous(tmp_path):
    source = _SOURCE.replace(
        "self.learned = nn.Parameter(torch.empty(config.heads))",
        """self.learned = nn.Parameter(torch.empty(config.heads))
            self.other = nn.Parameter(torch.empty(config.heads))""",
        1,
    ).replace(
        "lane = self.learned.reshape(1, 1, -1)",
        """lane = self.learned.reshape(1, 1, -1)
            other = self.other.reshape(1, 1, -1)""",
        1,
    ).replace(
        "torch.cat([scores, lane], dim=-1)",
        "torch.cat([scores, lane, other], dim=-1)",
        1,
    )
    result = _reader(tmp_path, source)
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_guarded_parameter_replacement_is_not_unconditional_proof(tmp_path):
    source = _SOURCE.replace(
        "lane = self.learned.reshape(1, 1, -1)",
        """lane = torch.empty(1, 1, 1)
            if self.training:
                lane = self.learned.reshape(1, 1, -1)""",
        1,
    )
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_real_gpt_oss_positive_and_llama_qwen_controls():
    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    cases = (
        ("gpt-oss-20b", (), "resolved"),
        ("llama-7b", (), "failed"),
        ("qwen2-vl-7b-instruct", ("text_config",), "failed"),
    )
    from model_unfolder.evidence.context import ParseContext
    for slug, config_path, status in cases:
        config = json.loads(
            (corpus / f"{slug}.json").read_text(encoding="utf-8"))["config"]
        context = ParseContext.build(config)
        result = decoder_attention_sinks_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True)
        assert result.status == status, (slug, result)


def test_wrong_input_types_are_rejected():
    import pytest

    with pytest.raises(TypeError):
        decoder_attention_sinks_for_path(
            object(), object(), (), allow_root_stage=True)
