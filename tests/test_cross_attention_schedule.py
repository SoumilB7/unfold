"""U3-F exact additive cross-attention schedule controls."""
from __future__ import annotations

import dataclasses
import json
import pathlib
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.cross_attention_schedule import (
    decoder_cross_attention_all_layers_for_path,
)
from model_unfolder.evidence.models import SourceBundle


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import functional as F

    def select(fallback):
        return fallback

    def kernel(module, q, k, v):
        weights = F.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(weights, v)

    class Projection:
        def __init__(self, config):
            self.q = nn.Linear(config.hidden, config.hidden)
            self.k = nn.Linear(config.hidden, config.hidden)
            self.v = nn.Linear(config.hidden, config.hidden)

        def forward(self, hidden, context=None, cache=None):
            source = context if context is not None else hidden
            q = self.q(hidden)
            k = self.k(source)
            v = self.v(source)
            operation = select(kernel)
            return operation(self, q, k, v)

    class Cell:
        def __init__(self, config):
            self.first = Projection(config)
            self.second = Projection(config)

        def forward(self, hidden, context=None, cache=None):
            hidden = self.first(hidden, cache=cache)
            if context is not None:
                hidden = hidden + self.second(
                    hidden, context=context, cache=cache)
            return hidden

    class Core:
        def __init__(self, config):
            self.cells = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, hidden, context=None):
            for cell in self.cells:
                hidden = cell(hidden, context=context)
            return hidden

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
    return decoder_cross_attention_all_layers_for_path(
        pi.build_program_index(bundle), bundle, (),
        allow_root_stage=True)


def test_two_exact_attention_occurrences_with_one_external_kv_input_are_positive(
        tmp_path):
    result = _reader(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.kv_formal == "context"
    assert result.value.self_attention_occurrence != \
        result.value.cross_attention_occurrence
    assert not result.value.self_evidence.invocation.call.guard
    assert result.value.cross_evidence.invocation.call.guard
    assert result.provenance[-1].spans


def test_complete_class_field_formal_and_local_rename_preserves_evidence(
        tmp_path):
    source = (_SOURCE
              .replace("Projection", "Mixer")
              .replace("context", "memory")
              .replace("source", "material")
              .replace("first", "alpha")
              .replace("second", "beta")
              .replace("Cell", "Unit")
              .replace("Core", "Engine")
              .replace('base_model_prefix = "core"',
                       'base_model_prefix = "engine"')
              .replace("self.core =", "self.engine =")
              .replace("self.cells", "self.items")
              .replace("for cell in self.items", "for item in self.items")
              .replace("hidden = cell(", "hidden = item("))
    result = _reader(tmp_path, source)
    assert result.status == "resolved", result.failures
    assert result.value.kv_formal == "memory"
    assert result.value.attention_symbol.qualified_name == "Mixer"


def test_one_attention_construction_called_twice_is_not_dual_attention(tmp_path):
    source = _SOURCE.replace(
        "self.second = Projection(config)", "", 1).replace(
        "self.second(", "self.first(", 1)
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_guarded_cross_attention_construction_does_not_prove_every_layer(
        tmp_path):
    source = _SOURCE.replace(
        "self.second = Projection(config)",
        """if config.enabled:
                self.second = Projection(config)""",
        1)
    result = _reader(tmp_path, source)
    assert result.status != "resolved"


def test_supplying_the_optional_kv_formal_to_both_lanes_is_not_cross_shape(
        tmp_path):
    source = _SOURCE.replace(
        "self.first(hidden, cache=cache)",
        "self.first(hidden, context=context, cache=cache)",
        1)
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_optional_input_must_feed_both_key_and_value(tmp_path):
    source = _SOURCE.replace(
        "v = self.v(source)",
        "v = self.v(hidden)",
        1)
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_two_different_attention_implementations_are_rivals_not_one_interface(
        tmp_path):
    source = _SOURCE.replace(
        "    class Cell:",
        """    class OtherProjection(Projection):
        pass

    class Cell:""",
    ).replace(
        "self.second = Projection(config)",
        "self.second = OtherProjection(config)",
        1,
    )
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_a_third_attention_occurrence_is_not_silently_discarded(tmp_path):
    source = _SOURCE.replace(
        "self.second = Projection(config)",
        """self.second = Projection(config)
            self.third = Projection(config)""",
        1,
    ).replace(
        "return hidden",
        """if context is not None:
                hidden = hidden + self.third(hidden, context=context)
            return hidden""",
        1,
    )
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_an_attention_sibling_outside_the_exact_block_cannot_vote(tmp_path):
    source = _SOURCE.replace(
        """            self.first = Projection(config)
            self.second = Projection(config)

        def forward(self, hidden, context=None, cache=None):
            hidden = self.first(hidden, cache=cache)
            if context is not None:
                hidden = hidden + self.second(
                    hidden, context=context, cache=cache)""",
        """            self.first = Projection(config)

        def forward(self, hidden, context=None, cache=None):
            hidden = self.first(hidden, cache=cache)
            if context is not None:
                hidden = hidden + self.first(
                    hidden, context=context, cache=cache)""",
        1,
    ).replace(
        "    class Cell:",
        """    class Unused:
        def __init__(self, config):
            self.first = Projection(config)
            self.second = Projection(config)
        def forward(self, hidden, context=None):
            hidden = self.first(hidden)
            return self.second(hidden, context=context)

    class Cell:""",
        1,
    )
    result = _reader(tmp_path, source)
    assert result.status == "failed"


def test_real_musicgen_positive_and_llama_negative():
    from model_unfolder.evidence.context import ParseContext

    corpus = pathlib.Path(__file__).parent / "sable_test_corpus"
    for slug, config_path, status in (
            ("musicgen-small", ("decoder",), "resolved"),
            ("llama-7b", (), "failed")):
        config = json.loads(
            (corpus / f"{slug}.json").read_text(encoding="utf-8"))["config"]
        context = ParseContext.build(config)
        result = decoder_cross_attention_all_layers_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True)
        assert result.status == status, (slug, result)


def test_result_closure_rejects_same_occurrence_for_both_lanes(tmp_path):
    result = _reader(tmp_path)
    assert result.status == "resolved"
    with pytest.raises(ValueError):
        dataclasses.replace(
            result.value,
            cross_attention_occurrence=result.value.self_attention_occurrence)


def test_wrong_input_types_are_rejected():
    with pytest.raises(TypeError):
        decoder_cross_attention_all_layers_for_path(
            object(), object(), (), allow_root_stage=True)
