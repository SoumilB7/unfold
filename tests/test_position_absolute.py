"""U8 learned-absolute application: exact coordinate lookup -> pre-stack add."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_absolute import (
    decoder_learned_absolute_position_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

class Cell(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, hidden):
        return self.proj(hidden)

class Stage(nn.Module):
    def __init__(self, config):
        self.tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.coords = nn.Embedding(config.max_positions, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.cells = nn.ModuleList([Cell(config) for _ in range(config.layers)])

    def forward(self, token_ids, coordinate_ids=None):
        token_vectors = self.tokens(token_ids)
        if coordinate_ids is None:
            coordinate_ids = torch.arange(token_vectors.shape[1])
            coordinate_ids = coordinate_ids.unsqueeze(0)
        coordinate_vectors = self.coords(coordinate_ids)
        hidden = token_vectors + coordinate_vectors.to(token_vectors.device)
        hidden = self.drop(hidden)
        for cell in self.cells:
            hidden = cell(hidden)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "body"
    def __init__(self, config):
        self.body = Stage(config)
"""


def _result(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_absolute.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    return decoder_learned_absolute_position_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_exact_coordinate_embedding_add_before_stack_resolves(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved", result
    assert result.value.kind == "learned_absolute"
    assert result.value.application == "embedding_add"
    assert result.value.coordinate_spans
    assert result.value.provenance_spans


def test_total_class_field_and_local_rename_is_invariant(tmp_path):
    source = (_SOURCE
              .replace("Cell", "Unit")
              .replace("Stage", "Body")
              .replace("self.tokens", "self.first")
              .replace("self.coords", "self.second")
              .replace("self.cells", "self.sequence")
              .replace("token_ids", "a")
              .replace("coordinate_ids", "b")
              .replace("token_vectors", "x")
              .replace("coordinate_vectors", "y")
              .replace("hidden", "z")
              .replace("cell in self.sequence", "unit in self.sequence")
              .replace("cell(z)", "unit(z)"))
    result = _result(tmp_path, source)
    assert result.status == "resolved", result


@pytest.mark.parametrize("old,new", [
    ("coordinate_ids = torch.arange(token_vectors.shape[1])",
     "coordinate_ids = token_ids"),
    ("hidden = token_vectors + coordinate_vectors.to(token_vectors.device)",
     "hidden = token_vectors"),
    ("hidden = token_vectors + coordinate_vectors.to(token_vectors.device)",
     "hidden = token_vectors * coordinate_vectors.to(token_vectors.device)"),
    ("self.coords = nn.Embedding(config.max_positions, config.hidden_size)",
     "self.coords = nn.Linear(config.hidden_size, config.hidden_size)"),
])
def test_missing_exact_protocol_remains_unknown(tmp_path, old, new):
    result = _result(tmp_path, _SOURCE.replace(old, new))
    assert result.status != "resolved"


def test_alias_then_addition_preserves_the_exact_relation(tmp_path):
    source = _SOURCE.replace(
        "hidden = token_vectors + coordinate_vectors.to(token_vectors.device)",
        "hidden = token_vectors\n"
        "        hidden = hidden + coordinate_vectors")
    assert _result(tmp_path, source).status == "resolved"


def test_guarded_add_does_not_become_a_general_stage_claim(tmp_path):
    source = _SOURCE.replace(
        "hidden = token_vectors + coordinate_vectors.to(token_vectors.device)",
        "if coordinate_ids is not None:\n"
        "            hidden = token_vectors + coordinate_vectors.to(token_vectors.device)\n"
        "        else:\n"
        "            hidden = token_vectors")
    assert _result(tmp_path, source).status != "resolved"


def test_two_coordinate_embeddings_are_preserved_as_ambiguity(tmp_path):
    source = _SOURCE.replace(
        "self.drop = nn.Dropout(config.dropout)",
        "self.coords_2 = nn.Embedding(config.max_positions, config.hidden_size)\n"
        "        self.drop = nn.Dropout(config.dropout)",
    ).replace(
        "coordinate_vectors = self.coords(coordinate_ids)",
        "coordinate_vectors = self.coords(coordinate_ids)\n"
        "        coordinate_vectors_2 = self.coords_2(coordinate_ids)",
    ).replace(
        "hidden = token_vectors + coordinate_vectors.to(token_vectors.device)",
        "first_hidden = token_vectors + coordinate_vectors.to(token_vectors.device)\n"
        "        hidden = first_hidden + coordinate_vectors_2")
    assert _result(tmp_path, source).status == "ambiguous"


def test_sibling_class_cannot_launder_coordinate_evidence(tmp_path):
    source = _SOURCE + """
class Decoy(nn.Module):
    def __init__(self, config):
        self.lookup = nn.Embedding(config.max_positions, config.hidden_size)
    def forward(self, x):
        p = torch.arange(x.shape[1])
        return x + self.lookup(p)
"""
    result = _result(
        tmp_path,
        source.replace(
            "coordinate_ids = torch.arange(token_vectors.shape[1])",
            "coordinate_ids = token_ids"))
    assert result.status != "resolved"


def test_arange_plus_a_tensor_is_not_a_coordinate_offset(tmp_path):
    source = _SOURCE.replace(
        "coordinate_ids = torch.arange(token_vectors.shape[1])",
        "coordinate_ids = torch.arange(token_vectors.shape[1]) + token_ids")
    assert _result(tmp_path, source).status != "resolved"


def test_arange_of_an_arbitrary_tensor_is_not_a_coordinate_origin(tmp_path):
    source = _SOURCE.replace(
        "torch.arange(token_vectors.shape[1])",
        "torch.arange(token_vectors)")
    assert _result(tmp_path, source).status != "resolved"


def test_method_name_alone_cannot_prove_a_cache_length_offset(tmp_path):
    source = _SOURCE.replace(
        "torch.arange(token_vectors.shape[1])",
        "torch.arange(token_vectors.shape[1]) + token_vectors.get_seq_length()")
    assert _result(tmp_path, source).status != "resolved"


def test_exact_optional_framework_cache_offset_is_a_scalar_coordinate(tmp_path):
    source = (_SOURCE.replace(
        "def forward(self, token_ids, coordinate_ids=None):",
        "def forward(self, token_ids, coordinate_ids=None, past_key_values=None):")
        .replace(
            "coordinate_ids = torch.arange(token_vectors.shape[1])",
            "past = (past_key_values.get_seq_length() "
            "if past_key_values is not None else 0)\n"
            "            coordinate_ids = torch.arange("
            "past, past + token_vectors.shape[1])"))
    assert _result(tmp_path, source).status == "resolved"


@pytest.mark.parametrize("old,new", [
    ("past_key_values=None", "past_key_values=object()"),
    ("past_key_values is not None", "coordinate_ids is not None"),
    ("past_key_values.get_seq_length()",
     "past_key_values.get_seq_length(1)"),
])
def test_cache_method_near_miss_cannot_certify_a_coordinate(
        tmp_path, old, new):
    source = (_SOURCE.replace(
        "def forward(self, token_ids, coordinate_ids=None):",
        "def forward(self, token_ids, coordinate_ids=None, past_key_values=None):")
        .replace(
            "coordinate_ids = torch.arange(token_vectors.shape[1])",
            "past = (past_key_values.get_seq_length() "
            "if past_key_values is not None else 0)\n"
            "            coordinate_ids = torch.arange("
            "past, past + token_vectors.shape[1])")
        .replace(old, new))
    assert _result(tmp_path, source).status != "resolved"


def test_arbitrary_cumsum_is_not_position_evidence(tmp_path):
    source = _SOURCE.replace(
        "class Stage(nn.Module):",
        """class CoordinateEmbedding(nn.Embedding):
    def forward(self, mask, coordinate=None):
        if coordinate is None:
            coordinate = torch.cumsum(mask, dim=1)
            coordinate = (coordinate * mask - 1).long()
        return super().forward(coordinate)

class Stage(nn.Module):""",
    ).replace(
        "self.coords = nn.Embedding(config.max_positions, config.hidden_size)",
        "self.coords = CoordinateEmbedding(config.max_positions, config.hidden_size)",
    ).replace(
        "coordinate_vectors = self.coords(coordinate_ids)",
        "coordinate_vectors = self.coords(token_ids)",
    ).replace(
        "coordinate_ids = torch.arange(token_vectors.shape[1])",
        "coordinate_ids = token_ids")
    assert _result(tmp_path, source).status != "resolved"


@pytest.mark.parametrize("old,new", [
    ("torch.cumsum(mask, dim=1)", "torch.sum(mask, dim=1)"),
    ("return super().forward(coordinate)", "return coordinate"),
    ("class CoordinateEmbedding(nn.Embedding):",
     "class CoordinateEmbedding(nn.Linear):"),
])
def test_internal_coordinate_protocol_rejects_near_misses(tmp_path, old, new):
    source = _SOURCE.replace(
        "class Stage(nn.Module):",
        """class CoordinateEmbedding(nn.Embedding):
    def forward(self, mask, coordinate=None):
        if coordinate is None:
            coordinate = torch.cumsum(mask, dim=1)
        return super().forward(coordinate)

class Stage(nn.Module):""",
    ).replace(
        "self.coords = nn.Embedding(config.max_positions, config.hidden_size)",
        "self.coords = CoordinateEmbedding(config.max_positions, config.hidden_size)",
    ).replace(
        "coordinate_vectors = self.coords(coordinate_ids)",
        "coordinate_vectors = self.coords(token_ids)",
    ).replace(
        "coordinate_ids = torch.arange(token_vectors.shape[1])",
        "coordinate_ids = token_ids").replace(old, new)
    assert _result(tmp_path, source).status != "resolved"


def test_dto_rejects_a_forged_application_kind(tmp_path):
    value = _result(tmp_path).value
    with pytest.raises(ValueError):
        replace(value, application="attention_side_input")


def test_real_gpt2_source_proves_the_same_relation():
    from transformers import AutoConfig
    from model_unfolder.evidence.sources import resolve_source_files

    config = AutoConfig.for_model("gpt2").to_dict()
    bundle = resolve_source_files(config)
    result = decoder_learned_absolute_position_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result
    assert result.value.kind == "learned_absolute"


def test_real_opt_waits_for_the_u8_mask_to_coordinate_join():
    from transformers import AutoConfig
    from model_unfolder.evidence.sources import resolve_source_files

    config = AutoConfig.for_model("opt").to_dict()
    bundle = resolve_source_files(config)
    result = decoder_learned_absolute_position_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status != "resolved"
