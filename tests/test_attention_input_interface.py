"""U11-E2c source-proven attention input-interface controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.attention_input_interface import (
    context_fallback_attention_interface_at_symbol,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
from torch.nn import functional as F

class Worker:
    def __call__(self, box, state, side=None):
        if box.pre is not None:
            state = box.pre(state)
        query = box.first(state)
        if side is None:
            side = state
        elif box.adjust:
            side = box.prepare(side)
        key = box.second(side)
        value = box.third(side)
        query = query.view(1, 1, 1, 1)
        key = key.view(1, 1, 1, 1)
        value = value.view(1, 1, 1, 1)
        return F.scaled_dot_product_attention(query, key, value)
"""


def _read(tmp_path, source=SOURCE, class_name="Worker"):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture=class_name,
        component_files={"root": (str(path),)},
        component_architectures={"root": class_name}))
    symbol = next(item.symbol for item in index.classes
                  if item.symbol.qualified_name == class_name)
    return context_fallback_attention_interface_at_symbol(index, symbol)


def test_sdpa_proves_primary_query_and_context_fallback_kv(tmp_path):
    value = _read(tmp_path).require_value()
    assert value.primary_formal.name == "state"
    assert value.context_formal.name == "side"
    assert value.container_formal.name == "box"
    assert [item.args[0].name for item in (
        value.query_projection, value.key_projection,
        value.value_projection)] == ["state", "side", "side"]
    assert value.fallback.value.name == "state"
    assert len(value.preserving_bindings) == 2


def test_complete_class_formal_field_and_local_renaming_is_powerless(tmp_path):
    source = (SOURCE.replace("Worker", "Unit")
              .replace("box", "holder")
              .replace("state", "main")
              .replace("side", "other")
              .replace("query", "alpha")
              .replace("key", "beta")
              .replace("value", "gamma")
              .replace("first", "a")
              .replace("second", "b")
              .replace("third", "c"))
    value = _read(tmp_path, source, "Unit").require_value()
    assert value.primary_formal.name == "main"
    assert value.context_formal.name == "other"


@pytest.mark.parametrize("old,new", [
    ("side=None", "side=False"),
    ("if side is None:\n            side = state",
     "if side is None:\n            side = unknown"),
    ("key = box.second(side)", "key = box.second(state)"),
    ("value = box.third(side)", "value = box.third(state)"),
    ("query = box.first(state)", "query = box.first(side)"),
    ("return F.scaled_dot_product_attention(query, key, value)",
     "return F.scaled_dot_product_attention(key, query, value)"),
    ("elif box.adjust:\n            side = box.prepare(side)",
     "elif box.adjust:\n            side = state"),
    ("query = query.view(1, 1, 1, 1)",
     "query = side"),
])
def test_every_qkv_source_and_fallback_edge_is_required(tmp_path, old, new):
    assert _read(tmp_path, SOURCE.replace(old, new)).status == "failed"


def test_dot_softmax_without_an_exact_sdpa_boundary_stays_unknown(tmp_path):
    source = SOURCE.replace(
        "return F.scaled_dot_product_attention(query, key, value)",
        "scores = query @ key\n        probs = F.softmax(scores, dim=-1)\n"
        "        return probs @ value")
    assert _read(tmp_path, source).status == "failed"


def test_compute_in_a_different_entry_callable_stays_typed_unknown(tmp_path):
    source = SOURCE.replace(
        "    def __call__(self, box, state, side=None):",
        "    def __call__(self, box, state, side=None):\n"
        "        return self.forward(box, state, side)\n"
        "    def forward(self, box, state, side=None):",
    )
    result = _read(tmp_path, source)
    assert result.status == "failed"
    assert "different entry callable" in result.failures[0].detail


def test_auxiliary_conditioning_is_preserved_without_becoming_context(tmp_path):
    source = (SOURCE.replace("side=None):", "side=None, mask=None):")
              .replace("box.prepare(side)", "box.prepare(side, mask)"))
    value = _read(tmp_path, source).require_value()
    assert [item.name for item in value.auxiliary_formals] == ["mask"]
    assert value.primary_formal.name == "state"
    assert value.context_formal.name == "side"


def test_shape_metadata_from_key_does_not_become_query_or_value_data(tmp_path):
    source = SOURCE.replace(
        "query = query.view(1, 1, 1, 1)\n"
        "        key = key.view(1, 1, 1, 1)\n"
        "        value = value.view(1, 1, 1, 1)",
        "width = key.shape[-1]\n"
        "        query = query.view(1, 1, 1, width)\n"
        "        key = key.view(1, 1, 1, width)\n"
        "        value = value.view(1, 1, 1, width)",
    )
    assert _read(tmp_path, source).status == "resolved"


def test_shape_call_cannot_hide_a_rival_tensor_data_operand(tmp_path):
    source = SOURCE.replace(
        "query = query.view(1, 1, 1, 1)",
        "query = query.view(key)",
    )
    assert _read(tmp_path, source).status == "failed"


def test_guarded_query_rewrite_from_context_cannot_be_skipped(tmp_path):
    source = SOURCE.replace(
        "query = query.view(1, 1, 1, 1)",
        "if box.adjust:\n"
        "            query = box.first(side)\n"
        "        query = query.view(1, 1, 1, 1)",
    )
    assert _read(tmp_path, source).status == "failed"


def test_dto_rejects_role_and_provenance_forgery(tmp_path):
    value = _read(tmp_path).require_value()
    with pytest.raises(ValueError, match="distinct formals"):
        replace(
            value,
            primary_formal=replace(value.primary_formal, name="forged"),
        )
    with pytest.raises(ValueError, match="Q uses primary"):
        replace(
            value,
            query_projection=value.key_projection,
            key_projection=value.query_projection,
        )
    with pytest.raises(ValueError, match="provenance"):
        replace(value, spans=())
