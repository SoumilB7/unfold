"""U11-E2c unanimous default attention-container controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.attention_container_interface import (
    default_attention_container_interface,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
import torch
from torch.nn import functional as F

class First:
    def __call__(self, box: Container, main, other=None):
        q = box.a(main)
        if other is None: other = main
        k = box.b(other)
        v = box.c(other)
        return F.scaled_dot_product_attention(q, k, v)

class Second:
    def __call__(self, box: Container, main, other=None):
        q = box.x(main)
        if other is None: other = main
        k = box.y(other)
        v = box.z(other)
        probs = box.scores(q, k)
        return torch.bmm(probs, v)

class Container:
    def __init__(self, worker=None):
        if worker is None:
            worker = First() if available() else Second()
        self.install(worker)
    def install(self, item):
        self.delegate = item
    def scores(self, q, k):
        scratch = torch.empty(q.shape[0], q.shape[1], k.shape[1])
        scores = torch.baddbmm(scratch, q, k.transpose(-1, -2))
        probs = scores.softmax(dim=-1)
        return probs
    def forward(self, value, context=None, **options):
        return self.delegate(
            self, value, other=context, **options)

class Root:
    def __init__(self): self.unit = Container()
"""


def _read(tmp_path, source=SOURCE):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}))
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "unit")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    return default_attention_container_interface(index, frame)


def test_all_runtime_default_alternatives_prove_one_interface(tmp_path):
    value = _read(tmp_path).require_value()
    assert len(value.implementations) == 2
    assert value.primary_formal.name == "value"
    assert value.context_formal.name == "context"
    assert value.selector_value.value is None
    assert {item.symbol.qualified_name for item in value.implementations} == {
        "First", "Second"}
    assert {item.interface.compute.protocol
            for item in value.implementations} == {
        "scaled_dot_product_attention", "dot_softmax"}


def test_complete_symbol_field_formal_and_local_renaming_is_powerless(tmp_path):
    source = (SOURCE.replace("First", "Alpha")
              .replace("Second", "Beta")
              .replace("Container", "Holder")
              .replace("Root", "Top")
              .replace("worker", "choice")
              .replace("install", "put")
              .replace("delegate", "run")
              .replace("value", "state")
              .replace("context", "side")
              .replace("main", "primary")
              .replace("other", "secondary"))
    path = Path(tmp_path) / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture="Top",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Top"}))
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Top"
                and item.target == "unit")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    value = default_attention_container_interface(index, frame).require_value()
    assert value.primary_formal.name == "state"
    assert value.context_formal.name == "side"


@pytest.mark.parametrize("old,new", [
    ("self.install(worker)", "self.other(worker)"),
    ("self.delegate = item", "self.delegate = choose(item)"),
    ("worker = First() if available() else Second()",
     "worker = First() if available() else opaque"),
    ("k = box.y(other)", "k = box.y(main)"),
    ("probs = scores.softmax(dim=-1)", "probs = scores"),
    ("torch.baddbmm(scratch, q, k.transpose(-1, -2))",
     "torch.baddbmm(scratch, q, q.transpose(-1, -2))"),
    ("return torch.bmm(probs, v)", "return torch.bmm(probs, q)"),
    ("return self.delegate(\n            self, value, other=context, **options)",
     "return self.delegate(\n            self, context, other=value, **options)"),
    ("def forward(self, value, context=None, **options):",
     "def forward(self, value, context=False, **options):"),
])
def test_every_install_alternative_and_delegate_edge_is_required(
        tmp_path, old, new):
    assert _read(tmp_path, SOURCE.replace(old, new)).status == "failed"


def test_explicit_runtime_processor_does_not_claim_the_default_route(tmp_path):
    source = SOURCE.replace("Container()", "Container(First())")
    assert _read(tmp_path, source).status == "failed"


def test_dto_rejects_missing_alternative_and_role_forgery(tmp_path):
    value = _read(tmp_path).require_value()
    with pytest.raises(ValueError, match="alternatives"):
        replace(value, implementations=value.implementations[:1])
    with pytest.raises(ValueError, match="primary"):
        replace(value, primary_formal=value.context_formal)
