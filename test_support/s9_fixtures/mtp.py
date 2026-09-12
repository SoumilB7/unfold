"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.mtp import decoder_mtp_construction_for_path
from model_unfolder.evidence.program_index import build_program_index


def _source(*, share_embedding: bool = True, share_head: bool = True,
            include_concat: bool = True, include_block: bool = True) -> str:
    predictor_embedding = (
        ""
        if share_embedding else
        "        self.own_embedding = Embedding(config.vocab_size, 4)\n"
    )
    predictor_head = (
        "        self.head = head\n"
        if share_head else
        "        self.head = Linear(4, config.vocab_size)\n"
    )
    second_input = "embedding" if share_embedding else "token_ids"
    embed_value = "embedding" if share_embedding else "self.own_embedding(token_ids)"
    concat = (
        "        joined = torch.cat((a, b), dim=-1)\n"
        if include_concat else
        "        joined = a + b\n"
    )
    block = (
        "        x = self.block(x)\n"
        if include_block else
        ""
    )
    stage_embedding = (
        "        embedding = self.embed(token_ids)\n"
        "        for predictor in self.predictors:\n"
        "            logits = predictor(hidden, embedding)\n"
        if share_embedding else
        "        for predictor in self.predictors:\n"
        "            logits = predictor(hidden, token_ids)\n"
    )
    return f"""import torch
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

class Layer:
    def __init__(self, config):
        self.proj = Linear(4, 4)
    def forward(self, hidden):
        return self.proj(hidden)

class Predictor:
    def __init__(self, config, head):
        self.hnorm = LayerNorm(4)
        self.enorm = LayerNorm(4)
        self.proj = Linear(8, 4)
        self.block = Layer(config)
{predictor_embedding}{predictor_head}
    def forward(self, hidden, {second_input}):
        a = self.hnorm(hidden)
        b = self.enorm({embed_value})
{concat}        x = self.proj(joined)
{block}        return self.head(x)

class Root:
    base_model_prefix = ""
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, 4)
        self.head = Linear(4, config.vocab_size)
        self.layers = ModuleList(
            [Layer(config) for _ in range(config.num_layers)])
        self.predictors = ModuleList(
            [Predictor(config, self.head)
             for _ in range(config.num_predictors)])
    def forward(self, token_ids):
        hidden = self.embed(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
{stage_embedding}        return self.head(hidden), logits
"""


def _read(tmp_path, **changes):
    path = tmp_path / "model.py"
    path.write_text(_source(**changes), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}, architecture="Root")
    return decoder_mtp_construction_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
