"""Real-parser qualification witnesses for mechanisms absent from the corpus.

The blessed corpus is the preservation population.  It is deliberately not a
complete mechanism catalogue: some exact source shapes only have focused HF or
synthetic frontier witnesses.  These helpers run those witnesses through the
same production parser and ledger as the corpus.  They return IRs, never a list
of fact names, so the registry cannot be greened by an allowlist.
"""
from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.parser import config_to_ir


_MTP_SOURCE = """\
import torch
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
        self.head = head
    def forward(self, hidden, embedding):
        a = self.hnorm(hidden)
        b = self.enorm(embedding)
        joined = torch.cat((a, b), dim=-1)
        x = self.proj(joined)
        x = self.block(x)
        return self.head(x)

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
        embedding = self.embed(token_ids)
        for predictor in self.predictors:
            logits = predictor(hidden, embedding)
        return self.head(hidden), logits
"""


def _mtp_ir(directory: Path):
    source = directory / "modeling_qualification_mtp.py"
    source.write_text(_MTP_SOURCE, encoding="utf-8")
    bundle = SourceBundle(
        source="qualification", files=(str(source),),
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"}, architecture="Root")
    document = {
        "architectures": ["Root"], "model_type": "qualification_mtp",
        "is_decoder": True, "vocab_size": 32, "hidden_size": 4,
        "intermediate_size": 8, "num_hidden_layers": 2,
        "num_layers": 2, "num_attention_heads": 1,
        "num_key_value_heads": 1, "num_predictors": 2,
        "hidden_act": "relu", "tie_word_embeddings": False,
        "max_position_embeddings": 32,
    }
    context = ParseContext(source_bundle=bundle, source="qualification")
    return config_to_ir(document, parse_context=context)


def qualification_irs(directory: Path):
    """Return named IRs from exact production-parser frontier witnesses."""
    out = []
    for model_type in ("gemma3n_text", "gemma4_text"):
        document = AutoConfig.for_model(model_type).to_dict()
        context = ParseContext.build(document)
        out.append((f"qualification:{model_type}",
                    config_to_ir(document, parse_context=context)))
    out.append(("qualification:mtp", _mtp_ir(directory)))
    return tuple(out)


__all__ = ["qualification_irs"]
