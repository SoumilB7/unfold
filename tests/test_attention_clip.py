"""U6 exact Q/K/V clipping evidence."""

from pathlib import Path
import textwrap

from model_unfolder.evidence.attention import (
    decoder_attention_qkv_clip_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _bundle(tmp_path, clamp_line: str) -> SourceBundle:
    source = f"""
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.qkv = nn.Linear(config.hidden, 3 * config.hidden)
        self.out = nn.Linear(config.hidden, config.hidden)
        self.limit = config.clip_qkv
    def forward(self, x):
        qkv = self.qkv(x)
        {clamp_line}
        q, k, v = qkv.chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-1, -2))
        context = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.out(context)

class Cell:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.ffn = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        return self.ffn(self.mix(x))

class Core:
    def __init__(self, config):
        self.layers = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Core(config)
"""
    path = Path(tmp_path) / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )


def _read(bundle):
    return decoder_attention_qkv_clip_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_fused_projection_clamp_reaching_attention_is_resolved(tmp_path):
    result = _read(_bundle(tmp_path, "qkv = qkv.clamp(max=self.limit)"))
    assert result.status == "resolved", result.failures
    assert result.value.config_path == ("clip_qkv",)
    assert result.value.clamp_call.callee.name == "clamp"


def test_same_config_field_without_clamp_cannot_author_clipping(tmp_path):
    result = _read(_bundle(tmp_path, "qkv = qkv"))
    assert result.status == "failed"


def test_unused_clamp_result_cannot_certify_the_live_qkv_lane(tmp_path):
    result = _read(_bundle(tmp_path, "qkv.clamp(max=self.limit)"))
    assert result.status == "failed"


def test_clamp_operand_path_is_exact_not_a_familiar_spelling(tmp_path):
    bundle = _bundle(tmp_path, "qkv = qkv.clamp(max=self.limit)")
    path = Path(bundle.files[0])
    path.write_text(
        path.read_text().replace(
            "self.limit = config.clip_qkv",
            "self.limit = config.unfamiliar_bound"),
        encoding="utf-8",
    )
    result = _read(bundle)
    assert result.status == "resolved", result.failures
    assert result.value.config_path == ("unfamiliar_bound",)
