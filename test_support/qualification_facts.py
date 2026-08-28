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


# U10-F4 registry qualification sources.  These are deliberately parsed by
# ``config_to_ir`` below; they do not manufacture ledger rows.  Together they
# cover source/config mechanisms that are real and supported but absent from
# the blessed preservation corpus.
_DIFFUSION_ATTENTION_SOURCE = """\
import torch
from torch import nn
from torch.nn import functional as F

class Norm:
    def __init__(self, dim): self.weight = torch.ones(dim)
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + 1e-6))

class Kernel:
    def __init__(self, config):
        self.width = config.hidden // config.query_heads
        self.scale = config.scale
        self.q = nn.Linear(config.hidden, config.query_heads * self.width)
        self.k = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.v = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.qn = Norm(config.hidden)
            self.kn = Norm(config.hidden)
    def forward(self, state, context):
        q, k, v = self.q(state), self.k(context), self.v(context)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        return torch.matmul(F.softmax(score, dim=-1), v)

class DensePath:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, state):
        return self.down(F.silu(self.gate(state)) * self.up(state))

class Scale:
    def __init__(self, config):
        self.proj = nn.Linear(config.hidden, config.hidden)
    def forward(self, value): return self.proj(value)

class Cell:
    def __init__(self, config):
        self.a = nn.LayerNorm(config.hidden)
        self.kernel = Kernel(config)
        self.b = nn.LayerNorm(config.hidden)
        self.dense = DensePath(config)
        self.mod = Scale(config)
    def forward(self, state, context, condition):
        gate = self.mod(condition)
        normalized = self.a(state)
        delta = self.kernel(normalized, context)
        state = state + gate * delta
        state = state + self.dense(self.b(state))
        return state

class Root:
    def __init__(self, config):
        self.sequence = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, state, context, condition):
        for element in self.sequence:
            state = element(state, context, condition)
        return state
"""


_DIFFUSION_POSITION_SOURCE = """\
import torch
from torch import nn
from torch.nn import functional as F

def half_turn(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)

def apply_pair(a, b, factor_a, factor_b):
    factor_a = factor_a.unsqueeze(1)
    factor_b = factor_b.unsqueeze(1)
    out_a = (a * factor_a) + (half_turn(a) * factor_b)
    out_b = (b * factor_a) + (half_turn(b) * factor_b)
    return out_a, out_b

class Kernel:
    def __init__(self, config):
        self.width = config.hidden // config.query_heads
        self.q = nn.Linear(config.hidden, config.query_heads * self.width)
        self.k = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.v = nn.Linear(config.hidden, config.kv_heads * self.width)
        self.use_rotation = config.use_rotation
    def forward(self, state, context, factor_a, factor_b):
        q, k, v = self.q(state), self.k(context), self.v(context)
        if self.use_rotation:
            q, k = apply_pair(q, k, factor_a, factor_b)
        return F.scaled_dot_product_attention(q, k, v)

class Cell:
    def __init__(self, config): self.kernel = Kernel(config)
    def forward(self, state, context, factor_a, factor_b):
        return self.kernel(state, context, factor_a, factor_b)

class Root:
    def __init__(self, config):
        self.sequence = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, state, context, factor_a, factor_b):
        for element in self.sequence:
            state = element(state, context, factor_a, factor_b)
        return state
"""


_DIFFUSION_FFN_SOURCE = """\
import torch
from torch import nn
from torch.nn import functional as F

class DensePath:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x): return self.down(self.act(self.up(x)))

class SplitPath:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Choice:
    def __init__(self, config):
        if config.choose_split:
            self.inner = SplitPath(config)
        else:
            self.inner = DensePath(config)
    def forward(self, x): return self.inner(x)

class Block:
    def __init__(self, config): self.ffn = Choice(config)
    def forward(self, x): return self.ffn(x)

class Root:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
"""


_DIFFUSION_CELL_SOURCE = """\
import torch
from torch import nn
from torch.nn import functional as F

class Compute:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, signal):
        q, k, v = self.q(signal), self.k(signal), self.v(signal)
        score = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), v)

class Transform:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, signal): return self.down(self.act(self.up(signal)))

class Cell:
    def __init__(self, config):
        self.compute = Compute(config)
        self.transform = Transform(config)
        self.first = nn.LayerNorm(config.hidden)
        self.second = nn.LayerNorm(config.hidden)
        self.residual_multiplier = config.residual_multiplier
    def forward(self, signal):
        residual = signal
        signal = self.first(signal)
        attention_output = self.compute(signal)
        signal = residual + attention_output * self.residual_multiplier
        residual = signal
        signal = self.second(signal)
        ffn_output = self.transform(signal)
        return residual + ffn_output * self.residual_multiplier

class Root:
    def __init__(self, config):
        self.stack = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, signal):
        for cell in self.stack: signal = cell(signal)
        return signal
"""


_DIFFUSION_GATED_DELTA_SOURCE = """\
import torch
from torch import nn
from torch.nn import functional as F

def first_kernel(q, k, v, **kwargs): return q, k
def second_kernel(q, k, v, **kwargs): return q, k

class Recurrent:
    def __init__(self, config):
        self.red = config.key_heads
        self.green = config.value_heads
        self.blue = config.key_dim
        self.gold = config.value_dim
        self.kw = config.kernel
        self.qk_width = self.blue * self.red
        self.v_width = self.gold * self.green
        self.conv = nn.Conv1d(
            self.qk_width * 2 + self.v_width,
            self.qk_width * 2 + self.v_width, kernel_size=self.kw)
        self.first = first_kernel
        self.second = second_kernel
    def forward(self, x):
        q, k, v = torch.split(
            x, [self.qk_width, self.qk_width, self.v_width], dim=-1)
        q = q.reshape(1, 1, -1, self.blue)
        k = k.reshape(1, 1, -1, self.blue)
        v = v.reshape(1, 1, -1, self.gold)
        beta = x.sigmoid()
        decay = F.softplus(x)
        if x.shape[0] == 1:
            out, state = self.first(q, k, v, decay=decay, beta=beta)
        else:
            out, state = self.second(q, k, v, decay=decay, beta=beta)
        if self.green // self.red > 1:
            q = q.repeat_interleave(self.green // self.red)
            k = k.repeat_interleave(self.green // self.red)
        return out

class Cell:
    def __init__(self, config): self.unit = Recurrent(config)
    def forward(self, x): return self.unit(x)

class Root:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for item in self.layers: x = item(x)
        return x
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


def _diffusion_ir(directory: Path, name: str, source_text: str, values: dict):
    source = directory / f"modeling_qualification_diffusion_{name}.py"
    source.write_text(source_text, encoding="utf-8")
    bundle = SourceBundle(
        source="qualification", files=(str(source),),
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"}, architecture="Root")
    document = {
        "_class_name": "FluxTransformer2DModel",
        "architectures": ["Root"],
        "model_type": f"qualification_diffusion_{name}",
        **values,
    }
    context = ParseContext(source_bundle=bundle, source="qualification")
    return config_to_ir(document, parse_context=context)


def qualification_irs(directory: Path):
    """Return named IRs from exact production-parser frontier witnesses."""
    out = []
    # PaliGemma is the frontier witness for the source-bound projector input
    # lane.  Its exact construction binds ``vision_config.hidden_size`` to the
    # input of ``PaliGemmaMultiModalProjector``; parsing it therefore authors
    # the registered ``projector_in_features`` fact through production code.
    # Keep this here (rather than satisfying the registry with a hand-written
    # fact row): the bidirectional registry gate must observe every definition
    # from a real parser population.
    for model_type in ("gemma3n_text", "gemma4_text", "paligemma"):
        document = AutoConfig.for_model(model_type).to_dict()
        context = ParseContext.build(document)
        out.append((f"qualification:{model_type}",
                    config_to_ir(document, parse_context=context)))
    out.append(("qualification:mtp", _mtp_ir(directory)))
    common = {
        "hidden": 64, "wide": 128, "layers": 2,
        "query_heads": 8, "kv_heads": 2,
    }
    out.extend((
        ("qualification:diffusion-attention", _diffusion_ir(
            directory, "attention", _DIFFUSION_ATTENTION_SOURCE,
            {**common, "scale": 0.125, "qk_norm": True})),
        ("qualification:diffusion-position", _diffusion_ir(
            directory, "position", _DIFFUSION_POSITION_SOURCE,
            {**common, "use_rotation": True})),
        ("qualification:diffusion-ffn", _diffusion_ir(
            directory, "ffn", _DIFFUSION_FFN_SOURCE,
            {"hidden": 64, "wide": 128, "layers": 2,
             "choose_split": True})),
        ("qualification:diffusion-cell", _diffusion_ir(
            directory, "cell", _DIFFUSION_CELL_SOURCE,
            {"hidden": 64, "wide": 128, "layers": 2,
             "residual_multiplier": 0.22})),
        ("qualification:diffusion-gated-delta", _diffusion_ir(
            directory, "gated_delta", _DIFFUSION_GATED_DELTA_SOURCE,
            {"layers": 2, "key_heads": 2, "value_heads": 4,
             "key_dim": 8, "value_dim": 4, "kernel": 3})),
    ))
    return tuple(out)


__all__ = ["qualification_irs"]
