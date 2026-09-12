"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention_sinks import decoder_attention_sinks_for_path
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
