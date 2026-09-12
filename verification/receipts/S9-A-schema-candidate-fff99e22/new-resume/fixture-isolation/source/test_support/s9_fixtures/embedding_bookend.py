"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root, resolve_declared_model_stage
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child


_SOURCE = """
    import torch
    from torch import nn

    class Unit:
        def __init__(self, config): pass
        def forward(self, x): return x

    class CustomNorm:
        def __init__(self, config):
            self.eps = config.eps
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)

    class Core:
        def __init__(self, config):
            self.embedding = nn.Embedding(config.vocab, config.hidden)
            self.entry = CustomNorm(config)
            self.units = nn.ModuleList(
                [Unit(config) for _ in range(config.layers)])
        def forward(self, token_ids, inputs_embeds=None):
            if inputs_embeds is None:
                inputs_embeds = self.embedding(token_ids)
            hidden = self.entry(inputs_embeds)
            for unit in self.units:
                hidden = unit(hidden)
            return hidden

    class Shell:
        base_model_prefix = "core"
        def __init__(self, config):
            self.core = Core(config)
"""


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _pipeline(tmp_path, source=_SOURCE):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local",
        files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": "Shell"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    return bundle, index, root, stage, inventory, repeated


def _with_final_norm(source=_SOURCE, *, return_expression="self.final(hidden)"):
    return source.replace(
        "            self.entry = CustomNorm(config)",
        "            self.entry = CustomNorm(config)\n"
        "            self.final = CustomNorm(config)",
    ).replace(
        "            return hidden",
        f"            return {return_expression}",
    )
