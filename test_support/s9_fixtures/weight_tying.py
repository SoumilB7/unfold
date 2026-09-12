"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.weight_tying import manual_weight_tying_for_path


def _source(
    *,
    wrapper="Wrapper",
    stage="Stage",
    block="Block",
    model_field="core",
    head_field="output",
    embedding_field="tokens",
    stage_embedding_call=None,
    returned=None,
    assignment=None,
    extra_root="",
    extra_stage="",
):
    stage_embedding_call = (
        stage_embedding_call
        or f"self.{embedding_field}(token_ids)")
    returned = (
        returned
        or f"self.{head_field}(self.{model_field}(token_ids))")
    assignment = (
        assignment
        if assignment is not None
        else (
            f"self.{head_field}.weight = "
            f"self.{model_field}.{embedding_field}.weight"))
    return f"""
        from torch import nn

        class {block}:
            def forward(self, hidden):
                return hidden

        class {stage}:
            def __init__(self):
                self.{embedding_field} = nn.Embedding(16, 8)
                self.layers = nn.ModuleList(
                    [{block}() for _ in range(2)])
                {extra_stage}

            def forward(self, token_ids):
                hidden = {stage_embedding_call}
                for layer in self.layers:
                    hidden = layer(hidden)
                return hidden

        class {wrapper}:
            base_model_prefix = "{model_field}"

            def __init__(self):
                self.{model_field} = {stage}()
                self.{head_field} = nn.Linear(8, 16, bias=False)
                {extra_root}
                {assignment}

            def forward(self, token_ids):
                return {returned}
    """


def _reader(tmp_path, source=None, architecture="Wrapper"):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(source or _source()), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture})
    return manual_weight_tying_for_path(
        pi.build_program_index(bundle), bundle, ())
