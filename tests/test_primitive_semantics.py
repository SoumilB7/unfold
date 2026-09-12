"""U3-F3b — primitive semantics are code/protocol facts, never name guesses."""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.construction_calls import resolve_construction_call
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.primitive_semantics import classify_primitive_call


_SOURCE = """
    import torch
    from torch import nn
    from torch.nn import LayerNorm

    class MisleadingLayerNorm:
        def __init__(self, config):
            self.eps = config.eps
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)

    class MisleadingRMSNorm:
        def __init__(self, config):
            self.eps = config.eps
        def forward(self, x):
            centered = x - x.mean(-1, keepdim=True)
            return centered

    class UnknownPrimitive:
        def __init__(self, config):
            self.scale = config.scale
        def forward(self, x):
            return x * self.scale

    class Base:
        def __init__(self, config):
            self.embedding = nn.Embedding(config.vocab, config.hidden)
            self.group = nn.GroupNorm(config.groups, config.hidden)
            self.norm = LayerNorm(config.hidden)
            self.rms_math = MisleadingLayerNorm(config)
            self.layer_math = MisleadingRMSNorm(config)
            self.unknown = UnknownPrimitive(config)
        def forward(self, x):
            a = self.embedding(x)
            grouped = self.group(a)
            b = self.norm(grouped)
            c = self.rms_math(b)
            d = self.layer_math(c)
            return self.unknown(d)

    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = Base(config)
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
        component_architectures={"root": "Wrapper"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    assert stage.status == "resolved"
    return index, root, stage


def _construction(index, root, stage, field):
    owner = root.graph.node_for(stage.occurrence).symbol
    forward = pi.SymbolId(owner.source, f"{owner.qualified_name}.forward")
    call = next(
        item for item in index.calls_in(forward)
        if item.callee.kind == "attribute"
        and item.callee.name == field
        and item.callee.children
        and item.callee.children[0].kind == "name"
        and item.callee.children[0].name == "self"
    )
    return resolve_construction_call(index, root, stage.occurrence, call)


@pytest.mark.parametrize(("field", "expected"), [
    ("embedding", "embedding"),
    ("group", "groupnorm"),
    ("norm", "layernorm"),
])
def test_exact_external_protocol_classifies_primitive(tmp_path, field, expected):
    index, root, stage = _pipeline(tmp_path)
    result = classify_primitive_call(
        index, _construction(index, root, stage, field))
    assert result.status == "resolved"
    assert result.value == expected
    assert result.provenance[0].kind == "external"


def test_custom_rms_math_overrules_misleading_layernorm_class_name(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "rms_math"))
    assert result.status == "resolved"
    assert result.value == "rmsnorm"
    assert result.provenance[0].kind == "source"


def test_custom_mean_centering_overrules_misleading_rms_class_name(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "layer_math"))
    assert result.status == "resolved"
    assert result.value == "layernorm"


def test_exact_external_base_protocol_is_inherited_without_name_guess(tmp_path):
    source = _SOURCE.replace(
        "class MisleadingLayerNorm:",
        "class MisleadingLayerNorm(nn.RMSNorm):",
    ).replace(
        """        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps)
""",
        "",
        1,
    )
    index, root, stage = _pipeline(tmp_path, source)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "rms_math"))
    assert result.status == "resolved"
    assert result.value == "rmsnorm"


def test_unknown_implementation_fails_instead_of_using_class_spelling(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"


def test_unknown_external_protocol_fails_instead_of_guessing(tmp_path):
    source = _SOURCE.replace(
        "self.unknown = UnknownPrimitive(config)",
        "self.unknown = nn.Linear(config.hidden, config.hidden)",
    )
    index, root, stage = _pipeline(tmp_path, source)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"
    assert result.failures[0].kind == "external_unavailable"


def test_two_construction_sites_are_typed_ambiguity_with_exact_sites(tmp_path):
    source = _SOURCE.replace(
        "            self.norm = LayerNorm(config.hidden)",
        """            if config.use_rms:
                self.norm = nn.RMSNorm(config.hidden)
            else:
                self.norm = LayerNorm(config.hidden)""",
    )
    index, root, stage = _pipeline(tmp_path, source)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "norm"))
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_shadowed_external_reference_never_recovers_by_familiar_spelling(tmp_path):
    source = _SOURCE.replace(
        "    class MisleadingLayerNorm:",
        "    nn = replacement\n\n    class MisleadingLayerNorm:",
    )
    index, root, stage = _pipeline(tmp_path, source)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "embedding"))
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def _repeated_primitive_source(*, element="nn.LayerNorm", output="op(chunk)",
                               clause=""):
    replacement = f"""
    class UnknownPrimitive:
        def __init__(self, config):
            self.parts = nn.ModuleList(
                [{element}(config.hidden) for _ in range(config.parts)])
        def forward(self, x):
            chunks = torch.split(x, 1, dim=1)
            return torch.cat(
                [{output} for op, chunk in zip(self.parts, chunks){clause}],
                dim=1)
"""
    start = _SOURCE.index("    class UnknownPrimitive:")
    end = _SOURCE.index("\n    class Base:", start)
    return _SOURCE[:start] + replacement + _SOURCE[end:]


def test_partition_map_reassemble_proves_repeated_layernorm_primitive(tmp_path):
    index, root, stage = _pipeline(tmp_path, _repeated_primitive_source())
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "resolved", result.failures
    assert result.value == "layernorm"


def test_repeated_non_norm_elements_cannot_impersonate_normalization(tmp_path):
    index, root, stage = _pipeline(
        tmp_path, _repeated_primitive_source(element="nn.Linear"))
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"


def test_repeated_container_without_element_application_is_not_a_norm(tmp_path):
    index, root, stage = _pipeline(
        tmp_path, _repeated_primitive_source(output="chunk"))
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"


def test_filtered_repeated_application_is_not_a_complete_norm_protocol(tmp_path):
    index, root, stage = _pipeline(
        tmp_path, _repeated_primitive_source(clause=" if chunk is not None"))
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"


def test_shadowed_zip_cannot_author_repeated_primitive_semantics(tmp_path):
    source = _repeated_primitive_source().replace(
        "    class UnknownPrimitive:",
        "    zip = replacement_zip\n\n    class UnknownPrimitive:")
    index, root, stage = _pipeline(tmp_path, source)
    result = classify_primitive_call(
        index, _construction(index, root, stage, "unknown"))
    assert result.status == "failed"
