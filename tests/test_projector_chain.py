"""U9-C exact owner-qualified operation-chain controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.projector_chain import read_projector_operation_chain


def _chain(tmp_path, source):
    path = tmp_path / "modeling_chain.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return read_projector_operation_chain(index, root, root.occurrence)


def test_sequential_protocol_emits_exact_ordered_primitive_chain(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear, GELU, Sequential
        class Root:
            def __init__(self):
                self.path = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
            def forward(self, x):
                return self.path(x)
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.operations] == [
        "linear", "activation", "linear"]
    assert [op.label for op in result.value.operations] == [
        "Linear (in)", "GELU", "Linear (out)"]


def test_exact_groupnorm_and_dropout_are_shared_operation_protocols(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Conv2d, Dropout, GroupNorm
        class Root:
            def __init__(self):
                self.norm = GroupNorm(4, 8)
                self.drop = Dropout(0.1)
                self.conv = Conv2d(8, 8, 3)
            def forward(self, x):
                x = self.norm(x)
                x = self.drop(x)
                return self.conv(x)
    """)
    assert result.status == "resolved"
    assert [(op.kind, op.label) for op in result.value.operations] == [
        ("norm", "GroupNorm"),
        ("dropout", "Dropout"),
        ("conv2d", "2D convolution"),
    ]


def test_only_return_reaching_calls_enter_the_chain(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self):
                self.used = Linear(4, 4)
                self.decoy = Linear(4, 4)
            def forward(self, x):
                ignored = self.decoy(x)
                return self.used(x)
    """)
    assert result.status == "resolved"
    assert len(result.value.operations) == 1
    assert result.value.operations[0].line == 9


def test_reassignment_trace_preserves_linear_activation_linear(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        from transformers.activations import ACT2FN
        class Root:
            def __init__(self, config):
                self.first = Linear(4, 8)
                self.act = ACT2FN[config.hidden_act]
                self.last = Linear(8, 4)
            def forward(self, x):
                x = self.first(x)
                x = self.act(x)
                x = self.last(x)
                return x
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.operations] == [
        "linear", "activation", "linear"]


def test_numeric_affine_input_preprocessing_reaches_the_projection(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, pixels):
                pixels = 2 * (pixels - 0.5)
                return self.out(pixels)
    """)
    assert result.status == "resolved", result.failures
    assert [op.kind for op in result.value.operations] == [
        "elementwise", "linear"]
    assert result.value.operations[0].fn == "affine"


def test_numeric_shape_arithmetic_does_not_become_a_tensor_operation(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x):
                width = 2 * (x.shape[-1] - 1)
                return self.out(x)
    """)
    assert result.status == "resolved", result.failures
    assert [op.kind for op in result.value.operations] == ["linear"]


def test_unconsumed_affine_input_decoy_does_not_enter_the_chain(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x):
                ignored = 2 * (x - 0.5)
                return self.out(x)
    """)
    assert result.status == "resolved", result.failures
    assert [op.kind for op in result.value.operations] == ["linear"]


def test_shape_call_is_kept_only_on_the_return_path(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import LayerNorm, Linear
        class Root:
            def __init__(self):
                self.norm = LayerNorm(4)
                self.out = Linear(4, 4)
            def forward(self, x):
                hidden = self.norm(x).reshape(-1, 4)
                unused = x.flatten()
                return self.out(hidden)
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.operations] == [
        "norm", "reshape", "linear"]


def test_unknown_return_producer_is_incomplete_not_default_linear(tmp_path):
    result = _chain(tmp_path, """
        class Root:
            def __init__(self):
                self.unknown = make_callable()
            def forward(self, x):
                return self.unknown(x)
    """)
    assert result.status == "failed"
    assert result.failures[0].kind in {"incomplete_graph", "external_unavailable"}


def test_exact_internal_child_is_followed_without_class_or_field_markers(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Linear
        class ArbitraryChild:
            def __init__(self):
                self.a = Linear(4, 8)
                self.b = Linear(8, 4)
            def forward(self, x):
                return self.b(self.a(x))
        class Root:
            def __init__(self):
                self.any_spelling = ArbitraryChild()
            def forward(self, x):
                return self.any_spelling(x)
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.operations] == ["linear", "linear"]


def test_exact_loop_append_value_path_is_a_symbolic_operation_template(tmp_path):
    result = _chain(tmp_path, """
        import torch
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, xs):
                gathered = []
                for item in xs:
                    value = item.reshape(-1, 4).transpose(0, 1)
                    gathered.append(value)
                joined = torch.cat(gathered, dim=0)
                return self.out(joined)
    """)
    assert result.status == "resolved"
    assert [op.kind for op in result.value.operations] == [
        "reshape", "reshape", "concat", "linear"]


def test_exact_frontend_primitives_are_code_classified(tmp_path):
    result = _chain(tmp_path, """
        from torch.nn import Conv2d, AvgPool2d, PixelShuffle, Sequential
        class Root:
            def __init__(self):
                self.path = Sequential(
                    Conv2d(3, 8, 4), AvgPool2d(2), PixelShuffle(2))
            def forward(self, x):
                return self.path(x)
    """)
    assert result.status == "resolved", result.failures
    assert [op.kind for op in result.value.operations] == [
        "conv2d", "pooling", "pixel_shuffle"]


def test_exact_functional_reduction_and_concat_are_not_config_inferences(
        tmp_path):
    result = _chain(tmp_path, """
        import torch
        import torch.nn.functional as F
        class Root:
            def forward(self, x, y):
                x = F.avg_pool2d(x, 2)
                return torch.cat((x, y), dim=1)
    """)
    assert result.status == "resolved", result.failures
    assert [op.kind for op in result.value.operations] == [
        "pooling", "concat"]


def test_frontend_primitive_spelling_without_framework_binding_proves_nothing(
        tmp_path):
    result = _chain(tmp_path, """
        class Conv2d:
            def __init__(self, *args): pass
            def forward(self, x): return x
        class Root:
            def __init__(self): self.path = Conv2d(3, 8, 4)
            def forward(self, x): return self.path(x)
    """)
    assert result.status != "resolved"
    assert result.value is None


def test_unknown_accumulator_mutation_keeps_chain_incomplete(tmp_path):
    result = _chain(tmp_path, """
        import torch
        from torch.nn import Linear
        class Root:
            def __init__(self): self.out = Linear(4, 4)
            def forward(self, x):
                gathered = []
                gathered.extend(x)
                return self.out(torch.cat(gathered, dim=0))
    """)
    assert result.status == "incomplete"
    assert result.failures[0].kind == "unsupported_syntax"
