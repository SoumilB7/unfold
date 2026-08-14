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
        "reshape", "reshape", "linear"]


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
