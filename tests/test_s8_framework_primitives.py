"""Serialized class addresses cannot impersonate framework objects."""
import torch.nn as nn

from physics.framework_primitives import capture_framework_types, witness_framework_type


def test_spoofed_module_and_qualname_are_not_primitive_authority():
    captured = capture_framework_types()
    class Impostor(nn.Module):
        def forward(self, value):
            return value + 10
    Impostor.__module__ = nn.SiLU.__module__
    Impostor.__qualname__ = nn.SiLU.__qualname__
    assert witness_framework_type(Impostor(), captured) is None
    assert witness_framework_type(nn.SiLU(), captured).key == "silu"


def test_subclass_is_not_the_closed_framework_object():
    class Custom(nn.SiLU):
        pass
    assert witness_framework_type(Custom(), capture_framework_types()) is None


def test_class_forward_replacement_invalidates_witness(monkeypatch):
    captured = capture_framework_types()
    monkeypatch.setattr(nn.SiLU, "forward", lambda self, value: value + 10)
    assert witness_framework_type(nn.SiLU(), captured) is None


def test_in_place_function_code_replacement_invalidates_witness():
    captured = capture_framework_types()
    original = nn.SiLU.forward.__code__
    try:
        nn.SiLU.forward.__code__ = (lambda self, value: value + 10).__code__
        assert witness_framework_type(nn.SiLU(), captured) is None
    finally:
        nn.SiLU.forward.__code__ = original
