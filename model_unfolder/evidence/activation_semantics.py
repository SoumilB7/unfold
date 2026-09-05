"""Shared, syntax-level activation protocol vocabulary.

This module normalizes only an already-resolved import target.  It never
selects a model component, scans by class name, or treats a config token as an
executed operation.  Ordinary and routed FFN readers share it so their
activation vocabularies cannot drift.
"""
from __future__ import annotations


FUNCTIONAL_ACTIVATIONS = {
    "torch.nn.functional.gelu": "gelu",
    "torch.nn.functional.glu": "glu",
    "torch.nn.functional.relu": "relu",
    "torch.nn.functional.silu": "silu",
}
MODULE_ACTIVATIONS = {
    "torch.nn.GELU": "gelu",
    "torch.nn.modules.activation.GELU": "gelu",
    "torch.nn.ReLU": "relu",
    "torch.nn.modules.activation.ReLU": "relu",
    "torch.nn.SiLU": "silu",
    "torch.nn.modules.activation.SiLU": "silu",
}
