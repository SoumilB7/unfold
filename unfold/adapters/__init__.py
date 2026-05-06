"""Adapter registry. Order matters: more specific adapters come first."""
from . import deepseek, llama

ADAPTERS = [deepseek, llama]


def find_adapter(cfg):
    for a in ADAPTERS:
        if a.matches(cfg):
            return a
    return None
