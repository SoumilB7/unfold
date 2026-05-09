"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``unfold.adapters.transformer``.
"""

from . import deepseek, llama

ADAPTERS = [deepseek, llama]

