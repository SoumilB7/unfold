"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``model_unfolder.adapters.transformer``.
"""

from . import deepseek, gemma4, llama

# Order matters: more specific adapters first.  ``gemma4`` claims its own
# top-level ``model_type`` / architecture and must run before the generic
# llama-family matcher.
ADAPTERS = [deepseek, gemma4, llama]
