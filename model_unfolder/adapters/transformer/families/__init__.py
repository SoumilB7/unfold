"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``model_unfolder.adapters.transformer``.
"""

from . import deepseek, gemma4, llama, mistral, qwen

# Order matters: more specific adapters first.
# ``gemma4`` must run before ``llama`` (which also matches ``gemma`` model_type).
# ``mistral`` and ``qwen`` must run before ``llama``.
ADAPTERS = [deepseek, gemma4, mistral, qwen, llama]
