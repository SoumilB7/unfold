"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``model_unfolder.adapters.transformer``.
"""

from . import deepseek, fallback, llama, minimax, mistral, qwen
from . import gemma  # gemma/ package — dispatches to gemma3/gemma4 internally

# Order matters: more specific adapters first.
# ``gemma`` must run before ``llama`` (llama previously caught "gemma" model_type).
# ``mistral``, ``qwen``, ``minimax`` before ``llama``.
# ``fallback`` always matches — must be last.
ADAPTERS = [deepseek, gemma, minimax, mistral, qwen, llama, fallback]
