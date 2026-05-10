"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``model_unfolder.adapters.transformer``.
"""

from . import (
    cohere, deepseek, falcon, fallback, jamba, llama,
    mamba, minimax, mistral, qwen, recurrent_gemma, rwkv, zamba,
)
from . import gemma  # gemma/ package — dispatches to gemma2/gemma3/gemma4 internally

# Order matters: more specific adapters first.
# ``gemma`` before ``llama``          (llama once matched "gemma" model_type).
# ``falcon`` before ``llama``         (FalconForCausalLM arches contain "falcon").
# ``recurrent_gemma`` before ``llama`` (distinct model_type, same safety net).
# ``mamba`` before ``llama``          (MambaForCausalLM arches contain no "llama").
# ``zamba`` before ``mamba``          (Zamba has both SSM and attention; more specific).
# ``fallback`` always matches — must be last.
ADAPTERS = [
    deepseek, gemma, jamba, falcon, minimax, mistral,
    qwen, recurrent_gemma, rwkv, cohere, zamba, mamba,
    llama, fallback,
]
