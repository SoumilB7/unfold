"""Transformer model-family adapters.

These modules translate family-specific HuggingFace config dialects into the
shared transformer IR pieces in ``model_unfolder.adapters.transformer``.
"""

from . import (
    cohere, deepseek, falcon, fallback, gpt_neox, jamba, llama,
    minimax, mistral, qwen, recurrent_gemma, rwkv, zamba,
)
from . import gemma  # gemma/ package — dispatches to gemma2/gemma3/gemma4 internally

# Order matters: more specific adapters first.
# ``gemma`` before ``llama``           (llama once matched "gemma" model_type).
# ``falcon`` before ``llama``          (FalconForCausalLM arches contain "falcon").
# ``recurrent_gemma`` before ``llama`` (distinct model_type, same safety net).
# ``zamba`` before ``llama``           (Zamba has both SSM and attention; more specific).
# ``gpt_neox`` before ``llama``        (distinct model_type, no overlap).
# ``fallback`` always matches — must be last.
# NOTE: ``mamba`` (pure SSM) is intentionally excluded — only LLM families supported.
ADAPTERS = [
    deepseek, gemma, jamba, falcon, minimax, mistral,
    qwen, recurrent_gemma, rwkv, cohere, zamba, gpt_neox,
    llama, fallback,
]
