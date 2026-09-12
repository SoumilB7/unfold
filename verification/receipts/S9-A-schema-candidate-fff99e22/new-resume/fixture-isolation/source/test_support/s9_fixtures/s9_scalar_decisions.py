"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from copy import deepcopy
from model_unfolder.adapters.transformer import parser
from model_unfolder.evidence import config_access
from model_unfolder.evidence.context import ParseContext, capture_facts
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument, checkpoint_provenance
from model_unfolder.evidence.models import SourceBundle


def _config():
    return {"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100,
            "num_attention_heads": 4, "intermediate_size": 128,
            "tie_word_embeddings": False}


def _parse(checkpoint, *, context=None, document=None):
    if context is None:
        context = ParseContext(SourceBundle(source="local", files=()))
    if document is None:
        document = PreparedDocument(deepcopy(checkpoint), deepcopy(checkpoint),
                                    provenance=checkpoint_provenance(checkpoint))
    with config_access.bound_document(DocumentBinding("root", (), document)), \
            config_access.capture_events(context.config_access), capture_facts(context.facts):
        ir = parser.parse(document.document, context=context)
    return ir, context.facts.typed_records(), context.config_access.events
