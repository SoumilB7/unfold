"""The ONE encoder-panel round-trip — adapter-neutral.

Any component slot that carries a transformers encoder/LM config (a diffusion
pipeline's text encoders, a seq2seq composite's conditioning encoder) is read
off the SAME universal transformer adapter that parses those models
standalone, and projected into the ONE facts-only sub-model spec.  Both
adapters import from here, so embedded ≡ standalone parity has a single
implementation (no transformer→diffusor dependency).
"""
from __future__ import annotations

def hydrate_encoder_config_facts(c: dict) -> dict:
    """Fill config-class DEFAULTS invisible in a raw component config.json.

    Thin shim over the ONE document primitive, kept for callers that only want
    the document.  The duplicate implementation that lived here is DELETED: two
    preparations of the same kind inevitably disagreed, and the disagreement was
    invisible because each discarded what the other needed — this one returned an
    EMPTY provenance map whenever the config class rejected a config, so every
    read of that encoder then had no recorded origin at all.  A failed hydration
    does not erase where a field came from: the document simply IS the
    checkpoint, and ``prepare_document`` says so."""
    from .evidence.document import LOADER_STAMPS, prepare_document
    return prepare_document(c, loader_keys=LOADER_STAMPS, merge=False).document


def normalize_encoder_config(c: dict, context=None, binding=None) -> dict:
    """Read an encoder's shape off the ONE universal transformer adapter.

    A pipeline's text-encoder config *is* a transformers config (CLIP, T5,
    Qwen-VL pressed into prompt-encoding duty), so it goes through the same
    parser that handles those models standalone — every dialect, nested
    ``text_config``, GQA, norm kind — and the neutral spec is projected from
    the resulting IR.  No second field-extraction vocabulary lives here.

    U2-R7: ONE preparation boundary.  An enclosing caller that already
    prepared and ENTERED this document (the diffusor slot walk) passes its
    ``binding``; this function then reads ``binding.prepared`` and does NOT
    prepare again or re-enter the scope (a second entry would compose the
    slot address twice).  With no binding, this function IS the boundary:
    it prepares once and holds the bound scope open for its WHOLE body —
    the post-parse evidence reads included — so no read here is unlocated.
    """
    from contextlib import nullcontext

    from .evidence.context import ParseContext
    from .adapters.transformer.parser import parse as _parse_transformer

    from .evidence.document import (
        DocumentBinding, LOADER_STAMPS, prepare_document,
    )
    _prepared = (binding.prepared if binding is not None
                 else prepare_document(c, loader_keys=LOADER_STAMPS,
                                       merge=False))
    c = _prepared.document
    try:
        if context is None:
            context = ParseContext.build(c, source="local")
        # U2-R1 (§5.1): this recursively-parsed slot is its OWN document.  Bind
        # it to the exact component owner, register it on the context (so a
        # later reader finds the right document without guessing from a module
        # name), and ENTER it via its binding — the migrated path where value,
        # address and provenance travel together and object identity is
        # verified.  The address stays () here: it is document-relative, and the
        # enclosing diffusor scope carries the slot's absolute address.
        _owner = getattr(context, "component_namespace", "root")
        _binding = binding if binding is not None \
            else DocumentBinding(_owner, (), _prepared)
        if hasattr(context, "prepared_documents"):
            context.prepared_documents[_owner] = _binding
        from .evidence import config_access as _config_access
        _scope = (_config_access.bound_document(_binding)
                  if binding is None else nullcontext())
        with _scope:
            ir = _parse_transformer(c, context=context)
            if not ir.layers:
                return {}
            return _project_encoder_spec(c, ir, context)
    except Exception:
        return {}


def _project_encoder_spec(c: dict, ir, context) -> dict:
    """Project the parsed IR into the neutral encoder spec.  Runs INSIDE the
    slot's bound document scope, so the evidence reads below stay located."""
    # Grouped, not layer-0: the flat summary fields describe the DOMINANT layer
    # type, and a heterogeneous stack (sliding/global alternation, hybrid
    # full/linear mixers) additionally carries one entry per distinct signature
    # so the tower renders every real layer type — same collapse the main
    # architecture view uses (ir.distinct_layer_groups).
    from .ir import distinct_layer_groups
    groups = distinct_layer_groups(ir.layers)
    dominant = max(groups, key=lambda group: len(group["indices"]))
    layer = dominant["layer"]
    ffn = layer.ffn

    # The recursively parsed ModelIR is the ONLY source.  Re-reading norm
    # spellings or relaying an FFN source owner here creates a second embedded-
    # only authority and breaks standalone parity.
    norm = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
        layer.norm_kind)
    gated = ffn.gated
    act = ffn.activation
    # Flat fields are PROSE/legacy-display only — attention geometry lives on
    # the sub-model spec's typed facts (attention_detail per group), never
    # duplicated here.
    fields = {
        "layers": len(ir.layers),
        "hidden": ir.hidden_size,
        "ffn": ffn.intermediate_size,
        "activation": act,
        "vocab": ir.vocab_size,
        "max_pos": ir.max_position_embeddings,
        "norm": norm,
    }
    out = {k: v for k, v in fields.items() if v}
    if gated is not None:
        out["gated"] = bool(gated)
    # The ONE facts-only sub-model spec — groups, schedule, per-group typed
    # attention/FFN facts, evidence envelopes — replaces every hand-plumbed
    # structural key.  Drill children/cards/regions derive from it at
    # projection time through the same canonical builders the root uses, so a
    # new IR fact reaches every embedded context (at any nesting depth) with
    # zero relay edits here.
    from .submodel import submodel_spec
    # The context is evidence metadata only: ``submodel_spec`` derives exact
    # callable citations from its already-computed typed ReaderResults.  The
    # caller cannot relay an FFN owner/file (the old drift-prone path), and the
    # architectural shape remains solely the typed ModelIR.
    out["sub_model"] = submodel_spec(
        ir, altitude="tower", evidence_context=context)
    # Flat prose fields (title/chips wording) derive from the spec's dominant
    # group — never hand-built a second time.
    spec_groups = out["sub_model"]["groups"]
    dominant_group = max(spec_groups, key=lambda group: group["count"])
    out["attention_detail"] = dominant_group["attention"]
    return out

__all__ = ["hydrate_encoder_config_facts", "normalize_encoder_config"]
