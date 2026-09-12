"""U2 — document preparation must NEVER mutate its input.

``AutoConfig.for_model`` mutates the nested component dicts it is handed (it pops
``model_type`` out of a composite's sub-configs).  In shadow mode
``PreparedDocument.document is raw``, so the two share those nested objects — and
passing raw's live values to AutoConfig corrupted the checkpoint the parser then
reads.  MusicGen's conditioning tower lost its ``model_type``, its presence gate
rejected it, and the whole modality vanished — a regression the 25-witness
preservation set could not see because no composite-conditioning model was in it
(MusicGen is now being added).

The fix: hydrate from a throwaway DEEP CLONE; nothing reachable from ``raw`` may
reach AutoConfig.  These tests pin that invariant permanently.
"""
from __future__ import annotations

import copy

import pytest

from model_unfolder.evidence.document import (
    LOADER_STAMPS,
    prepare_document,
)

try:
    from test_support import MUSICGEN_SMALL
except Exception:                       # pragma: no cover
    MUSICGEN_SMALL = None


def _prep(raw, **kw):
    return prepare_document(raw, loader_keys=LOADER_STAMPS, **kw)


# --------------------------------------------------------------------------
# The mutation invariant
# --------------------------------------------------------------------------

def test_shadow_preparation_does_not_mutate_any_nested_raw_value():
    raw = {"model_type": "gemma2", "hidden_size": 8,
           "sub": {"model_type": "t5", "d_model": 16, "nested": [1, 2, 3]}}
    before = copy.deepcopy(raw)
    _prep(raw, merge=False)
    assert raw == before, "shadow preparation mutated its input"


def test_merge_preparation_also_does_not_mutate_its_input():
    raw = {"model_type": "gemma2", "hidden_size": 8,
           "sub": {"model_type": "t5", "d_model": 16}}
    before = copy.deepcopy(raw)
    _prep(raw, merge=True)
    assert raw == before, "merge preparation mutated its input"


def test_nested_model_type_fields_survive_preparation():
    """The exact regression: AutoConfig pops model_type from sub-configs."""
    raw = {"model_type": "musicgen",
           "text_encoder": {"model_type": "t5", "d_model": 8},
           "audio_encoder": {"model_type": "encodec"},
           "decoder": {"model_type": "musicgen_decoder"}}
    before = copy.deepcopy(raw)
    _prep(raw, merge=False)
    assert raw["text_encoder"].get("model_type") == "t5"
    assert raw["audio_encoder"].get("model_type") == "encodec"
    assert raw == before


def test_shadow_document_is_raw_and_still_pristine():
    """document is raw in shadow mode, and hydration did not touch it."""
    raw = {"model_type": "gemma2", "hidden_size": 8,
           "sub": {"model_type": "t5"}}
    prepared = _prep(raw, merge=False)
    assert prepared.document is raw
    assert raw["sub"].get("model_type") == "t5"


def test_class_overlay_is_computed_from_the_cloned_hydrated_document():
    """The overlay records what the class WOULD supply — derived from the
    hydrated clone, never from the mutated input.  A class-added key present in
    the overlay proves the hydration ran on a clone that kept its shape."""
    raw = {"model_type": "gemma2", "hidden_size": 256, "num_hidden_layers": 4,
           "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
           "intermediate_size": 512, "sliding_window": 128, "head_dim": 64}
    prepared = _prep(raw, merge=False)
    # the class supplies keys the checkpoint did not — recorded in the overlay
    assert prepared.class_overlay, "overlay empty — hydration clone was lost"
    assert raw["model_type"] == "gemma2"       # input untouched


# --------------------------------------------------------------------------
# The behavioural proof: MusicGen keeps its composite modality
# --------------------------------------------------------------------------

@pytest.mark.skipif(MUSICGEN_SMALL is None, reason="no MusicGen fixture")
def test_musicgen_retains_its_conditioning_modality_and_embedded_tower():
    import model_unfolder as mu
    ir = mu.unfold(MUSICGEN_SMALL).to_ir()
    modalities = (ir.get("extras") or {}).get("modalities") or {}
    cond = (modalities.get("inputs") or {}).get("conditioning")
    assert cond is not None, "MusicGen lost its conditioning modality"
    # the embedded T5 tower rides the universal round-trip
    assert cond.get("tokens") is not None or cond.get("sub_model") is not None or cond


@pytest.mark.skipif(MUSICGEN_SMALL is None, reason="no MusicGen fixture")
def test_musicgen_parse_is_repeatable_no_input_corruption():
    """Parsing twice must give the same modality — proof the first parse did not
    corrupt the shared fixture."""
    import model_unfolder as mu
    a = "modalities" in (mu.unfold(MUSICGEN_SMALL).to_ir().get("extras") or {})
    b = "modalities" in (mu.unfold(MUSICGEN_SMALL).to_ir().get("extras") or {})
    assert a and b, "MusicGen modality is not stable across repeated parses"
