"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import json
from pathlib import Path
from model_unfolder.evidence.context import ParseContext, slot_parse_context


_CORPUS = Path(__file__).resolve().parents[2] / "tests" / "sable_test_corpus"


def _t5_context():
    config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    return slot_parse_context(ParseContext.build(config), "text_encoder_2")
