"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import json
from pathlib import Path
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.mixer_schedule import decoder_mixer_schedule_for_path


_CORPUS = Path(__file__).resolve().parents[2] / "tests" / "sable_test_corpus"


def _config():
    return json.loads(
        (_CORPUS / "qwen3-5-27b-text.json").read_text(encoding="utf-8")
    )["config"]


def _selector(document):
    def select(path):
        current = document
        for part in tuple(path):
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _real_result(config=None):
    config = config or _config()
    context = ParseContext.build(config)
    return decoder_mixer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(config))
