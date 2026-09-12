"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from model_unfolder.adapters.diffusor.projection_ir import project_diffusion_ir
from model_unfolder.evidence.context import FactLedger, capture_facts
from test_support.s9_fixtures.diffusion_config_binding import _bind


def _projected(tmp_path):
    result, _config_ledger = _bind(tmp_path)
    bound = result.require_value()
    facts = FactLedger()
    with capture_facts(facts):
        projection = project_diffusion_ir(bound)
    return bound, projection, facts.typed_records()
