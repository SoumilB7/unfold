"""A product input cannot re-enable the legacy UNet authority."""
from types import SimpleNamespace

import pytest

from model_unfolder.adapters.diffusor import parser
from model_unfolder.adapters.diffusor import unet_cutover
from model_unfolder.adapters.diffusor.unet_differential import (
    legacy_comparison_enabled, legacy_unet_comparison,
)


def test_old_author_cannot_be_called_without_the_comparison_flag():
    with pytest.raises(RuntimeError, match="restricted to differential"):
        parser._parse_unet_model({}, "unused", [])


def test_comparison_flag_is_scoped_and_restored_on_failure():
    assert not legacy_comparison_enabled()
    with pytest.raises(ValueError):
        with legacy_unet_comparison():
            assert legacy_comparison_enabled()
            with legacy_unet_comparison():
                assert legacy_comparison_enabled()
            assert legacy_comparison_enabled()
            raise ValueError("stop comparison")
    assert not legacy_comparison_enabled()


def test_production_u_shape_uses_cutover_even_with_misleading_config(monkeypatch):
    calls = []
    result_ir = object()
    monkeypatch.setattr(parser, "_shadow_diffusion_root_topology", lambda context:
                        SimpleNamespace(has_value=True, value=SimpleNamespace(kind="u_shaped")))
    monkeypatch.setattr(parser, "_projected_pipeline_handoffs", lambda *a, **k: {})
    monkeypatch.setattr(unet_cutover, "build_unet_cutover", lambda *a, **k:
                        calls.append("new") or SimpleNamespace(ir=result_ir))
    monkeypatch.setattr(parser, "_parse_unet_model", lambda *a, **k:
                        pytest.fail("production entered legacy authority"))
    config = {"_class_name": "Address", "legacy_unet_comparison": True,
              "UNFOLD_UNET_LEGACY": True}
    assert parser.parse(config, context=SimpleNamespace(source_overrides=())) is result_ir
    assert calls == ["new"]
