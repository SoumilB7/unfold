"""A product input cannot revive the deleted UNet compatibility authority."""
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

from model_unfolder.adapters.diffusor import parser
from model_unfolder.adapters.diffusor import unet_cutover
from model_unfolder.renderers.html.block_views import registry


def test_old_author_and_comparison_module_are_deleted():
    assert find_spec('model_unfolder.adapters.diffusor.unet') is None
    assert find_spec('model_unfolder.adapters.diffusor.unet_differential') is None
    assert not hasattr(parser, '_parse_unet_model')
    assert not {'unet', 'unet_stage', 'unet_resnet', 'unet_transformer',
                'encoded_text_concat'}.intersection(registry.VIEW_REGISTRY)


@pytest.mark.parametrize('result_ir', [object(), None])
def test_production_u_shape_uses_only_cutover_with_misleading_config(monkeypatch, result_ir):
    calls = []
    limited = SimpleNamespace(warnings=[])
    monkeypatch.setattr(parser, '_parse_projected_denoiser', lambda *a, **k: limited)
    monkeypatch.setattr(parser, '_shadow_diffusion_root_topology', lambda context:
                        SimpleNamespace(has_value=True, value=SimpleNamespace(kind='u_shaped')))
    monkeypatch.setattr(parser, '_projected_pipeline_handoffs', lambda *a, **k: {})
    monkeypatch.setattr(unet_cutover, 'build_unet_cutover', lambda *a, **k:
                        calls.append('new') or SimpleNamespace(ir=result_ir, inventory_result=SimpleNamespace(failure=None)))
    config = {'_class_name': 'Address', 'legacy_unet_comparison': True,
              'UNFOLD_UNET_LEGACY': True}
    actual = parser.parse(config, context=SimpleNamespace(source_overrides=()))
    assert actual is (result_ir if result_ir is not None else limited)
    if result_ir is None:
        assert actual.warnings == ['UNet investigation_missing: the exact selected source reader did not close the UNet root']
    assert calls == ['new']
