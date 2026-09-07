"""A supplied denoiser must not acquire an invented surrounding pipeline."""
from types import SimpleNamespace

from model_unfolder.adapters.diffusor.unet_projection import project_unet
from model_unfolder.expanded.loop import build_sampling_loop
from model_unfolder.renderers.html.views_diffusion import _build_loop_view, _stub_info


def project(handoffs):
    values = {
        "constructed_modules": {"": {"children": [], "class_name": "ExactRoot"}},
        "constructed_parameter_shapes": {"parameters": {}, "by_module": {"": 0}},
        "constructed_stage_relations": {"producer_stages": [], "intermediate_stages": [],
            "consumer_stages": [], "producer_field": "down", "consumer_field": "up",
            "unresolved_relations": []},
        "primary_state_ports": {"root_inputs": ["actual_input"], "regions": [{"position": 0, "kind": "assignment",
            "stage_fields": [], "constructed_stages": [], "receives_previous_state": False,
            "route": {"kind": "formal", "formal": "actual_input"}}]},
    }
    facts = {"root.denoiser." + key: SimpleNamespace(value=value) for key, value in values.items()}
    return project_unet(facts=facts, handoffs=handoffs, name="Component", architecture="ExactRoot").to_dict()


def test_denoiser_only_has_actual_input_and_no_sampling_pipeline():
    ir = project({})
    render = ir["extras"]["render"]
    assert {row["id"] for row in render["loop_blocks"]} == {"denoiser", "unet_root_input_0"}
    assert build_sampling_loop(ir["extras"]) is None
    svg = _build_loop_view(ir, _stub_info(), "scope")
    assert "actual_input" in svg
    assert all(text not in svg for text in ("VAE", "Scheduler", "Text prompt", "Token embedding", "Noise"))


def test_supplied_scheduler_document_survives_missing_display_label():
    ir = project({"component_presence": {"scheduler": True, "vae": False, "text_encoders": False}})
    ids = {row["id"] for row in ir["extras"]["render"]["loop_blocks"]}
    assert "scheduler" in ids and "vae_decode" not in ids and "text_encoder" not in ids
    assert ir["extras"]["render"]["component_scope"] == "partial_pipeline"
    assert build_sampling_loop(ir["extras"]) is None


def test_supplied_full_pipeline_retains_existing_component_cards():
    ir = project({"component_presence": {"scheduler": True, "vae": True, "text_encoders": True},
                  "text_encoders": ["Actual encoder"], "text_encoder_specs": [{"name": "Actual encoder"}],
                  "scheduler": "Declared scheduler", "vae": {"latent_channels": 4}})
    render = ir["extras"]["render"]
    assert "component_scope" not in render
    assert {"encoder_0", "scheduler", "vae_decode", "denoiser"} <= {row["id"] for row in render["loop_blocks"]}
    assert render["loop_region"]


def test_opaque_failure_preserves_component_scope_and_has_no_sampling_loop():
    from model_unfolder.adapters.diffusor.blocks import diffusion_opaque_render_spec
    from model_unfolder.expanded.loop import build_sampling_loop
    render = diffusion_opaque_render_spec({'component_presence': {
        'text_encoders': False, 'scheduler': False, 'vae': False}})
    assert [block['id'] for block in render['loop_blocks']] == ['denoiser']
    assert render['component_scope'] == 'denoiser'
    assert render['loop_edges'] == []
    assert 'loop_region' not in render
    assert build_sampling_loop({'render': render}) is None
