"""Diffusion (DiT/MMDiT) adapter — detection, IR shape, theme, render health.

These use public diffusers config values as plain dicts without network access.
UNet production tests perform isolated meta-device construction and source
investigation; they do not load weights or run model forward execution.
"""
import pytest

from model_unfolder.renderers.html.card_payload import expand_card_payloads
from model_unfolder import unfold, config_to_ir
from model_unfolder.adapters.diffusor import parser as diffusor
from model_unfolder.adapters.transformer import parser as transformer
from model_unfolder.block_schema import validate_block_tree, validate_click_coupling


# Shared model fixtures live in the importable top-level test_support package
# (§16.1 fixture isolation) — no test module imports another test module.
from test_support import FLUX, PIXART, LLAMA, SDXL_UNET, HYBRID_ENC, MOE_ENC


@pytest.fixture(scope="module")
def sdxl_render():
    """One real render of the unchanged shared input; consumers must not mutate it."""
    diagram = unfold(SDXL_UNET)
    return diagram, diagram.to_ir(), expand_card_payloads(diagram.to_html(standalone=True))


def _unet_fact(ir, leaf):
    fact = ir["extras"]["fact_provenance"]["root.denoiser." + leaf]
    assert fact["status"] == "code_proven"
    return fact["value"]


def _instance_card(ir, path):
    from model_unfolder.lint import _walk_blocks
    return next(block for block in _walk_blocks(ir)
                if block.get("source_instance_path") == path
                and block["id"].startswith("instance_"))


def _assert_instance_drawn(ir, html, path):
    html = expand_card_payloads(html)
    from model_unfolder.lint import _walk_blocks
    # A source-bound call card can be the drawn occurrence while its canonical
    # containment card remains a separate drill. Both carry the same exact path.
    card = next(block for block in _walk_blocks(ir)
                if block.get("source_instance_path") == path
                and f'data-id="{block["id"]}"' in html)
    assert f'data-card-id="{card["id"]}"' in html, path
    assert "root.denoiser.constructed_modules" in card["source_fact_keys"]
    return card


def _own_svgs(html, card_id):
    from model_unfolder.preview import svg_views
    return [svg for label, svg in svg_views(html) if label.split("/")[-1] == card_id]


def _unet_regression_input(name):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parents[1] / "test_support" /
                       "unet_regression_inputs" / (name + ".json")).read_text())


@pytest.mark.parametrize("name", ["kandinsky-original-invalid", "if-first-original-invalid"])
def test_original_explicit_head_inputs_keep_their_specific_failure(name, monkeypatch):
    """The original rejected fixtures cannot pass through an unrelated environment failure."""
    from model_unfolder.adapters.diffusor import unet_cutover
    results = []
    builder = unet_cutover.build_resolved_instance
    def observe(*args, **kwargs):
        result = builder(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(unet_cutover, "build_resolved_instance", observe)
    diagram = unfold(_unet_regression_input(name))
    assert len(results) == 1 and results[0].status == "failed"
    assert results[0].failure.kind == "ConstructionFailed"
    assert results[0].failure.stage == "construct"
    assert "num_attention_heads" in results[0].failure.detail
    assert "naming issue" in results[0].failure.detail
    assert results[0].inventory is None
    assert diagram.ir.construction_summary is None
    assert "UNet investigation_missing:" in diagram.to_html(standalone=True)


def test_incomplete_text_time_fixture_returns_visible_constructor_failure(monkeypatch):
    """The original incomplete fixture remains an actual typed failure control."""
    from model_unfolder.adapters.diffusor import unet_cutover
    from test_support import SDXL_UNET_INCOMPLETE_TEXT_TIME

    results = []
    builder = unet_cutover.build_resolved_instance

    def observe(*args, **kwargs):
        result = builder(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(unet_cutover, "build_resolved_instance", observe)
    diagram = unfold(SDXL_UNET_INCOMPLETE_TEXT_TIME)
    assert len(results) == 1
    result = results[0]
    assert result.status == "failed" and result.inventory is None
    assert result.failure.kind == "ConstructionFailed"
    assert result.failure.stage == "construct"
    assert "NoneType" in result.failure.detail
    assert diagram.ir.construction_summary is None
    assert "root.denoiser.constructed_modules" not in diagram.ir.extras["fact_provenance"]
    assert "UNet investigation_missing:" in diagram.to_html(standalone=True)


def test_diffusor_matches_dit_not_transformer():
    assert diffusor.matches(FLUX) is True
    assert diffusor.matches(PIXART) is True
    # The diffusor adapter must NOT claim ordinary transformer-LLM configs;
    # it is registered before the catch-all transformer adapter.
    assert diffusor.matches(LLAMA) is False
    # The catch-all transformer adapter still matches everything (by design).
    assert transformer.matches(FLUX) is True

    # A framework-protocol spelling may route to the diffusion adapter as an
    # ADDRESS, but cannot author any denoiser structure without source proof.
    # This closes the dangerous fallthrough into the catch-all token adapter.
    fabricated = {**FLUX, "_class_name": "FabricatedDiTTransformer2DModel"}
    assert diffusor.matches(fabricated) is True
    fabricated_ir = config_to_ir(fabricated)
    assert fabricated_ir.layers == []
    assert fabricated_ir.extras["render"]["opaque_layer_block"]["resolved"] is False


def test_flux_layer_count_and_geometry():
    ir = config_to_ir(FLUX)
    # The exact source contains two root stack occurrences with checkpoint-bound
    # counts (19 + 38).  Width/head geometry does not have the same closed
    # constructor binding, so raw config arithmetic must not fill it in.
    assert ir.num_layers == 57
    assert ir.hidden_size == 0
    # No token vocabulary in a denoiser.
    assert ir.vocab_size == 0
    # The dispatch call proves an attention-like lane but not its concrete
    # compute/mask/head protocol; no conventional full-MHA fact is invented.
    attn = ir.layers[0].attention
    assert attn.kind is None
    assert attn.num_kv_heads is attn.num_heads is None
    assert attn.mask == "unknown"
    assert attn.rope_dim is None
    assert ir.layers[0].ffn.kind is None


def test_diffusion_render_spec_theme():
    ir = config_to_ir(FLUX)
    render = ir.extras["render"]
    assert render["family"] == "diffusion"
    # Green (teal) for now — same family theme as LLMs.
    assert render["theme"] == "teal"


def test_diffusion_renders_in_the_green_llm_theme():
    diffusion_html = unfold(FLUX).to_html(standalone=True)
    assert "#0F6E56" in diffusion_html         # teal/green block colour present
    assert "#1E5FB0" not in diffusion_html      # blue palette not used for now
    # Same palette as a transformer.
    transformer_html = unfold(LLAMA).to_html(standalone=True)
    assert "#0F6E56" in transformer_html


def test_denoiser_skeleton_is_drawn():
    diagram = unfold(FLUX)
    html = diagram.to_html(standalone=True)
    for label in ("Denoiser state", "Input op: linear", "Output op: linear", "VAE decode"):
        assert label in html, f"skeleton stage {label!r} not drawn"
    for fabricated in ("Patchify", "Unpatchify", "AdaLN-Out", "-&gt; noise eps"):
        assert fabricated not in html


def test_source_projected_bookend_labels_stay_bounded_without_losing_detail():
    """The diagram label is compact; the card retains the exact operation list."""
    from model_unfolder.adapters.diffusor.blocks import _operation_block_label

    assert _operation_block_label("Input", []) == "Input unresolved"
    assert _operation_block_label("Input", ["linear"]) == "Input op: linear"
    assert _operation_block_label(
        "Output", ["concat", "linear", "reshape"]
    ) == "Output operations"


def test_zero_materialized_layers_never_claim_zero_repetitions():
    """An unresolved count is absence of proof, not a source-proven ``x 0``."""
    from model_unfolder.renderers.html.views import _build_architecture_view

    raw = {
        "layers": [],
        "extras": {"render": {
            "family": "diffusion",
            "opaque_layer_block": {
                "id": "opaque", "kind": "opaque", "role": "opaque",
                "label": "Structure unresolved", "resolved": False,
            },
            "model_blocks": [],
        }},
    }
    svg = _build_architecture_view(raw, {"dominant": None, "blocks": {}}, "uf-u10")
    assert "Structure unresolved" in svg
    assert "x 0" not in svg and "× 0" not in svg


def test_diffusion_model_blocks_are_typed():
    ir = config_to_ir(FLUX)
    blocks = {
        block["id"]: block
        for block in ir.extras["render"]["model_blocks"]
    }
    assert blocks["tok_text"]["diffusion_stage"] == "latent_input"
    assert blocks["embed"]["diffusion_stage"] == "input_projection"
    assert blocks["final_rms"]["diffusion_stage"] == "output_projection"
    assert blocks["lm_head"]["diffusion_stage"] == "denoiser_output"
    assert blocks["embed"]["detail"]["operations"] == ["linear"]
    assert blocks["final_rms"]["detail"]["operations"] == ["linear"]


def test_diffusion_name_from_model_tag_and_stats():
    """Header name comes from the model tag (repo id), and the stats banner shows
    diffusion-specific cells (Timesteps, Latent) instead of Vocab / Context."""
    cfg = {**FLUX, "_repo_id": "black-forest-labs/FLUX.1-dev"}
    ir = config_to_ir(cfg)
    assert ir.name == "FLUX.1-dev"           # not "transformer" / the component path
    html = unfold(cfg).to_html(standalone=True)
    assert "TIMESTEPS" in html and "1,000" in html
    # U13 still owns the scheduler display.  The denoiser's raw in_channels is
    # not an exact state-geometry proof, so the former LATENT/64ch cell is gone.
    assert "LATENT" in html and "64 ch" not in html
    assert "VOCAB" not in html and "CONTEXT" not in html
    # Transformers keep Vocab / Context.
    assert "VOCAB" in unfold(LLAMA).to_html(standalone=True)


def test_loop_blocks_are_typed_with_approved_stages():
    """Every sampling-loop node carries an approved diffusion_stage, so the hero
    view is under the same type guard as the denoiser."""
    from model_unfolder.block_schema import DIFFUSION_STAGES
    ir = config_to_ir(FLUX)
    loop = {b["id"]: b for b in ir.extras["render"]["loop_blocks"]}
    expected = {
        "noise": "noise_input", "timestep": "timestep", "prompt": "prompt",
        "encoder_0": "text_encoder", "encoder_1": "text_encoder",
        "denoiser": "denoiser", "scheduler": "scheduler",
        "vae_decode": "vae_decode", "image": "image_output",
    }
    for bid, stage in expected.items():
        assert loop[bid]["diffusion_stage"] == stage, bid
        assert stage in DIFFUSION_STAGES, stage


def test_pixart_caption_projection_is_explicit_and_code_shaped():
    """A caption dimension cannot manufacture PixArt's projection mechanism.

    This config-only fixture has no exact source owner, so the former
    Linear→GELU→Linear card and its denoiser edge must disappear together.
    """
    d = unfold(PIXART)
    render = d.ir.extras["render"]
    blocks = {b["id"]: b for b in render["loop_blocks"]}
    assert "text_projection" not in blocks
    edges = {(e["from"], e["to"]) for e in render["loop_edges"]}
    assert ("text_encoder", "denoiser") not in edges
    html = expand_card_payloads(d.to_html(standalone=True))
    assert 'data-id="text_projection"' not in html
    assert "PixArtAlphaTextProjection" not in html


def test_norm_elementwise_affine_config_cannot_author_a_norm_claim():
    """A bare config operand is not proof of the constructed norm module.

    Until U10 binds this field to the exact norm owner, changing it must not
    add an affine/non-affine claim or alter the repeated-cell architecture.
    """
    left = config_to_ir({**PIXART, "norm_elementwise_affine": False})
    right = config_to_ir({**PIXART, "norm_elementwise_affine": True})
    assert left.layers == right.layers == []
    for ir in (left, right):
        assert ir.extras["render"]["opaque_layer_block"]["resolved"] is False
        assert not any(
            item.endswith(":norm_elementwise_affine")
            for item in ir.extras["config_access"]["accessed_unconsumed"]
        )


def test_vae_decoder_surfaces_input_conv_and_attention_mid_block():
    cfg = {**PIXART, "_vae_config": FLUX["_vae_config"] | {"mid_block_add_attention": True}}
    d = unfold(cfg)
    vae = next(b for b in d.ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode")
    children = {b["id"]: b for b in vae["children"]}
    assert "vae_conv_in" in children and "vae_mid_block" in children
    assert [op["label"] for op in children["vae_mid_block"]["detail"]["ops"]] == [
        "ResNet", "Attention", "ResNet"
    ]
    html = expand_card_payloads(d.to_html(standalone=True))
    assert 'data-id="vae_conv_in"' in html and 'data-card-id="vae_conv_in"' in html
    assert 'data-id="vae_mid_block"' in html and 'data-card-id="vae_mid_block"' in html


def test_vae_mid_block_requires_evidence_and_honors_explicit_no_attention():
    silent = unfold({**PIXART, "_vae_config": FLUX["_vae_config"]})
    silent_vae = next(
        b for b in silent.ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode"
    )
    silent_ids = {b["id"] for b in silent_vae["children"]}
    assert "vae_conv_in" not in silent_ids and "vae_mid_block" not in silent_ids

    explicit = unfold({
        **PIXART,
        "_vae_config": FLUX["_vae_config"] | {"mid_block_add_attention": False},
    })
    explicit_vae = next(
        b for b in explicit.ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode"
    )
    explicit_children = {b["id"]: b for b in explicit_vae["children"]}
    assert "vae_conv_in" in explicit_children and "vae_mid_block" in explicit_children
    assert [op["label"] for op in explicit_children["vae_mid_block"]["detail"]["ops"]] == [
        "ResNet", "ResNet"
    ]


def test_unknown_diffusion_blocks_render_unresolved():
    from model_unfolder.renderers.html.views import _is_resolved_diffusion_block

    info = {"blocks": {}}
    assert _is_resolved_diffusion_block(True, info, "embed", {"diffusion_stage": "patchify"})
    assert not _is_resolved_diffusion_block(True, info, "new_slot", {"diffusion_stage": "not_approved"})
    assert not _is_resolved_diffusion_block(True, info, "new_slot", {"kind": "linear", "label": "New slot"})
    # An unapproved stage on a loop node renders pale (unresolved) in the hero view.
    assert not _is_resolved_diffusion_block(True, info, "scheduler", {"diffusion_stage": "made_up"})


def test_main_view_is_the_sampling_loop():
    """The hero image is the recursive sampling loop, not the transformer stack."""
    html = expand_card_payloads(unfold(FLUX).to_html(standalone=True))
    assert "SAMPLING LOOP" in html
    assert "sampling step" in html       # honest loop framing (no invented step count)
    assert "× T steps" not in html        # the old placeholder is gone
    assert "↺ t → 0" in html             # the loop frame's repeat pill (engine style)
    # The denoiser is a clickable loop node with a backing card.
    assert 'data-id="denoiser"' in html
    assert 'data-card-id="denoiser"' in html
    # Loop nodes present.
    for node in ("noise", "scheduler", "vae_decode", "image", "timestep"):
        assert f'data-id="{node}"' in html, f"loop node {node!r} missing"


def test_dit_layers_typed_correctly_not_as_llm():
    """Diffusion layers inherit neither LLM nor conventional DiT defaults."""
    html = unfold(FLUX).to_html(standalone=True)
    assert "NoPE" not in html             # Flux uses axial RoPE, not NoPE
    assert "causal" not in html.lower()   # DiT attention is bidirectional
    assert "MM-DiT" not in html
    assert "single-stream" not in html
    assert "Joined-input attention" in html
    ir = config_to_ir(FLUX)
    a0 = ir.layers[0].attention
    assert a0.mask == "unknown" and a0.no_rope is False


def test_denoiser_drills_into_the_dit_stack():
    """Clicking the denoiser opens the transformer architecture one panel deeper:
    its card must embed the DiT stack's clickable layer nodes (attention, etc.)."""
    import re
    html = expand_card_payloads(unfold(FLUX).to_html(standalone=True))
    m = re.search(
        r'data-card-id="denoiser"(.*?)</div>\s*<div class="uf-card-detail', html, re.S
    )
    assert m, "denoiser card not found"
    denoiser_card = m.group(1)
    assert 'data-id="attn"' in denoiser_card      # DiT layer attention is reachable
    assert 'data-id="embed"' in denoiser_card      # patchify
    # Three drill depths exist: L2 loop, L3 DiT blocks, L4 internals.
    depths = sorted(set(re.findall(r'data-depth="(\d+)"', html)))
    assert depths == ["2", "3", "4"]


def test_text_encoders_render_as_separate_blocks():
    """The conditioning shows one block per real encoder (Flux: CLIP + T5) fed by
    a shared prompt — not a single combined 'CLIP + T5' block."""
    ir = config_to_ir(FLUX)
    loop_ids = [b["id"] for b in ir.extras["render"]["loop_blocks"]]
    assert "prompt" in loop_ids
    assert "encoder_0" in loop_ids and "encoder_1" in loop_ids
    html = expand_card_payloads(unfold(FLUX).to_html(standalone=True))
    assert "CLIP" in html and "T5" in html
    # Each encoder is a clickable node with a backing card.
    for nid in ("prompt", "encoder_0", "encoder_1"):
        assert f'data-id="{nid}"' in html and f'data-card-id="{nid}"' in html


def test_text_encoder_breaks_into_drillable_ops():
    """Known encoder ops stay drillable while unknown wiring stays undrawn.

    CLIP has exact placement evidence and therefore norm/residual nodes. T5's
    owner remains unresolved, so its independently proven attention and FFN
    remain clickable without invented norm occurrences or residual adds.
    """
    html = expand_card_payloads(unfold(FLUX).to_html(standalone=True))
    for op in ("embed", "selfattn", "ffn"):
        assert f'data-id="encoder_0_op_{op}"' in html
        assert f'data-card-id="encoder_0_op_{op}"' in html
    assert 'data-id="encoder_1_op_embed"' in html
    assert 'data-card-id="encoder_1_op_embed"' in html
    # T5 is occurrence-exact: block 0 owns the relative-bias producer while
    # later blocks have no proved loop-carried bias transport.  Both groups
    # keep their independently known attention/FFN drills.
    for group in ("g0", "g1"):
        for op in ("selfattn", "ffn"):
            assert f'data-id="encoder_1_{group}_op_{op}"' in html
            assert f'data-card-id="encoder_1_{group}_op_{op}"' in html
    for op in ("norm", "add"):
        assert f'data-id="encoder_0_op_{op}"' in html
        assert f'data-id="encoder_1_op_{op}"' not in html


def test_text_encoder_shows_real_config_dims():
    """When the loader fetched the encoders' configs, the view shows their real
    depth/width/heads/FFN (not a schematic 'N'), distinctly per encoder."""
    specs = diffusor._text_encoder_specs(FLUX)
    structural = [{k: v for k, v in spec.items()
                   if k not in {"ffn_evidence", "ffn_projection_mode",
                                "attention_detail",
                                "sub_model"}}
                  for spec in specs]
    # U7 binds each activation to the exact selected FFN occurrence: CLIP's
    # ACT2FN dispatch consumes quick_gelu, while T5's exact final wrapper slot
    # and boolean construction branch consume dense_act_fn=gelu_new.  Neither
    # value is admitted from config presence alone.  T5's norm remains unknown
    # because its separate encoder/decoder norm owner is still unresolved.
    assert structural == [
        {"name": "CLIP", "family": "CLIP", "layers": 12, "hidden": 768, "ffn": 3072,
         "activation": "quick_gelu", "vocab": 49408, "max_pos": 77,
         "norm": "LayerNorm", "gated": False},
        {"name": "T5", "family": "T5", "layers": 24, "hidden": 4096, "ffn": 10240,
         "activation": "gelu_new", "vocab": 32128, "gated": True},
    ]
    # Attention geometry lives ONLY on the typed sub-model facts — never
    # duplicated as flat scalars (the dead add-on vocabulary this replaced).
    assert [(s["attention_detail"]["kind"], s["attention_detail"]["num_heads"],
             s["attention_detail"]["num_kv_heads"], s["attention_detail"]["head_dim"])
            for s in specs] == [("mha", 12, 12, 64), ("mha", 64, 64, 64)]
    # No parallel free-form FFN envelope OR caller-relayed owner/file remains.
    # The canonical recursively parsed FFN fact is the sole authority; its
    # storage result survives embedded projection without a second source path.
    assert all(
        not ({"ffn_source_owner", "ffn_source_file"} & set(group))
        for spec in specs for group in spec["sub_model"]["groups"])
    assert [s["sub_model"]["groups"][0]["ffn"]["projection_mode"]
            for s in specs] == ["dense", "split"]
    # The typed attention facts ride the same spec: positional scheme + score
    # scaling are evidence, per encoder (CLIP learned-absolute + scaled;
    # T5's exact first layer carries relative bias while its dominant 23-layer
    # group remains unknown + code-proven UNscaled scores).
    # Model-input position addition is not copied onto the attention fact.
    # The flat attention_detail is deliberately the dominant group, not a
    # majority-to-all claim.  The exact first-layer proof stays separately in
    # the canonical grouped schedule.
    assert [(s["attention_detail"]["position_kind"],
             s["attention_detail"].get("scores_scaled", True))
            for s in specs] == [
        ("unknown", True),
        ("unknown", False),
    ]
    t5_groups = specs[1]["sub_model"]["groups"]
    assert [(group["layers"], group["attention"]["position_kind"])
            for group in t5_groups] == [([0], "relative_bias"),
                                        (list(range(1, 24)), "unknown")]
    html = unfold(FLUX).to_html(standalone=True)
    # CLIP is homogeneous; T5 is occurrence-honestly split into its one
    # relative-bias owner and 23 reuse-unproved owners.  The tower still names
    # the total depth, but it may not collapse those into a false ×24 cell.
    assert "× 12" in html and "24 layers" in html
    assert "1 of 24" in html and "23 of 24" in html
    assert "12 heads" in html and "64 heads" in html
    # Width remains an exact typed operand in both group facts, but an
    # unresolved FFN mechanism may not be strengthened into a conventional
    # two-layer arrow merely because both dimensions are known.
    assert [s["sub_model"]["groups"][0]["ffn"]["intermediate_size"]
            for s in specs] == [3072, 10240]


def test_text_encoder_ffn_summary_drill_and_cards_share_one_region():
    """Supporting text towers are not allowed to flatten a gated FFN into the
    old prose-only "two-layer MLP" card. CLIP and T5 consume the same canonical
    FFN resolver but retain their exact dense vs gated source structure."""
    from model_unfolder.opgraph import ffn_region

    ir = config_to_ir(FLUX)
    loop = {b["id"]: b for b in ir.extras["render"]["loop_blocks"]}
    clip = next(c for c in loop["encoder_0"]["children"] if c["id"].endswith("_op_ffn"))
    t5 = next(c for c in loop["encoder_1"]["children"] if c["id"].endswith("_op_ffn"))

    clip_fact = clip["detail"]["ffn"]
    t5_fact = t5["detail"]["ffn"]
    assert ffn_region(clip_fact, clip_fact["hidden"]).template == "dense_mlp"
    assert ffn_region(t5_fact, t5_fact["hidden"]).template == "gated_mlp"
    # Bias is projected from the same selected DenseGated branch.  A second
    # mechanism scan used to lose this conditional owner and return unknown.
    assert t5_fact["bias"] is False
    assert "Two-layer MLP" in clip["description"]
    assert "Gated MLP" in t5["description"] and "SwiGLU" not in t5["description"]
    assert {c["id"] for c in t5["children"]} == {
        "encoder_1_g0_ffn_gate_proj", "encoder_1_g0_ffn_up_proj",
        "encoder_1_g0_ffn_activation", "encoder_1_g0_ffn_multiply",
        "encoder_1_g0_ffn_down_proj",
    }
    assert not ({c["id"] for c in clip["children"]} & {c["id"] for c in t5["children"]})

    diagram = unfold(FLUX)
    html = expand_card_payloads(diagram.to_html(standalone=True))
    # The summary card itself contains the canonical SVG, and every drawn op is
    # coupled to its namespaced leaf card at the next interaction depth.
    for cid in ("encoder_0_op_ffn", "encoder_1_g0_op_ffn"):
        start = html.index(f'data-card-id="{cid}"')
        assert '<div class="uf-card-svg"><svg' in html[start:start + 25000]
    for nid in ("encoder_1_g0_ffn_gate_proj", "encoder_1_g0_ffn_up_proj",
                "encoder_1_g0_ffn_multiply", "encoder_1_g0_ffn_down_proj"):
        assert f'data-id="{nid}"' in html and f'data-card-id="{nid}"' in html
    assert validate_click_coupling(html) == []
    owners = {(event.block_path, event.source_owner, event.component)
              for event in diagram.render_events() if event.view == "ffn"}
    # The drill cites the exact mechanism callable, not the enclosing model
    # stage.  This is what lets nested conformance inspect the same FFN that
    # supplied the typed shape without a family/role search.
    assert (("encoder_0_op_ffn",), "CLIPMLP", "text_encoder") in owners
    assert (("encoder_1_g0_op_ffn",),
            "T5DenseGatedActDense", "text_encoder_2") in owners


def test_text_encoder_ffn_missing_source_stays_opaque():
    from model_unfolder.submodel import submodel_ffn_block
    from model_unfolder.opgraph import ffn_region

    block = submodel_ffn_block(
        {"component": "", "evidence": {"ffn": {"status": "oracle_missing"}}},
        {"ffn": {"kind": "dense", "hidden": 256, "intermediate_size": 1024,
                 "activation": "silu", "gated": True,
                 "structure_status": "oracle_missing"}},
        "unknown_encoder")
    fact = block["detail"]["ffn"]
    region = ffn_region(fact, fact["hidden"])
    assert region.template == "unresolved_storage" and region.resolved is False
    assert [op.kind for op in region.ops] == ["opaque"]
    assert "exact projection storage" in block["description"]


def test_text_encoder_ffn_preserves_fused_gate_up_storage():
    from model_unfolder.submodel import submodel_ffn_block
    from model_unfolder.opgraph import ffn_region

    block = submodel_ffn_block(
        {"component": "",
         "evidence": {"ffn": {"status": "proven", "owner_class": "NovelFusedCell"}}},
        {"ffn": {"kind": "dense", "hidden": 256, "intermediate_size": 1024,
                 "activation": "silu", "gated": True,
                 "structure_status": "proven", "projection_mode": "fused_gate_up"}},
        "fused_encoder")
    fact = block["detail"]["ffn"]
    assert ffn_region(fact, fact["hidden"]).template == "fused_gated_mlp"
    ids = {child["id"] for child in block["children"]}
    assert "fused_encoder_ffn_gate_up_proj" in ids
    assert "fused_encoder_ffn_gate_up_split" in ids
    assert "fused_encoder_ffn_gate_proj" not in ids


def test_text_encoder_falls_back_when_no_config():
    """Without fetched encoder configs, the view stays honest: schematic '× N
    layers', no invented numbers."""
    flux_no_enc = {k: v for k, v in FLUX.items() if k != "_text_encoder_configs"}
    specs = diffusor._text_encoder_specs(flux_no_enc)
    assert specs == [
        {"name": "CLIP", "family": "CLIP"},
        {"name": "T5", "family": "T5"},
    ]
    html = unfold(flux_no_enc).to_html(standalone=True)
    assert "× N" in html


def test_vae_decoder_has_a_drill_view():
    """VAE decode opens its own view, built from the real VAE config (channels,
    upsampling) the loader fetched."""
    ir = config_to_ir(FLUX)
    vae_block = next(b for b in ir.extras["render"]["loop_blocks"] if b["id"] == "vae_decode")
    assert vae_block.get("view") == "vae_decoder"
    assert vae_block["detail"]["block_out_channels"] == [128, 256, 512, 512]
    html = unfold(FLUX).to_html(standalone=True)
    # Real decoder stages drawn compactly: 8x upscale (3 doublings), 128->3 output head.
    assert "Up stage" in html and "Output image head" in html
    assert "8× upscaled" in html
    assert "z₀ (clean)" not in html      # the removed loop-arrow label stays gone


def test_flux_splits_double_and_single_stream_groups():
    """The denoiser layer map must distinguish Flux's 19 double-stream (sequential)
    from its 38 single-stream (parallel) blocks — not collapse them into one."""
    from model_unfolder.renderers.html.metadata import _make_info
    info = _make_info(unfold(FLUX).to_ir())
    groups = info["groups"]
    assert len(groups) == 2
    assert sorted(len(g["indices"]) for g in groups) == [19, 38]


def test_flux_single_stream_block_has_no_text_rail_and_clean_labels():
    """Each stack receives only its positively proven stream relation.

    FLUX's first block returns two state roots and its second joins the inputs.
    Neither label comes from the old family split, and neither invents a
    text/AdaLN side rail.
    """
    ir = config_to_ir(FLUX)
    assert {layer.attention.variant["tag"] for layer in ir.layers} == {
        "two returned states", "explicit join",
    }
    joined = next(layer for layer in ir.layers
                  if layer.attention.variant["tag"] == "explicit join")
    dual = next(layer for layer in ir.layers
                if layer.attention.variant["tag"] == "two returned states")
    assert joined.attention.variant["label"] == ["Joined-input attention"]
    assert dual.attention.variant["label"] == ["Dual-state attention"]
    for layer in (joined, dual):
        assert not {"text_cond", "adaln_cond"} & {
            block["id"] for block in layer.blocks if block.get("lane")}
    html = unfold(FLUX).to_html(standalone=True)
    assert "Joined-input attention" in html


@pytest.mark.parametrize("cfg", [FLUX, PIXART])
def test_diffusion_blocks_and_clicks_valid(cfg):
    ir = config_to_ir(cfg)
    assert validate_block_tree(ir) == []
    html = unfold(cfg).to_html(standalone=True)
    assert validate_click_coupling(html) == []


# Real SDXL-base UNet config shape (+ pipeline wiring).


def test_unet_is_claimed_by_diffusor_not_transformer():
    assert diffusor.matches(SDXL_UNET) is True
    assert config_to_ir(SDXL_UNET).architecture == "UNet2DConditionModel"


def test_unet_does_not_fabricate_cross_attention_mid_when_undeclared():
    """A conv-U conditioned by `cross_attention_dim` but declaring no block-type
    lists (Kandinsky3UNet shape) must NOT get a fabricated cross-attention mid
    block, and must surface the code-defined-placement caveat as a warning."""
    kand = {
        "_class_name": "Kandinsky3UNet", "_repo_id": "kandinsky-community/kandinsky-3",
        "block_out_channels": [384, 768, 1536, 3072], "layers_per_block": 3,
        "attention_head_dim": 64, "cross_attention_dim": 4096, "in_channels": 4,
    }
    # A config signature cannot manufacture a U-shaped computation when the
    # source does not prove the route.
    ir = config_to_ir(kand)
    assert "unet" not in ir.extras
    assert ir.extras["render"]["family"] == "diffusion"
    assert ir.extras["render"]["opaque_layer_block"]["resolved"] is False


def test_sdxl_retains_both_encoders_and_conditional_added_embedding(sdxl_render):
    """Encoder components survive; own added-embedding bindings do not imply universal conditioning."""
    from model_unfolder.lint import _walk_blocks
    _, ir, html = sdxl_render
    encoders = {block["id"] for block in _walk_blocks(ir) if block.get("view") == "text_encoder"}
    assert len(encoders) == 2
    assert all(f'data-card-id="{node}"' in html for node in encoders)
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["add_embedding"]["class_name"] == "TimestepEmbedding"
    shapes = _unet_fact(ir, "constructed_parameter_shapes")["parameters"]
    assert shapes["add_embedding.linear_1.weight"]["shape"] == [1280, 2816]
    _assert_instance_drawn(ir, html, "add_embedding")
    assert "add_embedding" in ir["extras"]["unet"]["bound_target_paths"]
    ports = _unet_fact(ir, "primary_state_ports")
    assert ports["invocation_binding"]["kind"] == "conditional_wrapper"
    assert ports["invocation_binding"]["conditions"]
    assert "as AdaLN modulation" not in html
    assert modules["mid_block.attentions.0.transformer_blocks"]["children"] == [str(i) for i in range(10)]


def test_published_image_decoder_keeps_own_projection_without_text_tower():
    """Published Kandinsky input preserves image projection occurrences without asserting unproved K/V roles."""
    from model_unfolder.lint import _walk_blocks
    diagram = unfold(_unet_regression_input("kandinsky-published"))
    ir, html = diagram.to_ir(), diagram.to_html(standalone=True)
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["encoder_hid_proj"]["class_name"] == "ImageProjection"
    assert modules["add_embedding"]["class_name"] == "ImageTimeEmbedding"
    shapes = _unet_fact(ir, "constructed_parameter_shapes")["parameters"]
    assert shapes["encoder_hid_proj.image_embeds.weight"]["shape"] == [24576, 1280]
    assert shapes["encoder_hid_proj.norm.weight"]["shape"] == [768]
    for path in ("encoder_hid_proj", "encoder_hid_proj.image_embeds", "add_embedding"):
        _assert_instance_drawn(ir, html, path)
    assert "encoder_hid_proj" in ir["extras"]["unet"]["bound_target_paths"]
    assert not any(block.get("view") == "text_encoder" for block in _walk_blocks(ir))
    assert "root.denoiser.context_connections" not in ir["extras"]["fact_provenance"]
    assert "Encoded text" not in html
    assert validate_click_coupling(html) == []


def test_kandinsky3_mid_block_dropped_from_source_evidence():
    """F2: Kandinsky3UNet constructs NO mid block (forward is conv_in -> down -> up
    -> conv_out). With the class source resolved, no bottleneck stage is drawn and
    the false 'Declared by mid_block_type' provenance never appears."""
    kand3 = {
        "_class_name": "Kandinsky3UNet", "_repo_id": "kandinsky-community/kandinsky-3",
        "block_out_channels": [384, 768, 1536, 3072], "layers_per_block": 3,
        "attention_head_dim": 64, "cross_attention_dim": 4096, "in_channels": 4, "out_channels": 4,
        "scheduler": ["diffusers", "DDPMScheduler"],
    }
    ir = config_to_ir(kand3)
    assert "unet" not in ir.extras
    assert ir.extras["render"]["family"] == "diffusion"
    html = unfold(kand3).to_html(standalone=True)
    assert "Mid stage" not in html
    assert "unet_mid" not in html
    assert "Declared by mid_block_type" not in html
    assert "Denoiser internals" in html or "Repeated denoiser" in html


def test_kandinsky3_attention_placement_read_from_code():
    """F2: Kandinsky3UNet declares NO block-type lists — its per-level attention
    placement lives in the class code (add_cross_attention=(F,T,T,T)). Read it so
    the attention this model is known for is SHOWN, as a code-defined self+cross
    cell (Kandinsky3AttentionBlock), never a fabricated Transformer2D/GEGLU."""
    kand3 = {
        "_class_name": "Kandinsky3UNet", "_repo_id": "kandinsky-community/kandinsky-3",
        "block_out_channels": [384, 768, 1536, 3072], "layers_per_block": 3,
        "attention_head_dim": 64, "cross_attention_dim": 4096, "in_channels": 4, "out_channels": 4,
        "scheduler": ["diffusers", "DDPMScheduler"],
    }
    ir = config_to_ir(kand3)
    assert "unet" not in ir.extras
    html = unfold(kand3).to_html(standalone=True)
    # The old U11 interpreter's per-level result is not allowed to leak through
    # a config-authoritative route before the root skip topology is proven.
    assert "code-PLACED" not in html
    assert "Transformer block" not in html
    assert "structure unresolved" in html


def test_vq_decoder_labelled_from_config_declared_vq():
    """F7b: VQ fields prove vector quantization, not the narrower MoVQ family."""
    from model_unfolder.adapters.diffusor.blocks import _vae_decode_label, _vae_class_kind
    assert _vae_class_kind({"num_vq_embeddings": 16384}) == "vq"
    assert _vae_decode_label({"num_vq_embeddings": 16384}) == "VQ decode"
    # class name alone (no VQ config field) is NOT enough — never a class-name bucket.
    assert _vae_class_kind({"class": "VQModel"}) is None
    assert _vae_decode_label({"latent_channels": 4}) == "VAE decode"


def test_valid_synthetic_simple_cross_text_variant_retains_bounded_cells():
    """The documented default-head variant constructs; familiar attention classes cannot invent GEGLU."""
    from model_unfolder.lint import _walk_blocks
    cfg = _unet_regression_input("if-first-synthetic-default-head")
    assert "num_attention_heads" not in cfg and "attention_head_dim" not in cfg
    diagram = unfold(cfg)
    ir, html = diagram.to_ir(), diagram.to_html(standalone=True)
    default_fact = ir["extras"]["fact_provenance"]["root.denoiser.declared_constructor_defaults"]
    assert default_fact["status"] == "class_default"
    defaults = default_fact["value"]
    assert defaults["attention_head_dim"] == {
        "checkpoint": "omitted", "provenance": "class_default", "value": 8}
    line = "Declared class default · attention_head_dim: 8 (checkpoint omitted)"
    denoiser = next(block for block in _walk_blocks(ir) if block["id"] == "denoiser")
    assert line in denoiser["facts"]
    assert line in html
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["down_blocks.1"]["class_name"] == "SimpleCrossAttnDownBlock2D"
    assert modules["encoder_hid_proj"]["class_name"] == "Linear"
    shapes = _unet_fact(ir, "constructed_parameter_shapes")["parameters"]
    assert shapes["encoder_hid_proj.weight"]["shape"] == [4096, 4096]
    _assert_instance_drawn(ir, html, "encoder_hid_proj")
    _assert_instance_drawn(ir, html, "down_blocks.1.attentions.0")
    assert not any(row["class_name"] in {"Transformer2DModel", "GEGLU"} for row in modules.values())
    assert not any(block.get("view") == "runtime_ffn" for block in _walk_blocks(ir))
    assert "root.denoiser.ffn_mechanisms" not in ir["extras"]["fact_provenance"]
    assert "geglu" not in html.lower()
    assert validate_click_coupling(html) == []


SVD_UNET = {
    "_class_name": "UNetSpatioTemporalConditionModel",
    "_repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal",
                         "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal"],
    "up_block_types": ["UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal",
                       "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"],
    "cross_attention_dim": 1024, "num_frames": 25, "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 768, "layers_per_block": 2,
    "in_channels": 8, "out_channels": 4, "num_attention_heads": [5, 10, 20, 20],
    "scheduler": ["diffusers", "EulerDiscreteScheduler"], "_scheduler_config": {"num_train_timesteps": 1000},
}


def test_spatio_temporal_occurrences_and_proved_mixer_argument_are_drawn():
    """Temporal shapes and the proved temporal-result argument survive; no alpha/frame formula is inferred."""
    from model_unfolder.lint import _walk_blocks
    diagram = unfold(SVD_UNET)
    ir, html = diagram.to_ir(), expand_card_payloads(diagram.to_html(standalone=True))
    modules = _unet_fact(ir, "constructed_modules")
    convs = [path for path, row in modules.items() if row["class_name"] == "Conv3d"]
    mixers = [path for path, row in modules.items() if row["class_name"] == "AlphaBlender"]
    assert len(convs) == 44 and len(mixers) == 38
    for path in [*convs, *mixers]:
        _assert_instance_drawn(ir, html, path)
    shapes = _unet_fact(ir, "constructed_parameter_shapes")["parameters"]
    assert shapes["down_blocks.0.resnets.0.temporal_res_block.conv1.weight"]["shape"] == [320, 320, 3, 1, 1]
    assert shapes["down_blocks.0.resnets.0.time_mixer.mix_factor"]["shape"] == [1]
    cells = {block["id"]: block for block in _walk_blocks(ir)
             if block.get("source_owner") == "SpatioTemporalResBlock"}
    assert len(cells) == 22
    for card in cells.values():
        assert "root.denoiser.cell_connections" in card["source_fact_keys"]
        assert card["detail"]["connections"] == [{"source": "call_1", "target": "call_2", "argument_slot": 1}]
    first = _instance_card(ir, "down_blocks.0.resnets.0")
    calls = first["detail"]["connection_calls"]
    assert calls["call_1"]["target"].endswith("__temporal_res_block")
    assert calls["call_2"]["target"].endswith("__time_mixer")
    assert any(all(f'data-id="{calls[key]["target"]}"' in svg for key in ("call_1", "call_2"))
               and "marker-end" in svg for svg in _own_svgs(html, first["id"]))
    assert validate_click_coupling(html) == []


def test_image_unet_has_2d_occurrences_without_temporal_fabrication(sdxl_render):
    """Actual 2D operations remain visible; temporal operations are not fabricated."""
    _, ir, html = sdxl_render
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["conv_in"]["class_name"] == "Conv2d"
    _assert_instance_drawn(ir, html, "conv_in")
    assert not any(row["class_name"] in {"Conv3d", "AlphaBlender", "TemporalResnetBlock"}
                   for row in modules.values())
    assert "AlphaBlender" not in html
    assert "Temporal transformer" not in html
    assert "Spatio-temporal" not in html


def test_sdxl_draws_constructed_transformer2d_and_qualified_ffn(sdxl_render):
    """Constructed Transformer2D depth and its separately proved FFN stay visible."""
    _, ir, html = sdxl_render
    modules = _unet_fact(ir, "constructed_modules")
    path = "down_blocks.2.attentions.0"
    assert modules[path]["class_name"] == "Transformer2DModel"
    assert modules[path + ".transformer_blocks"]["children"] == [str(i) for i in range(10)]
    _assert_instance_drawn(ir, html, path)
    ffn_path = path + ".transformer_blocks.0.ff"
    ffn = _unet_fact(ir, "ffn_mechanisms")[ffn_path]
    assert ffn["gated"] is True and ffn["activation"] == "gelu"
    card = _assert_instance_drawn(ir, html, ffn_path)
    assert card["view"] == "runtime_ffn"
    for child in card["children"]:
        if child.get("role") == "operation":
            assert f'data-id="{child["id"]}"' in html


def test_unet_context_argument_does_not_assert_unproved_kv_role(sdxl_render):
    """A source-bound external argument remains distinct from an unproved K/V role."""
    _, ir, html = sdxl_render
    path = "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
    value = _unet_fact(ir, "context_connections")[path]
    assert value["source_formal"] == value["target_formal"] == "encoder_hidden_states"
    assert value["cross_attention"] == "investigation_missing"
    card = _assert_instance_drawn(ir, html, path)
    assert card["view"] == "runtime_context_connection"
    assert value["reason"] in card["facts"]


def test_simple_cross_stage_occurrences_are_constructed_and_carded():
    """The original valid variant retains all nine exact stages without positional role invention."""
    DEEPFLOYD = {
        "_class_name": "UNet2DConditionModel", "in_channels": 3, "out_channels": 6,
        "block_out_channels": [320, 640, 1280, 1280], "layers_per_block": 3,
        "cross_attention_dim": 4096,
        "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D",
                             "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D"],
        "up_block_types": ["SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D",
                           "SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
        "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    }
    diagram = unfold(DEEPFLOYD)
    ir, html = diagram.to_ir(), diagram.to_html(standalone=True)
    relation = _unet_fact(ir, "constructed_stage_relations")
    assert relation["producer_stages"] == [f"down_blocks.{i}" for i in range(4)]
    assert relation["consumer_stages"] == [f"up_blocks.{i}" for i in range(4)]
    assert relation["intermediate_stages"] == ["mid_block"]
    modules = _unet_fact(ir, "constructed_modules")
    for path in [*relation["producer_stages"], "mid_block", *relation["consumer_stages"]]:
        _assert_instance_drawn(ir, html, path)
    assert modules["down_blocks.0"]["children"] == ["resnets", "downsamplers"]
    assert modules["up_blocks.3"]["children"] == ["resnets"]
    assert "attentions" in modules["down_blocks.1"]["children"]
    assert diagram.ir.construction_summary.stage_count == 9
    assert validate_click_coupling(html) == []


def test_unet_structure_has_constructed_stages_and_nonzero_summary(sdxl_render):
    """Stage membership, own population and quantities come from qualified facts."""
    diagram, ir, html = sdxl_render
    relation = _unet_fact(ir, "constructed_stage_relations")
    assert relation["producer_stages"] == [f"down_blocks.{i}" for i in range(3)]
    assert relation["consumer_stages"] == [f"up_blocks.{i}" for i in range(3)]
    assert relation["intermediate_stages"] == ["mid_block"]
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["down_blocks.0.resnets"]["children"] == ["0", "1"]
    assert "attentions" not in modules["down_blocks.0"]["children"]
    assert "attentions" in modules["down_blocks.1"]["children"]
    assert diagram.ir.layers == []
    assert diagram.ir.construction_summary.stage_count == 7
    assert diagram.ir.construction_summary.parameter_count == _unet_fact(
        ir, "constructed_parameter_shapes")["total"] > 0
    for path in [*relation["producer_stages"], "mid_block", *relation["consumer_stages"]]:
        _assert_instance_drawn(ir, html, path)


def test_unet_component_entry_and_constructed_stages_couple(sdxl_render):
    """The component entry opens actual down/mid/up stages and backed skip routes."""
    diagram, ir, html = sdxl_render
    assert validate_block_tree(diagram.ir) == []
    assert validate_click_coupling(html) == []
    assert ir["component_entry"]["root_id"] == "denoiser"
    assert "DENOISER COMPONENT" in html
    assert "SAMPLING LOOP" not in html
    relation = _unet_fact(ir, "constructed_stage_relations")
    assert relation["skip_route"]["producer"] and relation["skip_route"]["consumer"]
    assert len(ir["extras"]["unet"]["skip_routes"]) == 6
    assert diagram.ir.name == "stable-diffusion-xl-base-1.0"
    assert "DENOISER LAYER MAP" not in html


def test_unet_unsupported_stage_factory_is_a_visible_typed_failure(monkeypatch):
    """A nonexistent factory is reported, without inventing a constructed stage."""
    from model_unfolder.adapters.diffusor import unet_cutover
    results = []
    builder = unet_cutover.build_resolved_instance
    def observe(*args, **kwargs):
        result = builder(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(unet_cutover, "build_resolved_instance", observe)
    cfg = {
        **SDXL_UNET,
        "down_block_types": ["DownBlock2D", "CustomMagicDown", "CrossAttnDownBlock2D"],
    }
    diagram = unfold(cfg)
    assert len(results) == 1 and results[0].status == "failed"
    assert results[0].failure.kind == "ConstructionFailed"
    assert results[0].failure.stage == "construct"
    assert "CustomMagicDown" in results[0].failure.detail
    assert results[0].inventory is None
    assert "root.denoiser.constructed_modules" not in diagram.ir.extras["fact_provenance"]
    html = diagram.to_html(standalone=True)
    assert "CustomMagicDown" in html
    assert "UNet investigation_missing:" in html


def test_dit_dialect_field_aliases_require_exact_source_bindings():
    """Familiar DiT fields supply operands only after source binds their role."""
    source_missing = config_to_ir({
        "_class_name": "UninstalledCustomTransformer2DModel",
        "num_mmdit_layers": 4, "num_single_dit_layers": 32,
        "attention_head_dim": 256, "num_attention_heads": 8,
        "joint_attention_dim": 2048, "in_channels": 4,
    })
    assert source_missing.num_layers == 0
    assert source_missing.hidden_size == 0
    assert source_missing.extras["render"]["opaque_layer_block"]["resolved"] \
        is False

    aura = config_to_ir({
        "_class_name": "AuraFlowTransformer2DModel", "num_mmdit_layers": 4,
        "num_single_dit_layers": 32, "attention_head_dim": 256,
        "num_attention_heads": 8, "joint_attention_dim": 2048, "in_channels": 4,
    })
    # Installed source proves two exact repeated containers and binds only
    # their count operands.  The familiar width-like fields still cannot
    # author a hidden width without an exact source use.
    assert aura.num_layers == 36 and aura.hidden_size == 0

    hunyuan = config_to_ir({
        "_class_name": "HunyuanDiT2DModel", "num_layers": 40,
        "num_attention_heads": 16, "hidden_size": 1408, "cross_attention_dim": 1024,
    })
    # Installed source closes Hunyuan's exact repeated container and its bound
    # num_layers operand as the same general rule used for Aura.
    assert hunyuan.num_layers == 40 and hunyuan.hidden_size == 0
    assert all(layer.attention.mask == "unknown" for layer in hunyuan.layers)


def test_pixart_single_stream_only():
    ir = config_to_ir(PIXART)
    # This local fixture has no exact source bundle. Neither the layer count nor
    # its supposed stream kind may be reconstructed from raw config fields.
    assert ir.num_layers == 0
    assert ir.hidden_size == 0
    assert ir.extras["render"]["opaque_layer_block"]["resolved"] is False


# --- Diffusion auto-depth harness: recursive denoiser conformance ------------
# The package bakes every drill depth into the standalone HTML up front, so
# click-coupling over the FULL html is the recursive-leaf guarantee (every
# clickable node at every depth resolves to a card). This locks the structural
# depth (cross-attn sublayer, AdaLN gates, VAE/encoder drills) across families.

_AURAFLOW = {"_class_name": "AuraFlowTransformer2DModel", "num_mmdit_layers": 4,
             "num_single_dit_layers": 32, "attention_head_dim": 256,
             "num_attention_heads": 8, "joint_attention_dim": 2048, "in_channels": 4}
_HUNYUANDIT = {"_class_name": "HunyuanDiT2DModel", "num_layers": 40,
               "num_attention_heads": 16, "hidden_size": 1408, "cross_attention_dim": 1024}

_DEPTH_FIXTURES = {"flux_mmdit": FLUX, "pixart_cross": PIXART, "sdxl_unet": SDXL_UNET,
                   "auraflow": _AURAFLOW, "hunyuandit": _HUNYUANDIT}


def _walk_blocks(blocks):
    for b in blocks or []:
        if isinstance(b, dict):
            yield b
            yield from _walk_blocks(b.get("children"))


def _explained_call_target_leaf(block, blocks, provenance):
    """An own-card explanation is a leaf; this does not assert target navigation."""
    from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY

    key = "root.denoiser.primary_state_ports"
    target_id = block.get("target")
    targets = [candidate for candidate in blocks if candidate.get("id") == target_id]
    if not target_id or target_id == block.get("id") or len(targets) != 1:
        return False
    target = targets[0]
    path = block.get("source_instance_path")
    return bool(
        path and target.get("source_instance_path") == path
        and block.get("source_fact_keys") == [key]
        and provenance.get(key, {}).get("status") == "code_proven"
        and block.get("facts") == [
            "Target of this call at the corresponding recorded loop slot"]
        and (target.get("view") in VIEW_REGISTRY
             or target.get("description") or target.get("children"))
    )


@pytest.mark.parametrize("poison", [
    None, "missing_target", "wrong_path", "empty_fact", "plain_bare",
    "self_target", "target_cycle", "unqualified", "foreign_fact", "duplicate_target",
])
def test_explained_call_target_leaf_keeps_reference_constraints(poison):
    from copy import deepcopy

    key = "root.denoiser.primary_state_ports"
    block = {"id": "slot", "target": "instance", "source_instance_path": "down_blocks.0",
             "source_fact_keys": [key],
             "facts": ["Target of this call at the corresponding recorded loop slot"]}
    target = {"id": "instance", "source_instance_path": "down_blocks.0",
              "description": "Constructed occurrence with its own recorded shapes"}
    provenance = {key: {"status": "code_proven"}}
    blocks = [block, target]
    if poison == "missing_target":
        block["target"] = "absent"
    elif poison == "wrong_path":
        target["source_instance_path"] = "up_blocks.0"
    elif poison == "empty_fact":
        block["facts"] = []
    elif poison == "plain_bare":
        block.clear()
        block["id"] = "slot"
    elif poison == "self_target":
        block["target"] = "slot"
    elif poison == "target_cycle":
        target.pop("description")
        target["target"] = "slot"
    elif poison == "unqualified":
        provenance[key]["status"] = "investigation_missing"
    elif poison == "foreign_fact":
        block["source_fact_keys"] = ["root.other_denoiser.primary_state_ports"]
    elif poison == "duplicate_target":
        blocks.append(deepcopy(target))
    assert _explained_call_target_leaf(block, blocks, provenance) is (poison is None)


@pytest.mark.parametrize("name", sorted(_DEPTH_FIXTURES))
def test_diffusion_recursive_depth_conforms(name):
    cfg = _DEPTH_FIXTURES[name]
    d = unfold(cfg)
    ir = d.ir
    # 1. Recursion bottoms out in a view, description, children, or an exact
    # source-qualified call-target explanation on the reference's own card.
    # Such a leaf does not provide alias-to-canonical navigation today.
    from model_unfolder.renderers.html.block_views.registry import VIEW_REGISTRY
    render = (ir.extras or {}).get("render") or {}
    trees = [getattr(L, "blocks", []) for L in ir.layers] + \
            [render.get("model_blocks") or [], render.get("loop_blocks") or []]
    blocks = [block for tree in trees for block in _walk_blocks(tree)]
    provenance = (ir.extras or {}).get("fact_provenance", {})
    reference_leaves = []
    for block in blocks:
        if block.get("static"):
            continue
        if block.get("view") in VIEW_REGISTRY or block.get("description") or block.get("children"):
            continue
        assert _explained_call_target_leaf(block, blocks, provenance), \
            f"{name}: block {block.get('id')!r} is a bare leaf (no view/description/explanation)"
        reference_leaves.append(block)
    if name == "sdxl_unet":
        expected = {
            f"{prefix}__slot_{slot}": f"instance_{direction}_blocks__{slot}"
            for prefix, direction in (
                ("unet_primary_region_3__iteration__when_true__call", "down"),
                ("unet_primary_region_3__iteration__when_false__when_true__operand_0__call", "down"),
                ("unet_primary_region_3__iteration__when_false__when_false__call", "down"),
                ("unet_primary_region_6__iteration__when_true__call", "up"),
                ("unet_primary_region_6__iteration__when_false__call", "up"),
            ) for slot in range(3)
        }
        assert len(reference_leaves) == 15
        assert {block["id"]: block["target"] for block in reference_leaves} == expected
    # 2. recursive coupling: every clickable node at every drill depth → a card.
    html = expand_card_payloads(d.to_html(standalone=True))
    assert validate_block_tree(ir) == []
    assert validate_click_coupling(html) == []
    from html import escape
    for block in reference_leaves:
        node_id = escape(block["id"], quote=True)
        assert f'<g class="uf-node" data-id="{node_id}">' in html
        own_card = html.split(f'data-card-id="{node_id}"', 1)[1].split(
            '<div class="uf-card-detail', 1)[0]
        assert f'<div class="uf-card-title">{escape(block["label"])}</div>' in own_card
        assert f'<span class="uf-fact">{escape(block["facts"][0])}</span>' in own_card


# Real FLUX.1-dev model_index.json wiring (the pipeline component map).
FLUX_INDEX = {
    "_class_name": "FluxPipeline",
    "_diffusers_version": "0.30.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def test_diffusion_loader_merges_pipeline_and_denoiser(tmp_path, monkeypatch):
    """By-ID loading: model_index.json + transformer/config.json -> merged config.

    Mocks hf_hub_download so the denoiser component config and the pipeline
    component map are combined the way a real diffusers repo would yield them.
    """
    import json
    import huggingface_hub
    from model_unfolder.adapters.diffusor.loader import load_diffusion_config_by_id

    def fake_download(repo_id, filename, subfolder=None, token=None):
        if subfolder == "transformer" and filename == "config.json":
            data = FLUX
        elif filename == "model_index.json" and not subfolder:
            data = FLUX_INDEX
        else:
            raise FileNotFoundError(f"404 {subfolder}/{filename}")
        p = tmp_path / f"{subfolder or ''}_{filename.replace('/', '_')}"
        p.write_text(json.dumps(data))
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    merged = load_diffusion_config_by_id("black-forest-labs/FLUX.1-dev")
    assert merged is not None
    # Denoiser identity is preserved (used for detection + arch name)...
    assert merged["_class_name"] == "FluxTransformer2DModel"
    # ...and the pipeline wiring is merged in for skeleton naming.
    assert merged["_pipeline_class_name"] == "FluxPipeline"
    assert merged["text_encoder"][1] == "CLIPTextModel"

    ir = config_to_ir(merged)
    assert ir.num_layers == 57
    assert ir.extras["render"]["theme"] == "teal"
    loop_ids = {block["id"] for block in ir.extras["render"]["loop_blocks"]}
    assert {"encoder_0", "encoder_1"} <= loop_ids
    assert not any(
        edge["from"] in {"encoder_0", "encoder_1"}
        and edge["to"] == "denoiser"
        for edge in ir.extras["render"]["loop_edges"])


def test_diffusion_loader_returns_none_for_non_diffusion(tmp_path, monkeypatch):
    """A repo with no model_index.json isn't a diffusion pipeline -> None (so the
    by-ID path falls through to the normal raw-config handling)."""
    import huggingface_hub
    from model_unfolder.adapters.diffusor.loader import load_diffusion_config_by_id

    def fake_download(repo_id, filename, subfolder=None, token=None):
        raise FileNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert load_diffusion_config_by_id("some/llm") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# DiT coverage audit classes (2026-06: video DiTs, UNet-by-id, MoE-DiT,
# non-KL VAEs, conditioning-style variants) — synthetic configs, no network.
# ---------------------------------------------------------------------------

WAN_STYLE = {
    "_class_name": "WanTransformer3DModel", "_diffusers_version": "0.33.0",
    "dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30,
    "attention_head_dim": 128, "text_dim": 4096, "freq_dim": 256,
    "in_channels": 16, "out_channels": 16, "patch_size": [1, 2, 2],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
}

COGVIDEO_STYLE = {
    "_class_name": "CogVideoXTransformer3DModel", "_diffusers_version": "0.31.0",
    "num_attention_heads": 48, "attention_head_dim": 64, "num_layers": 42,
    "text_embed_dim": 4096, "time_embed_dim": 512, "in_channels": 16,
    "patch_size": 2, "activation_fn": "gelu-approximate",
}


def test_dit_norm_kind_is_code_authoritative_and_wiring_stays_independent():
    """A norm selector spelling cannot manufacture kind, placement, or skips."""
    variants = [
        PIXART,
        {**PIXART, "norm_type": "rms_norm"},
        {k: v for k, v in PIXART.items() if k not in ("norm_type", "norm_eps")},
    ]
    diagrams = [unfold(cfg).ir for cfg in variants]
    assert all(ir.layers == [] for ir in diagrams)
    assert all(ir.extras["render"]["opaque_layer_block"]["resolved"] is False
               for ir in diagrams)
    assert all("rms_norm" not in str(ir.extras["render"]).lower()
               for ir in diagrams)


def test_clickable_highlight_is_image_only():
    """The Dable amber-border overlay marks clickable blocks in the IMAGE pass only —
    it is injected into the extracted svg before rasterizing and must NEVER appear in
    the shipped HTML."""
    from model_unfolder.preview import _with_clickable_highlight, _CLICKABLE_HIGHLIGHT
    svg = '<svg viewBox="0 0 10 10"><g class="uf-node" data-id="x"><rect/></g></svg>'
    out = _with_clickable_highlight(svg)
    assert out.startswith("<svg") and _CLICKABLE_HIGHLIGHT in out, "overlay must inject into the svg"
    # the shipped document never carries the overlay's amber stroke (it has its own
    # .uf-node hover/select CSS — that's the real product, distinct from this overlay).
    html = unfold(FLUX).to_html(standalone=True)
    assert "FFC400" not in html and _CLICKABLE_HIGHLIGHT not in html, "overlay leaked into shipped HTML"


def test_inspect_code_resolves_diffusion_norm_from_diffusers_source():
    """When the config is silent on the norm type (FLUX & most DiTs), `inspect_code`
    reads it from the diffusers BLOCK class (AdaLayerNormZero → LayerNorm), tier-2,
    so the typed layer retains that independently proven norm kind.  The first
    FLUX cell preserves its independently proven attention beside an opaque
    FFN and one wiring-unresolved marker; an unknown branch must not erase a
    proven sibling or manufacture norm placement.
    Since C4 the resolution is ALWAYS-ON in the diffusor adapter (root-scoped
    class evidence) — it no longer hides behind the inspect_code flag, so the
    plain unfold resolves too whenever the source is installed."""
    import importlib.util
    if importlib.util.find_spec("diffusers") is None:
        return  # diffusers not installed — the code path can't run
    from model_unfolder.evidence.sources import resolve_source_files
    if not resolve_source_files(FLUX, source="local").files:
        return  # installed diffusers doesn't define this class — skip

    layer = unfold(FLUX, inspect_code=True).ir.layers[0]
    assert layer.norm_kind == "layernorm"
    assert layer.norm_placement == "unknown"
    assert layer.residual_topology == "unknown"
    by = {b["id"]: b for b in layer.blocks}
    assert tuple(by) == ("attn", "wiring_unresolved", "ffn")
    assert by["wiring_unresolved"]["resolved"] is False
    assert by["ffn"]["detail"]["ffn"]["kind"] is None


def test_cross_attn_dit_preserves_mechanisms_without_inventing_wiring():
    """A config-only cross-attention dialect cannot author three sublayers."""
    d = unfold(PIXART)
    assert d.ir.layers == []
    with_norm_diagram = unfold({**PIXART, "cross_attn_norm": True})
    assert with_norm_diagram.ir.layers == []
    assert not any(
        item.endswith(":cross_attn_norm")
        for item in with_norm_diagram.ir.extras[
            "config_access"]["accessed_unconsumed"]
    )
    html = d.to_html(standalone=True)
    assert "Cross-Attention" not in html
    assert "Repeated denoiser" in html and "structure unresolved" in html
    assert validate_click_coupling(html) == []


def test_video_dit_detected_and_honest():
    """Transformer3DModel classes are diffusion denoisers — never the LLM
    adapter (the Wan misparse: hidden 0, no loop, fake decoder)."""
    d = unfold(WAN_STYLE)
    ir = d.to_ir()
    assert (ir["extras"].get("render") or {}).get("family") == "diffusion"
    assert ir["hidden_size"] == 0
    assert len(ir["layers"]) == 30
    assert all(layer["attention"]["kind"] is None for layer in ir["layers"])
    html = d.to_html(standalone=True)
    # Adapter routing remains correct, but raw dim/ffn/video fields cannot fill
    # an absent source boundary.
    assert "Cross-Attention" not in html
    assert "MM-DiT" not in html
    assert ">Frames<" not in html
    assert "Output domain unresolved" in html
    assert validate_click_coupling(html) == []


def test_concat_joint_video_dit_is_not_called_dual_stream():
    """A text_embed_dim spelling proves neither joint nor dual-stream attention."""
    html = unfold(COGVIDEO_STYLE).to_html(standalone=True)
    assert "Joint attention — concatenated text + latent sequence" not in html
    assert "MM-DiT" not in html


def test_unet_is_a_loadable_denoiser_key():
    """UNet pipelines (SD1.5/SDXL/Kandinsky) load by id — the by-id loader must
    accept 'unet' as a denoiser component, not only 'transformer'."""
    from model_unfolder.adapters.diffusor.loader import _DENOISER_KEYS
    assert "unet" in _DENOISER_KEYS


def test_moe_dit_counts_do_not_manufacture_a_router():
    """Expert counts are geometry, not proof that this exact DiT routes."""
    cfg = {"_class_name": "HiDreamImageTransformer2DModel",
           "num_layers": 4, "num_attention_heads": 20, "attention_head_dim": 128,
           "num_routed_experts": 4, "num_activated_experts": 2,
           "joint_attention_dim": 4096}
    ir = unfold(cfg).to_ir()
    assert len(ir["layers"]) == 4
    assert all(layer["ffn"]["kind"] is None for layer in ir["layers"])
    assert "num_experts" not in str(ir["extras"]["render"])
    assert "Mixture of Experts" not in unfold(cfg).to_html(standalone=True)


def test_non_kl_vae_stays_honest():
    """DC-AE (Sana) spells channels decoder_block_out_channels and mixes block
    types per stage: stages render with channel chips but NO invented
    'N× ResNet' claim and no dead output-head click."""
    cfg = dict(FLUX)
    cfg["_vae_config"] = {
        "_class_name": "AutoencoderDC",
        "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
        "latent_channels": 32,
    }
    html = expand_card_payloads(unfold(cfg).to_html(standalone=True))
    assert validate_click_coupling(html) is None or validate_click_coupling(html) == []
    i = html.find('data-card-id="vae_decoder_block_1"')
    assert i > 0
    assert "× ResNet" not in html[i:i + 600]


def test_vae_decoder_upsample_matches_diffusers_placement():
    """diffusers' Decoder puts an upsampler on every up block EXCEPT the final
    one (add_upsample = not is_final_block).  Execution order runs deepest ->
    shallowest, and the drill ids count block_N (first executed) down to
    block_1 (last).  So block_N..block_2 upsample; block_1 does NOT.  The old
    drawing had this exactly reversed (missing on the first, fabricated on the
    last) — in every AutoencoderKL gallery including blessed SDXL."""
    cfg = dict(SDXL_UNET)
    cfg["_vae_config"] = {
        "_class_name": "AutoencoderKL",
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2, "latent_channels": 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
        "norm_num_groups": 32,
    }
    html = expand_card_payloads(unfold(cfg).to_html(standalone=True))

    def facts_of(cid: str) -> str:
        i = html.find(f'data-card-id="{cid}"')
        assert i > 0, cid
        return html[i:i + 1200]

    # first executed (deepest, 512 ch) DOES upsample …
    assert "↑2× spatial" in facts_of("vae_decoder_block_4")
    # … the final (128 ch) does NOT — nothing fabricated after it.
    assert "↑2× spatial" not in facts_of("vae_decoder_block_1")
    assert "↑2× spatial" in facts_of("vae_decoder_block_2")
    assert validate_click_coupling(html) == []


def test_swiglu_video_dit_ffn_is_gated():
    """A swiglu config token cannot manufacture a gated FFN without source."""
    cfg = dict(COGVIDEO_STYLE, _class_name="MochiTransformer3DModel",
               activation_fn="swiglu")
    ir = unfold(cfg).to_ir()
    assert len(ir["layers"]) == 42
    assert all(layer["ffn"]["gated"] is None for layer in ir["layers"])
    assert "SwiGLU" not in unfold(cfg).to_html(standalone=True)


def test_video_latent_shape_uses_declared_temporal_geometry():
    """Raw temporal geometry cannot define the denoiser or output state shape."""
    cfg = dict(COGVIDEO_STYLE, sample_height=60, sample_width=90,
               sample_frames=49, temporal_compression_ratio=4)
    html = unfold(cfg).to_html(standalone=True)
    assert "16 × 13 × 30 × 45" not in html
    assert "Output domain unresolved" in html
    html_wan = unfold(WAN_STYLE).to_html(standalone=True)
    assert "shape [16 channels]" not in html_wan


def test_scheduler_step_renders_a_clean_combine_not_floating_ops():
    """The scheduler step opens the update rule as a purpose-built graph: the
    denoiser's prediction is scaled and combined with z_t into ONE ⊕ → z_{t-1}.
    (Regression: the declared-ops chain floated/duplicated the ⊕ because the
    combine merges the primary latent with a side-scaled input — same failure mode
    the self-conditioning view hit.)"""
    from model_unfolder.adapters.diffusor.blocks import _scheduler_step_view

    # flow-matching family (FLUX / FlowMatchEuler): velocity prediction, Euler step
    sv = _scheduler_step_view({"scheduler": "Flow Match Euler", "scheduler_flow_matching": True})
    assert sv["view"] == "scheduler_step"
    s = sv["detail"]["scheduler_step"]
    assert s["sym"] == "v̂" and "v̂" in s["step_label"] and "+" in s["step_label"]

    html = unfold(FLUX).to_html(standalone=True)
    assert "from denoiser" in html and "current latent" in html
    assert validate_click_coupling(html) == []            # no floating/orphan ⊕

    # epsilon family from a declared prediction_type → z_t − σ_t·ε̂
    eps = _scheduler_step_view({"scheduler": "DDIM", "scheduler_train_timesteps": 1000,
                                "scheduler_prediction_type": "epsilon"})
    assert eps["view"] == "scheduler_step"
    assert eps["detail"]["scheduler_step"]["sym"] == "ε̂"
    # an undeclared scheduler keeps the honest prose card (no fabricated step)
    assert _scheduler_step_view({}) == {}


def test_scheduler_display_names_handle_acronym_runs():
    """CogVideoXDDIMScheduler must not camel-split into 'Cog Video XDDIM' —
    oddballs live in typing.yaml's scheduler_display overrides."""
    from model_unfolder.adapters.diffusor.parser import _scheduler_geom
    assert _scheduler_geom({"scheduler": ["diffusers", "CogVideoXDDIMScheduler"]})["scheduler"] == "CogVideoX DDIM"
    assert _scheduler_geom({"scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]})["scheduler"] == "Flow Match Euler"
    assert _scheduler_geom({"scheduler": ["diffusers", "UniPCMultistepScheduler"]})["scheduler"] == "UniPC Multistep"


def test_ops_region_declared_side_inputs():
    """A declared {"kind": "input"} op is a side source: wired only by `from`,
    never advancing the implicit chain — and it gets a derived card."""
    from model_unfolder.labels import cards_from_region
    from model_unfolder.opgraph import ops_region
    r = ops_region([
        {"id": "pred", "kind": "input", "label": "prediction"},
        {"id": "scale", "kind": "elementwise", "fn": "mul", "from": ["pred"]},
        {"id": "step", "kind": "elementwise", "fn": "add", "from": ["hidden", "scale"]},
    ], rid="s")
    assert r.merges() == ["step"]
    assert [(e.src, e.dst) for e in r.edges] == [("pred", "scale"), ("hidden", "step"), ("scale", "step")]
    ids = [c["id"] for c in cards_from_region(r)]
    assert ids == ["pred", "scale", "step"]               # side input has a card; hidden doesn't


def test_sampling_loop_json_matches_html_nodes():
    """The JSON `sampling_loop` and the HTML loop view are two projections of ONE
    declared edge set (adapters/diffusor/blocks.diffusion_loop_*), so their node
    sets must be identical — the structural anti-drift gate."""
    import re
    d = unfold(FLUX)
    j = d.to_json()["sampling_loop"]
    json_nodes = {n["id"] for n in j["nodes"]}

    html = expand_card_payloads(d.to_html(standalone=True))
    seg = html[html.index("SAMPLING LOOP"):]
    loop_svg = re.search(r"<svg.*?</svg>", seg, re.S).group(0)
    html_nodes = set(re.findall(r'data-id="([^"]+)"', loop_svg))

    assert html_nodes == json_nodes, (
        f"loop drift — only in HTML: {html_nodes - json_nodes}; "
        f"only in JSON: {json_nodes - html_nodes}")

    # Every edge endpoint is a real node; the recurrence has its one back-edge.
    ids = json_nodes
    for e in j["edges"]:
        assert e["from"] in ids and e["to"] in ids, f"dangling edge {e}"
    backs = [(e["from"], e["to"]) for e in j["edges"] if e.get("back_edge")]
    assert backs == [("scheduler", "latent")], backs
    # Fan-in (connectors) and fan-out (splitters) are derived, present, honest.
    assert {c["at"] for c in j["connectors"]} == {"latent", "denoiser"}
    assert {s["at"] for s in j["splitters"]} == {"denoiser", "prompt"}


def test_diffusion_json_does_not_leak_llm_io_fields():
    """A denoiser has no token vocabulary or LM head; the expanded JSON's
    dimensions + io must not leak vocab_size / tie_word_embeddings / token_ids /
    token_embedding (the IR carries vocab_size=0 only for param honesty). Pins
    the FAIL-2 fix."""
    j = unfold(FLUX).to_json()
    dims = j["dimensions"]
    assert "vocab_size" not in dims and "tie_word_embeddings" not in dims
    assert "in_channels" not in dims and "hidden_size" not in dims

    io = j["io"]
    assert io["input"]["kind"] == "denoiser_state"
    assert "token_embedding" not in io and "lm_head" not in io
    assert io["output"]["kind"] == "denoiser_state"
    assert io["output"]["domain"] is None

    # A real LLM still reports the token fields — the branch is diffusion-only.
    ld = unfold(LLAMA).to_json()
    assert "vocab_size" in ld["dimensions"]
    assert ld["io"]["input"]["kind"] == "token_ids"


def test_dit_ffn_undeclared_structure_is_honest_not_fabricated():
    """When NOTHING declares the FFN activation — neither the config NOR a
    ``class_defaults.yaml`` entry — the inner structure (activation AND gating) is
    rendered honestly as undeclared (gated=null, structure_declared=false), never a
    fabricated non-gated GELU MLP.  Uses IDEO_STYLE: a custom DiT class with no
    class default.  (Flux, whose model class DOES fix the activation, is the
    code-derived case — see ``test_flux_ffn_activation_is_code_derived_gelu``.)
    Pins WEAK-3 / honest-unknown.  A real LLM (declares its activation) is
    unaffected."""
    diagram = unfold(IDEO_STYLE)
    assert diagram.to_json()["layer_groups"] == []
    assert diagram.ir.extras["render"]["opaque_layer_block"]["resolved"] is False
    html = diagram.to_html(standalone=True)
    assert "GELU" not in html
    # LLAMA declares its activation — gating/activation are real facts, not flagged.
    lf = unfold(LLAMA).to_json()["layer_groups"][0]["ffn"]
    assert lf["activation"] == "silu" and lf["activation_assumed"] is None
    assert lf["gated"] is True


def test_flux_stream_and_ffn_proofs_remain_independent():
    """Independent source proofs must neither erase nor launder each other.

    FLUX's exact joined-stack FFN occurrence proves a dense GELU path.  Its
    dual-stack return relation is also exact, but that stack's FFN mechanism is
    not closed by the current evidence and therefore remains unknown.  No
    config spelling or class-default table bridges the missing proof.
    """
    ir = config_to_ir(FLUX)

    joined = next(x for x in ir.layers
                  if x.attention.variant.get("tag") == "explicit join")
    dual = next(x for x in ir.layers
                if x.attention.variant.get("tag") == "two returned states")
    assert joined.ffn.activation == "gelu"
    assert joined.ffn.gated is False
    assert joined.ffn.activation_from_class is True
    # The first stack's dual-state RETURN relation is now proven, but that does
    # not launder its still-unresolved FFN storage/activation.  Independent
    # evidence axes strengthen independently.
    assert dual.ffn.kind is None
    assert dual.ffn.activation is None
    assert dual.ffn.gated is None

    # JSON parity keeps the known activation/gating. Projection storage is a
    # separate U10 fact. The joined stack proves dense GELU; the dual stack's
    # FFN remains opaque even though its stream relation is now exact.
    groups = unfold(FLUX).to_json()["layer_groups"]
    assert any(group["ffn"].get("activation") == "gelu"
               and group["ffn"].get("gated") is False for group in groups)
    assert [block["id"] for block in dual.blocks] == [
        "attn", "wiring_unresolved", "ffn"]
    assert next(block for block in dual.blocks
                if block["id"] == "ffn")["detail"]["ffn"]["kind"] is None

    html = unfold(FLUX).to_html(standalone=True)
    # Only the exact joined stack renders GELU; the dual stack stays opaque.
    assert "GELU" in html


# Ideogram-4-style DiT: custom class, LLM-feature conditioning, an AdaLN dim, a
# CFG twin, and NO declared activation / attention-text dim — exercises FAIL-1,
# WEAK-3 and GAP-4 offline (no network).
IDEO_STYLE = {
    "_class_name": "Ideogram4Transformer2DModel",
    "num_layers": 2, "num_attention_heads": 4, "attention_head_dim": 64,
    "in_channels": 16, "intermediate_size": 256,
    "adaln_dim": 512, "llm_features_dim": 53248,
    "unconditional_transformer": ["diffusers", "Ideogram4Transformer2DModel"],
}


def test_ideogram_style_dit_captures_declared_facts():
    """GAP-4: adaln_dim / llm_features_dim are captured; the CFG twin is a NOTE
    (by-design advisory, not a config gap — must NOT raise "partial config"); the
    AdaLN dimension remains inspected metadata, but cannot author a rail;
    and (FAIL-1) no text->attention rail appears without an attention-text
    signal."""
    ir = config_to_ir(IDEO_STYLE)
    assert ir.layers == []
    assert "diffusion" not in ir.extras
    assert ir.extras["render"]["opaque_layer_block"]["resolved"] is False
    assert "adaln_dim" not in str(ir.extras["render"])
    assert "llm_features_dim" not in str(ir.extras["render"])


def test_unet_overview_draws_the_source_bound_context_route(sdxl_render):
    """The actual external context source and backed stage destinations are visible."""
    import re
    from xml.etree import ElementTree
    _, ir, html = sdxl_render
    routes = ir["extras"]["unet"]["context_routes"]
    assert {row["target"] for row in routes} == {
        "instance_down_blocks__1", "instance_down_blocks__2",
        "instance_up_blocks__0", "instance_up_blocks__1"}
    assert all(row["source"] == "unet_context_0" for row in routes)
    overview = next(svg for svg in re.findall(r"<svg\b.*?</svg>", html, re.S)
                    if 'aria-label="Source port boundaries and constructed stages;' in svg)
    assert 'data-id="unet_context_0"' in overview
    assert 'data-card-id="unet_context_0"' in html
    assert all(f'data-id="{row["target"]}"' in overview for row in routes)
    drawn = {(group.get("data-source"), group.get("data-target"))
             for group in ElementTree.fromstring(overview).iter()
             if group.get("data-route-kind") == "external_context"
             and any(node.get("marker-end") for node in group.iter())}
    assert drawn == {(row["source"], row["target"]) for row in routes}
    assert validate_click_coupling(html) == []


def test_dual_encoder_internals_survive_without_an_inferred_concat():
    """Both configured encoders retain their own drawings; width addition does not prove a concat call."""
    from model_unfolder.lint import _walk_blocks
    from test_support import SDXL_TEXT_ENCODER_CONFIGS
    cfg = dict(SDXL_UNET, _text_encoder_configs=SDXL_TEXT_ENCODER_CONFIGS)
    diagram = unfold(cfg)
    ir, html = diagram.to_ir(), expand_card_payloads(diagram.to_html(standalone=True))
    encoders = {block["id"]: block for block in _walk_blocks(ir)
                if block.get("view") == "text_encoder"}
    assert len(encoders) == 2
    assert {(block["detail"]["hidden"], block["detail"]["layers"]) for block in encoders.values()} == {(768, 12), (1280, 32)}
    for card in encoders.values():
        assert f'data-card-id="{card["id"]}"' in html
        own = _own_svgs(html, card["id"])
        assert own and any("marker-end" in svg for svg in own)
        assert card["children"]
    assert 'data-id="text_concat_op"' not in html
    assert 'data-id="text_proj_op"' not in html
    assert "encoder_hid_proj" not in _unet_fact(ir, "constructed_modules")
    assert validate_click_coupling(html) == []


def test_encoder_bridge_own_linear_shape_and_conditional_call_ports():
    """The original valid bridge keeps its own applied Linear, without an inferred encoder-to-K/V claim."""
    from model_unfolder.lint import _walk_blocks
    from test_support import SDXL_UNET_INCOMPLETE_TEXT_TIME
    cfg = {k: v for k, v in SDXL_UNET_INCOMPLETE_TEXT_TIME.items()
           if k not in ("_repo_id", "text_encoder_2", "addition_embed_type")}
    cfg["encoder_hid_dim"] = 4096
    cfg["_text_encoder_configs"] = {
        "text_encoder": {"_class_name": "ChatGLMModel", "hidden_size": 4096,
                         "num_layers": 28, "num_attention_heads": 32,
                         "ffn_hidden_size": 13696, "vocab_size": 65024}}
    diagram = unfold(cfg)
    ir, html = diagram.to_ir(), expand_card_payloads(diagram.to_html(standalone=True))
    assert _unet_fact(ir, "constructed_modules")["encoder_hid_proj"]["class_name"] == "Linear"
    shapes = _unet_fact(ir, "constructed_parameter_shapes")
    assert shapes["parameters"]["encoder_hid_proj.weight"]["shape"] == [2048, 4096]
    assert shapes["parameters"]["encoder_hid_proj.bias"]["shape"] == [2048]
    assert shapes["by_module"]["encoder_hid_proj"] == 8_390_656
    _assert_instance_drawn(ir, html, "encoder_hid_proj")
    ports = _unet_fact(ir, "primary_state_ports")
    call = ports["regions"][3]["route"]["iteration_result"]["when_true"]["arguments"][2]["route"]["target_binding"]["helper_output"]["when_true"]
    assert call["arguments"] == [{"port": "0", "route": {"kind": "formal", "formal": "encoder_hidden_states"}}]
    assert call["target_binding"]["kind"] == "constructed_target"
    assert call["target_binding"]["targets"] == ["encoder_hid_proj"]
    assert len(call["target_binding"]["conditions"]) == 15
    assert call["result_slot"] == []
    route_id = "unet_primary_region_3__iteration__when_true__arg_2__helper_output__when_true"
    route = next(block for block in _walk_blocks(ir) if block["id"] == route_id)
    assert "root.denoiser.primary_state_ports" in route["source_fact_keys"]
    assert any(f'data-id="{route_id}__arg_0"' in svg and f'data-id="{route_id}__call"' in svg
               and svg.count("marker-end") == 2 for svg in _own_svgs(html, route_id))
    assert validate_click_coupling(html) == []


def test_unet_stage_drills_show_each_occurrences_own_weight_shape(sdxl_render):
    """Distinct stage cards retain each constructed convolution's own dimensions."""
    _, ir, html = sdxl_render
    shapes = _unet_fact(ir, "constructed_parameter_shapes")["parameters"]
    expected = {"down_blocks.0": [320, 320, 3, 3], "down_blocks.1": [640, 320, 3, 3],
                "down_blocks.2": [1280, 640, 3, 3], "mid_block": [1280, 1280, 3, 3],
                "up_blocks.0": [1280, 2560, 3, 3], "up_blocks.1": [640, 1920, 3, 3],
                "up_blocks.2": [320, 960, 3, 3]}
    ids = set()
    for stage, dimensions in expected.items():
        path = stage + ".resnets.0.conv1"
        assert shapes[path + ".weight"]["shape"] == dimensions
        card = _assert_instance_drawn(ir, html, path)
        assert "weight: " + " × ".join(map(str, dimensions)) in card["facts"]
        ids.add(card["id"])
    assert len(ids) == 7
    for suffix in ("0", "1"):
        path = "mid_block.resnets." + suffix + ".conv1"
        _assert_instance_drawn(ir, html, path)
        assert shapes[path + ".weight"]["shape"] == [1280, 1280, 3, 3]
    assert validate_click_coupling(html) == []


def test_unet_resnet_is_one_drawn_occurrence_not_a_repeated_cell(sdxl_render):
    """A stage's actual two ResNets remain distinct, with no invented cell repeat."""
    _, ir, html = sdxl_render
    modules = _unet_fact(ir, "constructed_modules")
    assert modules["down_blocks.1.resnets"]["children"] == ["0", "1"]
    for number in range(2):
        path = f"down_blocks.1.resnets.{number}"
        card = _assert_instance_drawn(ir, html, path)
        assert card["view"] == "runtime_cell_connections"
        own = _own_svgs(html, card["id"])
        assert own and any("GroupNorm" in svg for svg in own)
        assert all("× N" not in svg and "× 1" not in svg and "× 2" not in svg for svg in own)
        _assert_instance_drawn(ir, html, path + ".conv1")


def test_unet_attention_containment_does_not_invent_inner_compute(sdxl_render):
    """Own constructed projections are visible without an invented attention algorithm."""
    _, ir, html = sdxl_render
    path = "down_blocks.1.attentions.0.transformer_blocks.0.attn1"
    card = _assert_instance_drawn(ir, html, path)
    assert card["view"] == "constructed_children"
    own = _own_svgs(html, card["id"])
    assert own
    for suffix in ("to_q", "to_k", "to_v"):
        _assert_instance_drawn(ir, html, path + "." + suffix)
    for svg in own:
        for unsupported in ("scaled_scores", "kv_cache", "q_rope", "k_rope", "softmax"):
            assert unsupported not in svg
    assert "root.denoiser.context_connections" not in card["source_fact_keys"]
    assert validate_click_coupling(html) == []


def test_unet_context_reference_couples_at_each_proved_deep_drill(sdxl_render):
    """Each proved attention argument has its own drawn route and deep source card."""
    from model_unfolder.preview import svg_views
    _, ir, html = sdxl_render
    paths = _unet_fact(ir, "context_connections")
    views = svg_views(html)
    assert paths
    for path, value in paths.items():
        card = _assert_instance_drawn(ir, html, path)
        assert "root.denoiser.context_connections" in card["source_fact_keys"]
        assert card["detail"]["target_formal"] == value["target_formal"]
        own = [svg for label, svg in views if label.split("/")[-1] == card["id"]
               if 'aria-label="Context input proven;' in svg]
        # The argument is a static formal port, so it deliberately has no
        # clickable data-id. Inspect its actual text and incoming arrow here.
        assert own and all('data-id="unet_context_0"' in svg and "marker-end" in svg
                           and ">encoder_hidden_states</text>" in svg for svg in own)
        assert any(child["id"] == "unet_context_0" for child in card["children"])
    assert validate_click_coupling(html) == []


def test_unet_external_context_drill_is_distinct_from_self_attention(sdxl_render):
    """An external argument route is drawn only on the occurrence whose source proves it."""
    _, ir, html = sdxl_render
    prefix = "down_blocks.1.attentions.0.transformer_blocks.0."
    context = _unet_fact(ir, "context_connections")
    assert prefix + "attn2" in context and prefix + "attn1" not in context
    self_card = _assert_instance_drawn(ir, html, prefix + "attn1")
    external_card = _assert_instance_drawn(ir, html, prefix + "attn2")
    assert self_card["view"] == "constructed_children"
    assert external_card["view"] == "runtime_context_connection"
    assert "root.denoiser.context_connections" not in self_card["source_fact_keys"]
    assert "root.denoiser.context_connections" in external_card["source_fact_keys"]
    assert external_card["detail"]["source"] == "unet_context_0"


def test_unet_component_entry_uses_its_actual_denoiser_label(sdxl_render):
    """The entry's actual denoiser card supplies its label; no DiT name is invented."""
    import re
    from model_unfolder.preview import architecture_svg
    _, ir, html = sdxl_render
    svg = architecture_svg(html)
    texts = re.findall(r"<text[^>]*>(.*?)</text>", svg)
    assert "U-Net" in texts and "Denoiser" in texts
    assert "DiT Denoiser" not in texts
    assert ir["component_entry"]["title"] in html
    flux_svg = architecture_svg(unfold(FLUX).to_html(standalone=True))
    flux_texts = re.findall(r"<text[^>]*>(.*?)</text>", flux_svg)
    assert "Diffusion denoiser" in flux_texts and "DiT Denoiser" not in flux_texts


def test_unet_overview_stage_occurrences_and_skip_routes_are_carded(sdxl_render):
    """Actual stage occurrences and each proved skip route survive the overview projection."""
    import re
    from xml.etree import ElementTree
    from model_unfolder.lint import _walk_blocks
    _, ir, html = sdxl_render
    overview = next(svg for svg in re.findall(r"<svg\b.*?</svg>", html, re.S)
                    if 'aria-label="Source port boundaries and constructed stages;' in svg)
    nodes = set(re.findall(r'data-id="([^"]+)"', overview))
    cards = list(_walk_blocks(ir))
    for path in ir["extras"]["unet"]["stage_block_ids"]:
        visible = [block for block in cards if block.get("source_instance_path") == path
                   and block["id"] in nodes]
        assert visible, path
        assert all(f'data-card-id="{block["id"]}"' in html for block in visible)
    drawn = {(group.get("data-source"), group.get("data-target"))
             for group in ElementTree.fromstring(overview).iter()
             if group.get("data-route-kind") == "skip_accumulation"
             and any(node.get("marker-end") for node in group.iter())}
    assert drawn == {(row["source"], row["target"]) for row in ir["extras"]["unet"]["skip_routes"]}
    assert len(drawn) == 6
    assert sum(target == "unet_skip_bank" for _, target in drawn) == 3
    assert sum(source == "unet_skip_bank" for source, _ in drawn) == 3
    assert "stroke-dasharray" not in overview
    assert validate_click_coupling(html) == []


def test_unet_entry_arrows_are_declared_ports_not_a_sampling_claim(sdxl_render):
    """The old eight-edge pipeline template is replaced by the exact component interface."""
    import re
    from model_unfolder.preview import architecture_svg
    _, ir, html = sdxl_render
    entry = ir["component_entry"]
    assert len(entry["input_ids"]) == 13
    assert ir["extras"]["render"]["loop_edges"] == [
        {"from": node, "to": entry["root_id"]} for node in entry["input_ids"]]
    hero = architecture_svg(html)
    assert set(re.findall(r'data-id="([^"]+)"', hero)) == set(entry["input_ids"]) | {entry["root_id"]}
    assert hero.count("marker-end") == len(entry["input_ids"])
    assert "SAMPLING LOOP" not in html
    assert "sampling_loop" not in ir["extras"]["render"]


def test_unet_stage_drills_keep_occurrence_cards_and_qualified_ffn_opener(sdxl_render):
    """Distinct constructed cells keep their own cards and the proved FFN graph opener."""
    _, ir, html = sdxl_render
    prefix = "down_blocks.1."
    for suffix in ("resnets.0", "attentions.0", "attentions.0.transformer_blocks.0.attn1",
                   "attentions.0.transformer_blocks.0.attn2", "attentions.0.transformer_blocks.0.ff"):
        _assert_instance_drawn(ir, html, prefix + suffix)
    resnet = _instance_card(ir, prefix + "resnets.0")
    assert resnet["view"] == "runtime_cell_connections" and resnet["detail"]["connections"]
    ffn = _instance_card(ir, prefix + "attentions.0.transformer_blocks.0.ff")
    assert ffn["view"] == "runtime_ffn"
    assert "root.denoiser.ffn_mechanisms" in ffn["source_fact_keys"]
    operations = [child["id"] for child in ffn["children"] if child.get("role") == "operation"]
    assert len(operations) == 5
    assert any(all(f'data-id="{node}"' in svg for node in operations)
               and svg.count("marker-end") == 7 for svg in _own_svgs(html, ffn["id"]))
    assert validate_click_coupling(html) == []


def test_unet_mid_occurrences_preserve_shapes_with_stage_order_open(sdxl_render):
    """Preserve three actual mid occurrences; the old template's two edges are retired."""
    _, ir, html = sdxl_render
    modules = _unet_fact(ir, "constructed_modules")
    shapes = _unet_fact(ir, "constructed_parameter_shapes")
    assert modules["mid_block.resnets"]["children"] == ["0", "1"]
    assert modules["mid_block.attentions"]["children"] == ["0"]
    ids = set()
    for path in ("mid_block.resnets.0", "mid_block.attentions.0", "mid_block.resnets.1"):
        card = _assert_instance_drawn(ir, html, path)
        ids.add(card["id"])
        assert card.get("repeat") is None
    assert len(ids) == 3
    for path in ("mid_block.resnets.0", "mid_block.resnets.1"):
        assert shapes["by_module"][path] == 31_138_560
        assert shapes["parameters"][path + ".conv1.weight"]["shape"] == [1280, 1280, 3, 3]
        assert shapes["parameters"][path + ".conv2.weight"]["shape"] == [1280, 1280, 3, 3]
    relation = _unet_fact(ir, "constructed_stage_relations")
    assert any(row["kind"] == "direct_stage_order_open" for row in relation["unresolved_relations"])
    assert "inter-loop source position does not prove unconditional execution order" in html


def test_unet_resnet_draws_proved_fragments_and_conditional_arithmetic(sdxl_render):
    """Five proved call edges and conditional arithmetic survive; the bridge stays open."""
    _, ir, html = sdxl_render
    path = "down_blocks.1.resnets.0"
    card = _assert_instance_drawn(ir, html, path)
    assert "root.denoiser.cell_connections" in card["source_fact_keys"]
    assert "root.denoiser.cell_arithmetic" in card["source_fact_keys"]
    detail = card["detail"]
    assert [(row["source"], row["target"], row["argument_slot"]) for row in detail["connections"]] == [
        ("call_0", "call_1", 0), ("call_1", "call_6", 0),
        ("call_9", "call_12", 0), ("call_12", "call_13", 0), ("call_13", "call_14", 0)]
    own = _own_svgs(html, card["id"])
    calls = detail["connection_calls"]
    for edge in detail["connections"]:
        source, target = calls[edge["source"]]["target"], calls[edge["target"]]["target"]
        assert any(f'data-id="{source}"' in svg and f'data-id="{target}"' in svg
                   and "marker-end" in svg for svg in own)
    assert detail["return_arithmetic"]["scale"] == "divide"
    assert detail["conditioning_arithmetic"] and all(
        row["conditional"] for row in detail["conditioning_arithmetic"])
    for row in detail["conditioning_arithmetic"]:
        assert f'data-id="{row["merge"]}"' in html
    assert "Connections across optional branches and helpers remain under investigation." in card["facts"]
    _assert_instance_drawn(ir, html, path + ".time_emb_proj")


def test_text_encoder_attention_drills_are_canonical_and_positionally_honest():
    """Every fetched-config encoder bakes a REAL attention drill (canonical
    region, namespaced), and its positional lane matches the encoder's own
    code: CLIP learned-absolute -> no RoPE nodes; T5 -> relative-bias score
    add + code-proven unscaled QK^T.  A spec without a fetched sub-config
    keeps the description-only card (no fabricated Q/K/V)."""
    d = unfold(FLUX)
    html = expand_card_payloads(d.to_html())
    ir = d.to_ir()

    def find_block(blocks, bid):
        for b in blocks or []:
            if b.get("id") == bid:
                return b
            hit = find_block(b.get("children"), bid)
            if hit is not None:
                return hit

    loop = ((ir.get("extras") or {}).get("render") or {}).get("loop_blocks")
    clip = find_block(loop, "encoder_0_op_selfattn")
    # T5 constructs the learned relative-bias table only in block 0.  The
    # returned bias is loop-carried to later blocks, but that transport is not
    # yet an exact execution-flow fact, so U8 must split the tower rather than
    # laundering block 0's producer across all 24 layers.
    t5 = find_block(loop, "encoder_1_g0_op_selfattn")
    t5_later = find_block(loop, "encoder_1_g1_op_selfattn")
    assert clip and clip.get("view") == "attention" and clip.get("children")
    assert t5 and t5.get("view") == "attention" and t5.get("children")
    assert t5_later and t5_later.get("view") == "attention"

    clip_ids = {c["id"] for c in clip["children"]}
    t5_ids = {c["id"] for c in t5["children"]}
    # Namespaced so two encoders at the same depth cannot satisfy each other.
    assert all(i.startswith("encoder_0_attn_") for i in clip_ids)
    assert all(i.startswith("encoder_1_g0_attn_") for i in t5_ids)
    # CLIP: learned absolute positions live at the embedding, NOT in attention.
    assert not any(i.endswith(("q_rope", "k_rope")) for i in clip_ids)
    assert not any(i.endswith(("rel_pos_bias", "alibi_bias")) for i in clip_ids)
    # T5: the learned relative bias enters the scores; RoPE would be fabricated.
    assert {"encoder_1_g0_attn_rel_pos_bias",
            "encoder_1_g0_attn_rel_bias_offsets",
            "encoder_1_g0_attn_score_bias_add"} <= t5_ids
    assert not any(i.endswith(("q_rope", "k_rope")) for i in t5_ids)
    assert not any(i.endswith(("rel_pos_bias", "rel_bias_offsets",
                               "score_bias_add"))
                   for i in {c["id"] for c in t5_later["children"]})
    # Unscaled scores: the T5 drill draws raw QK^T (no fabricated sqrt(dim)),
    # while CLIP keeps the standard scaled fraction.
    assert 'data-id="encoder_1_g0_attn_scaled_scores"' in html
    t5_panel = html.split('data-card-id="encoder_1_g0_op_selfattn"', 1)[1]
    t5_svg = t5_panel.split("</svg>", 1)[0]
    assert "sqrt(dim)" not in t5_svg
    clip_panel = html.split('data-card-id="encoder_0_op_selfattn"', 1)[1]
    clip_svg = clip_panel.split("</svg>", 1)[0]
    assert "sqrt(dim)" in clip_svg
    # No fetched sub-config -> honest description-only card, never a guessed drill.
    from model_unfolder.adapters.diffusor.blocks import _text_encoder_ops
    bare = _text_encoder_ops("CLIP", None, None, "enc_x", spec={})
    bare_attn = next(b for b in bare if b["id"] == "enc_x_op_selfattn")
    assert "view" not in bare_attn and not bare_attn.get("children")
    assert bare_attn.get("description")


def test_config_only_heterogeneous_encoder_schedule_cannot_split_the_tower():
    """A ``layer_types`` list is not execution evidence.

    This fixture resolves to ordinary Llama source, whose repeated block always
    receives one causal mask builder.  Injecting alternating familiar tokens
    must therefore leave one homogeneous source-proven group; preserving the
    former two-group expectation would restore the config-authored U8 bug.
    """
    import re
    d = unfold(HYBRID_ENC)
    ir = d.to_ir()

    def find_block(blocks, bid):
        for b in blocks or []:
            if b.get("id") == bid:
                return b
            hit = find_block(b.get("children"), bid)
            if hit is not None:
                return hit

    loop = ((ir.get("extras") or {}).get("render") or {}).get("loop_blocks")
    enc = find_block(loop, "encoder_0")
    det = enc.get("detail") or {}
    sub_model = det.get("sub_model") or {}
    groups = sub_model.get("groups")
    assert [(g["count"], g["tag"]) for g in groups] == [(24, "")]
    assert sub_model.get("schedule", {}).get("period") == 1

    child_ids = [c.get("id") for c in enc.get("children") or []]
    assert "encoder_0_op_selfattn" in child_ids
    assert not any(item.startswith("encoder_0_g") for item in child_ids)

    html = expand_card_payloads(d.to_html())
    seg = html.split('data-card-id="encoder_0"', 1)[1]
    svg = seg.split("</svg>", 1)[0]
    node_ids = set(re.findall(r'data-id="([^"]+)"', svg))
    assert {"encoder_0_op_selfattn", "encoder_0_op_ffn"} <= node_ids
    assert "sliding window" not in svg
    assert 'data-card-id="encoder_0_op_selfattn"' in html
    assert 'data-card-id="encoder_0_g0_op_selfattn"' not in html
    assert d.wiring_problems() == []

    # A homogeneous encoder is untouched: no groups, the original single-cell ids.
    d_flat = unfold(FLUX)
    flat = find_block(((d_flat.to_ir().get("extras") or {}).get("render") or {})
                      .get("loop_blocks"), "encoder_0")
    assert len(((flat.get("detail") or {}).get("sub_model") or {}).get("groups") or []) <= 1
    assert "encoder_0_op_selfattn" in [c.get("id") for c in flat.get("children") or []]


def test_config_tokens_do_not_change_typed_grouping_or_period_detection():
    """Typed grouping follows source-proven layer facts, not a config list."""
    from model_unfolder.ir import detect_layer_period, distinct_layer_groups
    from model_unfolder.evidence.context import ParseContext
    sub = HYBRID_ENC["_text_encoder_configs"]["text_encoder"]
    ir = transformer.parse(sub, context=ParseContext.build(sub, source="local"))
    groups = distinct_layer_groups(ir.layers)
    assert len(groups) == 1
    assert groups[0]["indices"] == list(range(24))
    assert groups[0]["runs"] == [(0, 23)]
    sigs = [layer.signature() for layer in ir.layers]
    assert detect_layer_period(sigs) == 1
    assert detect_layer_period(sigs[:1]) is None
    assert detect_layer_period([sigs[0]] * 6) == 1


def test_moe_text_encoder_opens_the_canonical_moe_drill():
    """An MoE text encoder opens the SAME router/top-k/expert drill a decoder
    MoE opens — serialized off the one decoder builder, at the ENCODER's own
    width — and the tower cell is labelled MoE, not Feed-forward."""
    import re
    d = unfold(MOE_ENC)
    html = expand_card_payloads(d.to_html())

    seg = html.split('data-card-id="encoder_0_op_ffn"', 1)[1]
    svg = seg.split("</svg>", 1)[0]
    nodes = set(re.findall(r'data-id="([^"]+)"', svg))
    assert {"router", "expert_1", "add_moe"} <= nodes
    assert not {"expert_k", "expert_kp1", "expert_n"} & nodes
    assert "in · 4,096" in svg          # the ENCODER's width, not FLUX's inner dim
    assert "top-2 of 8" in svg
    assert "template × 2" in svg
    # Full canonical depth: the router gate pipeline and the expert FFN leaves.
    assert 'data-card-id="router"' in html
    assert 'data-card-id="expert_1"' in html
    # Mixtral stores experts as ONE fused gate_up tensor — the drill and its
    # cards show the faithful fused projection + split (storage evidence
    # composing with the embedded sub-model).
    assert 'data-card-id="expert_gate_up_proj"' in html
    assert 'data-card-id="expert_gate_up_split"' in html
    assert 'data-card-id="expert_gate_proj"' not in html
    # Tower cell names the real block.
    tower_svg = html.split('data-card-id="encoder_0"', 1)[1].split("</svg>", 1)[0]
    assert "Mixture of Experts" in tower_svg
    assert d.wiring_problems() == []
    from model_unfolder.block_schema import validate_click_coupling
    assert validate_click_coupling(html) == []
    # A dense encoder never gains an expert subtree.
    flat = expand_card_payloads(unfold(FLUX).to_html())
    flat_seg = flat.split('data-card-id="encoder_0_op_ffn"', 1)[1].split("</svg>", 1)[0]
    assert "router" not in flat_seg


def test_scalar_sample_size_never_fabricates_a_square_grid():
    """U-D0 (audio plan): a square side may only be inferred from a scalar
    sample_size when 2D-ness is evidenced — a declared patchify (DiT) or a
    conv-UNet family.  A bare scalar on anything else is just a length:
    Stable Audio's sample_size=1024 is 1-D latent frames, and the old code
    drew a fabricated 1024 x 1024 grid from it."""
    from model_unfolder.adapters.diffusor.blocks import diffusion_loop_blocks

    def latent_facts(geom):
        blocks = {b["id"]: b for b in diffusion_loop_blocks(geom)}
        return " ".join(blocks["latent"].get("facts") or [])

    # 1-D shaped: no patch_size, not a UNet -> honest channels + declared length.
    audio = latent_facts({"in_channels": 64, "sample_size": 1024,
                          "denoiser_family": "dit"})
    assert "1,024" in audio and "x 1,024" not in audio and "x 1024" not in audio
    assert "64 channels" in audio

    # Declared patchify keeps the square (PixArt: 128 / patch 2 = 64).
    dit = latent_facts({"in_channels": 4, "sample_size": 128, "patch_size": 2,
                        "denoiser_family": "dit"})
    assert "4 × 64 x 64" in dit

    # A conv-UNet's constructor reads scalar sample_size as H = W.
    unet = latent_facts({"in_channels": 4, "sample_size": 128,
                         "denoiser_family": "unet"})
    assert "4 × 128 x 128" in unet
