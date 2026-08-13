"""Tests for the Sable harness: the two new mechanical nets (label-lint,
wiring-conformance), the orchestrator, and the CI regression lock.

Each net has a NEGATIVE CONTROL — proof it actually fires — because a net that
can't fail is worthless (the doctrine's rule). The corpus uses the offline config
fixtures from test_diffusion so everything runs without network.
"""
from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

import model_unfolder as mu
from model_unfolder import lint_labels, sable, bless, check_regression, load_corpus
from model_unfolder.evidence import check_fact_conformance, check_wiring_conformance
from test_support import FLUX, PIXART, LLAMA, MUSICGEN_SMALL

CORPUS = [("FLUX", FLUX), ("PIXART", PIXART), ("LLAMA", LLAMA)]


def test_plain_parse_resolves_model_source_once(monkeypatch):
    """Every architectural detector consumes one call-local source bundle."""
    from model_unfolder.evidence import context as context_module

    real = context_module.resolve_source_files
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(context_module, "resolve_source_files", counted)
    mu.unfold(LLAMA)
    assert len(calls) == 1


def test_sable_parse_and_all_conformance_nets_share_one_source_bundle(monkeypatch):
    """Sable must not rediscover source separately for parse and each net."""
    from model_unfolder.evidence import context as context_module

    real = context_module.resolve_source_files
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(context_module, "resolve_source_files", counted)
    sable(LLAMA, render_images=False)
    assert len(calls) == 1


def test_concurrent_transformer_and_diffusion_renders_are_call_local():
    """Theme and graph diagnostics cannot cross-contaminate concurrent calls."""
    from model_unfolder.renderers.html.document import render_document
    from model_unfolder.renderers.html.render_context import (
        RenderContext,
        activate_render_context,
    )

    llama_ir = mu.unfold(LLAMA).to_ir()
    flux_ir = deepcopy(mu.unfold(FLUX).to_ir())
    # Production currently chooses teal for both domains. Force the registered
    # blue palette here so the test exercises actual cross-theme isolation.
    flux_ir["extras"]["render"]["theme"] = "blue"

    def render(ir, mount, theme):
        context = RenderContext(theme=theme)
        with activate_render_context(context):
            html = render_document(ir, mount)
        return html, context

    llama_expected, _ = render(llama_ir, "concurrent-llama", "teal")
    flux_expected, _ = render(flux_ir, "concurrent-flux", "blue")

    for _ in range(3):
        with ThreadPoolExecutor(max_workers=2) as pool:
            llama_future = pool.submit(render, llama_ir, "concurrent-llama", "teal")
            flux_future = pool.submit(render, flux_ir, "concurrent-flux", "blue")
            llama_html, llama_context = llama_future.result()
            flux_html, flux_context = flux_future.result()

        assert llama_html == llama_expected
        assert flux_html == flux_expected
        assert "#0F6E56" in llama_html and "#1E5FB0" not in llama_html
        assert "#1E5FB0" in flux_html and "#0F6E56" not in flux_html
        assert llama_context.events and flux_context.events
        assert not llama_context.wiring_findings
        assert not flux_context.wiring_findings


def test_render_events_carry_block_path_component_and_variant():
    diagram = mu.unfold(FLUX)
    diagram.to_html(standalone=True)
    events = diagram.render_events()
    assert events
    attn = next(event for event in events
                if event.view == "attn" and event.component == "root")
    assert attn.block_path == ("attn",)
    assert "MM-DiT" in attn.variant
    assert attn.source_owner == "FluxTransformer2DModel"
    assert attn.drawn_ops == frozenset({"opaque", "port"})
    # The supporting text encoders now bake their OWN canonical attention drills,
    # each carrying its qualified component — never unioned with the denoiser's.
    encoder_attn = [event for event in events
                    if event.view == "attn" and event.component != "root"]
    assert encoder_attn and all(
        event.block_path[-1].endswith("_op_selfattn") for event in encoder_attn)


# --------------------------------------------------------------------------- #
# label-lint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", CORPUS)
def test_label_lint_clean_on_corpus(name, cfg):
    assert lint_labels(mu.unfold(cfg).to_ir()) == []


def test_label_lint_flags_nested_parens_and_raw_activation():
    """Negative controls for text/design classes that repeatedly slipped through
    non-visual checks: nested labels, raw backend activation, numeric facts on
    blocks, backend alternatives, and static Tier-2 connectors."""
    ir = {
        "layers": [{"blocks": [
            {"id": "attn", "kind": "attention", "label": ["Joint Attention", "(MM-DiT (dual-stream))"]},
            {"id": "activation", "kind": "activation", "label": "gelu-approximate"},
            {"id": "clip", "kind": "embedding", "label": "CLIP (768-d)"},
            {"id": "patch", "kind": "linear", "label": "Linear / Conv2d"},
            {"id": "encoder", "kind": "attention", "label": "Encoder ×30"},
            {"id": "bad_add", "kind": "residual_add", "label": "⊕", "static": True},
            {"id": "ok_attn", "kind": "attention", "label": ["Joint Attention", "(dual-stream)"]},
            {"id": "ok_act", "kind": "activation", "label": "GELU"},
            {"id": "ok_conv", "kind": "linear", "label": "Conv2d"},
        ]}],
        "extras": {},
    }
    problems = lint_labels(ir)
    assert any("nested/doubled parentheses" in p and "attn" in p for p in problems)
    assert any("raw backend" in p and "activation" in p for p in problems)
    assert any("dimensions/counts" in p and "clip" in p for p in problems)
    assert any("backend ops" in p and "patch" in p for p in problems)
    assert any("dimensions/counts" in p and "encoder" in p for p in problems)
    assert any("Tier-2 connector" in p and "bad_add" in p for p in problems)
    # the clean siblings are NOT flagged
    assert not any("ok_attn" in p or "ok_act" in p or "ok_conv" in p for p in problems)


def test_numeric_lint_separates_dimensions_from_topology_descriptors():
    """A *dimension* (768-d / 1024d) is a fact that belongs on a chip and must
    flag; a bare single-digit + D (2D / 3D / axial 3D) is an N-dimensional
    TOPOLOGY descriptor (3D-RoPE, 2D patch grid) — operation identity, not a
    channel count — and must NOT flag.  This is the false-positive that would
    otherwise reject an honest ``3D RoPE`` attention label."""
    from model_unfolder.lint import _leaks_numeric_fact

    # dimensions / counts -> flag
    for fact in ("768-d", "1024d", "1,280-d", "768d", "8-d", "12 heads", "Encoder ×30"):
        assert _leaks_numeric_fact(fact), f"{fact!r} is a numeric fact and must flag"
    # qualitative N-dimensional topology -> do NOT flag
    for topo in ("3D", "2D", "axial 3D", "3D RoPE", "2D patch", "3D-RoPE"):
        assert not _leaks_numeric_fact(topo), f"{topo!r} is a topology descriptor, not a dimension"

    # …and the same at the label-lint level: a 3D-RoPE attention block is clean.
    ir = {
        "layers": [{"blocks": [
            {"id": "rope", "kind": "attention", "label": "Attention (3D RoPE)"},
            {"id": "patch", "kind": "linear", "label": "2D Patchify"},
            {"id": "dim", "kind": "embedding", "label": "Embedding 768-d"},
        ]}],
        "extras": {},
    }
    problems = lint_labels(ir)
    assert not any("rope" in p or "patch" in p for p in problems)
    assert any("dimensions/counts" in p and "dim" in p for p in problems)


def test_config_access_capture_survives_nested_reset_and_reports_dotted_paths():
    """Sable's outer audit cannot be erased by a nested component parser.

    Post-H3 the capture is the owner-scoped ``capture_events()`` ledger and
    ``debug.reset()`` is a no-op — so this is now a regression guard proving a
    nested parser's legacy ``reset()`` still cannot erase the enclosing capture.
    """
    from model_unfolder.adapters.transformer import debug
    from model_unfolder.evidence.config_access import capture_events

    cfg = {
        "model_type": "outer",
        "vision_config": {"hidden_size": 128, "new_architecture_switch": True},
        "torch_dtype": "float16",  # intentionally ignored vocabulary
    }
    with capture_events() as ledger:
        debug.note_access("model_type")
        debug.note_access("vision_config")
        debug.reset()  # a nested parser's legacy reset() (now a no-op) must not erase the outer capture
        debug.note_access("hidden_size")
    touched = ledger.touched_names()
    assert "model_type" in touched and "hidden_size" in touched
    assert debug.unparsed_fields([cfg], touched=touched, recursive=True) == [
        "vision_config.new_architecture_switch"
    ]


def test_config_field_audit_is_blocking():
    """Promoted 2026-07-04 (owned-field backlog reached zero): an unread config
    switch now FAILS the mechanical pass — a new field must be parsed, chipped
    via config_facts.yaml, or consciously declared ignored."""
    cfg = {**LLAMA, "brand_new_architecture_switch": True}
    report = sable(cfg, render_images=False)
    audit = next(c for c in report.checks if c.name == "config_field_audit")
    assert audit.blocking is True
    assert any("brand_new_architecture_switch" in finding for finding in audit.findings)
    assert not report.mechanical_passed


def test_config_field_audit_clears_only_a_read_mapping_parent():
    """An exact child read covers its address container, not its siblings."""
    from model_unfolder.adapters.transformer import debug

    cfg = {"rope_parameters": {"rope_theta": 10000, "factor": 8}}
    assert debug.unparsed_fields(
        [cfg], recursive=True,
        owner_touched={"root": {"rope_theta"}}, root_owner="root",
        owner_paths={"root": {"rope_parameters.rope_theta"}},
        owner_exact_leaves={"root": {"rope_theta"}},
    ) == ["rope_parameters.factor"]


def test_u2_projection_and_census_nets_are_wired_blocking_and_clean():
    """U2 P4: net #13 (projection-audit) and net #14 (zero-asserted census) are
    both blocking mechanical nets and both pass on a real decoder; the
    accessed-but-unprojected upgrade is advisory."""
    report = sable(LLAMA, render_images=False)
    by_name = {c.name: c for c in report.checks}
    assert by_name["projection_audit"].blocking is True
    assert by_name["projection_audit"].passed, by_name["projection_audit"].findings
    assert by_name["zero_asserted_census"].blocking is True
    assert by_name["zero_asserted_census"].passed, by_name["zero_asserted_census"].findings
    assert by_name["config_accessed_unprojected"].blocking is False


# --------------------------------------------------------------------------- #
# wiring-conformance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", [("FLUX", FLUX), ("PIXART", PIXART)])
def test_wiring_conformance_clean_on_corpus(name, cfg):
    """Every conditioning rail the diffusion diagrams draw maps to a real forward
    argument (FLUX/PixArt blocks all take temb + encoder_hidden_states)."""
    ir = mu.unfold(cfg).to_ir()
    assert [p.message for p in check_wiring_conformance(cfg, ir)] == []


def test_wiring_conformance_flags_fabricated_text_rail():
    """NEGATIVE CONTROL: a text-conditioning rail drawn into a block whose
    forward() takes no text argument is flagged. LlamaDecoderLayer.forward has no
    encoder_hidden_states, so a fabricated text rail on a llama layer must fire."""
    ir = mu.unfold(LLAMA).to_ir()
    ir["layers"][0]["blocks"].append({
        "id": "text_cond", "lane": "external_bottom_right",
        "diffusion_stage": "text_conditioning", "kind": "conditioning",
    })
    probs = check_wiring_conformance(LLAMA, ir)
    assert any(p.kind == "fabricated_input" and p.op == "text" for p in probs), \
        [p.message for p in probs]


def test_wiring_conformance_flags_missing_text_rail():
    """NEGATIVE CONTROL (missing direction): when a block's forward() TAKES a text
    input (Flux's dual block has encoder_hidden_states) but the diagram draws no
    text rail and shows no joined-sequence indication, the dropped text is flagged.
    This is the direction that caught PRX (text K/V concatenated, drawn as plain
    self-attention)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:                       # strip the rail from a dual-stream layer
        tag = str((L.get("attention") or {}).get("variant", {}).get("tag") or "").lower()
        if "dual-stream" in tag:
            L["blocks"] = [b for b in L["blocks"] if b.get("id") != "text_cond"]
            break
    probs = check_wiring_conformance(FLUX, ir)
    assert any(p.kind == "missing_input" and p.op == "text" for p in probs), \
        [p.message for p in probs]


# --------------------------------------------------------------------------- #
# fact-conformance — the SAME-op-kind, different-SEMANTICS axis (positional
# scheme, attention algorithm) that op-presence conformance is blind to.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", [("FLUX", FLUX), ("PIXART", PIXART)])
def test_fact_conformance_clean_on_corpus(name, cfg):
    """FLUX (axial RoPE, softmax) and PixArt (learned-pos, softmax) make no
    fabricated-NoPE or wrong-attention claim — fact-conformance is clean."""
    ir = mu.unfold(cfg).to_ir()
    assert [p.message for p in check_fact_conformance(cfg, ir)] == []


def test_fact_conformance_flags_fabricated_nope():
    """NEGATIVE CONTROL: a block whose forward() applies rotary (FLUX threads
    image_rotary_emb) but whose diagram asserts NoPE must fire. This is the
    recurring fabricated-NoPE class (Wan/CogVideoX/Mochi/LTX/Lumina2)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:                       # strip the positional scheme
        att = L.get("attention") or {}
        att["no_rope"], att["rope"], att["rope_dim"] = True, False, None
    probs = check_fact_conformance(FLUX, ir)
    assert any(p.kind == "missing_position" for p in probs), [p.message for p in probs]


def test_fact_conformance_is_symmetric_for_fabricated_rope_and_missing_learned_position():
    """StarCoder's old drawing must fail in BOTH directions: invented RoPE and
    omitted learned-position addition, using one typed source decision."""
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("gpt_bigcode").to_dict()
    ir = mu.unfold(cfg).to_ir()
    for layer in ir["layers"]:
        attn = layer["attention"]
        attn["rope"] = True
        attn["position_kind"] = "rope"
        attn["position_application"] = "qk_rotation"
    ir["extras"]["render"]["model_blocks"] = [
        block for block in ir["extras"]["render"]["model_blocks"]
        if block.get("id") not in {"position_ids", "position_embed", "position_add"}
    ]
    from model_unfolder.evidence.qualification import qualification_findings
    findings = qualification_findings(ir)
    assert any("decoder.attention.position_schedule projects" in item
               for item in findings)
    assert any("decoder.input.position_addition is ledgered" in item
               for item in findings)


def test_fact_conformance_flags_one_non_rope_scheme_substituted_for_another():
    """Wrong alternatives must fail even though both correctly omit RoPE."""
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("bloom").to_dict()
    ir = mu.unfold(cfg).to_ir()
    for layer in ir["layers"]:
        layer["attention"]["position_kind"] = "learned_absolute"
        layer["attention"]["position_application"] = "embedding_add"
    from model_unfolder.evidence.qualification import qualification_findings
    findings = qualification_findings(ir)
    assert any("decoder.attention.position_schedule fact schedule" in item
               for item in findings)


def test_position_schedule_without_its_exact_fact_is_blocking():
    ir = mu.unfold(LLAMA).to_ir()
    ir["extras"]["fact_provenance"].pop(
        "decoder.attention.position_schedule", None)
    from model_unfolder.evidence.qualification import qualification_findings
    assert any("decoder.attention.position_schedule projects" in item
               for item in qualification_findings(ir))


def test_unresolved_position_withheld_as_unknown_is_qualification_clean():
    """Unknown is the honest projection of unresolved source, not a mismatch."""
    ir = mu.unfold(LLAMA).to_ir()
    ir["extras"]["fact_provenance"].pop(
        "decoder.attention.position_schedule", None)
    for layer in ir["layers"]:
        attention = layer["attention"]
        attention["rope"] = None
        attention["no_rope"] = False
        attention["position_kind"] = "unknown"
        attention["position_application"] = "unknown"
        attention["rope_dim"] = None
    from model_unfolder.evidence.qualification import qualification_findings
    assert not any("position_schedule" in item
                   for item in qualification_findings(ir))


def test_true_oracle_missing_remains_visible_in_sable_report():
    cfg = {
        "model_type": "definitely_uninstalled_decoder",
        "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 4,
    }
    report = sable(cfg, render_images=False)
    assert report.oracle.startswith("MISSING")


def test_fact_conformance_flags_wrong_attention_kind():
    """NEGATIVE CONTROL: a diagram that draws LINEAR attention for a block whose
    code uses softmax (FLUX has no *LinearAttn* processor) must fire — the inverse
    of the Sana miss (softmax drawn for a linear-attention block)."""
    ir = mu.unfold(FLUX).to_ir()
    for L in ir["layers"]:
        (L.get("attention") or {})["kind"] = "linear"
    probs = check_fact_conformance(FLUX, ir)
    assert any(p.kind == "wrong_attention" for p in probs), [p.message for p in probs]


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,cfg", CORPUS)
def test_sable_mechanical_pass_on_corpus(name, cfg):
    r = sable(cfg, render_images=False)
    assert r.oracle == "present", r.oracle      # modeling source resolved
    assert r.mechanical_passed, r.summary()
    assert r.view_hashes                         # at least one distinct view locked
    assert r.visual_review == "PENDING"          # never auto-passes the eye step


# --------------------------------------------------------------------------- #
# the CI lock
# --------------------------------------------------------------------------- #

def test_bless_refuses_without_visual_artifacts(tmp_path):
    """The visual gate is ARTIFACT-verified, in every environment:
    a CLEAN string cannot bless without real on-disk evidence."""
    r = sable(FLUX, render_images=False)
    # mechanical-clean but visual PENDING -> NOT blessable.
    with pytest.raises(ValueError):
        bless(r, FLUX, corpus_dir=str(tmp_path))
    # CLEAN with NO gallery (rsvg absent / render skipped) -> refused loudly.
    r.visual_review = "CLEAN"
    with pytest.raises(ValueError, match="without images is not a review"):
        bless(r, FLUX, corpus_dir=str(tmp_path))
    # CLEAN with a STALE gallery (files deleted since review) -> refused.
    r.gallery = [str(tmp_path / "gone" / "00__architecture.png")]
    with pytest.raises(ValueError, match="missing on disk|gallery/view"):
        bless(r, FLUX, corpus_dir=str(tmp_path))
    # CLEAN with a PARTIAL gallery (count != distinct views) -> refused.
    partial = tmp_path / "partial"; partial.mkdir()
    one = partial / "00__architecture.png"; one.write_bytes(b"png")
    r.gallery = [str(one)]
    if len(r.view_hashes) != 1:
        with pytest.raises(ValueError, match="gallery/view mismatch"):
            bless(r, FLUX, corpus_dir=str(tmp_path))


def test_bless_reuses_the_existing_fixture_for_an_identical_frozen_config(tmp_path):
    """A loader-lost display name cannot fork one input into two witnesses."""
    from model_unfolder.sable import _fixture_path_for_config

    legacy = tmp_path / "musicgen-small.json"
    legacy.write_text(json.dumps({
        "model": "musicgen-small",
        "config": MUSICGEN_SMALL,
        "hash_signature": ["old-lock"],
    }))

    selected = _fixture_path_for_config(
        tmp_path, "MusicgenForConditionalGeneration", deepcopy(MUSICGEN_SMALL)
    )

    assert selected == legacy
    assert not (tmp_path / "musicgenforconditionalgeneration.json").exists()


def test_bless_refuses_two_fixture_paths_for_the_same_frozen_config(tmp_path):
    """Divergent locks for one exact input are ambiguous, never auto-selected."""
    from model_unfolder.sable import _fixture_path_for_config

    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(json.dumps({
            "model": name,
            "config": MUSICGEN_SMALL,
            "hash_signature": [name],
        }))

    with pytest.raises(ValueError, match="duplicate corpus fixtures freeze the same config"):
        _fixture_path_for_config(
            tmp_path, "MusicgenForConditionalGeneration", deepcopy(MUSICGEN_SMALL)
        )


def test_corpus_has_one_fixture_per_exact_frozen_config():
    """The CI lock has one verdict per exact input, never contradictory twins."""
    seen: dict[str, str] = {}
    for filename, fixture in load_corpus():
        identity = json.dumps(
            fixture.get("config"), sort_keys=True, separators=(",", ":"), default=str
        )
        assert identity not in seen, (
            f"{filename} and {seen.get(identity)} freeze the same config; "
            "keep one stable fixture and one visual verdict"
        )
        seen[identity] = filename


def test_bless_round_trips_with_real_gallery_and_records_supersession(tmp_path):
    """Success path: a real gallery blesses, reproduces drift-free, and a
    RE-bless after drift carries the superseded signature (a visible
    transition, never a silent overwrite)."""
    import shutil
    if not shutil.which("rsvg-convert"):
        pytest.skip("rsvg-convert not installed — bless success path needs real PNGs")
    r = sable(FLUX, outdir=str(tmp_path / "gallery"))
    assert r.gallery and len(r.gallery) == len(r.view_hashes)
    assert (tmp_path / "gallery" / "MANIFEST.txt").exists()
    r.visual_review = "CLEAN"
    path = bless(r, FLUX, corpus_dir=str(tmp_path))
    fixture = json.loads(open(path).read())
    assert check_regression(fixture) == []
    assert fixture["visual_evidence"]["png_count"] == len(r.view_hashes)
    assert "superseded_hash_signature" not in fixture
    # Tamper the locked signature -> drift is detected.
    tampered = dict(fixture)
    tampered["hash_signature"] = ["deadbeef"] + fixture["hash_signature"][1:]
    # The reviewed witness name may be a repository id that an offline config
    # cannot reconstruct.  A same-config re-bless must preserve it.
    tampered["model"] = "reviewed-flux-name"
    assert any("view drift" in m for m in check_regression(tampered))
    # Simulate an older lock with a different signature on disk, then re-bless:
    # the new fixture must carry the superseded signature.
    with open(path, "w") as fh:
        json.dump(tampered, fh)
    gallery_home = tmp_path / "galleries" / fixture["model"]
    review = gallery_home / "her_eyes_review.md"
    review.write_text("human review evidence must survive a re-bless\n")
    path2 = bless(r, FLUX, corpus_dir=str(tmp_path))
    refreshed = json.loads(open(path2).read())
    assert refreshed["hash_signature"] == fixture["hash_signature"]
    assert refreshed["superseded_hash_signature"] == tampered["hash_signature"]
    assert refreshed["model"] == "reviewed-flux-name"
    assert review.read_text() == "human review evidence must survive a re-bless\n"


def test_sable_regression_corpus():
    """Every blessed model retains its SVG lock.  Old blessings newly invalidated
    by exact source attribution stay pinned as explicit unresolved debt; they are
    not silently re-blessed without a fresh Dable review."""
    # Unit 10 (2026-07-03) resolved the last pinned unresolved debt: the UNet
    # string-factory (get_down_block) is followed generally now, so NO fixture
    # may carry unresolved nested-conformance findings.
    expected_unresolved: dict[str, set] = {}
    corpus = load_corpus()
    if not corpus:
        pytest.skip("no blessed models in the corpus yet")
    for filename, fixture in corpus:
        drift = check_regression(fixture)
        expected = expected_unresolved.get(filename, set())
        actual_expected = {
            view for view in expected
            if any(item.startswith(f"nested_conformance: {view}: no code unit resolved")
                   for item in drift)
        }
        unexpected = [item for item in drift
                      if not any(item.startswith(
                          f"nested_conformance: {view}: no code unit resolved"
                      ) for view in expected)]
        assert actual_expected == expected, f"{filename} lost pinned unresolved coverage: {drift}"
        assert unexpected == [], f"{filename} regressed:\n  " + "\n  ".join(unexpected)


# --------------------------------------------------------------------------- #
# evidence_ambiguity — the present-but-ambiguous advisory net
# --------------------------------------------------------------------------- #

def test_evidence_ambiguity_flags_ambiguous_envelopes_and_passes_clean_trees():
    """The net fires on a block whose evidence envelope says ``ambiguous`` (the
    source was scanned but the extractor could not resolve it — a stub the code
    could have answered), dedupes repeated layers, and exempts honest
    ``oracle_missing`` / ``proven`` envelopes."""
    from model_unfolder.sable import _ambiguous_evidence_findings

    def _ir(status):
        block = {
            "id": "enc_op_ffn", "label": "Feed-forward",
            "detail": {"evidence": {"status": status, "component": "text_config",
                                    "reason": "no exact feed-forward projection callable"}},
        }
        return {
            "layers": [
                {"blocks": [dict(block)]},
                {"blocks": [dict(block)]},          # repeated layer -> one finding
            ],
            "extras": {"render": {"loop_blocks": [
                {"id": "encoder_0", "children": [dict(block)]},
            ]}},
        }

    ambiguous = _ambiguous_evidence_findings(_ir("ambiguous"))
    assert len(ambiguous) == 2                       # layerN dedup + the loop block
    assert all("ambiguous" in f_ and "text_config" in f_ for f_ in ambiguous)
    assert _ambiguous_evidence_findings(_ir("proven")) == []
    assert _ambiguous_evidence_findings(_ir("oracle_missing")) == []


def test_evidence_ambiguity_is_wired_into_sable_as_blocking():
    """Promoted 2026-07-03 (backlog reached zero): an ambiguous evidence
    envelope — installed source scanned, callable unresolved — now blocks a
    bless like any other mechanical failure."""
    report = sable(FLUX, render_images=False)
    check = next(c for c in report.checks if c.name == "evidence_ambiguity")
    assert check.blocking
    assert check.passed, check.findings              # FLUX resolves all envelopes


def test_every_fixture_gallery_is_durable_and_complete():
    """The reviewed pixels are part of the lock: every fixture's visual_evidence
    must point INSIDE the corpus (galleries/<slug>, never a scratch/session
    directory), with the PNG count it certifies and the save_images MANIFEST."""
    from model_unfolder.sable import DEFAULT_CORPUS, load_corpus

    for fname, fix in load_corpus():
        evidence = fix.get("visual_evidence") or {}
        gallery_dir = str(evidence.get("gallery_dir") or "")
        assert gallery_dir.startswith("galleries/"), \
            f"{fname}: gallery_dir {gallery_dir!r} is not corpus-relative"
        home = DEFAULT_CORPUS / gallery_dir
        pngs = list(home.glob("*.png"))
        assert home.is_dir(), f"{fname}: durable gallery missing: {home}"
        assert len(pngs) == evidence.get("png_count"), \
            f"{fname}: {len(pngs)} PNGs on disk vs {evidence.get('png_count')} certified"
        assert (home / "MANIFEST.txt").exists(), f"{fname}: gallery MANIFEST missing"
