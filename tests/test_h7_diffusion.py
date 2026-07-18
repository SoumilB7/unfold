"""H7 (§16.5/§16.6) — diffusion fact-family migration.

The full H7 migrates every diffusion fact family to a typed evidenced fact with a
render projection, one family at a time — a large, ongoing effort.  This slice
lands the concrete, verifiable piece §16.5 named explicitly: the three
audit-clearing diffusion reads removed in ``procedure 2`` return as REGISTERED
typed facts carrying DECLARED PENDING-PROJECTION debt (name/owner/canonical/
reason/projection), so they are neither silent reads nor forgotten removals — the
H7-full reader + its render projection are the named next step.

It also proves the H9-core metamorphic harness holds on a diffusion reference, so
each subsequent diffusion family migration has a ready contract to satisfy.
"""
from __future__ import annotations

from model_unfolder.evidence.structural_debt import (
    STRUCTURAL_DEBT, pending_projection_paths)
from test_support import FLUX, metamorphic

# the exact reads procedure 2 removed (§16.5) — the debt must cover all three
_REMOVED_READS = {"max_sequence_length", "act_fn", "temporal_compression_ratio"}


def test_pending_projection_debt_covers_the_three_removed_reads():
    """REC-4 grew the register into the living EXACT-debt ledger (§10.3); U2-R6
    moved it into the ONE StructuralDebt register: the original 3 removed reads
    stay covered, and the 8 fields whose U1 config-authored rows were deleted
    are now visible owner-exact debt."""
    pairs = pending_projection_paths()
    covered = {path.rsplit(".", 1)[-1] for _, path in pairs}
    assert covered >= _REMOVED_READS, covered
    assert {"in_channels", "patch_size", "norm_num_groups",
            "scaling_factor"} <= covered
    assert ("root.vae", "_vae_config.in_channels") in pairs
    assert ("root.denoiser", "norm_num_groups") in pairs


def test_pending_projection_debt_is_fully_qualified():
    # REC-4/REC-5 grew the exact-debt ledger across owners; U2-R6 rows carry a
    # real component owner, a projection target, a checkable deletion condition
    # and the excusal writer/consumer (constructor-enforced).  U2-R7 added the
    # standing-occurrence dispositions, which legitimately span the parse root
    # and the pipeline text-encoder slots.
    owners = {"root", "root.denoiser", "root.vae", "root.vision",
              "root.text_encoder", "root.text_encoder_2",
              "root.text_encoder_3", "root.text_encoder.vision"}
    rows = [r for r in STRUCTURAL_DEBT if r.sink_kind == "config_read"
            and not r.deletion_condition.startswith("classified:")]
    for row in rows:
        assert row.source_occurrence and row.reason and row.structural_target
        assert row.owner in owners, row
    pairs = [(r.owner, r.source_occurrence) for r in rows]
    assert len(pairs) == len(set(pairs)), "duplicate pending-projection rows"


def test_the_three_reads_are_still_removed_from_the_diffusor():
    """The reads must remain REMOVED (the honest state) until the H7-full reader
    reintroduces them as evidenced facts — no silent re-read crept back."""
    import pathlib

    import model_unfolder
    parser = (pathlib.Path(model_unfolder.__file__).parent
              / "adapters" / "diffusor" / "parser.py").read_text()
    assert '"max_sequence_length": _resolve(cfg, "max_sequence_length")' not in parser
    assert '"act_fn": _g(vcfg, "act_fn")' not in parser
    assert '"temporal_compression_ratio": _g(vcfg, "temporal_compression_ratio")' not in parser


def test_config_field_audit_excuses_the_pending_projection_fields():
    """The reads stay removed, so the BLOCKING config_field_audit must EXCUSE a
    field registered as pending-projection debt — a declared classification, not
    unread coverage debt.  Regression guard for the ``procedure 9`` re-vet finding:
    ``procedure 2`` removed these reads assuming the audit was advisory; it is
    blocking (the render-suite regression net that the fast smoke had skipped
    caught it), so the removal alone left it red until the registry is recognized.
    """
    import model_unfolder as mu

    ir = mu.unfold(FLUX).to_ir()  # FLUX carries _vae_config.act_fn, a pending fact
    unread = ((ir.get("extras") or {}).get("config_audit") or {}).get("unread", [])
    pending = {path.rsplit(".", 1)[-1] for _, path in pending_projection_paths()}
    offending = [p for p in unread if p.rsplit(".", 1)[-1] in pending]
    assert not offending, (
        "a pending-projection field is a DECLARED classification and must not be "
        f"reported as unread config debt; got {offending}")


def test_metamorphic_harness_holds_on_a_diffusion_reference():
    """H9-core contract on diffusion: rename-invariance, provenance integrity, and
    owner-separated siblings all hold on FLUX — so every diffusion family
    migration has a ready harness to satisfy."""
    metamorphic.assert_rename_invariant(FLUX)
    metamorphic.assert_partial_source_invariant(FLUX)
    metamorphic.assert_collision_invariant(FLUX)


# --------------------------------------------------------------------------- #
# REC-4 (§10.6) — real diffusion census, blocking conflicts, no yaml growth
# --------------------------------------------------------------------------- #

def test_diffusion_consumed_census_is_non_empty_and_exact():
    """§10.6.1-2: the denoiser AND the VAE consume their geometry declarations
    into exact fact owners/keys — an unmigrated adapter can no longer look
    empty-clean."""
    from model_unfolder.evidence.config_access import capture_events, owner_scope
    import model_unfolder as mu

    with capture_events() as ledger:
        with owner_scope("root"):
            mu.unfold(FLUX).to_ir()
    consumed = [e for e in ledger.events if e.intent == "consumed"]
    assert consumed, "diffusion consumed census is EMPTY"
    owners = {e.fact_owner for e in consumed}
    assert "denoiser.stack" in owners and "denoiser.attention" in owners
    assert "vae.geometry" in owners, owners
    # exact keys ride along
    keys = {(e.fact_owner, e.fact_key) for e in consumed}
    assert ("denoiser.stack", "num_layers") in keys
    assert ("denoiser.attention", "num_heads") in keys
    # and the VAE events carry the exact container path
    vae_paths = {e.config_path for e in consumed if e.fact_owner == "vae.geometry"}
    assert any(path.startswith("_vae_config.") for path in vae_paths), vae_paths


def test_conflicting_diffusion_aliases_block_not_default():
    """§10.6.3: rival head-count spellings are a typed blocking ambiguity."""
    from model_unfolder.sable import sable

    cfg = {**FLUX, "n_heads": 16}   # FLUX declares num_attention_heads=24
    rep = sable(cfg, render_images=False)
    amb = next(c for c in rep.checks if c.name == "config_ambiguity")
    assert amb.blocking and amb.findings, amb.findings
    assert any("num_attention_heads" in f for f in amb.findings)
    assert not rep.mechanical_passed


def test_absent_head_geometry_stays_unknown_not_zero_as_known():
    """§10.6.4: a denoiser with no head geometry renders honest emptiness —
    no fabricated positive head/hidden claims."""
    import model_unfolder as mu
    from model_unfolder.errors import ConfigParseError

    cfg = {k: v for k, v in FLUX.items()
           if k not in ("num_attention_heads", "attention_head_dim",
                        "hidden_size", "num_kv_heads")}
    try:
        ir = mu.unfold(cfg).to_ir()
    except ConfigParseError:
        return  # a LOUD refusal is honest — never a fabricated geometry
    prov = (ir.get("extras") or {}).get("fact_provenance") or {}
    for key, row in prov.items():
        if key.endswith((".num_heads", ".hidden_size", ".head_dim")):
            assert not row.get("value"), f"fabricated geometry claim: {key}={row}"


def test_config_facts_yaml_cannot_grow_structural_rows():
    """§10.6.5 poison: the (bucket, field) surface is PINNED — a new row fails
    this census until it lands with a registered fact/receipt (Law F).  The
    U1-added rows are gone; the pre-existing legacy table only shrinks."""
    from model_unfolder.everchanging import load_diffusion_config_facts

    table = load_diffusion_config_facts()
    pairs = {(bucket, row["field"]) for bucket, rows in table.items() for row in rows}
    FROZEN = {('conditioning', 'max_text_seq_length'), ('conditioning', 'resolution_embeds'), ('denoiser', 'added_kv_proj_dim'), ('denoiser', 'addition_embed_type_num_heads'), ('denoiser', 'attention_out_bias'), ('denoiser', 'attention_type'), ('denoiser', 'axes_lens'), ('denoiser', 'bottleneck_size'), ('denoiser', 'boundary_ratio'), ('denoiser', 'center_input_sample'), ('denoiser', 'class_embeddings_concat'), ('denoiser', 'conv_in_kernel'), ('denoiser', 'conv_out_kernel'), ('denoiser', 'cross_attention_head_dim'), ('denoiser', 'cross_attention_input_dim'), ('denoiser', 'double_self_attention'), ('denoiser', 'downsample_padding'), ('denoiser', 'dual_cross_attention'), ('denoiser', 'encoder_hid_dim'), ('denoiser', 'encoder_hid_dim_type'), ('denoiser', 'eps'), ('denoiser', 'ffn_dim_multiplier'), ('denoiser', 'global_states_input_dim'), ('denoiser', 'image_dim'), ('denoiser', 'mid_block_scale_factor'), ('denoiser', 'multiple_of'), ('denoiser', 'num_cross_attention_heads'), ('denoiser', 'num_embeds_ada_norm'), ('denoiser', 'num_refiner_layers'), ('denoiser', 'only_cross_attention'), ('denoiser', 'pos_embed_seq_len'), ('denoiser', 'resnet_out_scale_factor'), ('denoiser', 'resnet_skip_time_act'), ('denoiser', 'resnet_time_scale_shift'), ('denoiser', 'rope_max_seq_len'), ('denoiser', 'spatial_interpolation_scale'), ('denoiser', 'temporal_interpolation_scale'), ('denoiser', 'theta'), ('denoiser', 'upcast_attention'), ('denoiser', 'use_additional_conditions'), ('denoiser', 'use_linear_projection'), ('latent', 'default_sample_size'), ('latent', 'sample_size'), ('timestep', 'condition_dim'), ('timestep', 'flip_sin_to_cos'), ('timestep', 'freq_dim'), ('timestep', 'freq_shift'), ('timestep', 'guidance_embeds_scale'), ('timestep', 'time_cond_proj_dim'), ('timestep', 'time_embed_dim'), ('timestep', 'time_embedding_act_fn'), ('timestep', 'time_embedding_type'), ('timestep', 'time_factor'), ('timestep', 'time_max_period'), ('timestep', 'time_proj_dim'), ('timestep', 'timestep_activation_fn'), ('timestep', 'timestep_guidance_channels'), ('timestep', 'timestep_post_act'), ('vae', 'add_attention_block'), ('vae', 'attn_scales'), ('vae', 'batch_norm_eps'), ('vae', 'batch_norm_momentum'), ('vae', 'decoder_act_fns'), ('vae', 'decoder_block_types'), ('vae', 'decoder_causal'), ('vae', 'decoder_layers_per_block'), ('vae', 'decoder_norm_types'), ('vae', 'decoder_qkv_multiscales'), ('vae', 'downsample_block_type'), ('vae', 'encoder_block_out_channels'), ('vae', 'encoder_block_types'), ('vae', 'encoder_causal'), ('vae', 'encoder_layers_per_block'), ('vae', 'encoder_qkv_multiscales'), ('vae', 'invert_scale_latents'), ('vae', 'resnet_norm_eps'), ('vae', 'sample_size'), ('vae', 'spatial_compression_ratio'), ('vae', 'spatial_expansions'), ('vae', 'spatio_temporal_scaling'), ('vae', 'temperal_downsample'), ('vae', 'temporal_expansions'), ('vae', 'upsample_block_type')}
    grown = pairs - FROZEN
    assert not grown, f"config_facts.yaml grew structural rows without a registered fact: {sorted(grown)}"
