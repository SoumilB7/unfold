# U2-R6 StructuralDebt rows — the reviewed writer/consumer audit (Fable agent, 2026-07-18)

Source content for the ONE register in `model_unfolder/evidence/structural_debt.py`.
All writers verified by reading (not guessed line numbers). The register as landed
normalizes this content: per-(writer, target) EXACT rows (67), U3–U14 units, closed
deletion-condition DSL. `mu/` = `model_unfolder`.

## Key resolutions (formerly writer-less rows)

- rope / softcap / attention → `mu/adapters/transformer/parser.py::parse`
  (setdefault sites :1871/:1883/:1897, :1903, :1911)
- unet → `mu/adapters/diffusor/parser.py::_parse_unet_model` (AnnAssign dict-literal :474)
- render → 3 authors: `diffusor/parser.py::_parse_unet_model`, `diffusor/parser.py::parse`,
  `transformer/assembly.py::decoder_extras`
- diffusion → 2 authors: `diffusor/parser.py::_parse_unet_model` (:486, U11 side) and
  `diffusor/parser.py::parse` (:1203, U10 side) — split into two rows
- modalities → payload authored by `modalities/schema.py::multimodal_payload` (:21) via
  `builder.py::multimodal_extras`, merged by `assembly.py::_merge_extras` (dynamic-keyed
  `target[key] = value` — censused as `<dynamic>` after the R6 scanner extension)
- pass-through flags `attention_k_eq_v` / `use_double_wide_mlp` →
  `transformer/parser.py::parse` :1925-1928 variable-keyed loop (censused as `<dynamic>`)

## Consumer findings

- 12 legacy extras keys have ZERO downstream readers — consumed only via
  `ModelIR.to_dict` (ir.py:234) JSON serialization (U14 surface): moe, sliding_window,
  qk_norm, dual_kv, irope, num_kv_shared_layers, parallel_residual,
  partial_rotary_factor, rope, softcap, codebooks, attention.
- Live render/conformance consumers: mtp (views/cards/expanded/validate),
  position_encoding (architecture view + BLOCKING `_drawn_position_kinds`),
  diffusion (`sections._diffusion_stats`, `unet._text_source_label`),
  unet (`build_unet_view` + drills), block_diffusion (`_build_block_diffusion_view`),
  modalities (architecture view, metadata_modalities, modality views, 4 conformance nets),
  render (document theme, stats banner, `_is_diffusion_architecture`, sampling loop,
  storage conformance).
- 10 of 12 PENDING_PROJECTION rows and all 7 PENDING_CONFIG rows have NO production
  reader (reads removed by `procedure 2` / REC-4, or never existed) — the excusal
  writer is `parser.py::config_to_ir` and the consumer is the excusal + sable dashboard;
  vision_out_width is the only live-read row
  (`vision.py::vision_encoder_hidden_size` + COR-4 projector consumption).

## DRAWN_UNLEDGERED writers (drawn_leaf rows)

- position_kind → `mu/opgraph.py::_sdpa_region` (bias_kind :544) + labels chips;
  declared in `renderers/html/fact_projection.ATTENTION_DRAWN`
- qk_norm → `blocks/attention.py::_sdpa_detailed_child_blocks` (q/k-norm cards)
- q_norm / k_norm → `blocks/attention.py::attention_detail` (derived leaves :39/:40)
- logit_softcap → `mu/opgraph.py::_sdpa_core_ops` (Op:attn_softcap)

## legacy_convention (registered facts drawn by convention; drawn_leaf rows)

- norm_placement → abstain still draws pre-norm + banner (transformer parse);
  retires via `unknown_policy_retired:norm_placement`
- scores_scale → silent input drawn as sqrt(d) denominator (`_sdpa_core_ops`)
- projection_mode → absent mode drawn as solid split projections (`_sdpa_region`);
  retires via `status_retired:projection_mode:legacy_asserted`
- attention_kind → diffusor parse asserted-fold (543 records / 13 fixtures, json-only)
- ffn_storage → transformer parse asserted-fold (94 records / 2 fixtures, json-only)

## Flags carried forward

- `add_watermarker`: candidate for scoped-ignore DELETION rather than a fact
  (pipeline flag, no structural target) — decided at U11.
- Unit judgment calls: mtp (U7 vs U14), codebooks (U7 vs U9), dual_kv (U6 vs U8),
  moe.every_layer (U8 content inside U7 rows) — units chosen in the register;
  revisit at the owning unit.
- `diffusion` two-writer split landed as two rows (U11 + U10).
- 12 zero-reader keys: if U14 drops them from the JSON schema they can be deleted
  without any renderer change.
