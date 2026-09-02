# U2-R7 execution map (recon by Fable agent, 2026-07-18)

## Root cause of ALL text_encoder* unlocated+origin rows
`adapters/diffusor/parser.py:2282-2283` — `document_scope(("_text_encoder_configs", key))`
with NO obj= and NO provenance=. It holds `sub` (line 2269) + the address: build
`prepare_document(sub) → DocumentBinding(f"root.{key}", (...), prepared)`. NOTE
encoder_panel.py:50 re-prepares (merge=False) — R7 must pick ONE preparation boundary.

## The 15 unlocated leaves → reader fixes (all have a nameable object in scope)
- parser.py:748/749/753 (`use_sliding_window`, `max_window_layers`, `layer_types`+754) —
  reads of text_cfg; `_text_path = _wrapper_path(cfg, text_cfg)` at :615 →
  `config_container(_text_path, obj=text_cfg)` (pattern at 1057/1085/1887).
- parser.py:1042 and :1874 — bare `_g(text_cfg, "rope_parameters")` (only SUBKEYS are
  containered) → same fix.
- parser.py:2348-2350 `_norm_kind_evidence_src(cfg,…)` (rms_norm_eps/layer_norm_eps/
  layer_norm_epsilon) — called from parser.py:826-827 (has _text_path) and
  encoder_panel.py:96-100 which runs AFTER its bound_document exits at :68 → extend the
  binding scope or re-enter. Helper has no path; fix at callers.
- `text_config` (2w) — `_unwrap_text` probe (parser.py:472-473) invoked outside the binding
  at encoder_panel.py:94; becomes a lawful bare leaf of a NAMED document once diffusor:2283
  passes a binding.
- Same-shape (not yet in list): diffusor/parser.py:1879-1889 `_dit_norm_kind` bare reads.
- INVISIBLE to census: evidence/position.py:115,:444-466 walk nested text_config RAW (no
  funnel) — flag for review.

## Loose document_scope callers (deletion unit for the overload = R7)
- Production: ONLY diffusor/parser.py:2283 (fix above). config_access.py:1245 is internal
  delegation inside bound_document (absorbed when overload dies).
- Tests: test_projection_audit.py:543,556; test_config_paths.py:222,259,273,278,288,302,317
  (convertible via _prep helper, pattern in test_u2_r1:26-36); test_config_paths.py:417,431
  PIN the legacy overload (path-only) — rewrite against bound_document or delete with it;
  test_u2_r2_consumed_decision.py:17,22 helpers (+ :121-122).

## Standing 227 rows — disposition first pass
- root (52): U6 set (attention_bias, qk-norms, MLA geometry, partial_rotary…), U7 set
  (mlp_bias, eps rows, MoE routing, parallel residual, MTP), U8 set (layer_types,
  sliding-window, is_encoder_decoder 10w, rope_parameters+6 subrows), display-only
  (final_logit_softcapping, 4 token-id rows), address-only (text_config, vision_config).
- root.denoiser (100): geometry-only ~40 trivially classifiable; mechanism U10/U11 set
  (qk_norm, activation_fn, norm_type, rope…); address-only (text_encoder 15w etc read at
  diffusor/parser.py:2249-2254); metadata (flip_sin_to_cos…).
- root.scheduler (1): `scheduler` (15w) display-only at diffusor/parser.py:1957-1969; real
  scheduler debt = origin-unknown _scheduler_config.* → boundary fix (U13).
- root.text_encoder (27): mirrors root (U6/U7/U8 + display + address).
- vision owners (25+15): U9 geometry-mostly; mechanism (fullatt_block_indexes,
  vision rope, hidden_act, projector_hidden_act).
- te_2 (2), te_3 (5): T5 gating (U7) + decoderness (U8).
- Class-overlay (17): _name_or_path metadata; is_decoder/is_encoder_decoder class-default →
  U8 candidates; 10 rope_parameters* class rows → permissible-default vs syntax-normalization
  (document.py:188); 4 loader rows address-only, author nothing.
- Origin-unknown 80 collapse at boundaries: root 5 (same leaves as unlocated), scheduler 6,
  te 15, te_2 9, te_3 1, vae 44 — mostly the diffusor:2283 fix + vae/scheduler subtree
  provenance already mapped (verbatim-fetched).
- Census doc truncates te(15)/vae(44) origin lists with "…" — regenerate after boundary fixes.
