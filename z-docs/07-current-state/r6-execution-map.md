# U2-R6 execution map (recon by Fable agent, 2026-07-18)

Replace 4 debt registers with ONE StructuralDebt schema (spec §R6, lines 627-663).

## The registers
1. **LEGACY_EXTRAS** — structural_writes.py:200-255, 18 rows (target/owner/reason/unit/deletion).
   Units H8×14, H7×3, scoped×1 — NONE are U3–U14. 4 rows writer-less in the static census
   (rope, softcap, unet, modalities — dynamic/setdefault writers the scanner misses; render's
   writers exist as extras_nested). Top-level keys excuse ALL nested leaves (family-wide by
   construction). Consumers: test_structural_writes.py:60-91, test_fact_registry.py:258-283.
2. **DRAWN_UNLEDGERED_DEBT** — registry.py:405-422, 5 rows (position_kind, qk_norm, q_norm,
   k_norm, logit_softcap), all unit=H8, writer-less (writers = spec author + renderer drawing
   sites). Consumers: test_projection_obligations.py:21-80, test_h8_transformer.py:22,37.
3. **PENDING_PROJECTION_DEBT** — registry.py:650-715, 12 rows. NO unit field — units live in
   PROSE inside reason strings (U12/V-02 etc). NO stale/growth gate (parser comment: stale rows
   "cost nothing" — violates R6). Production joins: parser.py:153-176 (unread excusal), :287,
   :310; sable.py:390 (fabrication debt keys). Tests: test_h7_diffusion.py:16-75.
4. **PENDING_CONFIG_CLASSIFICATION** — registry.py:624-647, 7 rows, deletion_unit="U11" (the
   ONLY typed U3–U14 unit anywhere). NO test gate. Joins: parser.py:176-186, :222-226, sable.
5. **legacy_convention statuses** — 5 REGISTRY rows (norm_placement 294, scores_scale 304,
   projection_mode 313, attention_kind 334, ffn_storage 346); debt context prose-only in notes
   (names opgraph.py default-drawing sites).

## Adjacent (scope decisions)
- ASSERTED_BASELINE (test_fact_registry.py:128-132; exact-eq 641): population pins, family-wide;
  keep as pins tied to debt rows (not writer debt).
- _RAW_CONSUME_DEBT (test_u2_r2_raw_consume_debt.py:24-28): per-file counts, prose units —
  reader-migration debt, scope OUT with note (or upgrade rows).
- Broad-except _BASELINE (test_reader_exceptions.py:50-72): reader-failure debt, scope OUT.
- FROZEN (bucket,field) pin (test_h7_diffusion.py:155-157, ~85 pairs, growth-only): bare-leaf
  YAML-table pins — name in R6, resolve in R7/YAML territory.
- MIGRATED_SCOPES: claims register, NOT debt — but sable.py:390-392 fabrication _debt_keys join
  must keep working after replacement.
- _INFRA_EXTRAS (structural_writes.py:50-53, 7 keys) vs registry.INFRA_EXTRAS_KEYS (720-723,
  6 keys, lacks config_ambiguity): near-duplicates "kept in sync" — unify in R6.
- Identity_guard._LAWFUL_TABLES carries a `consumers` field — the only precedent for
  last_consumer.

## Population math
309 writer keys + 40 multi-count pins = the full writer census to partition. 18+5+12+7 = 42
typed debt rows + 5 legacy_convention statuses = current debt side. 9 writer-less rows (4
LEGACY_EXTRAS + 5 DRAWN_UNLEDGERED) need writer discovery. Scanner gap: top-level
`extras.setdefault("key", ...)` (Call, not Assign) is missed — extending the scanner backs
rope/softcap/unet/modalities rows with real writers.

## Required gates R6 must author
- PENDING_PROJECTION_DEBT + PENDING_CONFIG_CLASSIFICATION have NO stale/growth gates.
- Every debt row → live writer join (blocking). Stale + growth blocking on the ONE register.
- Unit vocabulary: U3–U14 only; tests pinning {H7,H8}/{H7,H8,scoped} vocabularies
  (test_structural_writes.py:63, test_projection_obligations.py:63) must be rewritten.
- Deletion conditions must be CHECKABLE (all current ones are prose).

Row-level content (writers, units, deletion conditions per row): see the row-authoring agent
report (r6-rows, pending at recon time).
