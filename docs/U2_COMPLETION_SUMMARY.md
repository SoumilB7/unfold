# U2 Definitive Completion — final summary (R9 commit, 2026-07-19)

Commits: ac860e6 (substrate+R1+R2) · 313307c (R3) · 2b1689a+a443661 (R4) ·
478e34a+a778fb7 (R5+vet correction) · 97a7d68 (R6) · 9498eb4 (R7) ·
7110be6 (R8) · THIS commit (R9: final-vet corrections + witness 26 +
re-bless + receipts).

## Soumil's final vet (2026-07-19) — both counterexamples killed
1. is_encoder_decoder (ARCHITECTURE: seq2seq half, mask causality,
   cross-attn schedule) was globally ignored by a bare-key vocabulary →
   removed from the list; CONSUMED into decoder.attention.mask at its
   deciding read; the global ignore-by-leaf ledger rail is DELETED —
   scoped ignores are exact rules (owner pattern + exact path + reason,
   everchanging/evidence/ledger_ignores.yaml); address keys match
   top-level reads only (nested paths never launder).
2. fabrication_debt_keys let pending config debt authorize a WHOLLY
   FABRICATED receipt → the debt-key lane is DELETED end-to-end; a receipt
   cites an actual typed fact or an exact migration-claim target, nothing
   else. Both counterexamples are permanent poisons
   (tests/test_u2_r9_final_corrections.py, verbatim).
R6 joins completed: fact-side deletion conditions are OWNER-BOUND; the
extras growth gate is WRITER-EXACT; drawn debt carries owner identity
(drawn_unledgered_pairs).

## Witness 26 + the explicit re-bless (delegated by Soumil in the vet)
musicgen-small blessed after a full 10-view visual pass (rubric clean;
T5 drill draws UNSCALED QK^T + relative position bias; dense-ReLU FFN;
correct codebook/cross-attn story). Reaching mechanical-clean fixed real
producers: composite MAIN slots enter the wrapper vocabulary
(wrapper_path + prefix-owner map), the conditioning slot got its
DocumentBinding boundary, codebooks/audio_channels/scale_embedding are
consumed, coverage joins compose document_path + config_path.
Manifest rebuilt (25 -> 26): PIXEL surfaces (ir/expanded/params/
html_meta/gallery) byte-identical on ALL 25 existing witnesses; only
ledgers/sable moved (the intended U2 evidence-surface change standing
red since 28a66ad — cleared by this re-bless); musicgen-small added.

## Honesty record (unchanged from the R6/R7 commits)
The zero-drift net was red since 28a66ad (U2.2a path truth); my R4/R5
"broad green" receipts were defective on that one net. Pixels never
drifted at any point (hash-proven at R6, R7, and here).

## Carried forward (visible, never silent)
evidence/position.py raw scope walks (funnel-invisible tie-break reads);
the pending config_read register rows (owner-bound checkable conditions,
gate-enforced); Lumina FFN width / Sana cross-attn geometry /
add_watermarker flags at their owning units (U10/U11).

## R9 bracket receipt (2026-07-19, on this commit — ZERO FAILURES)

Round-2 vet corrections included (owner-qualified DRAWN_PAIRS as the
authoritative gate input; drawn_leaf_is_lawful(owner, leaf) joins
registration/debt for THAT exact owner; sibling-owner laundering poison;
drawn_unledgered_names() compatibility-only; classified: evaluates
(row.owner, exact_path) with the U11 pair-shaped table contract).

Parallel bracket, fingerprint-enveloped:
- FULL SUITE: 1493 passed / 11 skipped / 3 xfailed / 0 FAILED (38m25s)
- Isolated files: config_paths 83p/11s . arbitration 17p . config_access
  46p . config_intents 8p . structural_writes 13p . fact_registry 29p .
  evidence_facts 330p . projection_obligations 7p . receipts 49p .
  projection_audit 38p . authority_probes 28p/3xf . isolation 6p .
  reader_exceptions 4p . preservation 20p (zero-drift GREEN against the
  re-blessed 26-witness manifest) . sable 33p (regression corpus incl.
  witness 26)
- Semantic controls: 280 passed
- Collect-only: clean
- Fingerprint before == after: IDENTICAL — the bracket is valid.
