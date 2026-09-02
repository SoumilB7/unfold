# U2 Definitive Completion — summary for Soumil (2026-07-18)

All nine units of `docs/U2_DEFINITIVE_COMPLETION_SPEC.md` are landed and
pushed on `audio-composite-support` (R9 bracket receipt appended when it
finishes):

| Unit | Commit | One line |
|---|---|---|
| substrate+R1+R2 | ac860e6 | prepared-document boundary + typed consumed decisions |
| R3 | 313307c | generic evidence arbitration |
| R4 | 2b1689a | writer-identity structural census (+ a443661 forward fix) |
| R5 | 478e34a + a778fb7 | projection routes + receipt rail; correction per your vet |
| R6 | 97a7d68 | ONE StructuralDebt register (114 rows), 4 allowlists deleted |
| R7 | 9498eb4 | census standing/unlocated/origin-unknown all ZERO; loose document_scope deleted |
| R8 | 7110be6 | all ten U2 nets blocking + anti-vacuous poisons |

## What you asked to be told honestly

1. **The zero-drift manifest net is RED and has been since 28a66ad
   (U2.2a).** Path-truth changed ledger bytes on all 25 witnesses by
   design; `preservation_expected_manifest.json` was last rebuilt at
   c061c92 and its re-bless is deferred to you. Bisected in pristine
   worktrees; environment exonerated (c061c92 reproduces its own manifest
   byte-exactly today).
2. **My R4/R5 "broad gate green" receipts were defective on that one
   net** — it fails deterministically at 2b1689a/478e34a/a778fb7, so those
   receipts cannot have contained a passing run of it. Every other net was
   and is green. Admitted in the 97a7d68 commit message.
3. **Pixel surfaces never drifted**: ir/expanded/params/html_meta verified
   byte-identical to the blessed manifest through R6/R7/R8 (25×6-surface
   hash proof at R6; 5-witness spot proof at R7). Only ledgers/sable moved,
   which is the intended evidence-surface change awaiting your bless.

## What is YOURS to do (never mine)

- **Manifest re-bless**: regenerate `preservation_expected_manifest.json`
  (the bless), which clears the standing red. The MusicGen witness
  (`musicgen-small.json`, deferred to R9 by your instruction) should enter
  as witness 26 in the same bless — the manifest pins witness_count, so
  adding it is inseparable from the rebuild.
- Gallery blesses for any witnesses you want re-captured.

## Open items carried forward (visible, not silent)

- `evidence/position.py` `_config_value/_config_scopes` walk configs RAW
  (funnel-invisible tie-break reads, e.g. layer_types) — needs wrapper-path
  naming per hop; named in R7/R8 commit messages and memory.
- 47 PENDING config_read rows in the StructuralDebt register (U7/U8/U10/
  U11/U12 units, checkable deletion conditions) + the extras/drawn rows —
  the ONE register is the live tracker; its gates block growth/staleness.
- Flags from the R7 disposition audit worth a look at their owning units:
  Lumina FFN width (ffn_dim_multiplier vs mlp_ratio=4 fallback — possible
  mis-drawn width, U10 row), Sana cross-attention head geometry (cross
  sublayer reuses self spec, U10 rows), add_watermarker (candidate scoped
  deletion at U11).

## R9 bracket receipt (2026-07-18, tree at 7110be6)

Isolated files (spec order): config_paths 81p/10s · arbitration 17p ·
config_access 47p · config_intents 8p · structural_writes 13p ·
fact_registry 29p · evidence_facts 330p · projection_obligations 5p ·
receipts 49p · projection_audit 38p · authority_probes 28p/3xf ·
isolation 6p · reader_exceptions 4p · preservation 19p + THE known red
(zero-drift manifest, yours to bless) · sable 33p.
Semantic controls (fact_ledger/layer_schedules/submodel_parity/diffusion/
code_evidence): 280 passed.
Full suite: 1476 passed / 10 skipped / 3 xfailed / 1 failed (the known
red only).
**Fingerprint before == after: IDENTICAL — the bracket is valid** (no test
mutated the tree; no code was edited while it ran).
