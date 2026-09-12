# S7 payload renewal on `b45bab26` (Q12)

The Linux gate `scripts/generate_s7_shadow.py --ci-shadow` re-runs S7 live and
compares against the **committed** payloads. It went red at `b45bab26` on the
first corpus model:

```
ValueError: live Linux S7 model artifact for 'auraflow-v0-3' disagrees:
  $.blocking_findings: committed length=1823, live length=1784
```

S9-A's claim-kind stamping changed the S7 result; only the *source stamp* was
renewed under Q11, so the payloads were stale against the code. This receipt
renews them.

## Authority

Soumil, 2026-09-12, verbatim: **"I approve the S7 payload renewal for S9-A."**
Persisted verdict [`owner-verdict.json`](owner-verdict.json), sha256
`cd51764b8c12faf77c0d62e0a0fd6ba84d9da21a7be927cb1b9d614fa8ca40e0`,
`covers: ["Q12"]`, `persisted_by` present, reviewer
"owner (Claude), z-docs/11-execution/S9-A-review.md" ≠ implementer. Soumil
persisted it himself; the executor drafted the shape and neither authored nor
edited it.

## Method

A **fresh full 39-model S7 shadow regenerated on the committed head**
`b45bab26539eca773c2f03519ebfd72ca7719d93` with a clean tree
([`run-matrix.py`](run-matrix.py), [`generate-matrix.py`](generate-matrix.py),
[`matrix-lane-result.json`](matrix-lane-result.json)): rc 0, 1,355.83 s, source
bracket `4c1ca4691fb95f5a…` and artifact bracket `5aed5480485a877c…` equal
before and after, `active_matrix_renewed: false`. No row was hand-edited.

That output is **byte-identical to the candidate37 matrix the owner audited** in
`z-docs/11-execution/S9-A-review.md` §2 — 39/39 model payloads, 39/39
observations, 39/39 relations — which is stronger than the logical identity the
ruling required.

## Enumeration

[`renewal-reconciliation.json`](renewal-reconciliation.json) (sha256
`15f887c3c6b2ebb289ce0f10211ef035a929f3e7151c27b7a2b20ccc64110cf1`) compares
fresh against the committed tree and reconciles every item against the owner's
`../S9-A-owner-review-37/matrix_audit.json`. **All twelve items agree:**

| item | value |
|---|---|
| occurrences | 44,088 = 44,088, identity and order identical on all 39 |
| construction / execution / schedules | identical on all 39 |
| citations removed | 0 |
| formerly unqualified now qualified in place | 16,324; 0 still unqualified |
| new unqualified | 481, all `scores_scale` (declared, unproven; S9-C) |
| claim kinds changed on existing citations | 0 |
| newly cited | 18,565 |
| blocking findings | 70,186 → 54,257 |
| relations | counts equal on all 39; five tying rows gain `config_paths` |
| provenance | additive only, 17 models |

### The 86 DeepSeek-V4-flash norm placements

The one enumerated difference outside "pure stamping": 43
`model.layers.N.input_layernorm` + 43 `model.layers.N.post_attention_layernorm`
move `projection_unresolved (structure_unaccounted)` → `rendered`, reason class
cleared. Placed by the decoder-norm reader's exact occurrence citation (proof
kind `retained_reader_projection`, reader `decoder_norm_kind_for_path`). **No
occurrence became unresolved.**

The **drawing is identical**: the owner rendered deepseek-v4-flash on the
pre-S9-A baseline worktree (`verify-s9-a-closure-baseline`, `fff99e22`) and on
HEAD — 7 nodes, 13 cards, 4 SVGs, architecture SVG hash `806af6b0ab8acae7` on
both. This is accounting: already-drawn norm blocks now cite their occurrences.
These are the "86 DeepSeek-V4 direct norm placements … named re-proofs" the
S9-A sheet named. Ruled **inside** the Q12 scope; the owner corrected the §2 row
that had claimed projection kind was identical on every occurrence and added the
§9 addendum.

Summary-row effect: `rendered` 2 → 88, `projection_unresolved` 1,413 → 1,327,
`structure_unaccounted` 1,413 → 1,327, `blocking_findings` 1,604 → 1,431.

## What was installed — 61 files

**60 of 117 payloads** ([`payload-split.json`](payload-split.json)): models 39,
observations 1, relations 20. The other 57 were pinned as writer controls and
verified unchanged.

`matrix.json` field changes: `artifacts` 39/39, `logical_artifacts` 39/39,
`relation_artifacts` 20/39, `logical_relation_artifacts` 19/39,
`observation_artifacts` 1/39, `models` summary rows 35/39 (fields
`qualified_fact_citations`, `unqualified_fact_citations`, `investigation_missing`,
`blocking_findings`; plus `rendered` / `projection_unresolved` /
`structure_unaccounted` on deepseek-v4-flash alone).

**`sources` byte-identical** (394 entries, the Q11 stamp) and **`targets.json`
byte-identical** — asserted by the writer before and re-verified after.

## Checks

- **S7 check on a scratch overlay: PASS**, 9.54 s
  ([`scratch-check-result.json`](scratch-check-result.json)). The committed
  `verification/s7` was copied to scratch, each of the 61 files verified against
  its `old_sha256` before overlay and `new_sha256` after, then the unchanged
  generator's `check(output=scratch)` was run. The active tree was verified
  untouched by that run. `tests/test_s7_artifacts.py` hardcodes
  `ROOT/verification/s7` (line 41) and could only run after installation.
- After installation: `tests/test_s7_artifacts.py` **42 passed** (14.84 s);
  `tests/test_s5_release.py` **13 passed** (37.43 s).

## The writer

[`install.executed.py`](install.executed.py) with
[`manifest.executed.json`](manifest.executed.json), manifest pin
`5e45f3c95ce53b85df48655f6919a8f0c39e336735153b15abca34ba295195cf`. Result
[`result.json`](result.json): 61 files, `controls_and_inputs_unchanged: true`.

Two changes from the S9-A output-delta writer, both owner-approved and recorded
in `z-docs/11-execution/S9-A-review.md`:

1. **`covers` is plan-pinned.** It was hardcoded `== ['Q10', 'Q11']`; this
   installation is approved under Q12, so the expected ids come from the
   manifest, which is itself sha256-pinned on the command line. Same equality,
   pinned value.
2. **The enumeration predicates are selected by a plan-declared `kind`.** The
   `s9a_output_delta` branch is byte-for-byte the original. The new
   `s7_payload_renewal` branch asserts this unit's own predicates: the
   enumeration's `DISAGREE` list must equal the manifest's `allowed_disagree`
   **verbatim** (so any entry beyond the ruled-in-scope 86 aborts), the
   projection-kind change map must equal `allowed_projection_kind_changes`
   exactly, fresh must be byte-identical to candidate37 on all 39 models plus
   observations and relations, all twelve reconciliation items must agree,
   removed citations / changed claim kinds / still-unqualified must be 0, and
   `sources` and `targets.json` must be byte-identical between the live tree and
   the candidate. An unknown `kind` aborts.

[`install-attempt-1-returned/`](install-attempt-1-returned) retains the first
run, which returned `KeyError: 'STOP'` — the writer still expected the S9-A
enumeration shape. It wrote nothing (`attempted_paths: []`, controls unchanged)
and is kept rather than discarded.
