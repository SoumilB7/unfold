# S7 source-stamp renewal — S10-1 candidate, then the S9-A superset

`verification/s7/matrix.json` carries a `sources` map: the sha256 of every file
in the generator's closed dependency surface. It is a freshness stamp, not
evidence. Renewing it re-hashes those files and changes nothing else — no row
is regenerated, no payload is rewritten, no observation is re-run.

Two writes, in the order the owner ruled
(`z-docs/11-execution/S9-A-review.md` §8), each through the same guarded writer
against a clean tree, because the writer's "no tracked change may precede
installation" guard was left untouched (owner ruling **(B)**).

| write | from | to | entries | deltas |
|---|---|---|---|---|
| A — S10-1, Soumil's Q11 candidate | `47f372e4…` | `efe7d97e…` | 388 → 385 | 11: 8 changed, 3 removed |
| B — S9-A superset over the current tree | `efe7d97e…` | `6c29d740…` | 385 → 394 | 81: 72 changed, 9 added, 0 removed |

Both writes assert, before and after: **all 39 model rows byte-identical, all
117 payloads byte-identical (and matching their bytes on disk), every
non-source matrix field byte-identical.** The 117 payloads are additionally
pinned as writer controls, so a payload change would abort the install.

## Authority

Soumil, 2026-09-12, verbatim **"i approve both"** — Q11 is the S10-1 source
stamp, candidate `efe7d97e2504c38d3679e05521f556068284fe638ad9cb83c646cdd196dd8b2c`
(`z-docs/12-design/00-design-decisions-verbatim.md`). The superset in write B is
the owner's ruling in `z-docs/11-execution/S9-A-review.md` §8: because the S9-A
commit changed production sources, the stamp must be produced again over the
current tree with the unchanged generator's source-hash step only. Persisted
verdict: `../S9-A-owner-review-37/owner-verdict.json`,
sha256 `4534786fc484cf30725535c651b9cce9c4af22ae79dd2081cfc883efd105c562`.

## Method

Write A installs the exact bytes retained in the S10-1 receipt
(`S10-1-unet-deletion-fff99e22`, artifact `deletion/s7-stamp-candidate-v2/matrix-candidate.json`),
independently re-verified here against the live matrix before installation.

Write B is `matrix['sources'] = _source_hashes(_targets())` from the **unchanged**
`scripts/generate_s7_shadow.py` (sha256 `e415c5d0a4efac1ae655d4a992a08087daa015e6d2438dc8e4771fddbbead70c`),
with the generator's own serialization (`json.dumps(..., indent=2, sort_keys=True)` + newline)
and a deep copy of every other field. Nothing else in the module is called: no
target is built, no model is parsed, no observation runs.

Write B's 9 added entries are the S9-A modules — `evidence/activation_registry.py`,
`checkpoint_metadata.py`, `class_default_value.py`, `presentation_census.py`,
`presentation_projection.py`, `reader_claims.py`, `reader_placement.py`,
`presentation.py`, `uncertainty.py`. Its 72 changed entries are the S9-A-edited
production sources. Full lists: `install-a/enumeration.json`,
`install-b/enumeration.json`.

## Result

`tests/test_s7_artifacts.py` green on the final tree; `tests/test_s5_release.py`
re-run green on the same tree. Per-install writer results in
`install-a/result.json` and `install-b/result.json`, each with its executed tool
and executed manifest.
