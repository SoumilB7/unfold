# Remaining preservation capture: independent pre-use review

**ACCEPT the released four capture scripts for their bounded diagnostic use.**
No model, pytest, render, baseline update or production change was performed for
this review. This verdict does not accept any future output delta.

`accepted-source/` and `accepted-source-pins.json` identify the reviewed scripts.
`initial-source/` preserves the immediately preceding version returned for a
missing expected-input check in the eight-example wrapper. The corrected wrapper
now verifies each executable fixture hash against the expected manifest before
generation, requires the local source used by the actual generator, and compares
the ordered actual config with that fixture before calling the parser.

Verified boundaries:

- Each remaining current witness uses one actual `sable` invocation. Its existing
  source performs one `config_to_ir` and retains the resulting Diagram. The wrapper
  records that actual IR/page; subsequent serialization/rendering uses the same
  parsed object. It does not invoke another model parse for comparator sidecars.
- The existing eight-row release generator remains unchanged. Its wrapped
  `unfold` returns the original Diagram after capture; all eight rows and the
  existing checker are retained. Public example mount/provenance bytes remain
  untouched. Canonical counterparts are explicitly separate fresh renders of a
  deep-copied, equality-checked IR, not rewritten public HTML.
- Graph and Region observers call their original producer and return its exact
  original object/string. Their copies are observation records only. Retained
  graph references prevent object-id reuse from associating a later graph with
  an unrelated captured Region. Missing Region observations remain absent.
- The old-renderer diagnostic holds IR and mount equal and patches only the
  old `_draw_parallel`. The independent read-only helper check confirms all
  **25 transitive same-module functions/constants/import bindings are AST equal**.
  The harness repeats and records that check at use. Independent-input parallels
  refuse this comparison. Imported helpers resolve to the current pinned package;
  this is not a historical whole-product replay or new mechanism proof.
- Expected surfaces come from `tests/preservation_expected_manifest.json`, not
  the U0 baseline. Existing comparator sidecar/canonicalization functions remain
  unchanged. Old view hashes must recover the exact expected ordered distinct
  views before renderer-only attribution can continue; mismatch leaves actual
  artifacts and stops. Recovery alone is explicitly not output approval.
- Both wrappers verify actual local `capture_common` import identity. Production
  source plus actual helper, comparator, expected manifest, fixture and script
  pins are recorded and compared in `finally`, including parse/render/assertion
  failures. Their earlier missing-helper-path and missing-finally issues are
  corrected. Initialization failure before entering model work is not reported
  as a successful capture.
- Writes target new capture/candidate directories. Reviewed examples are read or
  copied outward, never overwritten. The candidate manifest intentionally lacks
  a new PNG seal; `rasterize_hero=False` remains explicit. Neither script blesses
  an output or edits expected baselines.

The run manifest's intended denominator remains 20 current Sable captures plus
the existing eight-example generator, with Bloom reused separately; the accepted
historical sidecar lane is a distinct already-reviewed script. Final execution
must pin the actual immutable source tree and these released script bytes, keep
failure/result/finally receipts together, and retain every observed output delta
for owner review. `CAPTURED_REVIEW_REQUIRED` is not a green preservation verdict.
