# Transport consumer audit — read-only disposition

The production transport changes actual delivered HTML bytes and DOM residency.
Its explicit inverse is an exhaustive inspection interface; it must not replace
the raw HTML submitted to preservation or byte-identity controls.

| Consumer | Current boundary | Disposition |
|---|---|---|
| `preview.svg_views` and `render_images` leaf count | All precomputed SVGs and description-only cards | Patched to strict inverse before existing extraction; dedup rules unchanged. |
| `block_schema.validate_click_coupling`, `validate_unique_ref_ids`, `validate_no_dotted_arrows`, `validate_no_dotted_boundaries` | Exhaustive precomputed markup | Patched to strict inverse before unchanged validators. Inverse rejects malformed references, placements, shell, and interaction script. |
| `sable.sable` view signatures at line 672 | Calls `svg_views` | Already uses adapted entrypoint. Fact projection/reconciliation consumes original render events, generated before packing. |
| `scripts/generate_examples.py` | Saves actual `to_html`; hero uses preview | Keep actual compact bytes; preview handles hero extraction. |
| `scripts/report_s8_demonstration.py` `PageEvidence`, `condition_checks`, `_spatial_render_evidence` | Direct card parsing, prose windows, card SVG slices | Needs explicit inverse inside semantic inspection only. `read_case` actual file/hash verification and raw `html.diff` must remain original bytes. Patched at those semantic entrypoints; raw checks remain unchanged. |
| `scripts/dit_coverage.py:109` unknown-fact text count | Direct HTML text count | Needs explicit inverse for its semantic count. Actual saved page remains raw. Patched at those semantic entrypoints; raw checks remain unchanged. |
| `tests/test_diffusion.py` card-slice helpers and direct card/overview assertions (e.g. 1413, 1436, 1668, 1693, 1857, 2002, 2051, 2094) | Expects eager card tags | Needs explicit inverse at the relevant semantic assertion/helper boundary. No assertions removed; patched at explicit semantic call sites only. |
| `tests/test_submodel_parity.py:209–212` | Direct encoder card identity assertions | Same explicit inspection boundary needed for packed pages. Patched only at the explicit semantic call sites listed in `semantic-test-entrypoints.json`. |
| `tests/test_smoke.py` direct card identities (e.g. 681–687, 1244–1249) and `tests/test_code_evidence.py` | Some semantic checks inspect raw tags | Review per assertion; small pages remain eager, but negative assertions must never pass just because cards are stored. Patched only at the explicit semantic call sites listed in `semantic-test-entrypoints.json`. |
| Dedicated graph/card renderer tests | Receive individual SVG/card fragments before document packing | No change needed. |
| `test_support.preservation.html_meta` | Actual raw delivered page under its existing one-mount normalization | Leave unchanged, including its raw metadata. Enumerate packaging changes rather than replacing it with expanded HTML. A dedicated two-mount codec control verifies the existing normalizer remains sufficient. |
| `scripts/verify_release_install.py` | Raw document/root presence and warning banner | Root/banner remain eager; raw SHA remains actual output. |

This is a bounded source census, not a passing broad-gate receipt. Browser
navigation and actual per-occurrence facts still require independent controls.
