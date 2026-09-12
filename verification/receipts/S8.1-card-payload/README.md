# S8.1 compact card transport — released bounded receipt

The transport stores precomputed card/SVG strings and materialises selected
cards on click. Occurrence bindings, own quantities/provenance, parent variants,
and panel depth remain distinct. Exact template sharing and a reversible
declared-mount token reduce duplication; no model semantics are inferred here.

The inverse validates references, original ordered card spans, canonical
parent/depth membership, compact container placement, mount binding, and the
exact declared lazy interaction script. It reconstructs the original eager
HTML, including its original interaction script. That inverse is for exhaustive
inspection; actual compact bytes remain the raw preservation surface.

Current explicit presentation deltas are the JSON payload, one marker per
contiguous card parent, lazy interaction code, and absence of unselected cards
from the live DOM. Precomputed SVG strings/CSS and original eager HTML remain
recoverable exactly. Unsupported interspersed non-card layouts or reserved-token
collisions keep their original eager output.

Executed focused controls: `python3 -m pytest tests/test_card_payload.py -q`:
46 passed in 0.50 s, followed by 47 passed in 0.16 s after adding the
cross-consumer control. Both exact source snapshots and raw logs are archived
under `focused-46` and `focused-47`. These include two-nonce comparison
under the unchanged preservation normalizer, non-BMP prefixes, same-ID cards in
different parents, own-fact/SVG differences, malformed references, and omitted
or altered interaction scripts.

The first public render uses the immutable 318-file snapshot at
`/private/tmp/unfold-s81-render-snapshot-v1`, from the retained actual SDXL
pre-render IR. Its actual page is
`/private/tmp/unfold-s81-html-generation-v1/page.html`:
2,171,535 bytes, SHA-256
`51d318d35ea6ae6abc23af285b63cb7d556a5740d2e39d22849274607801c7d2`.
Its inverse is exactly the original 23,748,293-byte page with SHA-256
`d0ce8587ddf7aad17f4528ab314237e9d296315e59210462263023772237dc06`.
The run then **failed** encoding a frozenset in its event archive. Source,
input, and import-origin checks passed. The original FAILED receipt/log stays
intact; its timing locals were not persisted, so it is not a complete timing
receipt. The prepared harness correction explicitly tags frozensets and saves
timings before event archival.

The corrected, separately retained run at
`/private/tmp/unfold-s81-html-generation-v2` passes: public HTML generation
5.205 s, comprising 2.162 s canonical generation and 3.043 s packing. Inverse
inspection takes 1.256 s. Its raw compact and canonical hashes equal v1 exactly;
the input IR and pre-/post-packing render events are unchanged, and all
source/input/import checks pass. This is one retained-IR render measurement,
not an end-to-end model construction measurement.

The synthetic variant pair at `/private/tmp/unfold-s81-variant-fixture-v1`
contains two same-depth parent variants with shared card IDs and different
quantities. Its 617,453-byte eager page becomes 239,266 bytes with exact inverse.
The fixture-only threshold override is restored and the production size guard
remains active. This is a navigation control, not a model witness.

Independent static design/code review and the narrowly scoped consumer/test
migration are accepted. The root-owned
[browser timing receipt](../S8.1-browser-candidate-51d318d/summary.json) reports
443.7 ms median ready and 19 ms median click-to-painted card. Its
[settled controls](../S8.1-browser-settled-51d318d/summary.json) establish exact
active-card markup/text/identity/placement through all 16 SDXL steps, initial
and 14/16 screenshot byte equality, and 8/8 synthetic variant screenshot
equality. The two remaining SDXL screenshots differ by 15 pixels total; no
whole-page pixel identity or unsupported explanation for those pixels is
claimed here.

The archive contains 348 injectively named streams, including both public
render attempts, the exact frozen snapshot, retained input IR, tool/test
sources, logs, and synthetic pages. Every stored stream was decompressed,
hashed, size-checked, and restored into a new directory. Run
`python3 verify_restore.py --out /new/inspection/directory` from this receipt
to independently restore them. `growth.json` and the compressed exact source
diff distinguish 549/4 production lines, 9/3 inspection-tool lines, and 240/34
test lines added/deleted; no authority paths or debt waivers are introduced.

Broader model-backed consumer verification, end-to-end generation performance,
and baseline disposition remain separate. This receipt does not declare
S8.1 complete or authorize any baseline update.
