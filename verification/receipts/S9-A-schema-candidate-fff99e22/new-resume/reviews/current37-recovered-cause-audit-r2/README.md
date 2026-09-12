# current37-recovered-cause-audit — r2 (source only) and r3 (the run)

`audit.py` is **r2**, retained byte-for-byte (`7e4c596da18db32dfb482d8cebd76d5ffffd6b9d8098df97f45cc91376bcb939`,
listed in `manifest.json`). It was source-reviewed but never executed on a real
packet, and it crashes there — owner finding **A4**
(`../../../../S9-A-owner-review-37/lane4-cause-audit-FAILED.log`).

`audit-r3.py` is the corrected tool. **The exact change:** r2 asserted
`assert len(originals) == 1` (one render flagged `is_original_html_surface` per
witness); the retained packets carry **four** (three `canonical_surfaces` phases
plus `_view_hashes`). r3 accepts the four, asserts the phase order, asserts they
are the same page under the tool's existing first-mount law (each render
identical after replacing its own first `uf-<hex>` mount id — anything else is a
hard failure), records per witness the four raw hashes / mount ids / whether the
difference is mount-only in a new `original-render-audit.json`, and uses the
first render as the page, which is the surface r2 intended to use. No other
semantics were changed; `diff -u audit.py audit-r3.py` shows only the docstring,
the `first_mount_normalised` helper, that block, and two output additions.

Measured on the real packets: **58/58 (packet, witness) rows carry four original
renders, 58/58 are identical under the first-mount law, and 58/58 differ raw by
mount id only — 0 are byte-identical.** The mount-only difference is therefore
universal in both packets, not specific to the compact-transport witness.

Run (2026-09-12, from `unfold-pkg`, completed):

```
python3 verification/receipts/S9-A-schema-candidate-fff99e22/new-resume/reviews/\
current37-recovered-cause-audit-r2/audit-r3.py \
  --output /private/tmp/unfold-s9a-thirtyseventh/preservation-cause-audit
{"witnesses": 29, "findings": 17, "causes": {...}}
```

Its enumeration is reconciled against the owner's in
`../../../../S9-A-owner-review-37/cause-audit-reconciliation.json`; the run's
`summary.json` and `original-render-audit.json` are copied there.
