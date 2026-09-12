# S4 approved bless — manifest-delta inspection

- prior manifest: committed tree `08fc8cd`
- rebuilt from: implementation tree `8f9862e` plus receipt-only `08fc8cd`
- denominator: **29 witnesses before and after**
- tool/version record changed: **no**
- input hashes changed: **none**
- view sequences or hashes changed: **none**
- gallery hashes changed: **none**
- parameter hashes changed: **none**
- ledger hashes changed: **none**
- unexpected witnesses or surfaces: **none**

The actual changed set equals the approved set exactly:

- all 29 witnesses: canonical `sable` surface only;
- `granite-3-0-8b-instruct`: additionally `ir`, `expanded`, and `html_meta`;
- `stable-diffusion-xl-base-1-0`: additionally `ir`, `expanded`, and
  `html_meta`.

The Granite product delta consists of two exact unresolved-evidence receipts for
`root.embedding_multiplier` and `root.logits_scaling`. The SDXL product delta
consists of the eleven exact `root.denoiser` unresolved-evidence receipts listed
in the independent review. No SVG/gallery byte changed.

Comparison method: generate the complete canonical manifest into a temporary
path with `test_support.preservation.build_expected_manifest`; compare the old
and candidate documents by witness, input hash, each named canonical surface,
ordered view rows, witness count, and version record; only then replace the
tracked manifest.
