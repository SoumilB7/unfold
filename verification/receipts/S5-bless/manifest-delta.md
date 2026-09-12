# S5 approved bless — manifest-delta inspection

- prior manifest: committed tree `d153ad8`
- regenerated from: implementation tree `fb3e2fa` plus receipt-only `d153ad8`
- regenerated candidate SHA-256:
  `a1fe7d578c16bbed0d44b77ad9c60f77a210ded565473b25777c2b61716743a9`
- pre-review candidate comparison: **byte-for-byte equal**
- denominator: **29 witnesses before and after**
- tool/version record changed: **no**
- input hashes changed: **none**
- ordered view sequences or hashes changed: **none**
- gallery hashes changed: **none**
- IR or expanded-JSON hashes changed: **none**
- parameter hashes changed: **none**
- ledger or Sable hashes changed: **none**
- unexpected witnesses or surfaces: **none**

The actual changed set equals the approved set exactly:

1. `granite-3-0-8b-instruct`: canonical `html_meta` only;
2. `stable-diffusion-xl-base-1-0`: canonical `html_meta` only.

For Granite, the two existing `config_accessed_unprojected` rows are presented
under one producer-authored summary and retained exactly under disclosure. For
SDXL, the eleven existing rows receive the same presentation. No architectural
claim, evidence status, exact warning string, SVG, pixel, or non-HTML surface
changed.

Comparison method: regenerate the complete canonical manifest to a temporary
path with `test_support.preservation.build_expected_manifest`; compare the old
and candidate documents by witness, input hash, every named canonical surface,
ordered view rows, witness count, and version record; compare the regenerated
candidate byte-for-byte with the independently reviewed candidate; only then
replace the tracked manifest with those exact bytes.
