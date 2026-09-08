# S8.1 browser baseline

Actual saved S8 SDXL page at `788abbd`: 23,748,293 bytes, SHA-256
`d0ce8587ddf7aad17f4528ab314237e9d296315e59210462263023772237dc06`.

The three quiet fresh-browser samples (`before-1b`, `before-2b`, `before-3`)
have median readiness **1,138 ms** and median denoiser click **109.6 ms**.
Every sample opened the denoiser's own card in panel 0 without a runtime
exception, and the input HTML hash stayed unchanged. Readiness saw 192,833
DOM elements and 11,717 card containers; DOM elements are not architectural
node counts.

`summary.json` lists every included sample and exclusion. `before-1` failed
to initialize Chrome under the terminal sandbox. `before-2` completed but
overlapped a codec profile and is excluded from the quiet median. Both raw
receipts remain present.

Method: Chrome for Testing 151.0.7922.34 in a fresh temporary profile,
1440 × 1000, device scale 1, local file navigation. HTTP(S) requests,
including Google Fonts, were explicitly blocked before navigation. This
measures a fixed offline fallback-font condition, not online loading.
Readiness is load plus fonts-ready plus two animation frames; click time is
the native click event through two animation frames and the visible own-card
assertion. It measures a paint opportunity, not hardware presentation.

The exact observer is retained as `browser_benchmark.py` (SHA-256
`ca0a13d375252bddc51bf25f5b8708b1770e6efb39e6ce4cd34371752e0f22b6`).
Raw timings, exceptions, network failures, screenshots and input hashes are
retained per sample. This is a baseline only, not an S8.1 acceptance receipt.
