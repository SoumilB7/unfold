# Bloom actual attribution at 736fdab — geometry RETURN

The accepted-S7 333-file snapshot reproduced the committed expected Bloom ledger hash exactly: cb0f1c08fb94057f4400c55e893d680c3a6d428d45f1f9582942b2d7edd1a520. Source pins matched before and after the parse. The current capture invokes Sable once with the exact corpus input and source=local, retaining actual typed IR, canonical Graph/Region declarations, HTML, events and native SVG views.

Only one of four distinct views changes: attention. Both architecture views and FFN retain the locked hashes. Re-rendering the identical typed IR with only the old83140 _draw_parallel function restores all four expected hashes. The exact source cause is two empty-lane edges already present in the canonical Region: kv_cache→scaled_scores and kv_cache→attn_apply_v. The old renderer omitted them. The XML difference consists only of two circles and four paths.

Actual native inspection RETURNS the new geometry: the value rail at SVG x=78 crosses the scaled-scores and Softmax boxes before it reaches the apply-values node. Restoring a declared edge is warranted, but this routing is not accepted. The native before/after images and exact XML diff are preserved. No baseline or output is blessed.

The separate old/current ledger comparison has exactly one value delta: config_access.projection_coverage.receipted_scopes adds twelve root.denoiser scoped declarations and removes none. Every other ledger value is identical. This declaration-list change is not a claim that Bloom contains a denoiser. Current exact ledger hash:51795351432a269f64405cf393a323aba9f9ab30822692f2e21ae6bd6b0fa25c. Full values are in current/ledger-deltas.json.gz.
