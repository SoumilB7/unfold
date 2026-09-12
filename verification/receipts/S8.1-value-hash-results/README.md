# S8.1 value-hash controls and actual saved-SDXL replay

The only production change owned here adds memoized process-local field-tuple hashes to SourceId, SourceSpan and ExprNode. Full equality and content fingerprints remain unchanged. The nonfield cache is omitted by pickle/deepcopy and naturally absent from dataclass fields/asdict; replace resets it. Hash collisions still require full equality, and malformed unhashable fields still raise TypeError without caching errors.

**23 cheap tests passed, 6 unrelated existing static tests were deselected** by the explicit `hash or pickle or copy` selection. Actual complete stdout is retained. Source scope comparison in the preparation receipt proves all AST outside the three class bodies equals the already accepted static candidate, preserving ProgramIndex methods/maps.

The independently reviewed actual SDXL source-bundle replay passed using one exact historical BuildRequest/InventoryResult from the original baseline. No model was constructed. The full returned IR bytes and original inventory/request match exactly; source package membership/content, input, selected external and script hashes match before/finally. An isolated uncommitted source snapshot was copied with exact before/after/copy equality so concurrent source preparation could not restamp the replay.

Actual field-tuple computations were exactly once per retained object:

| Type | Computations | Distinct strongly retained objects |
| --- | ---: | ---: |
| SourceId | 32 | 32 |
| SourceSpan | 238,826 | 238,826 |
| ExprNode | 165,766 | 165,766 |

Rehashing every retained object against its independently assembled declared-field tuple leaves computation counts unchanged. The observed replay interval was 11.727612s wall and 10.669958s own process CPU, including diagnostic observer work. This is not a fresh model run, persistent cache-hit test, cold/warm budget result, or an apples-to-apples timing comparison with the earlier pressured/profiled run.

The archive includes the actual snapshot, exact historical inputs, actual output/logs, executed tools/tests and independent pre-use review. Appended `.gzip` paths are injective; every stream was decompressed and checked byte-for-byte. No baseline, family blessing, page regeneration or user document was changed.
