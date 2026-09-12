# Retained source closure — bounded ownership assessment

The SDXL matrix rejection is a real caller/index integration failure, not missing mechanism evidence or permission to weaken `_fact_source_keys`. The producing family readers already retain the nested attention source in `evidence.bindings.index`; the family verifier consumes it. The generic matrix caller currently returns the earlier index. This assessment gives required premises/controls; it does not claim an implementation correction is verified.

## Earliest stale ownership

`scripts/generate_s7_shadow.py:310–323` builds an index, resolves root and static claims, sets context._program_index, runs config_to_ir, then returns the earlier local index/root/claims. The cutover's final reader closure is extended through cells/nested/lookup at unet_runtime.py:94–100 but is not retained as the context's final source index. A later fact citation outside the earlier index correctly fails strict projection qualification.

The generator also replaces its local SourceBundle.component_architectures from the resolved inventory class while leaving context.source_bundle unchanged. Even when SDXL's names agree, the source/index/root consumer pair should share the same context-owned bundle. Actual runtime class is an address witness, not permission to infer architecture from its name.

## Small lawful boundary

1. Retain the exact reader-produced final immutable closure in the originating parse context. It must be an extension of the same parse's existing observations/source ownership; no scan over fact-key spellings, guessed source spans, reread-by-name, or parallel product fact layer.
2. The generator must read the final context-owned index **after** parsing, then derive its root/static construction claims from that same index and matching source bundle. Changing only returned index leaves root.graph/claims tied to the old closure.
3. Preserve original evidence objects and observation identity. `_merge_index` already retains base observations and adds exact records, with source fingerprints. Added source availability does not become an execution or mechanism proof without its reader.
4. Keep strict source-key validation. The nested attention claim must pass because its exact span's owning source/observations are now retained; truly absent or foreign source citations must still fail.

## Concrete cache boundary

ParseContext._component_inventory caches `ComponentOwnerInventory`, whose root and active entries embed `ComponentRootResolution` graphs from the index used on first call. Its cache has no index key. If a context adopts a new closure after this cache is primed, the next component_inventory() otherwise returns the old graph/availability result. Clear this one derived cache when the retained index changes; an unchanged index may keep it.

Current production search finds no direct `.component_inventory()` call outside its definition; its memoization is exercised in tests. Therefore this is a concrete API/cache-contract risk during index replacement, **not evidence that it caused the observed SDXL matrix failure**.

`reader_results` is different: it preserves reader-authored attempts/results, some holding their own immutable producing indexes. Do not indiscriminately erase it, projection receipts or facts. Parser's cached `root.denoiser.component_root` and `root.denoiser.topology` were produced before cutover; if a later consumer requests them as *current-closure* queries, they need an explicit targeted refresh or a clearly retained producing-index contract. Their historical results need not be falsely restamped as results of the new closure. No generic cache framework is required for this bounded caller fix.

## Required focused controls for the correction

- A nested source citation absent from the initial index is rejected before adoption and accepted using the actual final reader closure; an actually absent/foreign citation still rejects.
- Generator fake-parse/controlled fixture that extends the context verifies returned index, resolved root.graph and static construction claims all use the retained final closure, not the saved local.
- Prime component_inventory under index A, retain extension B, then verify its next resolution uses B; same-index retention does not spuriously invalidate; another context remains unchanged.
- Same context/source bundle and runtime root address remain coherent, including the generator's component_architectures replacement.
- Existing typed fact/reader-result evidence is preserved and is not re-authored by adoption. Prior source/observation rows remain present, with unchanged owner/hash identity.
- Replay the exact saved SDXL failure through strict source-key/projection reconciliation using the final retained closure, then complete the already-authorized matrix freshness lane. No gate weakening or per-model exception.

The code/source snapshots and hashes reviewed are in source-pins.json. No production edits, pytest, model runs, new grammar or cache framework were performed.
