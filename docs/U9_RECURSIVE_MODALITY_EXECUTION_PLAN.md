# U9 execution plan — recursive modality ownership and mechanism authority

> **Authority:** bounded execution plan for master plan §20.12 (M-01..M-16).
> U3 is closed on `4bd1395`; U9 consumes that substrate and must not add a
> second parser, owner resolver, or family/config mechanism table.

## 1. End state

```text
one SourceBundle + one ProgramIndex
  -> exact root construction occurrence
  -> exact active nested component occurrences
  -> shared U6/U7/U8 mechanism readers at each occurrence
  -> exact fusion/projector/tower facts
  -> parser, renderer, JSON, params and conformance project those facts
```

A config-declared component that is not reached by a proven construction is
inventory only.  A width mismatch, field presence, model/family name, class
substring, or renderer convention may not create architecture.

## 2. Finite implementation units

| Unit | Exact deliverable | Old authority deleted in the same unit | Exit condition |
|---|---|---|---|
| U9-A | Closed `ComponentOwnerInventory`: root plus every non-pipeline component classified as active, declared-unused, ambiguous, failed, or unavailable from exact U3 construction evidence | Any new component selection by “only candidate”, class/field substring, or source-file order is forbidden | Active entries round-trip through their component owner graph; declared-unused never becomes an active path; rival and broken-source poisons pass |
| U9-B | Fusion reader over the one ProgramIndex and exact root occurrence; typed result shared by parser and conformance | `fusion.py::_class_node`, its `ast.parse`, reachable-class best-depth selection, whole-forward `ast.unparse` substring decisions | placeholder replacement, prefix concat, grid-position route, cross-attention route, rivals/partial source/renaming controls; no raw source reopen |
| U9-C | Projector/merger owner and op-chain reader over exact active component/root occurrences | `projector.py::_class_node`, shallowest/field-rank pick, config mechanism classification and default-linear fallback | exact construction occurrence, exact ordered op chain, width operands only after code proves their use; width-mismatch-without-projector negative |
| U9-D | Recursive vision/audio/text/codec tower facts reuse U6 attention, U7 FFN/norm/cell, and U8 position/schedule readers | modality-local MHA/dense/cache conventions and duplicate AST helpers | dense-vision/gated-text sibling and mixed-attention variants remain occurrence-separated; unknown stays unknown |
| U9-E | Exact patch/embed/tiling/pooling/pixel-shuffle/feature-selection/token-emission readers | config-presence position, tiling, reduction and feature-selection mechanisms | positive local relations plus unused-signal, partial-source and rival-owner negatives |
| U9-F | Parser projections consume only U9 facts; `tower_submodel_spec` becomes a pure typed-fact projection | `is_unified_grid_stream`, `has_cross_attention_adapter`, detailed provisional modality paths, width-mismatch projector creation, fusion fallback | config can discover candidates and supply operands, but cannot choose any mechanism |
| U9-G | Renderer/metadata/params/conformance consume canonical component facts only | renderer-local encoder classification and `_MODALITY_BLOCK_SPECS` structural fallback | reverse-fabrication and projection-receipt gates cover every changed modality surface |
| U9-H | Delete remaining U9 parse authorities, close debt, run witness/artifact acceptance | all four U9 legacy parse sites and satisfied U9 debt rows | generated reader inventory shrinks; U2 authority gates green; witness matrix inspected; unchanged-tree full receipt |

## 3. Required witness matrix

- Qwen2-VL unified grid and vision projector;
- PaliGemma/LLaVA placeholder replacement;
- Mllama cross-attention;
- Qwen2-Audio and MusicGen conditioning;
- one real prefix-concat route;
- Gemma multimodal multi-input routing;
- dense vision versus gated text siblings;
- mixed attention/FFN variants within one tower;
- declared but unused component;
- width mismatch without a constructed projector;
- same class constructed at two sites;
- rival component owner, broken source, unsupported syntax and source missing;
- complete class/field renaming controls where address syntax remains equivalent.

## 4. Non-negotiable gates per behavior-changing unit

1. The focused unit tests include an anti-vacuous positive and every relevant
   ambiguity/absence counterexample.
2. `scripts/generate_u3_reader_inventory.py --check` remains current; a legacy
   reader deletion shrinks its generated count in the same commit.
3. U2 authority/identity/structural-write nets remain green.
4. Parser and conformance consume the same typed result; neither reruns a
   separate semantic reader.
5. Renderers and parameter estimators receive facts/specs only—never config,
   source, ProgramIndex, or owner graphs.
6. Every intentional artifact delta is listed and inspected before re-bless.
7. The full committed-tree receipt uses one unchanged fingerprint.  Focused
   work may use the parallel coordinator; the serial full suite is reserved for
   unit boundaries and U9-H acceptance.

## 5. Stop conditions

Stop the current unit instead of guessing when an exact owner does not resolve,
the ProgramIndex lacks a needed neutral observation, a code relation is only
locally positive but the proposed fact requires whole-callable completeness,
or a config value would have to select the mechanism before code proves the
operation that consumes it.

## 6. Execution tracker

| Unit | Status | Receipt / exact remaining boundary |
|---|---|---|
| U9-A | DONE | `a6e334a`, corrected by `dd58aea`: exact active/unused/rival component inventory |
| U9-B | DONE | `ed7e2fc`, corrected by `dd58aea` and `db02668`: exact executable fusion closure, including unresolved-rival negative proof |
| U9-C0 | DONE | `b9708a6`: ProgramIndex observes direct positional constructors in the closed `Sequential(A(), B())` protocol as element storage sites without asserting execution order. |
| U9-C1 | DONE | `c54abc8`: explicit-owner return-path operation reader for exact affine, norm, registered activation and Sequential operations; it does not select a projector. |
| U9-C2 | DONE | `77ee915`: exact fusion-operand producer lineage, occurrence-qualified width operands and guarded-constructor provenance. The counterexample matrix covers PaliGemma, LLaVA, Qwen2-VL, Mistral-3, Gemma-4 and Mllama; shape/device/index metadata and helper calls cannot launder sibling producers. The U9-D1 correction restores Qwen2-VL's factory-forwarded `vision_config.hidden_size` through the closed framework-address protocol. |
| U9-D1 | DONE | `150c0c9`: exact returned-local output descent plus recursive repeated-stage composition. Idefics2 proves, from one connector occurrence, its affine prefix, Perceiver repetition/count path, exact attention, gated FFN, block/final RMS norms and source-ordered elementwise multiply. No class/field/model spelling selects the mechanism. |
| U9-C | CUTOVER IMPLEMENTED — ACCEPTANCE PENDING | The parser and conformance now consume the exact ordinary/repeated-projector inventory. Widths enter only through source-bound projector operands; config mismatch cannot create a connector. The unreachable legacy projector AST implementation has been deleted. Final status waits for the U9-H artifact/full receipt. |
| U9-D2 | DONE | `ee81c46`: component-neutral recursive tower inventory partitions every active nested component, retains the exact U3 candidate census, and reuses U6 attention, U7 FFN/norm/cell and U8 position `ReaderResult`s without collapsing failures. Qwen2-VL permanently pins gated text versus dense vision siblings, exact component paths and the honest boundary that unresolved vision attention/ordinary position evidence cannot become MHA/RoPE. |
| U9-E0 | DONE | `6f88847`: the shared exact operation reader recognizes framework-bound convolution, pooling, pixel shuffle/unshuffle, embedding, resize, concatenate, stack and split/chunk primitives. Same-spelled local classes cannot author those operations. |
| U9-E1 | DONE | `5f77a7a`: positive local def-use routes join exact operation-bearing calls to caller-supplied repeated-stage templates. The reader makes no completeness/absence claim; Qwen2-VL proves a real Conv3d frontend route while retaining its unresolved competing invocation. |
| U9-E2a | DONE | `650df2b`: multi-axis position construction requires an exact fused wrapper, exact `position_ids` child-input binding, return-dependent helper chain and framework-bound stack of at least three axes. Real Qwen2-VL and complete helper/local renaming controls pass; a positionish name, uncalled helper or overwritten result proves nothing. |
| U9-E2b | DONE | `a8fe557`: wrapper feature selection is projected only from exact def-use operations. Config-declared layer/strategy fields cannot author selection. Preprocessing tiling remains deliberately outside model-code authority until an external-processor source boundary exists. |
| U9-F | IMPLEMENTED — ACCEPTANCE PENDING | Config builders now emit opaque address-only vision/audio/video/conditioning lanes. The atomic overlay installs only recursive component, wrapper-feature, fusion and projector evidence. Unknown/source-missing paths retain navigation but no MHA/dense/RoPE/projector/token-route story. |
| U9-G | IMPLEMENTED — ACCEPTANCE PENDING | Renderer and expanded JSON consume canonical modality facts. Removed head-dimension arithmetic, scalar patch-grid synthesis, grid-route→patch-merger inference, unknown-projector relabelling and config-derived summary-cell defaults. Conformance joins the same component/fusion/projector results and checks every exact stage/variant. |
| U9-H | APPROVED + RE-BLESSED — FINAL COMMITTED-TREE RECEIPT PENDING | Soumil approved the finite §8 honesty delta. All 29 witnesses passed the guarded mechanical/oracle/gallery/reproduction path; the canonical baseline and expected manifest were rebuilt independently. Focused U9 215p, authority 90p and preservation 52p are green. Remaining: explicitly stage the reviewed U9 tree/artifacts, commit, run the detached-worktree exhaustive coordinator on that commit, record the receipt and push. |

## 7. Current cutover laws (binding for U9-H)

1. A component-level summary exists only when every exact repeated stage and
   every exact block variant agrees.  Mllama's local/global vision stages keep
   separate counts and residual gates; no last-stage-wins projection is legal.
2. Attention kind, Q/KV counts, head width, QKV storage and separate rotary
   application are occurrence-qualified facts.  One tower-level verdict may
   not be copied into heterogeneous variants.
3. FFN mechanism is occurrence-qualified.  The exact projection protocol and
   gated/activation facts may be drawn; `intermediate_size` stays unknown in a
   nested drill until an exact FFN affine-operand reader binds it.  A component
   config width cannot be borrowed by a variant.
4. A fusion route, a multi-axis position route and a projector implementation
   are independent facts.  Grid tokens cannot prove a patch merger; position
   cannot create fusion; a declared lane cannot prove participation.
5. Numeric model geometry is retained only where code proves the owning use
   (projector widths, repeat counts, attention operands).  Patch/image/tiling
   declarations remain withheld rather than being restored from config after
   an operation-kind-only proof.
6. Renderer vocabulary maps canonical facts to labels only.  It may neither
   calculate missing facts nor promote one fact into another mechanism.

### Exact remaining acceptance sequence

1. Run the 29-witness preservation comparison without writing artifacts and
   publish every changed surface/view.
2. Inspect representative Qwen2-VL, Mllama, Qwen2-Audio, Gemma multimodal,
   MusicGen, placeholder-replace, prefix-concat, source-missing and
   declared-unused outputs; classify each delta as intended honesty or defect.
3. Stop for Soumil's explicit artifact/re-bless decision.  Do not modify the
   manifest or galleries before that decision.
4. Rebuild only approved manifest/gallery artifacts, explicitly stage the U9
   files (never `git add -A`), commit, and run the parallel committed-tree
   coordinator with a serial exhaustive boundary lane.
5. Mark U9 DONE and push only on zero unexplained findings, zero failures and
   identical source/artifact fingerprints.

## 8. U9-H pre-bless artifact audit (2026-08-17)

No manifest, baseline or gallery was written during this audit.  The comparison
ran against the old 29-witness contract after the final producer corrections.

### Gate results before the artifact ruling

- exact U9 mechanism/projection matrix: **92 passed**;
- reader-inventory check: current;
- U2 authority/identity/structural-debt matrix: **81 passed**;
- preservation harness infrastructure/poisons: **23 passed**;
- old-contract witness comparison: **29 expected failures**, because every
  witness's evidence ledger records the new owner-qualified readers;
- every Sable report remains mechanically passing.  The four Sable byte deltas
  are shrinking findings, never newly excused or newly failing claims.

### Exact structural blast radius

Only four of the 29 witnesses change a structural surface:

1. **Qwen2-VL** — the visual frontend is now the exact primary tensor route
   `reshape -> Conv3d -> reshape`; a positional side branch ending in `cat` is
   no longer flattened into that route.  The exact projector is
   `LayerNorm -> reshape -> Linear -> GELU -> Linear`.  Config-only patch/grid
   geometry, default MHA/dense/RoPE/cell claims and unbound input width are
   removed.  Its old standard-cell attention/FFN drills disappear because the
   current exact occurrence readers do not close that cell; the encoder stays
   present and honest-opaque.  Seven retained modality views change to show the
   exact operations, two fabricated cell drills are removed, and both outer
   architecture views plus the language attention/FFN views remain unchanged.
2. **MusicGen** — exact task-specific AutoModel dispatch resolves the embedded
   `T5EncoderModel`; exact source evidence projects the 12-layer, 12-head,
   64-head-dimension T5 attention stage, relative bias, final RMSNorm, and the
   real `Linear(768 -> 1024)` conditioning projector.  The prior class/config
   authored T5 identity and generic standard-cell attention drill are removed
   because the exact FFN/cell proof is incomplete.  The six existing outer
   views remain byte-identical, the two unsupported encoder/detail views are
   removed, and one exact projector-operation view is added.  The duplicated
   pipeline encoder stage now receives its layer count, hidden width and owner
   from the same tower projection rather than drifting from the encoder card.
3. **HunyuanVideo** — no view hash changes.  Internal secondary-refiner specs
   stop copying the denoiser's host width into each refiner attention/FFN
   occurrence without an occurrence-specific affine binding.  The visible
   diagrams remain byte-identical; only the over-strong internal claim and its
   derived fact strings are removed.
4. **Lumina Image 2** — the same host-width correction for its context/entry
   refiners.  No view hash changes; only the unsupported internal width claims
   are removed.

The shared tower projector initially added an unknown `num_kv_heads: null`, an
empty conditioning `embedding` object and an empty `post_ops` list on unrelated
surfaces.  Those three serialization-only deltas were removed before asking for
the artifact ruling; unknown is absence of a claim, not a new null-shaped claim.

### Evidence-only blast radius

- All 29 witnesses have ledger changes because U9 records exact component,
  stage, fusion, projector and operation-reader provenance.
- Flux removes one stale accessed-but-unprojected finding; Qwen-Image removes
  four; Qwen2-VL removes its two stale vision occurrences; MusicGen removes its
  stale conditioning-head occurrence.  These are discharged by exact source
  ownership or exact, owner-qualified U14 carry-forward debt—not bare-key
  ignores.
- The remaining 25 witnesses have no IR, expanded JSON, parameters, HTML/card
  or view-sequence delta beyond the evidence sidecar described above.

### Explicit carry-forward, not hidden completion

U9 proves the mechanism and exact owner, but raw modality `extras` still lack
the FactLedger/ProjectionReceipt representation used by fully migrated facts.
The exact Qwen2-VL vision operands and MusicGen conditioning operands therefore
remain owner/path-qualified `StructuralDebt(config_read)` rows assigned to
**U14**.  They may neither fabricate architecture nor be reported as silently
consumed.  U9 completion does not close that U14 representation migration.

### Required ruling

At the pre-bless stage Soumil had to explicitly approve or reject this finite
honesty delta. Only approval could authorize regenerating the 29
baselines/manifest and affected gallery artifacts. Section 9 records that
approval and rebuild; the committed-tree exhaustive gate is still required.

## 9. U9-H approved artifact transition (2026-08-18)

Soumil approved the finite §8 honesty delta and instructed the implementation
to proceed with complete checking. The guarded production path then completed
all 29 witnesses:

- every Sable report was mechanically green with a present source oracle;
- every report produced one PNG per occurrence-exact distinct view;
- every frozen config reproduced the same view signature offline;
- every fixture retained its stable corpus identity;
- the project bless path preserved all human-review sidecars;
- the canonical preservation baseline and 29-witness expected manifest were
  regenerated independently after the bless pass.

The resulting gallery delta is exactly the approved blast radius: MusicGen
replaces the unsupported conditioning-encoder drills with the exact conditioning
projector view, and Qwen2-VL replaces config/default-authored visual internals
with the exact frontend/projector routes while removing its two unsupported
cell drills. No other witness gallery changes.

One pre-commit gate caught stale test authority: two synthetic
`AutoModel.from_config` positives still expected a bare framework call plus
`component_architectures` to select the child class. They now carry the same
official Transformers lazy-registry source and exact config-class key required
by production. The lookalike/unregistered negatives remain negative; this
strengthens the test rather than restoring the removed shortcut.

Working-tree receipts after that correction:

- U9 semantic/consumer matrix: **215 passed**;
- U2 authority/identity/config/quarantine matrix: **90 passed**;
- preservation matrix: **52 passed**, all 29 witnesses at zero drift.

These are pre-commit receipts. U9 remains open until the explicitly staged
commit passes the detached-worktree exhaustive coordinator with identical
source and external-artifact fingerprints.
