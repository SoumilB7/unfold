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
| U9-C2 | READY FOR COMMIT RECEIPT | Exact fusion-operand producer lineage, occurrence-qualified width operands and guarded-constructor provenance. The counterexample matrix covers PaliGemma, LLaVA, Qwen2-VL, Mistral-3, Gemma-4 and Mllama; shape/device/index metadata and helper calls cannot launder sibling producers. |
| U9-C | ACTIVE | The exact reader is intentionally not a production authority yet. Idefics2's repeated Perceiver connector is completed in U9-D; U9-F/G then atomically cut parser + conformance to the typed result, consume its config dependencies, add facts/receipts and delete `_legacy_projector_evidence`. |
| U9-D..H | PENDING | Begin only after U9-C's committed-tree receipt |
