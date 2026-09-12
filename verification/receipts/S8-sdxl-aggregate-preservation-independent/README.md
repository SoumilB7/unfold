# SDXL legacy aggregate preservation — independent closure

**ACCEPT the finite preservation/limitation mapping. No actual lost old proved drawing was found in the 53 unique legacy UNet subtree cards.** This closes the aggregate-geometry review gap identified in S8-sdxl-differential-independent; it does not approve output blessing or close config-accounting findings.

Reviewed the exact final c38b008 ordinary/legacy artifacts. `ledger.json` retains every old card's complete assertion payload (title, description, facts, detail, view), followed by exact source-author dispositions and target occurrence/card/fact addresses. It covers 101 semantic entries: 54 carried as proved, 10 carried with an explicitly bounded limitation, 37 demoted because the old author did not establish that stronger assertion. These are review entries, not a new fact layer or a claim that 101 independent mechanisms were proved.

## What the old positive source readers actually established

| Old source author | Supported information | Exact new preservation |
| --- | --- | --- |
| parser.py:72–94 → patterns.py:624 onward | Mid construction presence | `instance_mid_block`, existence fact plus root stage relation/conditional call-port projection; visible in actual denoiser overview |
| parser.py:112–125 → patterns.py:532–582 | Reachable attention+FFN construction (returned transformer2d) | Exact Transformer2DModel instances, their transformer_blocks children, and each attn1/attn2/ff occurrence; actual containment cards |
| parser.py:55–71 → patterns.py:425–492 | Anchored FFN activation declaration `geglu` | All **70** exact FFNs have qualified fused gate/up + GELU + multiplication + output projection; all **70 actual FFN SVGs have seven arrows** |

The construction traversal does **not** read/prove self→cross→FFN forward wiring, full masks, latent-query lineage or text-K/V provenance. The old adapter's statement that transformer2d proves that entire placement must not enlarge the reader's actual result. Mid presence likewise does not prove the renderer's ResNet–attention–ResNet execution sandwich. None of those stronger claims was deleted from an old connection proof.

## Stage-by-stage quantities and visible addresses

| Old stage | Exact new card | ResNet children | Transformer wrappers × nested depth | Old width retained in actual conv2 output-weight extent |
| --- | --- | ---: | --- | ---: |
| unet_down_0 | instance_down_blocks__0 | 2 | none | 320 |
| unet_down_1 | instance_down_blocks__1 | 2 | 2 × 2 | 640 |
| unet_down_2 | instance_down_blocks__2 | 2 | 2 × 10 | 1280 |
| unet_mid | instance_mid_block | 2 | 1 × 10 | 1280 |
| unet_up_0 | instance_up_blocks__0 | 3 | 3 × 10 | 1280 |
| unet_up_1 | instance_up_blocks__1 | 3 | 3 × 2 | 640 |
| unet_up_2 | instance_up_blocks__2 | 3 | none | 320 |

All seven stage nodes are present in the actual denoiser SVG. Each count is checked against the qualified constructed population, not inferred from the old layer count. Each width is checked against the exact conv2 parameter shape and displayed child card. Containment/card count and raw shape dimensions are the retained claims; a shape does not establish activation tensor shape, execution order or head splitting.

## Specific retained claims and named demotions

**Residual cells and SiLU.** All 17 exact cells retain constructed norm/activation/conv/time projection/optional shortcut objects. Both norms have exact GroupNorm applied-function facts; every nonlinearity has exact SiLU; every conv1/conv2 is exact Conv2d with 3×3 weight extents. Existing cell connection claims retain the five positive local edges, while conditional conditioning/return arithmetic expose unresolved lineage and scale operands. The old complete nine-arrow cell was installed literally by unet.py:290–337 and block_views/unet.py:669–721. Those functions accept labels/geometry; they never receive a selected forward proof. Complete bypass/time-add lineage, generic padding/stride1 prose, and division by a presumed scale therefore become named limitations. This loses an apparently complete schematic, not a proved complete cell drawing.

**Attention and transformer ordering.** Every previously source-supported attention/FFN construction remains visible. Old num_heads/head_dim arose from unet.py:115–146's config convention (`attention_head_dim` treated as head count when num_attention_heads is missing); full mask and self/cross flags were assigned in :374–380, and latent-query/text-KV prose in :403–450. The self→cross→FFN sequence came from :636–660 and the literal transformer renderer. New occurrence/projection shapes remain; 60 exact formal→context-port claims remain. Query role, mask, head splitting, stronger K/V semantics and unproved ordering stay limited. Mid attention is not silently upgraded where the context reader has no positive result.

**Spatial operations.** The two actual downsampler cards retain qualified stride-2 Conv2d claims, with exact convolution children/shapes. Stride 2 is the precise retained operation; it does not promise exact half-sized outputs for every possible odd input. The two upsampler cards retain qualified interpolate operations and constructed convolution children. Their operand is None and they visibly say resize direction is under investigation. Old nearest-neighbour/2×/then-convolution wording came directly from unet.py:735–743; per-stage sample position and global downscale `2**(n-1)` came from :80–113/:172. No spatial reader supplied the universal old composition claim, so it is demoted rather than imported.

**Bookends and skips.** Input4→320 and output320→4 are retained through exact `[320,4,3,3]` and `[4,320,3,3]` weight cards. Output norm/SiLU/conv components and conditional root call-port bindings remain visible. The old predicted-noise meaning is not proved by shape. Exact up-stage concat→child connections and source skip-bank relations remain; matching-stage pairing and all-path execution are not inferred from a symmetrical U layout. Actual optional helpers and unresolved operand lineage remain in the new route drills.

**Text assembly.** Old `unet_text_cond`, `text_concat_op` and shared cross-state prose claimed 768+1280 feature concatenation, same text K/V for every cross stage and no projection/mixing. unet.py:849–940 generated those claims from component widths/counts/config, without a pipeline source connection reader. Encoder components/internals remain; the denoiser only keeps its narrower proved external-context ports. The independent named-removal receipt already established the same boundary for outer encoder arrows and the default Image label.

## Reproduction and review limits

    python3 verification/receipts/S8-sdxl-aggregate-preservation-independent/replay.py

The replay asserts exact coverage of all 53 old unique cards, current actual target cards, qualified claim kinds, seven visible overview stages, every stage's actual child count/depth/width, all 70 routed FFNs, and exact primitive/shape witnesses for the 17 residual cells. Old-author source SHA256 pins are in summary.json. No source/model mutation, new grammar capability, full model, pytest or output blessing.

The visible tradeoff is substantial: the new product preserves supported operations and quantities while showing less certainty about whole-stage/residual/attention flow. This receipt accepts that named honesty cost because the removed certainty came from identified template/config assumptions. It does not authorize hiding newly proved drawings, and it does not assert universal execution closure.
