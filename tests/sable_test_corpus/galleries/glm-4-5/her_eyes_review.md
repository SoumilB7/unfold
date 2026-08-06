# Her Eyes — GLM-4.5

```
   _______
  /  -O-  \
  \       /
   ¯¯¯¯¯¯¯
```
11 images reviewed · LOVE 8 · FINE 3 · DISLIKE 0 · APPROVE 9 / SUGGEST 2

## Every view
| image | delight | verdict | her sentence |
|-------|---------|---------|--------------|
| 00__architecture | LOVE | APPROVE | A calm, tall spine — token enters, token leaves, the two residual ⊕ breathe, and the stacked "MTP head · +1 future token" is a lovely closing whisper. |
| 01__architecture_v1 | FINE | SUGGEST | Beautiful, but almost pixel-for-pixel the previous spine with "Feed-Forward → MoE" and "×3 → ×89" swapped; as a pair they read as a spot-the-difference puzzle. |
| 02__attn | LOVE | APPROVE | The Q/K/V tree with QK-Norm and the honest "rot 64 · pass 64 dims" partial-RoPE detail; the side KV-sharing card (Q0–Q11 → KV0 … cache 12× smaller) tells the whole GQA story quietly. |
| 03__ffn | LOVE | APPROVE | Textbook SwiGLU — two lanes, SiLU, a gate ×, a down-projection. Nothing to add, nothing to remove. |
| 04__mtp | LOVE | APPROVE | The prettiest view in the set — two normed lanes braid at ‖ and rise through Linear 2d→d to the shared head; "logits → token t+k+1" makes the future-token idea instantly legible. |
| 05__ffn__1 | LOVE | SUGGEST | The sparse fan-out finally *looks* sparse and "top-8 of 160 · +1 shared (always on)" is a perfect caption; only quibble — it is wide and the ⊕ inputs crowd with little headroom at the top. |
| 06__mtp_block | LOVE | APPROVE | A whole transformer layer folded neatly into the MTP — attention then MoE, two residual ⊕ — reads like a mini-architecture from eh_proj to the shared head. |
| 07__router | LOVE | APPROVE | The new **sigmoid** now sits between Linear (Gate) and Top-k, so the gate logits visibly become *scores* before selection — the pipeline finally reads as a real scoring receipt: Linear → squash → select → renormalize → ×2.5. |
| 08__expert_1 | FINE | APPROVE | Correct and clean — fused "Linear (gate + up) → Split" then SiLU × up → down → weighted sum; a touch generic, but that is the nature of one expert. |
| 09__g_topk | FINE | APPROVE | Minimal and honest — "expert scores → Top-k experts → Gather weights → selected weights"; and those "expert scores" now have an on-screen origin (07's sigmoid), so the two views join seamlessly. |
| 10__architecture__1 | LOVE | APPROVE | The single best "whole model at a glance" — a 92-cell strip that says "3× Dense (L0–L2), 89× MoE (L3–L91), all GQA (QK-Norm, +bias, Partial RoPE)" in one calm bar. The variety the two spines lack. |

## What she suggests (she cannot edit — only point)
1. **01__architecture_v1** — two full-height spines that differ only in one box (Feed-Forward ↔ MoE) and a repeat count (×3 ↔ ×89) read as redundancy. Present the dense↔MoE variants as *one* spine a viewer toggles; the layer strip (10) already carries the "2 types" story beautifully, so lean on it and let the gallery lead with variety (attention, MoE, MTP).
2. **05__ffn__1** — the widest view; the representative experts + the shared expert sit high and the ⊕ merge at the very top has little breathing room, with the expert elbows crowding into it. Drop the expert row a touch / tighten the fan so the merge can breathe.

*(soft, low-priority)* **07__router** — now complete and continuous, but the composition is right-anchored: the pipeline hugs the right third while "learned bias" floats far left on a long horizontal connector, leaving a big empty middle. Pull the bias nearer / center the pipeline so the frame balances.

## Her answers
- **Prettiest this format can look?** Very close already. The MTP braid (04), the KV-sharing attention (02), and the layer strip (10) are the ceiling of the vertical-tree format, and the router (07) now reads as a genuine scoring pipeline rather than a bare stack. The only remaining lift is calming the two near-identical spines into one and balancing the router frame.
- **Bundle into one (drills preserved):** the dense and MoE architecture spines (00/01) into a single toggled spine; nothing else — every drill earns its own picture.
- **Would a newcomer get it?** Yes — entry/exit is obvious, GQA and SwiGLU are canonical, the KV-sharing card and "+1 future token" whisper carry the two GLM-specific ideas, and the layer strip orients them before they drill. And the router now flows: the sigmoid squashes the gate logits into *scores*, Top-k selects on those scores, and the Top-k drill (09) opens on "expert scores" a newcomer can trace straight back to that sigmoid — one continuous story, no more mystery input. The one lingering first-glance stumble is the bare acronym "MTP head" (a one-word card gloss would carry them).
- **Where the journey should end:** a newcomer understands GLM-4.5 at the layer strip (10) + attention (02) + MoE fan-out (05) + router (07) + MTP (04). The expert internal (08), the Top-k drill (09), and the mtp_block (06) are for the curious; the natural stop is "grouped-query attention with QK-norm + partial RoPE, a sparse MoE with an always-on shared expert, and it predicts one extra future token."
