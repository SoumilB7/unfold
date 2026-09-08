# Her Eyes — llama-7b

```
   _______
  /  -O-  \
  \       /
   ¯¯¯¯¯¯¯
```

4 images reviewed · LOVE 2 · FINE 1 · DISLIKE 1 · APPROVE 2 / SUGGEST 2

## Every view

| image | delight | verdict | her sentence |
| --- | --- | --- | --- |
| 00__architecture.png | LOVE | APPROVE | One warm spine, residual loops with real air, the ×32 badge breathing in its corner — this is what a transformer should feel like the first time you see one. |
| 01__attn.png | FINE | SUGGEST | The Q/K/V triptych rhymes beautifully and V's long rail reads clean, but the tiny ⌃⊥ badges on K and V are confetti to a newcomer — meaning-bearing marks deserve to be legible. |
| 02__ffn.png | LOVE | APPROVE | The SwiGLU diamond is genuinely pretty — symmetric split, the gate lane through SiLU, × at the crown; I would hang this one up. |
| 03__architecture__1.png | DISLIKE | SUGGEST | A flat brick of 32 identical cells in a different, plain UI font — it speaks a second design language the rest of the gallery never agreed to. |

## What she suggests (she cannot edit — only point)

1. **03__architecture__1.png** — Unify the typography with the diagram family (same hand-drawn face, same ink), and when a model has exactly ONE layer type, collapse the strip to a single caption chip — "all 32 layers identical: MHA + Dense" — instead of a full panel of indistinguishable bricks. Save the brick wall for models where the pattern IS the story (alternating types).
2. **01__attn.png** — Give the KV-cache badges a legible affordance: either a one-line whisper legend inside the panel ("⌃⊥ = written to / read from the KV cache") or grow them into small labeled tags on the box edge. A mark a newcomer cannot decode is decoration, not information.

## Her answers

- **Prettiest this format can look?** Very close. The architecture and FFN views are at ceiling. The gaps: the strip panel's font clash (two design languages in one gallery), and micro-glyphs that assume prior knowledge. Fix those two and the llama gallery is the reference look for every dense LLM.
- **Bundle into one (drills preserved):** Nothing needs bundling — the norm → attention → norm → FFN rhythm is already calm. One weight suggestion instead of a bundle: let the two quiet RMSNorm boxes sit visually lighter (thinner stroke or slightly smaller) so attention and FFN carry the eye; placement is their information, not presence.
- **Would a newcomer get it?** The architecture view — yes, immediately, and that is rare and precious. The attention drill — mostly; `sqrt(dim)` and the cache badges assume comfort the picture could grant instead.
- **Where the journey should end:** architecture → attention → FFN is the perfect three-step arc for a dense LLM; it ends exactly where understanding completes (the SwiGLU leaf). The layer strip belongs as a coda, not a stop on the journey. Depth pacing here is RIGHT — do not add more floors.
