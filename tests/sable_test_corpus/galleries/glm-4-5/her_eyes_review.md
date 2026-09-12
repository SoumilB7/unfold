# Her Eyes — glm-4-5

```
   _______
  /  -O-  \
  \       /
   ¯¯¯¯¯¯¯
```

9 images reviewed · LOVE 3 · FINE 5 · DISLIKE 1 · APPROVE 8 / SUGGEST 1

Judgments below come from individually opened current PNGs. Suggestions are visual review notes, not architecture changes or a fresh Sable run.

## Every view

| image | delight | verdict | her sentence |
|---|---|---|---|
| 00__architecture.png | FINE | APPROVE | The three-layer dense tower reads cleanly with clear residual margins. |
| 01__architecture_v1.png | FINE | APPROVE | The MoE tower keeps its larger label contained and both return paths distinct. |
| 02__attn.png | LOVE | APPROVE | Q and K normalization remain visible while the cache branches clear the score card. |
| 03__ffn.png | LOVE | APPROVE | The two branches form a balanced FFN with an obvious multiplication join. |
| 04__ffn__1.png | DISLIKE | SUGGEST | The shared-expert input rail disappears behind the wide Router box; route it outside that box without dropping the branch. |
| 05__router.png | FINE | APPROVE | The stored-bias side card joins Top-k clearly and the scale label has room. |
| 06__expert_1.png | LOVE | APPROVE | The split has a clearly visible unactivated branch returning into the multiplier. |
| 07__g_topk.png | FINE | APPROVE | The two-step selection drill is concise and readable. |
| 08__architecture__1.png | FINE | APPROVE | The small dense prefix and long MoE remainder are legible in the layer strip. |

## What she suggests (she cannot edit — only point)

1. Give the shared-expert input its own unobstructed route around the Router card.

## Her answers

- **Prettiest this format can look?** The expert drill and attention inset are strong; the MoE overview still has a visible rail obstruction.
- **Bundle into one (drills preserved):** Keep dense and MoE summaries separate, with router and expert drills preserved.
- **Would a newcomer get it?** Mostly; the shared branch obstruction makes the MoE overview harder than its own drill.
- **Where the journey should end:** Stop at the selected expert and router operations rather than expanding every repeated expert.
