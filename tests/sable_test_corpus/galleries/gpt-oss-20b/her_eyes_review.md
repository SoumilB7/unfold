# Her Eyes â gpt-oss-20b
8 images reviewed Â· LOVE 2 Â· FINE 6 Â· DISLIKE 0 Â· APPROVE 7 / SUGGEST 1

## Every view
| image | delight | verdict | her sentence |
|---|---|---|---|
| 00__architecture.png | LOVE | APPROVE | A textbook pre-norm spine â the Ã12 frame, twin residual elbows, and the "SW Â·" prefix tell the whole layer story without one wasted pixel. |
| 01__architecture_v1.png | FINE | APPROVE | Honest and clean, but it is 00's near-twin whose only difference is a missing two-letter prefix, so on its own it feels like dÃ©jÃ  vu rather than new information. |
| 02__attn.png | LOVE | APPROVE | The format at its peak â the sink column sits naturally on the spine, the KV-sharing card earns its corner, and the little token strip with "local window 128" explains SWA better than a paragraph could. |
| 03__ffn.png | FINE | APPROVE | The fan-out/fan-in symmetry reads instantly; only the jump from "Expert 1" to "Expert k" asks the reader to supply the ellipsis themselves. |
| 04__attn__1.png | FINE | APPROVE | Identical to 02 minus the window strip, which is exactly the right way to say "full attention" â correct, if unavoidably repetitive as a standalone page. |
| 05__router.png | FINE | APPROVE | Two boxes and two grey port labels is all a router is, and the drill has the confidence not to decorate that fact. |
| 06__expert_1.png | FINE | SUGGEST | The fused "Linear (gate + up) â Split" story is lovely, but the Ã's up-operand edge ducks behind the SiLU box â route that vertical clear of SiLU's left edge (or chip it "up") so both operands are traceable at a glance. |
| 07__architecture__1.png | FINE | APPROVE | The alternating stripe makes the SWA/full interleave visible in one sweep, though the twin legend lines are dense enough that "alternating" has to be decoded rather than seen. |

## What she suggests (she cannot edit â only point)
1. **06__expert_1.png** â pull the up-operand vertical left of the SiLU box (or add a small "up" chip on it) so the Ã visibly receives two distinct operands instead of one arrow emerging from behind SiLU.

## Her answers
- Prettiest this format can look? Very nearly â 02__attn is this format's best self (mechanism, cache economics, and window semantics in a single glance); what keeps the folder from perfection is not any one view but the two pairs of near-twins (00/01, 02/04) diluting the reveal.
- Bundle into one (drills preserved): merge 00 and 01 into one architecture view with the layer-strip (07) acting as the SWA/full selector, and let 02/04 share one attention canvas where a variant chip toggles the window strip on and off â every op stays drawn, nothing is hidden, half the pages disappear.
- Would a newcomer get it? Yes for the spine, the MoE fan-out, and the router; the single spot needing a hint is "Append sink column," where a newcomer won't guess the column is learned â a three-word annotation ("learned sink logits") would seal it without breaking the one-box honesty.
- Where the journey should end: at 06__expert_1 â token â layer â MoE â router â one expert's fused gate+up arithmetic is the natural floor of the model; 07's stripe is a map for orientation, not a destination.
