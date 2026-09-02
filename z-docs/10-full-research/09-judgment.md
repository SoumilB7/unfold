# Judgment — after the whole thing has been read

Issued 2026-09-02 on the evidence in `00`–`07` and `11`, with every earlier
claim adjudicated in `08-findings-register.md`. The goal it is judged against
(reconciled in `00-goal-and-intent.md` §8): a source-grounded explainer that
renders the **complete** structure of any supported model (tier 1), with
mechanism internals **honest** (tier 2) and values **exact** (tier 3), as an
interactive diagram a person can learn from — never guessing, never silently
omitting.

## 1. What the project is, in one paragraph

An evidence engine of genuine quality — the only tool in its landscape that
binds diagram nodes to config keys and shows "could not resolve" inside the
picture (`07` landscape) — wrapped in a product that has not shipped in eight
weeks, whose public face still describes the May design, whose learner half
was built in three abandoned Next.js labs and never reconnected, and whose
diffusion side renders its two most recognisable models as empty boxes with a
header reading "LAYERS 0". The honesty doctrine is real and enforced at the
reader layer; it is *not* enforced at the consumer layer, where unknown still
turns into a norm box and a label string still chooses a modality. The
campaign's soundness gates work; its recall is measured by one narrow check;
and nothing in the system can see what the parser did not see, because every
oracle is the parser's own AST.

## 2. The verdict, tier by tier

| tier | promise | state | verdict |
|---|---|---|---|
| 1 structure (must be complete) | components, stacks, ×N, block sequence, wiring, drills | text: complete; multimodal towers: partly opaque; diffusion: 33 % of provable facts, two flagships 0 % (`E` A, `04`) | **failing the promise where it matters most** |
| 2 mechanism (must be honest) | proven or visibly unknown | readers honest; one wrong `code_proven` fact (`B7`); consumers convert unknown→known in three places and sniff labels (`B16`); provable negatives drawn as unknown (`B9`); silent absences (`B6`, `B13`) | **honest at the core, not at the edges** |
| 3 values (must be exact) | widths, θ, counts | correct where drawn; DeepSeek params off by ~1 % and ~11 % active; PixArt none | **mostly** |
| the person | can learn from it | 4 pictures / 2 floors / prose leaves for Llama; no journey, no *why*, no param counts on blocks (`04` §2); labs that had journeys are dead (`11`) | **not built** |

## 3. The five root causes, ranked by what they cost

1. **The bundle boundary** (A1–A5) — one decision in U3, never revisited,
   compounded by U10 building on it. Cost: the diffusion tier-1 collapse and
   the encoder-drill collapse.
2. **No independent oracle** (B1, D5) — every check reads the parser's own
   AST; the one recall direction is narrow; the visual layer is self-marked.
   Cost: four consecutive blesses approved losses.
3. **Doctrine enforced at the reader and waived at the consumer** (B16, C2)
   — the firewall stops backward *imports*, not local reconstruction. Cost:
   the honesty guarantee is not actually total.
4. **Process without release** (D8, D1–D3) — 254 commits, no release, 111/111
   empty August commit bodies, docs planes unversioned and stale. Cost: the
   user has seen none of it; nobody can reproduce the receipts.
5. **The learner was never specified** (E1, E7, `11`) — the product served is
   the auditor; the product sold and once prototyped is the learner. Cost:
   the thing the author actually wants does not exist yet.

The latency complaint (A7) is **withdrawn as a root cause**: it is a
one-site quadratic bug.

## 4. What is right and must be kept

The reader-layer honesty law and its poison culture; the text-decoder
readers (every spot-checked geometry correct); the receipt discipline as a
*tool*; occurrence-exact ownership as an *idea* (even if its physics changes);
the Sable gallery as the perception substrate; U7 as the model unit; the
Gemma lab's "go with the config" seed; the three-tier framing.

## 5. Grades, re-issued

| subject | grade | one line |
|---|---|---|
| reader layer (evidence/) | A− | honest, general, verified — and oversized |
| consumer layer (adapters/blocks, renderers) | C | hand-authored prose, three unknown→known conversions, label sniffing, phantom boxes |
| verification | B− | 42 checks, one recall direction, no independent oracle, self-marked visual |
| product surface (README, Space, examples, npm) | D | eight weeks unshipped, stale and partly wrong examples, Space hides errors, JS port dead |
| plan | B− | right gates for soundness; no user, no recall, no release |
| method | B | arbiter model correct; the witnesses were never subpoenaed |
| **project against its goal** | **C+** | the hard engine is done and the visible product regressed |

## 6. The one sentence

The engine can already tell the truth; it cannot yet see everything, it does
not yet notice what it forgot, and nobody has been shown it since July.
