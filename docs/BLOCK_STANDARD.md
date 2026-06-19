# The Block & Diagram-Part Standard

*The single rulebook for what becomes a block, what becomes wiring, and what becomes a
footnote — for **every** transformer and diffusion model `model-unfolder` draws.*

This generalizes the block-worthiness paradigm first proven on DiffusionGemma (CLAUDE.md
Gate C) into a standard that applies family-by-family. It is the reference the optimization
sweep (`docs/MODULARIZATION.md`) is measured against, and the contract a new architecture's
**Sable** pass must satisfy.

> **The law.** *A block is a thing a researcher would draw on a whiteboard and give a name.
> Everything else is either an arrow (wiring) or a footnote (a property).*

The disease this standard cures is **over-blocking**: when a layer renders nine boxes, the
two that carry the architecture (attention, FFN) drown in plumbing, and the diagram *hides*
information instead of surfacing it. Every candidate element sorts into exactly one of three
tiers, and the choice is mechanical once you apply the earning test.

---

## 1. The three tiers

| Tier | Name | Rendered as | Clickable? | Card? | Mechanism |
|------|------|-------------|------------|-------|-----------|
| **1** | **BLOCK** | solid box | **yes — always** | yes | default (a normal block dict / `Node`) |
| **2** | **CONNECTOR** | glyph on the join/arrow (`⊕ × split concat`) | no | no | `static: True` / `Node(static=True)` |
| **3** | **ANNOTATION** | chip, label suffix, or frame caption | n/a | n/a | `facts=[…]`, label text, `render.layer_annotations`, `render.repeat_note` |

### Tier 1 — BLOCK
Substantial **named computation** *or* a salient architectural module a researcher points at:
attention, FFN / MoE, embedding, LM head, a conv stage, a VAE, a parallel-FFN branch,
self-conditioning, an MTP head, a vision/audio tower. A Tier-1 block is a **clickable box
with a card**, optionally drilling into a sub-view. **There is never an unclickable Tier-1
block** — if it can't be clicked, it isn't Tier 1.

### Tier 2 — CONNECTOR
Pure wiring / topology: the residual `⊕`, a gate `×`, a fan-out split, a `concat`. It is
drawn as a **glyph on the join** and carries no card. The loop-back arrow already says
"residual"; a `⊕` is not a module. Set `static: True` on the block dict (architecture view +
cards) or construct the graph node with `Node(static=True)` (drill-down views).

### Tier 3 — ANNOTATION
A **property** of a block or of the layer, not a computation with its own identity:
QK-norm, a sliding-window width, a RoPE θ, a learned per-layer scalar, norm *placement*,
a clamped-activation limit. It rides as a **chip** on the owning block's card (`facts`), a
**label suffix**, or — when it characterizes the whole repeated layer — a **frame caption**
(`render.layer_annotations`) or a note under the `× N` badge (`render.repeat_note`). **Never
a box.**

### The earning test (apply to every candidate, in order)
1. **Is it where real or named computation lives, or a salient architectural *choice*?**
   → Tier 1 (box + card).
2. **Is it pure wiring — an add, a multiply, a split, a concat?** → Tier 2 (`static` glyph).
3. **Is it a single scalar or a property of something else?** → Tier 3 (chip / caption).

**Norms are the one principled borderline.** They stay quiet Tier-1 boxes because the field
draws them to show *placement* (pre / post / sandwich). The *placement* is the information,
not the norm's existence — so the box is allowed but stays visually quiet, and the placement
is what the prose and ordering must convey.

### The two complementary filters
Tier assignment sits between two gates that pull in opposite directions, and a block must
pass **both**:

- **Floor — code signature (CLAUDE.md Gate B.7).** The operation must be visible in the HF
  modeling source as a real signature: a call, a tensor op, a `for` loop, a `+` /
  `torch.cat` / `.split`, or a config flag that gates it. No signature → the element does
  not exist; do not fabricate it.
- **Ceiling — block-worthiness (this document).** Existing in the code is *necessary, not
  sufficient*. `hidden * layer_scalar` has a signature and is still Tier 3, not a box.

---

## 2. The five diagram parts

An architecture is made of exactly five kinds of structural thing. Every Sable pass conforms
each, **in both directions**, against the HF `forward()` (CLAUDE.md Gate A.6). This is the
vocabulary the tiers are expressed in:

| # | Part | What it is | Graph primitive | Tier |
|---|------|-----------|-----------------|------|
| 1 | **arrows** | dataflow edges (with ports) | `Edge(kind="flow")`, chain `flow=[…]` | — (wiring) |
| 2 | **blocks** | ops / modules | `Node`, block dict | 1 |
| 3 | **repeating regions** | loops (`for` / `nn.ModuleList`) + loop-carried back-edges | `Group(members, repeat)` | frame |
| 4 | **connectors** | fan-in: `+` `×` `concat` | `Node(kind="residual_add"/"gate_mul"/"concat", static=True)`, `Edge(kind="residual")` | 2 |
| 5 | **splitters** | fan-out: one value used in several places | `Parallel(src, dst, lanes)`, `Lane`, `SideInput` | wiring |

Two-way conformance means: **code → structure** (every op, connection, loop, merge, and
fan-out the `forward()` performs is present and wired to the right port) **and structure →
code** (every arrow, block, region, connector, and split the diagram shows is verifiable in
the source — nothing fabricated). Anything that fails either direction is red-flagged, not
quietly drawn.

---

## 3. The mechanism — every switch, with code pointers

The paradigm is **opt-in per block** and requires **no engine change** to roll a family on:
you tag blocks, the engine honours the tags. Each switch below is the single place a tier
decision is expressed.

### 3.1 Tier-2 connector — `static`
- **Block dict:** `"static": True`. The architecture view computes
  `clickable = not block.get("static")` ([views.py:203](../model_unfolder/renderers/html/views.py#L203))
  and the card builder skips static blocks ([cards.py:39](../model_unfolder/renderers/html/cards.py#L39),
  [cards.py:117](../model_unfolder/renderers/html/cards.py#L117)).
- **Graph node:** `Node(static=True)` ([graph.py:87](../model_unfolder/renderers/html/graph.py#L87)).
  The engine emits `clickable=not node.static`
  ([graph_engine.py:200](../model_unfolder/renderers/html/graph_engine.py#L200)).
- A non-clickable node never gets a `data-id`, so `validate_click_coupling` cannot fault it.
- **Schema:** `static: bool` is a blessed `Block` key
  ([block_schema.py](../model_unfolder/block_schema.py)).

### 3.2 Inline parallel branch — `branch_side` + `feeds`
For two computations that fan out from one source and converge at one merge, drawn
**side-by-side off the central column** (DiffusionGemma's dense MLP ∥ MoE):
- `"branch_side": "left" | "right"` marks the block as a branch (not in the chain);
  `"feeds": "<merge_id>"` names the `static` `residual_add` they converge into. The view
  reserves a branch row and routes the fan-out + merge
  ([views.py:241](../model_unfolder/renderers/html/views.py#L241), `_draw_branch_split`).
- In a drill-down, the same shape is a `Parallel(src, dst, lanes)`
  ([graph.py:168](../model_unfolder/renderers/html/graph.py#L168)).

### 3.3 Side rail — `lane` + `tap_from` + `feeds` + `side_align`
For a pathway that runs *beside* the main column and rejoins it (parallel-residual FFN,
per-layer embedding, a cross-attention adapter). `lane` places it; `tap_from` says which
block it reads its input from; `feeds` says where it merges; `side_align` aligns it to its
tap. In drill-downs the lateral-input form is `SideInput(node, target, side)`
([graph.py:189](../model_unfolder/renderers/html/graph.py#L189)).

### 3.4 Residual connector — `residual_from`
A `residual_add` block with `"residual_from": "<earlier_block_id>"` draws the skip as a loop
from that earlier block into the `⊕` ([views.py:258](../model_unfolder/renderers/html/views.py#L258)).
Per this standard the `⊕` itself is **Tier 2** (`static: True`); the loop arrow carries the
"residual" meaning.

### 3.5 Tier-3 annotations — chips & captions
- **Chip:** add to the owning block's `facts=[…]` (numbers/specs only; never on the box).
- **Layer caption:** `extras["render"]["layer_annotations"]` (list[str]) — a property of the
  whole repeated layer, drawn top-left of the `× N` frame
  ([views.py](../model_unfolder/renderers/html/views.py)).
- **Repeat note:** `extras["render"]["repeat_note"]` (list[str]) — caption under the `× N`
  badge (e.g. DiffusionGemma's "shared by encoder (causal) & decoder (bidirectional)",
  [parser.py:457](../model_unfolder/adapters/transformer/parser.py#L457)).

### 3.6 Approved stages — the typing guard
Each adapter declares an **exhaustive** list of approved block stages in
`everchanging/<adapter>/typing.yaml`. A block tags itself (`diffusion_stage` /
`diffusion_part_kind` for diffusion; the transformer taxonomy is in
`transformer/typing.yaml`). A stage **not** on the list renders **pale / label-only** to flag
that its place in the diagram is *not decided yet* — a guardrail so a new adapter fact can't
silently become a first-class block. Today the **diffusion** renderer enforces pale-when-
unapproved ([block_schema.py:74-88](../model_unfolder/block_schema.py#L74)); the **transformer**
taxonomy is blessed and documented but the pale guard is not yet wired (see
`docs/MODULARIZATION.md`, the symmetric-guard item).

### 3.7 The graph primitives (drill-down views)
Every drill-down is a `Graph` of typed `Node`s laid out by the shared engine — no view
computes coordinates or draws a residual by hand
([graph.py](../model_unfolder/renderers/html/graph.py)):

`Node` (typed by `kind` → glyph in `KIND`) · `Edge` (`flow` | `residual`) ·
`Group` (the `× N` repeat frame) · `Parallel` (branch-and-merge / splitter) ·
`SideInput` (a lateral feed into one node) · `Lane` (one parallel mini-column).
Adding an architectural primitive = adding a `kind` to `KIND` and tagging nodes with it; the
engine and every view pick it up with no further wiring.

---

## 4. Tier tables — transformer (LLM) families

The **target** assignment for the recurring elements. Where today's code diverges (notably
the standard decoder layer still drawing residual adds as clickable Tier-1 boxes), the gap is
the optimization sweep's job — the table is the destination.

### 4.1 The decoder layer (Llama · Mistral · Qwen · Gemma · Phi · GPT-OSS · DeepSeek · GLM · Kimi · …)

| Element | Tier | How |
|---|---|---|
| Input / pre-attention norm | 1 (quiet box) | norm block; placement in prose |
| Attention (MHA/GQA/MQA/MLA) | 1 | `view:"attention"`, drills to Q/K/V/RoPE/scores/cache |
| Residual add (post-attn) | **2** | `static` `residual_add`, `residual_from` |
| Pre-FFN norm | 1 (quiet box) | norm block |
| FFN / MoE | 1 | `view:"ffn"`/`"moe"`, drills to gate/up/act/down or router/experts |
| Residual add (post-FFN) | **2** | `static` `residual_add`, `residual_from` |
| Parallel-residual topology | — | `lane`+`tap_from`+`feeds` (one shared norm, combined add) |
| QK-norm | **3** | chip / label suffix on attention |
| Sliding-window / chunked mask | **3** | chip; window width is a fact |
| RoPE θ / NoPE / iRoPE | **3** | chip on attention |
| Attention/MLP bias flags | **3** | chip |
| Learned per-layer scalar | **3** | layer caption / chip (never a box) |
| Clamped-SwiGLU limit (`swiglu_limit`) | **3** | chip on the FFN |

### 4.2 MoE specifics (DeepSeek · Qwen3-MoE · Mixtral · GLM-MoE · Kimi · Phi-MoE · GPT-OSS)

| Element | Tier | How |
|---|---|---|
| Router / gate | 1 | clickable; drills to the routing pipeline (scoring → bias → group-limit → top-k → norm → scale) |
| Routed expert | 1 | `expert` block; drills to the expert FFN |
| Shared (always-on) expert | 1 | a parallel always-on lane merging at the routed-sum `⊕` |
| Weighted sum of experts | **2** | `static` `residual_add` |
| Scoring fn (sigmoid/softmax), group-limit (`n_group`/`topk_group`), `norm_topk_prob`, `routed_scaling_factor` | gate sub-flow + **3** chips | router drill-down steps; summarized as facts |

### 4.3 Attention internals

| Element | Tier | How |
|---|---|---|
| Q / K / V / O projections | 1 (inside the attention drill) | `linear` nodes |
| MLA Q-LoRA / KV-LoRA latent | 1 | `view:"mla_query_path"` / `"mla_kv_cache_path"` |
| RoPE application | 1 (quiet) or **3** | a `rope` node in the drill; θ as a chip |
| Scaled-dot-product scores + softmax | 1 (quiet `formula`) | `formula` node |
| KV cache read/write | 1 (quiet) | `cache` node with K/V ports |
| Sparse-attention indexer (DeepSeek-V3.2 DSA) | 1 | a small indexer sub-block that selects top-k keys |
| Splitter Q→K (one value, two uses) | 5 (splitter) | `Parallel` / tap dot — **not** a box |

### 4.4 Model bookends & side towers

| Element | Tier | How |
|---|---|---|
| Token embedding | 1 | embedding block |
| Final norm | 1 (quiet) | norm block |
| LM head | 1 | linear-output block |
| MTP head(s) | 1 | `view:"mtp_head"` |
| Per-layer embedding (Gemma 3n PLE) | 1 + side lane | `view:"per_layer_embedding"` |
| Vision / audio / video tower | 1 | `view:"*_encoder"` / `"*_path"` |
| Modality projector / fusion | 1 | `view:"multimodal_fusion"` |
| Tokenized input ids (a label, not a tensor op) | **3** | a port/caption, never a green I/O box |

### 4.5 Alternative token mixers — **out of scope**
SSM / Mamba and RWKV are explicitly **not** drawn (CLAUDE.md scope: LLMs + diffusion only).
The stages exist in the taxonomy for completeness; do not build views for them.

---

## 5. Tier tables — diffusion families (DiT · MMDiT · UNet · VAE)

### 5.1 Pipeline level

| Element | Tier | How |
|---|---|---|
| Noise / image / mask input | 1 | input block (a real conditioning source, not a dim label) |
| Text encoder(s) (CLIP/T5), pooled embed | 1 | `text_encoder` block |
| Timestep / guidance embedding | 1 | conditioning block |
| Denoiser (DiT/MMDiT/UNet) | 1 | drills to the backbone |
| Scheduler / sampler step | 1 | scheduler block; the `z_t → z_{t-1}` loop is a region |
| VAE encode / decode | 1 | `vae_decoder` / encoder views |
| Final image output | 1 | output block |
| CFG twin / guidance scale value | **3** | a note/chip (it's a sampling choice, not a module) |

### 5.2 Denoiser block (DiT / MMDiT)

| Element | Tier | How |
|---|---|---|
| Patchify / unpatchify | 1 | stage block |
| Positional embed (axial RoPE / learned) | 1 (quiet) or **3** | block or chip |
| Norm (LayerNorm / AdaLN) | 1 (quiet) | placement is the info |
| AdaLN modulation (shift/scale/gate from timestep) | **3** (the scale/shift/gate scalars) + 1 (the modulation block) | the modulation *block* is Tier 1; the per-op shift/scale/gate are not separate boxes |
| Self / joint attention | 1 | attention view (single- vs dual-stream is a `variant`) |
| Cross-attention to text (PixArt) | 1 | `cross_attention` |
| Feed-forward | 1 | ffn view |
| Residual add | **2** | `static` `residual_add` |
| Dual-stream (image ∥ text) MMDiT | — | two lanes / `branch_side`, joining at joint attention |

### 5.3 VAE / UNet internals

| Element | Tier | How |
|---|---|---|
| Conv in / out | 1 | conv stage |
| ResNet block | 1 | `unet_resnet` |
| Down / up / mid stage | 1 | `unet_stage` / `vae_decoder_block` |
| Spatial up/downsample | 1 (quiet) | stage block |
| Skip-connection concat (UNet) | **2** | `static` `concat` glyph — **not** a box (use the existing concat, never a "skip connection" block) |

### 5.4 DiT / MMDiT block — structural rules (the diffusion-native depth)

The denoiser block is where diffusion departs most from an LLM layer, and where the
diagram must be *structural*, not annotation-only (conformed against `FluxTransformerBlock`
/ `JointTransformerBlock` / `PixArtTransformerBlock`):

- **AdaLN modulation is a connector, not just a side note.** The timestep (`temb`,
  optionally + pooled text) produces per-block **shift · scale · gate**. These are **drawn
  connections**, not prose on a side block:
  - **gate** (`gate_msa`, `gate_mlp`) → a **Tier-2 `gate_mul` `×` glyph** on the attention
    output and the FFN output, *before* each residual `⊕` (`h = h + gate · sublayer(...)`).
  - **shift / scale** → modulate the pre-sublayer norm; drawn as the `adaln_cond` side rail
    feeding the norm (Tier-3 detail on the norm), with `×(1+scale)+shift` named in prose.
  - The `adaln_cond` block is the **source** of these gates (a splitter — one temb feeds
    every gate), tagged so its fan-out to the `×` glyphs reads as conditioning.
- **MM-DiT double-stream = two lanes + one joint attention.** Image tokens and text tokens
  are **two separate streams** (each its own norm · FFN · gated residuals) that meet **only
  at the joint attention** (one attention node fed by both, returning to both). Draw it as
  two columns sharing one `attention` node — **not** one stream with a "dual-stream" label.
- **Single-stream = parallel attn ∥ MLP → concat → gated proj_out.** Not a plain combined
  add: the attention and MLP run in parallel off one norm, their outputs **concat**, a
  single `proj_out` projects, then the AdaLN `×gate` and the residual `⊕`.
- **Cross-attention DiT** (PixArt / Hunyuan-DiT): a `cross_attention` sub-block to the
  encoded-text K/V (reuse the transformer `cross_attention` + `encoded text` side source),
  plus AdaLN-single (one shared modulation table).
- **Patchify / unpatchify** are Tier-1 stage blocks; the **split** of patches and the axial
  RoPE are wiring/Tier-3.
- **VAE ResNet** must show the **nonlinearity** (SiLU) node between norm and conv
  (`norm → SiLU → conv`), and the up/downsample *before* conv1 — matching `ResnetBlock2D`.

---

## 6. Rolling a family onto the standard (the Sable checklist)

Bringing any architecture or block onto the paradigm is **additive and mechanical** — tag,
don't rewrite:

1. **Render first.** `unfold(<model>).save('../<model>.html')`; work against the rendered
   output, never imagined code (CLAUDE.md Gate A/B Step 1).
2. **Decompose downward** into the five diagram parts (§2). List every block, arrow, region,
   connector, splitter — each from a declared config field or a code signature.
3. **Sort each candidate into a tier** with the earning test (§1). Default suspicion:
   a thing that looks like a box is usually a connector or annotation.
4. **Tag, don't fork:**
   - connectors → `static: True` (§3.1, §3.4)
   - inline parallels → `branch_side` + `feeds` (§3.2); side rails → `lane`/`tap_from` (§3.3)
   - properties/scalars → `facts` chips or `layer_annotations` (§3.5) — move them *off* the
     boxes
   - any genuinely new stage → bless it in `everchanging/<adapter>/typing.yaml` (§3.6)
5. **Conform both directions** against the HF `forward()` (§2 / Gate A.6). Red-flag anything
   the code does that the diagram omits, and anything the diagram shows that the code
   doesn't do.
6. **Verify on pixels + gates:** re-render, look at it; `validate_block_tree` and
   `validate_click_coupling` clean; `pytest -q` green.

A change is **not done** until every Gate-A/B question is answered in writing and the render
has been inspected.

---

## 7. Enforcement

| Guard | What it catches | Where |
|---|---|---|
| `Block` TypedDict + `KNOWN_BLOCK_KEYS` | typo'd / unblessed keys (now incl. `static`, `branch_side`) | [block_schema.py](../model_unfolder/block_schema.py) |
| `validate_block_tree` | missing/dup `id`, unknown key, unregistered `view`, malformed children | block_schema.py |
| `validate_click_coupling` | a clickable node whose click opens no card (view↔block-id drift) | block_schema.py |
| typing.yaml + pale guard | a new/mistyped stage silently becoming a solid block | everchanging + renderer |
| `test_block_schema.py` corpus | the above, over a topology corpus that now includes the **block-paradigm** model | [tests/test_block_schema.py](../tests/test_block_schema.py) |

**Invariants** (a violation is a bug, not a style nit):
- Every Tier-1 block is clickable and resolves to a card. **Never an unclickable Tier-1 box.**
- Every connector is `static` and cardless. `⊕` is the only addition glyph.
- No number sits on a box; numbers are chips on cards.
- No light-green "this is the dim / this is what goes out" I/O block.
- No dotted arrows; no dotted recursive boundaries.

---

## 8. Anti-patterns (the disease catalogue)

- **Over-blocking.** Nine boxes in a layer. Fix: residual `⊕`, gate `×`, splits → Tier 2;
  scalars/flags → Tier 3.
- **A "skip connection" box.** A residual/skip is the loop arrow + a `⊕`/`concat` connector,
  never its own module.
- **A number on a box.** `4096→11008` belongs on the card as a chip.
- **A green I/O bookend that only restates the dim.** Use a port caption, or nothing.
- **A scalar as a box.** `hidden * layer_scalar`, an AdaLN gate, a guidance scale — Tier 3.
- **A prose-only structural card with no view and no real description.** Every block carries
  a real `view` *or* a real description (Gate B.2).
- **Special-casing one repo.** A branch tuned to one config is wrong; read the field
  generically and put new vocabulary in `everchanging/`.
