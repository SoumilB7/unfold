# 04 — Consumers and surface: what the product puts in front of a person

Research pass 04 of `z-docs/10-full-research/` (2026-09-02). Read-only on the
repository; renders under the session scratchpad `r04/`. The working tree is not
a git checkout (`git log` → "not a git repository"), so "HEAD" below means the
tree as found; every render made here was byte-identical to the blessed
galleries under `unfold-pkg/tests/sable_test_corpus/galleries/` (checked with
`cmp` for Llama-7B `00/01/02` and SD3.5 `00/04`), so what is described is the
blessed product, not a drift.

Renders made (all from corpus fixtures, `unfold(fixture["config"])`, local
transformers, no network):

| model | parse | HTML bytes | distinct views (PNG) | description-only leaf cards | layers · hidden · params banner |
|---|---|---|---|---|---|
| llama-7b | 21.2 s | 55,207 | 4 | 24 | 32 · 4,096 · `~6.74B*` + `⚠ unresolved evidence` |
| stable-diffusion-3.5-large | 59.4 s | 159,634 | 19 | 72 | **0 · 0 · `?`** + `ⓘ note` |
| deepseek-v3 | 27.0 s | 154,689 | 11 | 79 | 61 · 7,168 · `~670B (36.4B act.)*` |
| fluxtransformer2dmodel | 64.1 s | 176,791 | 20 | — | **57 · 0 · `?`** + `⚠ partial config` |

(`to_html` itself is 0.01 s; `save_images` 1–7 s. Parse time is pass 02's
subject; it is recorded here only because it is the user's wait.)

---

## 1. Surface by surface — what each consumes, decides, emits

### 1.1 `ir.py` — the contract every consumer projects from (612 lines)

**Consumes**: nothing; populated by the adapters. **Emits**: `ModelIR.to_dict()`
(`ir.py:431-450`), the dict every renderer/JSON/param path actually reads.

Fields, sorted by nature:

* **Typed facts (tri-state, `None` = unresolved)** — `AttentionSpec` (`ir.py:16-121`):
  `kind, num_heads, num_kv_heads, head_dim, mask, window_size, qk_norm, bias,
  cached, output_projection, projection_mode, scores_scaled, scores_scale,
  rope/position_kind/position_application/rope_theta/rope_dim/rope_initialization,
  sinks, logit_softcap, qkv_clip, cross_attention/cross_kv_source, MLA dims, DSA
  indexer dims, mrope_section, conv_kernel_size, output_gate, mixer_state`.
  `FFNSpec` (`ir.py:125-157`): `kind, activation, intermediate_size, gated,
  bias, projection_mode, expert_projection_mode, expert_activation_formula,
  num_experts, num_experts_per_tok, num_shared_experts, expert_intermediate_size,
  routing`. `LayerSpec` (`ir.py:218-238`): `norm_kind, norm_placement,
  residual_topology, parallel_norm_count, residual_scale, cross_attention`.
  `ModelIR` (`ir.py:399-422`): `hidden_size (None = unknown), vocab_size,
  max_position_embeddings, tie_word_embeddings (tri-state), embedding_norm_kind,
  final_norm_kind, cross_layer_edges`.
* **Provenance/debt** — `asserted: tuple` on both specs (`ir.py:121,156`),
  `activation_assumed`, `activation_from_class`.
* **Presentation living in the IR** —
  1. `AttentionSpec.variant: dict` (`ir.py:113-116`) — "self-describing label
     override … keys: short, tag, label (list[str]), title, desc". The
     diffusion adapter uses it to name MM-DiT joint attention; `labels.py`
     returns `variant["desc"]` verbatim (`labels.py:272-276, 377-379`). This is
     prose stored in the IR.
  2. `LayerSpec.blocks: list` (`ir.py:234`) — the entire render tree (untyped
     dict list; see §1.4) rides inside the IR, and `ModelIR.extras: dict`
     (`ir.py:420`) carries `render.model_blocks`, `render.loop_blocks`,
     `loop_edges` (with `route/gap/lane_index/label_size` SVG hints),
     `view_variants`, `render.theme`, `fact_provenance`, `config_audit`,
     `source_provenance`. The IR docstring says "renderers … project it; they
     do not reinterpret" (`ir.py:3-5`), but the block tree is *authored*
     upstream of the IR, so the IR is both a fact record and a pre-rendered
     layout document.
* **Grouping law** — `layer_signature` (`ir.py:352-386`) includes the block
  tree's structural keys (`_BLOCK_STRUCTURAL_FIELDS`, `ir.py:272-275`), so two
  layers with identical facts but different authored blocks form two groups.
  `distinct_layer_groups`/`detect_layer_period` (`ir.py:468-509`) are the
  consumer-facing collapse.
* **Zero sentinels**: for diffusion the IR is built with `hidden_size=0,
  vocab_size=0` (SD3.5 IR dump: `hidden: 0 vocab: 0`); `expanded/sections.py:36-43`
  has to special-case `hidden == 0` back to `None`, and `params.py:133` treats
  `not h` as unresolved. The stats banner does **not** — it prints `HIDDEN 0`
  (SD3.5, FLUX).

### 1.2 `opgraph.py` — the canonical op graph (1,250 lines)

**Consumes** the dict spec (`attn`/`ffn` dicts + `hidden`). **Decides** the inner
structure of a drill from facts alone: `ffn_structure_state` (`opgraph.py:96-122`)
is the one closed FFN vocabulary (`moe | conv_glu | mechanism_unresolved |
unsupported | gating_unresolved | storage_unresolved | gated | dense`);
`attention_region` (`opgraph.py:467-493`) dispatches on `kind` to SDPA/MLA/
gated-delta/SSM/LRU/RWKV/linear, and `None` → one opaque node
(`_unknown_attention_region`, `opgraph.py:496-554`). **Emits** a `Region`
(ops + edges, `opgraph.py:61-82`) with a 17-symbol op alphabet (`opgraph.py:37-41`).

What it decides on its own (not from a fact):

* Op **labels and card prose** are authored here: `"Linear (gate)"`, `"Concat
  heads"`, `"Append sink column"` + a 6-line description (`opgraph.py:774-782`),
  the four `scores_meta` narratives (`opgraph.py:583-609`), the T5 relative-bias
  sentence (`opgraph.py:842-846`), the conv-GLU descriptions with a **hard-coded
  `"Depthwise Conv 3×3"`** kernel (`opgraph.py:295-309`) — kernel size is not a
  fact on `FFNSpec`.
* `_cross_kv_label` (`opgraph.py:653-662`) decides the external-input node
  label by **substring-sniffing prose**: `any(w in src for w in ("text",
  "prompt", "encoder", "caption"))` → `"Encoded text"`.
* MoE: the expert is always one opaque node `["Expert FFN", "storage
  unresolved"]` (`opgraph.py:409-417`) — the renderer's expert drill comes from
  `blocks/feed_forward.py`, not from this region.
* Cache ports (`meta={"cached": True}` on K/V ops, `opgraph.py:691-724`), the
  `kv_cache` node (`opgraph.py:893-911`) and the `hidden` input pseudo-op are
  structural additions the forward() does not literally contain as modules.
* `ops_region` (`opgraph.py:1188-1250`) is the "universal declarer": a card can
  declare its internals as data; a typo'd kind raises.

### 1.3 `labels.py` — the vocabulary (976 lines)

Pure functions over the dict spec. **Emits** short/long/title strings, chip
lists, and card dicts. Almost every sentence a learner reads on a Llama card
originates here or in the adapter blocks:

* Tables: `_MASK_*`, `_KIND_*`, `_ACTIVATION_LABELS` (`labels.py:20-85`);
  `_OP_TITLES`/`_OP_SENTENCES` (`labels.py:899-934`) — e.g. `"Linear projection
  (a learned weight matrix applied to every position)."`.
* `attention_summary` (`labels.py:365-519`) writes the description sentence per
  kind and the honesty chips (`"mask unresolved"`, `"QK norm unresolved"`,
  `"cache unresolved"`, `"QKV storage unresolved"`, `"score scaling
  unresolved"`, `"position application unresolved"`, `"bias unresolved"`).
* `ffn_summary/ffn_label/ffn_title/ffn_short` (`labels.py:538-726`) derive from
  `ffn_structure_state` — a genuine projection.
* **Identity leak inside the vocabulary**: `attention_label` decides the
  cross-attention side label by `"prompt states" in str(attention.cross_kv_source)`
  → `"Prompt"` else `"Vision"` (`labels.py:786-789`), eight lines below a comment
  that says "never recover a source modality by searching prose here"
  (`labels.py:777-779`). A cross-attention whose K/V source is audio or video
  will be labelled "Vision".
* `activation_label` (`labels.py:253-265`) collapses `gelu_new/gelu_fast/
  gelu_pytorch_tanh` → `"GELU"` (the card chip still shows the raw fn name,
  e.g. `gelu_new` on the T5 activation card).

### 1.4 `block_schema.py` — the render-tree contract (400 lines)

`Block` is a `TypedDict` with 30 keys (`block_schema.py:33-84`); the docstring
is explicit that `id` is the only required key and "everything else is
presentation, drill-down, or layout". Presentation keys: `label, title,
description, facts, view, w, h, font, offset_y, lane, branch_side, tap_from,
feeds, also_feeds, side_align, residual_from`. Fact-ish keys: `detail,
components, resolved, static, source_*`. Validators: unknown key, unregistered
view, duplicate id (`block_schema.py:155-195`); `validate_click_coupling`
(`:209-238`) enforces node→card by panel depth; dotted-stroke bans (`:334-369`).
`DIFFUSION_STAGES/BLOCK_IDS/PART_KINDS` come from `everchanging/diffusor/
typing.yaml` (`:108-111`); a diffusion block outside the approved stage set
renders pale. `TRANSFORMER_STAGES` is loaded but "the transformer renderer
doesn't draw pale-when-unapproved yet" (`:113-117`).

### 1.5 The adapter "blocks" layer — where the drawn tree is authored

`adapters/transformer/blocks/{attention,feed_forward,layers,model}.py`,
`special_parts/*`, `adapters/diffusor/{blocks,unet,projection_ir,compound}.py`.
Audited in full by a sub-pass (line counts and quotes verified at the cited
lines). Headline: **every sentence the diagram renders is hand-written in the
adapter; the typed IR supplies operands (numbers, booleans, enum kinds) and
selects which pre-written card set to emit.** Across the nine authoring files
there are 279 `"title":` literals and ~850 presentation-key assignments against
~490 fact-reading keys.

| file | lines | titles | prose/layout share (est.) | fact-reading | worst offender |
|---|---|---|---|---|---|
| `transformer/blocks/attention.py` | 1,087 | 97 | ~94 % | `attention_detail` `:11-77` | 26-card MLA catalogue `:594-883`; prose sniff `"text" in cross_src` `:239-240` |
| `transformer/blocks/feed_forward.py` | 637 | 43 | ~95 % | `ffn_detail` `:21-38` | **four literal expert boxes** `expert_1/expert_k/expert_kp1/expert_n` `:582-608` drawn regardless of expert count; `ffn_view` returns `"dense_ffn"` when kind/gated unknown `:14-17` |
| `transformer/blocks/layers.py` | 498 | 14 | ~80 % | delegation `:432-465` | `wiring_unresolved` pale block `:137-158`; `{…}.get(norm_kind, "Normalization")` `:482` |
| `transformer/blocks/model.py` | 434 | 31 | ~85 % | `:290-433` | `block_diffusion_loop_blocks` `:103-279` with invented `30.0` softcap `:120`, `"up to 48 steps"` `:165`, `"bound ε=0.1"` `:268` under a docstring that says "never invented" `:116` |
| `special_parts/per_layer_embedding.py` | 220 | 8 | ~95 % | — | prose inside `detail` `:105-116`; `"(token + context) / sqrt(2)"` label `:177` with no operand |
| `special_parts/modalities/*` (9 files) | 1,494 | 0 | ~5 % | ~95 % | no sentences at all, but identity tables: `MODALITY_REGISTRY` `registry.py:81-103`, `_COMPONENT_TO_MODALITY` `evidence_projection.py:21-25` |
| `diffusor/blocks.py` | 1,393 | 40 | ~85 % | bookends `:137-188` | **placeholder text-encoder tower** for a bare denoiser config `:1247-1262`; 9 hand-drawn loop nodes `:426-672` incl. `"typically ~20-50 — it is not a config field"` `:606-621`; `text_projection` with an invented GELU `:675-722`; VAE scaffold `:849-968` |
| `diffusor/unet.py` | 971 | 38 | ~75 % | `parse_unet` `:57-182` | 7× `"text" in str(kv_label).lower()` `:366,479,517,550,641,769`; `gated = "glu" in ffn_act` `:389`; absent `layers_per_block` silently draws 2 ResNets `:88` |
| `diffusor/projection_ir.py` | 1,428 | 8 | ~8 % | ~92 % | `_stream_variant` prose table `:50-95`; `cell_structure_unresolved` template `:185-199` |
| `diffusor/compound.py`, `schema.py`, `loader.py`, `transformer/assembly.py`, `common.py` | — | 0 | ≤5 % | ≥95 % | `type_name.endswith("Block2D")` `compound.py:105`; `_ENCODER_NAMES` class-name → label yaml `diffusor/parser.py:40` |

The only text genuinely *derived* from facts rather than pre-written is the
diffusion bookend description `" → ".join(ops)` (`diffusor/blocks.py:36-42`).

### 1.6 `renderers/html/*` — document, cards, interaction, drawing

* **Document assembly** (`document.py:14-215`): header + stats banner +
  `ARCHITECTURE` `<details open>` + inspect panel (`data-depth="2"`) + nested
  panels (`data-depth="3..N"`) + `LAYER MAP` (`<details>` closed) + optional
  `CODE EVIDENCE`. One architecture SVG per (layer-group × `view_variant`),
  toggled by pure-CSS radio pills (`document.py:59-133`). Diffusion routes to
  `views_diffusion.render_diffusion_fragment` by `extras.render.family ==
  "diffusion"` (`document.py:32-37`). Empty stack → a prose card "Repeated layer
  structure unavailable" (`document.py:143-153`).
* **Cards** (`cards.py`): every clickable node gets one `<div class="uf-card-
  detail" data-card-id=…>` containing title, description, `uf-fact` chips, and
  (rich cards only) the drill SVG. Hard-coded id lists decide which model-level
  blocks get cards: `("tok_text","embed","embed_norm","join_concat",
  "position_ids","position_embed","position_add")` (`cards.py:21-22`),
  `("vision_path","video_path","audio_path","conditioning_path","fusion")`
  (`:39`), `("final_rms","lm_head")` (`:73`), `entry_stage`, `mtp` (`:77-93`).
  Nested panels are built from `children` levels (`cards.py:217-227`); a child
  with no SVG is a description-only leaf.
* **Metadata** (`metadata.py`): grouping (`_make_info` `:26-76`), card meta
  strictly from authored blocks (`_block_meta` `:165-172` — "a fact/spec value
  … cannot create a card when no block was authored", `:96-105`), badges
  (`_arch_badges` `:233-279`; UNet badges keyed on `extras["unet"]` presence),
  pills (`_group_label` `:175-188`). Identity: `_has_cross_attention_adapter`
  checks `block["id"] == "cross_attention_adapter"` (`:463-469`).
* **View registry** (`block_views/registry.py:177-230`): 31 views keyed by
  `block["view"]`. `render_view` (`:83-111`) infers the *component* of a block
  with no `source_component` by sniffing `"vision" in marker` / `"audio" in
  marker` over the block's `id view kind` strings (`:95-99`) — the render-event
  oracle binding can be decided by an id substring.
* **Interaction** (`interactions.py:5-130`): one inline `<script>`; a click on
  a `.uf-node` in the architecture shows panel index 0 (`data-depth=2`), a click
  inside panel N shows panel N+1 and clears deeper panels; clicking the
  selected node again returns to the `default` hint card. JS toggles
  visibility only — every SVG is baked at render time (`preview.py:9-12`).
  Panel size classes come from `data-card-size` (`cards.py:186-198`). No
  keyboard navigation, no URL state, no back/breadcrumb; the layer map and
  code-evidence sections are native `<details>`.
* **Theme/styles** (`theme.py`, `styles.py`): two palettes (teal, blue; the
  blue "diffusion" palette is declared but SD3.5/FLUX rendered **teal** — see
  §4.2, the adapters hard-set `teal`), Google-Fonts `Caveat`
  (`theme.py:23-25`), `max-width:720px` (`styles.py:13`), `.uf-frame
  max-width:820px` (`document.py:234`). Header comment says "No hover anywhere"
  (`sections.py:28-32`) but the params cell carries the assumption list only as
  a hover `title=` (`sections.py:110`), so the `*` on `~6.74B*` is unexplained
  on touch devices and in PNGs.
* **Evidence section** (`evidence.py:53-99`): rendered only when
  `extras.code_evidence` exists (the `inspect_code=True` path); none of the
  four renders had it. Chip labels are a hand table (`:14-50`).
* **Fact projection witness** (`fact_projection.py`): key-granularity sets of
  which ledger leaves each surface claims to draw (`ATTENTION_DRAWN` …
  `DRAWN_PAIRS`, `:45-126`); used by the projection-audit net, not by the
  drawing.
* **Render context** (`render_context.py`): call-local theme, wiring findings,
  `RenderEvent`s with `drawn_ops/node_ids/facts_projected/receipts`
  (`:12-40`) — the channel Sable's nets read.

The renderer-internal authorship (graph engine, drawn glyphs, per-view
conditionals) is tabulated in §4.2.

### 1.7 `expanded/*` — the JSON (`schema_version "3.2"`)

`build_expanded` (`expanded/__init__.py:46-95`) emits `model, dimensions,
parameters, io, stack, layer_groups` plus optional `sampling_loop,
external_pathways, modalities, multi_token_prediction, cross_layer_edges,
code_evidence, warnings`. Per group: `attention` (spec + `operation_graph`
projected from `opgraph.attention_region` with public renames
`scaled_scores→scores` etc., `expanded/attention.py:167-185`), `ffn`
(`expanded/ffn.py`), `norm`, `residual_topology`, `block_graph` (one node per
authored block, sequential edges — `expanded/block_graph.py:14-26`, so the
JSON's coarse layer DAG is the *authored* block order, not a computed one).
`io` is built from model-block ids (`expanded/sections.py:102-133`) — the JSON
reads the render tree. Diffusion: `dimensions` is `{}` and `parameters.
incomplete` explains why (SD3.5 JSON, 5,081 bytes; it carries the sampling
loop but zero layer groups). The test contract asserts "structural, not
renderer copy": no `summary/features/description` keys
(`tests/test_expanded_json.py:96-110`). Stability: one version constant, no
migration or changelog; `trace.code_finding_ids` is always `[]`
(`expanded/attention.py:60-63`).

### 1.8 `params.py` — the estimate (311 lines)

Formulas: attention `h·(nq+2·nkv)·hd + nq·hd·h` (`:51-55`), MLA per DeepSeek
geometry (`:21-42`), FFN `g·h·inter` with `g = 3 if gated else 2` (`:110-112`),
MoE `experts·per_expert + shared + router` (`:84-109`), norms by kind ×
placement count (`:233-247`), embeddings `v·h` and an untied head when
`tie_word_embeddings` is not True (`:145-149`). Every unknown appends a
sentence to `assumptions` (`:143-255`) and the banner stars the number.
Assumptions that a learner never sees unless they hover: Llama-7B carries
`"embedding-stage normalization not proven — its parameters omitted"` (a
harmless 0 for Llama, but it is why the number is starred). No biases, no
gated-delta formula (`:44-49`), no per-block counts surface anywhere in the
HTML — `per_layer` exists in the dict (`:257`) but is not drawn.

### 1.9 `diagram.py` + `__init__.py` — the public API

`unfold(cfg_or_id, token, inspect_code, code_source, return_json)`
(`__init__.py:54-98`); `Diagram.to_html(standalone)`, `save(.html|.json)`,
`to_ir`, `to_json`, `to_json_string`, `param_count`, `warnings`,
`wiring_problems`, `render_events`, `to_png`, `save_images`, `_repr_html_`
(`diagram.py:17-195`). `__all__` (`__init__.py:29-51`) also exports **`sable,
SableReport, bless, check_regression, load_corpus, lint_labels,
inspect_model_code`** — the QA harness, including the corpus-mutating `bless`,
is part of the public library surface with the same standing as `unfold`. The
docstring of the package still says "turn any HuggingFace transformer into a
… diagram" and the notebook example is Kimi-K2 (`__init__.py:1-12`).
`__version__ = "0.2.15"` (`__init__.py:27`) while `pyproject.toml` says
`0.2.17` and the Space pins `model-unfolder[hf]==0.2.17`.

### 1.10 `encoder_panel.py` / `submodel.py` — embedded towers

A pipeline text encoder is parsed by the same transformer adapter
(`encoder_panel.py:75-78`) and projected by `submodel_spec` (`submodel.py:154`)
into a facts-only spec rendered by one tower projector. `submodel_group_tags`
(`submodel.py:296-327`) names layer groups only by axes that differ
(kind/mask/ffn/norm); when nothing differs it falls back to **`"type A"`, `"type
B"`** (`:326`) — which is exactly what the T5 tower in SD3.5 shows (the real
difference, "layer 0 owns the relative-position bias", is not one of the four
axes). `ALTITUDE_TRANSFORMS["tower"]` is empty (`:35-44`): encoder-ness no
longer removes cache ports; that is left to evidence, so CLIP/T5 cards say
`"cache unresolved"`.

### 1.11 `preview.py` — PNG oracle

`save_images` extracts every baked `<svg>` and dedups by visual hash
(`preview.py:170-221`); needs `rsvg-convert` (`:58-63`) — a system binary, not a
Python dependency. Labelling bug: any SVG outside a card and outside a
`uf-arch-variant-N` div is labelled `architecture` (`:93-99`), so the **layer
map** is written as `03__architecture__1.png`/`18__architecture__1.png` in every
gallery and MANIFEST — a reviewer reads it as a second architecture view (the
Her-Eyes review of Llama did exactly that, `galleries/llama-7b/her_eyes_review.md`).

---

## 2. The user journey

### 2.1 Llama-7B — four pictures, two floors

**Level 0 — the page.** Header `llama-7b · LlamaForCausalLM`, badges `MHA`,
`Gated FFN`, and a clickable `⚠ unresolved evidence` badge whose bar reads
*"Unresolved code-defined facts (drawn honestly, never asserted): parallel norm
count — modeling source is present but unresolved."* Stats banner `LAYERS 32 ·
HIDDEN 4,096 · VOCAB 32,000 · CONTEXT 2,048 · PARAMS ~6.74B*`. A first-time
reader meets a warning on the most ordinary model in existence, about a fact
("parallel norm count") that only applies to parallel-residual layers Llama does
not have; the `*` has no visible explanation.

**Level 1 — `00__architecture.png`.** Tokenized text → Token Embedding layer →
[RMSNorm → Multi-Head Attention → ⊕ → RMSNorm → Gated FFN → ⊕] ×32 → Final
RMSNorm → Linear output layer. This is the strongest picture the product makes:
one spine, two residual loops, a `× 32` badge. Amber borders mark every box as
clickable; the ⊕ glyphs are static. A learner gets the shape of a decoder
immediately. What is absent: any tensor shape on the picture, any parameter
count on a box, any indication that attention is causal (that is a chip on the
layer map: `MHA + Gated FFN (causal · RoPE)`), any hint of *why* the norm sits
before the sublayer.

**Level 2 — `01__attn.png`.** Clicking *Multi-Head Attention* opens a card:
title "Attention", sentence "Multi-head self-attention — every head attends
over the sequence.", chips `32 heads · head dim 128 · RoPE θ 10,000 · bias-free
projections · QK norm unresolved · in/out 4,096`, then the drill: `in (4,096)`
fans to Linear (K)/(V)/(Q); K and Q pass through `apply RoPE θ 10000`; K and V
enter a `K/V cache update + read` box (with ⌃⊥ glyphs on K, V and the cache);
`Q K^T / sqrt(dim)` → Softmax → ⊙ → Concat heads → Linear (out). Every box is
clickable and opens a **description-only leaf** (24 of them across the
document): e.g. `[k_proj] Key projection — "Linear over the hidden state
producing the keys." · 4,096 → 4,096 · 32 KV heads · cache ports: ⌃ write · ⊥
read`. The journey ends there — the third click is text, and the third panel
(`data-depth=3`) is the last one that exists for this model.

**Level 2 — `02__ffn.png`.** `in (4,096)` → Linear (gate) → SiLU ⤳ × ⤲ Linear
(up) → Linear (down). Card chips `SiLU · hidden 11,008 · split gate/up · in/out
4,096`. Leaves: `Gate projection 4,096 → 11,008`, `Gate product — "SiLU(gate) ×
up"`, etc.

**Coda — `03__architecture__1.png`** is the LAYER MAP: `32 layers - 1 type`, a
bar of 32 identical cells, legend `MHA + Gated FFN (causal · RoPE) · L0–L31 ·
32×`. Different font family (UI sans) from the diagrams (Caveat).

Honesty chips a learner meets on Llama-7B: `QK norm unresolved` (attention
card), the `⚠` bar, the `*`. Things a learner cannot find: per-block parameter
counts (`params["per_layer"]` is computed but never drawn), weight shapes on
the picture (only inside leaf cards as `a → b`), the causal mask as a drawn
object (it is a word on the layer map and nothing in the drill), what the KV
cache is for, any "why" for RoPE/RMSNorm/gating, the vocabulary/head-tying
fact (`tie_word_embeddings` is in the IR, the LM-head card says only "Projects
the final hidden state into vocabulary logits").

### 2.2 SD3.5-Large — nineteen pictures, and the model is not among them

**Level 0.** Header `stable-diffusion-3.5-large · SD3Transformer2DModel`, one
badge `ⓘ note` ("Scheduler and codec panels remain explicit U13/U12
compatibility handoffs; they do not author the denoiser structure."), banner
**`LAYERS 0 · HIDDEN 0 · TIMESTEPS - · LATENT - · PARAMS ?`**. The section is
titled `SAMPLING LOOP — Denoiser applied iteratively · click it to open its
architecture`, and below it `DENOISER LAYER MAP — Denoiser layers · all
structurally identical`, which opens to `18__architecture__1.png`: **"0 layers
- 0 types"** and an empty bar.

**Level 1 — `00__architecture.png`.** Text prompt → CLIP / CLIP / T5 (left
column); Noise → `z_T · once` → latent → `z_t` → **Diffusion denoiser** →
`ε̂` → Flow Match Euler → `z_t-1 · each step` back to latent; denoiser → `z_0` →
VAE decode → Output; Timestep t → denoiser. A `↺ t → 0` badge on the loop
frame and a small patch-grid glyph. This is a good sampling-loop picture and
a first-time reader understands "denoise in a loop, decode once".

**Level 2 — `04__denoiser.png`.** Clicking the hero opens: Denoiser state →
*Input unresolved* (pale) → [**Repeated denoiser structure unresolved**, pale,
static, not clickable, inside an empty ×-less frame] → *Output unresolved*
(pale) → Denoiser output. Card: "Source-proven diffusion denoiser — The
denoiser detail is projected from exact source occurrences and their
checkpoint-bound operands. Unresolved mechanisms remain opaque." **The MM-DiT —
joint attention over image+text streams, AdaLN-Zero modulation, patchify,
unpatchify, 24 or 38 blocks — is entirely absent.** `fact_provenance` has one
row (`root.denoiser.diffusion_root_topology = repeated_stack, code_proven`).
The `_text_encoder_configs`, `_vae_config`, `_scheduler_config` produce
everything else on the page; the model the user asked about produces one pale
box. This is the blessed state (gallery `04__denoiser.png` is byte-identical).

**Level 2 — the supporting cast.** `01__encoder_0.png` (CLIP-L): Token
embedding → [LayerNorm → **Self-attention mechanism unresolved** → ⊕ →
LayerNorm → Feed-forward (FFN) → ⊕] ×12 → "→ denoiser conditioning". Clicking
the attention (`07__encoder_0_op_selfattn.png`) shows a single pale box
"Attention mechanism unresolved" with one arrow — while its card sentence
says *"Each token attends to the others in the prompt, mixing context across
the sequence…"* (`diffusor/blocks.py:1072`), i.e. prose asserts what the
picture refuses to. `03__encoder_2.png` (T5-XXL): Token embedding → `MHA type
A` → *Wiring unresolved* (pale) → Gated FFN, then [`MHA type B` → Wiring
unresolved → Gated FFN] ×23. `10__encoder_2_g0_op_selfattn.png` is a real
drill: Linear V/Q/K, `Relative positions` → `Relative position bias` → ⊕ →
Softmax → ⊙ → Concat heads → **Attention output path unresolved** (T5's
`o` projection is not proven). Chips: `64 heads · head dim 64 · bias
unresolved · QK norm unresolved · cache unresolved · type A · 1 of 24 layers`.
`05__scheduler.png`: `z_t current latent` ⊕ `Δσ · v̂` ← `v̂ velocity from
denoiser`, caption `z_{t-1} (one step)`. `06__vae_decode.png`: Conv-in → Mid
block → Up stage 4..1 → Output head; `13__vae_mid_block.png` ResNet → Attention
→ ResNet; `14__vae_decoder_block_4.png` [GroupNorm+SiLU → Conv 3×3 ×2 → ⊕] ×3 →
Upsample. Three drill depths exist (`data-depth` 2, 3, 4).

Where it becomes "blocks and blocks": the VAE (`06` → four near-identical
`Up stage n` boxes → `14..17` four near-identical stage drills, differing only
in channel count) and the two CLIP towers (`01`/`02` identical shapes, both
ending in an unresolved single box). 72 description-only leaf cards. Where
honesty chips appear: everywhere in the denoiser and CLIP towers; `HIDDEN 0`
and `PARAMS ?` in the banner; `Output` titled "Output domain unresolved".

### 2.3 Cross-checks on other blessed galleries

* **DeepSeek-V3** (`deepseek-v3_images/`): two pills (`MLA · Gated FFN L0–L2 ·
  3×`, `MLA · MoE L3–L60 · 58×`); the *default* architecture view is the
  3-layer dense group (`00__architecture.png` shows `× 3`), because the group
  order is encounter order and `checked` is `dominant_idx and j == 0`
  (`document.py:94`) — verified that `01__architecture_v1.png` is the `× 58`
  MoE view; which one opens first depends on `groups.index(dominant)`. MLA
  drill (`02__attn.png`) → `Query path` / `Compressed KV path` sub-drills
  (`05`, `06`) — a real third floor. MoE drill (`04__ffn.png`): `Router` →
  `Expert 1 · Expert k · Expert k+1 · Expert 256 · Shared expert` → ⊕ with the
  caption `top-8 of 256 experts active · +1 shared (always on)`; the four
  expert boxes are literal (`blocks/feed_forward.py:582-608`). `07__router.png`
  Linear (Gate) → sigmoid → Top-k ← *stored bias selection only* (pale) → sum
  renormalize → × 2.5; `09__g_topk.png` Group scores → Top-k groups → Mask
  groups → Top-k experts → Gather weights. This is the deepest and best journey
  in the corpus (four floors) and it ends in prose leaves like the rest.
* **Qwen2-VL-7B** (`galleries/qwen2-vl-7b-instruct/00__architecture.png`):
  Vision → grid / Token Embedding / Video → grid → Multimodal fusion →
  [RMSNorm → Grouped-Query Attention → ⊕ → RMSNorm → **Feed-forward mechanism
  unresolved** → ⊕] ×28 → **Pre-head path unresolved** (pale) → Linear output
  layer. A standard Qwen2 SwiGLU MLP is shown as unresolved in the blessed
  state.
* **FLUX** (HEAD, `fluxtransformer2dmodel_images/03/04__denoiser.png`): the
  denoiser *is* materialized here (unlike SD3.5): `Input op: linear` → [`Dual-
  state attention (unresolved)` → `LayerNorm wiring unresolved` → `Feed-forward
  mechanism unresolved`] ×19 → [‖ → `Joined-input attention (unresolved)` →
  `Wiring unresolved` → `Feed-forward (FFN)`] ×38 → `Output op: linear`. The
  attention drill (`07__attn.png`) is one pale "Attention mechanism unresolved"
  box. Banner `57 · 0 · - · - · ?`, badge `⚠ partial config` ("denoiser.stacks[0]:
  FFN mechanism is not uniquely source-resolved"). Pills read `unresolved ·
  Dual-state attention · two returned states · unresolved · FFN unresolved
  (L0–L18 · 19×)`.

---

## 3. The deployed public surface

### 3.1 The Hugging Face Space (`hf/app.py`, `hf/README.md`)

Gradio app titled **Unfold**, short description *"View any model's internal
architecture just by HF tag"* (`hf/README.md:12`), SDK `gradio 6.15.2`, Python
3.13, `requirements.txt` → `model-unfolder[hf]==0.2.17` (the PyPI release, not
this tree). Last Space commits: "version update" ×2, "moe fixes", "nitty
gritties" (`hf/.git`, July 2026).

What a visitor experiences:

1. A textbox pre-filled with `Qwen/Qwen2.5-0.5B-Instruct` (`app.py:9,135-140`),
   an optional password field for an HF token, an **Unfold** button; below, a
   dashed placeholder "Enter a Hugging Face model ID and unfold it."
   (`app.py:69-73`). Subtitle: "Generated purely from the Hugging Face config
   map. No model weights are downloaded." (`app.py:129`).
2. Click → `show_progress="full"` spinner (`app.py:157`) while
   `unfold(model_id, token)` runs (`app.py:117`). Measured locally with warm
   caches: 21 s (Llama-7B), 27 s (DeepSeek-V3), 59 s (SD3.5), 64 s (FLUX); on a
   cold Space (hub round-trips, `transformers` import, free CPU tier) the
   coordinator's measured 30–100 s per call is consistent — **the visitor
   stares at a spinner for the better part of a minute with no intermediate
   output**, and every subsequent model is another full minute. (Unverified:
   whether Gradio's default request timeout on the free tier cuts the longest
   ones.)
3. On success: a status line `Rendered `id` · ~6.74B params[, ~x active] ·
   Warnings: …` (`app.py:89-106`) and the standalone HTML injected as an
   `<iframe sandbox="allow-scripts" srcdoc=…>` of min-height 860 px
   (`app.py:80-86,18-24`). Inside the iframe the full document works: clicks,
   pills, `<details>`, the `⚠`/`ⓘ` bars, Google-Fonts load (network permitted
   in the sandbox). The Space exposes nothing else: no JSON, no param
   breakdown, no PNG, no permalink, no history, no download of the HTML.
4. On any failure the visitor sees exactly one message — **"That model card
   isn't accessible."** (`app.py:122-123`) — for `ModelNotFoundError`,
   `ModelAccessError`, `ConfigParseError`, a diffusers pipeline needing a
   loader, an `ImportError`, or a renderer crash alike. The typed error
   hierarchy in `errors.py:19-47` never reaches the public surface.

Given §2.2, the Space's promise "view any model's internal architecture" is
met for dense/MoE LLMs and not met for SD3.5/FLUX-class models, where the
visitor waits a minute and receives a pale box labelled "structure unresolved"
(on 0.2.17 the picture may still be the older, populated DiT — see §3.2; the
tree here is not what the Space runs, and this pass cannot say which behavior
PyPI 0.2.17 has).

### 3.2 The showcase (`unfold-pkg/examples/*.html`) — stale and partly mislabeled

Thirteen files dated 10 May – 10 June 2026 (`ls -la`). Compared with HEAD output:

| example | what it actually is | HEAD today |
|---|---|---|
| `deepseek-v3.html` | **`gemma-4-E2B` / `Gemma4ForConditionalGeneration`** (`uf-name`), 35 layers, 1.69B — the file is mislabeled | DeepSeek-V3 renders 61 layers, `~670B (36.4B act.)*`, `⚠ unresolved evidence` (parallel norm count, projection mode); README claims "~675B (~41B active)" (`unfold-pkg/README.md:79`) |
| `flux-1-dev.html` | full DiT drawn: cards `attn, adaln_cond, text_cond, q_proj … silu`, `6.46B` params, **0 occurrences of "unresolved"** | 57 layers · `HIDDEN 0` · `PARAMS ?` · 71 "unresolved" strings · `⚠ partial config`; attention drill is one opaque box |
| `llama-3-8b.html`, `gemma-4-e4b.html` | pre-nested-panel era: no `<meta generator>`, no `data-depth`, no `data-card-size`, 3 SVGs | 4 SVGs, depth-2/3 panels, `⚠` bar |
| `unet-*.html` (6) | `LAYERS 0`, `PARAMS 1.28K / 1.54K` — a parameter "estimate" equal to the hidden width | UNet path is "U11 compatibility handoff" (`expanded/sections.py:186-188`) |
| `kimi-k2.html`, `mistral-7b-v0.3.html`, `gemma-4-31b.html` | generator meta present, no panels | — |

`examples/images/` holds a single `llama-3-8b.png` used by the README hero.
None of the thirteen contains a single "unresolved" chip; HEAD emits them on
every one of the four models rendered here. The showcase advertises a
pre-honesty product.

---

## 4. Authorship audit — projection vs local authoring

### 4.1 By surface

| surface | projected from typed facts | authored locally | where |
|---|---|---|---|
| Architecture SVG (transformer) | block order, `× N`, group pills, norm kind words, badge text | every block `label`/`title`/`description` (adapter), glyph choice, layout constants (`w`, `lane`, `offset_y`) | `blocks/layers.py:99-117,137-158`; `blocks/model.py:314-433`; `renderers/html/views.py` |
| Architecture SVG (diffusion loop) | encoder count/names (from `_ENCODER_NAMES` yaml), scheduler class, VAE geometry | all 9 loop nodes + 10 edges + labels (`z_T · once`, `ε̂`, `↺ t → 0`), placeholder encoder tower when none is declared | `diffusor/blocks.py:288-361,426-672,1247-1262` |
| Attention drill SVG | op set and edges (`opgraph.attention_region`), widths, θ, softcap value, sink op, bias lane | op labels, `scores_meta` narratives, `"Encoded text"` by prose sniff, cache-port glyphs | `opgraph.py:568-912`, `:653-662` |
| FFN drill SVG | `ffn_structure_state` → template | op labels, conv-GLU `3×3` kernel, clamp labels | `opgraph.py:140-331` |
| MoE drill SVG | N, k, shared count, routing policy facts | four literal expert boxes; router step wording | `blocks/feed_forward.py:507-637` |
| Inspect cards (title/desc/chips) | numbers in chips, honesty chips (`*unresolved`) | every sentence; `variant.desc` verbatim from the IR | `labels.py:365-726,899-968`; `blocks/attention.py:80-1087` |
| Header badges / pills | kind, k/N, layer-type count, window | wording tables | `metadata.py:233-279`; `labels.py:20-85` |
| Stats banner | layers/hidden/vocab/context/params | `HIDDEN 0` for diffusion (zero sentinel not mapped to "?"); hover-only assumptions | `sections.py:84-133` |
| Layer map | groups, period, runs | legend wording, palette | `views.py` (`_build_layer_map`) |
| Embedded towers (CLIP/T5) | full ModelIR via the transformer adapter | `type A/B` fallback tags, `→ denoiser conditioning`, `in (prompt tokens)` | `submodel.py:296-327`; `block_views/text_encoder.py:57-60` |
| Expanded JSON | spec fields, op graphs, grouping | `block_graph` = authored block order; `io` from block ids; `code_finding_ids: []` | `expanded/block_graph.py:14-26`; `expanded/sections.py:102-133` |
| Params | formulas over facts | conventions when unknown, listed as strings | `params.py:143-255` |

### 4.2 Renderer-internal authoring (graph engine, views, glyphs)

Audited file-by-file by a second sub-pass; the highest-impact citations were
re-read at the lines quoted here.

**Genuinely projected (Region/IR-driven, no local shape)**: `op_render.py`
(Region → Graph; lane slot order `:183-190`), `graph.py` structure,
`graph_engine.py` layout — **no layout rule keys on an op id or canonical_id**;
the only identity-sensitive branches are `node.kind == "cache"` (`:307-309`)
and glyph-shape strings (`:274-309`) — `block_views/feed_forward.py`,
`declared_ops.py`, `block_facts.py`, the graph body of
`block_views/attention.py`, the expert region of `mixture_of_experts.py`.

**Authored locally (structure or text invented in the renderer)**:

| where | what is invented | file:line |
|---|---|---|
| `graph.py` glyph table | 25 default on-screen labels (`"Multi-head self-attention"`, `"Feed-forward (FFN)"`, `"Scaled scores"`, `"Context window"` …); **unknown kind silently renders as a norm box** | `graph.py:47-81`, `:105` |
| `op_render.py` | `region_out` port anchor appended to every graph; **unknown op kind falls through to a `"norm"` node** | `op_render.py:199`, `:287` |
| `svg.py` | the sliding-window strip: **15 cells with a fixed 5-cell active window** regardless of any fact; cache ⌃/⊥ port glyphs; tap dots; tooltips deliberately emptied | `svg.py:333-372` (`:345`), `:374-432`, `:270-272` |
| `block_views/attention.py` | on `mask == "sliding"` the canonical `hidden` node is **mutated** into the context-window glyph; receipts gated on exact node-id tuples; KV-sharing aside prose (`"KV cache Nx smaller"`) | `attention.py:343-352`, `:69-186`, `:356-405` |
| `mixture_of_experts.py` | four phantom experts `Expert 1 / k / k+1 / N`; `moe_hidden`/`add_moe`/`moe_out` ports | `:27-30`, `:38-46` |
| `moe_router.py` | the whole top-k selection node list (`Group scores → … → Gather weights`); invented `g_scale` ×, `g_bias` "stored bias" source | `:144-158`, `:102-119` |
| `views.py` (architecture SVG) | `_KIND_LAYOUT` shape table keyed on kind strings; solid-vs-pale by **id-set membership** (`DIFFUSION_BLOCK_IDS`); position scaffold triggered by the id triple `{position_ids, position_embed, position_add}`; `join_concat` ‖ glyph; the arrow chain `[stack_input] + … + [final_rms, lm_head]` synthesized rather than read from edges; ~20 magic pads (`mtp_pad 108`, `position_pad 56` …); fallback labels `"Embedding stage unresolved"`, `["Pre-head path","unresolved"]`, `"LayerNorm"` for `embed_norm` | `views.py:34-52`, `:106-109`, `:155-163`, `:340-349`, `:455-460`, `:226-239`, `:266-300`, `:330`, `:352` |
| `views_diffusion.py` | both loop canvases are hand-placed pixel layouts (`w,h = 760,640`; `den_x…`), the `latent` register node is renderer-invented and always `resolved=True`, `↺ t → 0` badge, decorative 5×5 Gaussian glyph, per-id dispatch `if bid == "denoiser"`, route-string dispatch, **`denoiser_view` defaults to `"dit"`**, block-diffusion **`canvas_length` defaults to 256** and its badge reads `"↺ up to 48 steps"` | `views_diffusion.py:316-345`, `:370-372`, `:582-603`, `:223-232`, `:436-450`, `:214`, `:756`, `:773-785` |
| `tower.py` | `_sublayer` synthesizes the whole cell (pre-norm, post-norm, gate ×, residual ⊕) around the region; **resizes the FFN box when the label contains `"unresolved"`/`"unsupported"`** (string-sniffing prose); two nodes aliased onto one card for post-norm | `tower.py:94-115`, `:129-137`, `:161-168` |
| `metadata_modalities.py` | ~120 authored sentences, ~35 invented card ids (`fusion_boi/eoi/boa/eoa/…`, `unified_*`); per-modality presentation table `_MODALITY_PRESENTATION_SPECS`; `_PROJECTOR_TITLES/DESCS`; **three copies of a norm-placement table with `where.get(placement) or where['pre']`** — an unknown placement is narrated as pre-norm; `"conditioning" in str(s)` substring tests | `:155-192`, `:913-930`, `:494,668,897`, `:1026-1029`, `:1065-1365` |
| `modality_views/fusion_placeholder.py`, `fusion_grid.py`, `fusion_prefix.py` | the token-strip diagrams (`tok · BOI · v0 · v1 · … · EOI`) are fully invented with fixed tile counts; canvas size chosen by modality count | `fusion_placeholder.py:37-52`, `:146-222`; `fusion_grid.py:96-149` |
| `modality_views/audio.py`, `vision_details.py` | `"add" in label` converts a position op into a ⊕ and invents a `"Fixed positions"` source; `"learned"/"fixed" in pos` decides whether a position node is drawn | `audio.py:216-225`; `vision_details.py:32` |
| `block_views/unet.py` | ResNet cell and Transformer cell authored (not read); `_concat_node` ‖ at every up-merge; text-conditioning buses; phantom `{"name": "Text encoder"}` when encoders are fewer than expected; `"CLIP" in str(e)` substring match; a second parallel `_rect_block` implementation (`_box`) | `unet.py:327-342`, `:368-379`, `:132-137`, `:152-212`, `:430`, `:36-48`, `:509-547` |
| `vae.py`, `scheduler_step.py`, `self_conditioning.py`, `dsa_indexer.py`, `mtp_head.py`, `per_layer_embedding.py` | ResNet cell authored (`GroupNorm + SiLU`, `Conv 3×3`, `nearest 2× → conv`); four printing defaults (`ε̂`, `noise`, `scale`, `z_t → z_{t-1}`); **`self_conditioning` and `dsa_indexer` graphs are 100 % hard-coded and read zero facts**; MTP and PLE use absolute hand-placed coordinates and a `"RMSNorm"` default label | `vae.py:94-105`; `scheduler_step.py:19-32`; `self_conditioning.py:22-41`; `dsa_indexer.py:24-31`; `mtp_head.py:33-41`; `per_layer_embedding.py:36-53` |
| `registry.py` | `"vision" in marker` / `"audio" in marker` over `id view kind` to pick evidence provenance | `registry.py:95-100` |
| `utils.py` | `_fmt_int(None) → "?"` prints inside otherwise-asserted prose | `utils.py:10` |

**How drill depth is built** (verified): a node is emitted as `<g
class="uf-node" data-id=…>` by four primitives (`svg.py:191-194, 263-266,
325-328`; `unet.py:542-545` has its own); at graph level
`clickable = not node.static`, and `static` is set by the three view entry
points as `bool(block.get("children"))` (`feed_forward.py:37`,
`registry.py:173`, `declared_ops.py:26`) — **a node is clickable iff its block
declared child cards**, and a block gets a rich (SVG) card only if its `view` is
registered (`cards.py:43-71`). Panel depths are `2` then `i+3` without bound
(`document.py:170-176`, `views_diffusion.py:102,130`; `cards.py:217-227` walks
`children` recursively). In practice the corpus reaches depth 3 for dense LLMs,
4 for MLA/MoE and diffusion pipelines. Only L1 nodes get `cursor:pointer` via
CSS (`styles.py:173-175`); nested ones get it via JS (`interactions.py:112-114`).

The `theme` question in §1.6 is answered: the diffusion adapters set
`"theme": "teal"` explicitly (`adapters/diffusor/blocks.py:222,267`,
`unet.py:963`), so `theme.BLUE` (`theme.py:51-68`) is dead code.

### 4.3 What the U4-E/U5/U9 claims removed, and what remains

* Removed (verified): the renderer no longer manufactures cards from spec
  values when no block was authored (`metadata.py:96-105`); FFN inner shape is
  the opgraph's alone (`registry.py:154-163`); op cards derive from regions
  (`labels.py:971-976`); dict-IR twins of grouping route through
  `ir.layer_signature` (`metadata.py:219-221`, `expanded/grouping.py:13-15`).
* Remaining authoring, by weight: (1) the adapter block catalogues (§1.5, ~850
  presentation keys); (2) `labels.py` sentences and `_OP_SENTENCES`; (3)
  opgraph op labels/desc; (4) prose sniffing on `cross_kv_source` in three
  places (`labels.py:787`, `opgraph.py:658`, `blocks/attention.py:239-240`) and
  seven in `diffusor/unet.py`; (5) `_MODALITY_*`-style identity tables in
  `special_parts/modalities/registry.py:81-103` and
  `evidence_projection.py:21-25`; (6) the bare-config text-encoder placeholder
  tower (`diffusor/blocks.py:1247-1262`) and the VAE/scheduler scaffold
  (`:754-968`) whose own note admits they "do not author the denoiser
  structure"; (7) `AttentionSpec.variant` prose inside the IR; (8) the `"type
  A"` fallback (`submodel.py:326`); (9) `block_diffusion_loop_blocks` with
  literal `30.0`, `48`, `0.1` (`blocks/model.py:120,165,268`); (10) inside the
  renderer: the four phantom experts, the 15-cell window strip, the fusion
  token strips, the hard-coded `self_conditioning`/`dsa_indexer` graphs, the
  unknown-kind → norm-box fallthroughs (`graph.py:105`, `op_render.py:287`),
  and the three `or where['pre']` norm-placement narrations
  (`metadata_modalities.py:494,668,897`) — the last three are the only places
  found where an *unknown* still silently draws or narrates a *known* shape.
* Net: the "renderer authoring" the U-units removed was the *card-from-spec*
  manufacturing path. What remains is (a) the entire sentence layer, which was
  never in the renderer to begin with but in the adapter `blocks/` catalogues,
  and (b) a long tail of renderer-invented scaffolding for the non-LLM
  surfaces. Both are outside the evidence gates: no net checks a sentence.

---

## 5. Output formats and integration

* **Jupyter**: `_repr_html_` → `render_fragment` with the `@import` Google-Fonts
  line and the inline click script (`diagram.py:122-124`, `document.py:14-38`);
  54.6 KB for Llama-7B; mount ids are per-Diagram UUIDs so several diagrams
  coexist. Offline notebooks fall back to `"Patrick Hand","Comic Sans MS",
  cursive` (`theme.py:25`).
* **Standalone HTML**: self-contained except the Google-Fonts `<link>`
  (`document.py:229-231`); no CDN scripts; 55 KB (Llama) – 177 KB (FLUX).
  Viewport meta present; `max-width 720/820 px`; no dark theme; no print CSS.
* **JSON**: `save("x.json")` writes `to_json()` (`diagram.py:143-149`);
  `to_ir()` is a separate, larger dict (339 KB for Llama-7B — it carries
  `config_access`/`config_audit`/`fact_provenance`/`source_provenance`, i.e.
  the QA ledgers ride along with the IR). `to_json()` is 20 KB for Llama and
  5 KB for SD3.5.
* **PNG**: `to_png`/`save_images` require `rsvg-convert` on `PATH`
  (`preview.py:58-63`); PNG rendering is not exposed by the Space.
* **Dependencies**: `pyproject.toml` declares `dependencies = []`; `hf` extra =
  `transformers>=4.40`. Model-id strings need `transformers.AutoConfig`
  (`parser.py:511-517`) and the diffusers rung needs `huggingface_hub`
  (`adapters/diffusor/loader.py:27`) — neither is declared; `yaml` is optional
  with a flow-parser fallback (`everchanging/__init__.py:36-38`).
* **Errors**: typed hierarchy (`errors.py`), `MODEL_UNFOLDER_DEBUG=1` prints
  causes; soft problems go to `Diagram.warnings` and the `⚠` bar; the Space
  flattens all of it to one sentence.
* **Version drift**: `__version__ 0.2.15` vs `pyproject 0.2.17`.

---

## 6. Design-debt list for the learner experience

1. **The hero model can be the one thing not drawn.** SD3.5 (blessed) shows
   scheduler, VAE, three encoders and an empty denoiser; FLUX shows a denoiser
   whose every mechanism is unresolved. The honesty program removed fabricated
   internals but did not replace them with proven ones; the product's promise
   ("view any model's internal architecture") now holds for LLMs only. The
   banner (`HIDDEN 0`, `PARAMS ?`, `LAYERS 0`) and an empty layer map (`0
   layers - 0 types`) confirm to the reader that nothing was found.
2. **Journeys end in prose, not in understanding.** Every drill's last floor is
   a description-only card (`24` on Llama, `72` on SD3.5, `79` on DeepSeek).
   The leaves repeat generic sentences (`"Linear projection (a learned weight
   matrix applied to every position)."` ×N). There is no terminal object — no
   shape table, no parameter roll-up, no formula card — that tells the reader
   "you have now seen the whole layer".
3. **Blocks and blocks.** Diffusion pages multiply near-identical panels: two
   CLIP towers with the same unresolved shape; four VAE `Up stage` boxes each
   opening a drill that differs by one number. No bundling ("4 up-stages,
   512→128 ch, 3 ResNets each") exists at the picture level; bundling happens
   only inside the layer-signature collapse of the *transformer* stack.
4. **Honesty chips outnumber facts on well-known models.** T5 attention card:
   `bias unresolved · QK norm unresolved · cache unresolved · position
   application unresolved`; Qwen2-VL: `Feed-forward mechanism unresolved`,
   `Pre-head path unresolved`; Llama-7B: a `⚠` badge about parallel norms. The
   reader cannot tell "the code genuinely does not say" from "the reader was
   not written yet" — both are the same chip.
5. **Prose contradicts pictures.** CLIP attention card says tokens "attend to
   the others in the prompt" beside a box that says the mechanism is
   unresolved (`diffusor/blocks.py:1072` vs `opgraph.py:496-554`). `type A /
   type B` labels the T5 groups without saying what differs.
6. **No quantities on the canvas.** Shapes appear only in leaf chips; per-layer
   parameter counts are computed (`params.py:257`) and never drawn; the `*`
   on the total is explained only by a hover title.
7. **No "why", no narrative order.** Nothing says what a residual is for, why
   the norm precedes attention, what the KV cache saves, or reads the diagram
   in inference order (the loop is drawn bottom-up while cards are listed
   top-down).
8. **Interaction is one-dimensional.** Click-in only; no breadcrumb, no
   keyboard, no deep links, no "expand all"; the layer map lives in a closed
   `<details>` with a different typeface; `⚠`/`ⓘ` bars are the only
   explanation affordance; the diffusion palette is declared but not applied.
9. **The public surfaces are out of step.** Space pins a PyPI release, error
   surface is one sentence, showcase HTML is three months stale, one example
   file is the wrong model, the README's DeepSeek number no longer matches.
10. **The API bundles QA with the product.** `bless`/`sable`/`check_regression`
    sit in `__all__` beside `unfold`; a library user importing `*` gets the
    corpus-mutating harness.

---

## 7. Loose ends

* Not verified: what PyPI `model-unfolder 0.2.17` actually renders for SD3.5/
  FLUX (the Space's behavior), nor the Space's cold latency or Gradio timeout;
  only this tree was run.
* Not rendered: any UNet pipeline, Qwen2-VL, MusicGen (the `views_modalities`/
  `audio` surfaces); the `inspect_code=True` CODE EVIDENCE section was never
  populated in these renders — whether the default `unfold(id)` path attaches
  `code_evidence` for hub models is unknown here.
* `expanded` JSON schema stability has no changelog; whether external
  consumers exist (beyond `tests/test_expanded_json.py`) is unknown.
* The two sub-pass audit tables (§1.5, §4.2) were spot-checked at ~20 cited
  lines (all held); the per-file percentage estimates are the sub-passes'
  estimates, not measured.
* Not answered: whether a well-known LLM's chips such as `QK norm unresolved`
  (Llama) or `Feed-forward mechanism unresolved` (Qwen2-VL, blessed) are reader
  gaps or genuine source ambiguity — that is pass 03's matrix; here they are
  recorded only as what the learner sees.
* The Her-Eyes review files under `galleries/*/her_eyes_review.md` are a
  parallel, human-voice record of the same journey; only Llama's was read.
