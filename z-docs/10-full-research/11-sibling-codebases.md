# 11 — Sibling codebases: the rest of the `Understand/` umbrella

Written 2026-09-02 against the working tree (engine branch `audio-composite-support`,
HEAD `9b3cb7b` 2026-09-01). Read-only pass; every path below is under
`/Users/soumil/Code/Projects/Understand/` unless absolute. Where a statement rests on a
network fetch made today rather than on files, it is marked **[fetched 2026-09-02]**.
Where I could not verify, it is marked **[unverified]**.

## 0. The umbrella at a glance

| dir | what it is | stack | files (excl. deps) | LOC | first → last commit | status | touches the engine? |
|---|---|---|---|---|---|---|---|
| `llmvisualizer/unfold-pkg/` | **the engine** (`model-unfolder`) | Python | 267 `.py` in package | — | 2026-05-06 → 2026-09-01, 487 commits | **live** | is the engine |
| `llmvisualizer/hf/` | Gradio Space `SoumilB7/Unfold` | Python/Gradio | 4 | ~150 | 05-31 → 07-07, 10 commits | frozen, Space **sleeping** | consumes PyPI `0.2.17` |
| `llmvisualizer/done/` | author's catalogue + rendered previews + review reports | md/html/png | 1,225 | — | not git; mtime 07-14 | stale | outputs of the engine |
| `llmvisualizer/finalize.ipynb`, `push.txt` | manual smoke notebook; release recipe | ipynb | 2 | — | 06-29 / 07-07 | stale | calls `unfold()` |
| `diffusiongemma-26B-A4B-it.html`, `sdxl-base.html` (root) | two standalone exports | HTML | 2 | — | 06-14 | stale | engine output |
| `unfold-npm/` | JS port of the engine, `model-unfolder@0.2.11` | ES modules | 147 (122 src modules) | 22,983 | not git; mtimes 06-01 → 06-21 | **dead, un-importable** | ports engine source; nothing consumes it |
| `Flowy/` | **Slive** — macOS hold-to-talk dictation | Swift + Python | ~130 source (36,296 with `.venv`/`.build`) | — | 07-12 → 08-23, 200 commits | live | **no** |
| `Transformers/` | Next.js "Story Cards" attention explainer | Next 16 / React 19 | 43 | 5,264 | 03-13 → 03-15, 16 commits | abandoned | no (predates engine) |
| `Concepts/` | "Concepts Lab" physics simulations | Next 15 / React 19 / framer-motion | 81 | 12,560 | 02-10 → 04-26, 63 commits | dormant; site 404 | no |
| `Neurons/` | "Neurons Lab" / "Cognition" concept explainers | same | 36 | 3,995 | 02-21 → 06-10, 22 commits | dormant; site 404 | no |
| `Gemma/` | Gemma 4 E2B rebuilt from scratch + experiments | PyTorch, notebooks, md | 98 (+10 GB weights) | 7,606 | 04-13 → 05-06, 31 commits | finished/stale | predecessor (same day handoff) |
| `Architecture/` | generic ML training notebooks | ipynb | 17 | 5,939 | not git; 04-04 | stale | no |
| `Dynamic-island/` | empty directory | — | 0 | 0 | 06-05 | nothing | no |

Counts: `find -type f` excluding `node_modules`, `.git`, `.next`; LOC by `wc -l` over js/ts/tsx/py/ipynb/css.
Git ranges from `git log --date=short`. Cross-reference grep: `grep -rIl 'model_unfolder|unfold-pkg|llmvisualizer|unfold\('`
over every sibling hits **only** `unfold-npm/` (8 files); nothing else in the umbrella imports, embeds or links the engine.

## 1. Flowy — actually "Slive", unrelated (one paragraph, per coordinator)

`Flowy/README.md:3-5` is titled **Slive — "Your whisper, truly yours."** with macOS / Swift /
Python 3.12 / Neural-Engine badges; `ARCHITECTURE.md:1-5` describes two processes — a Swift app
owning hotkey → audio → WhisperKit transcription → paste, and a Python backend spawned on
`127.0.0.1:50711`. Source lives in `Frontend/Sources/Slive/` (Dictation, Overlay, Hotkey,
Transcription, Training, Assistant …) and `Backend/src/flowy/`; `z-docs/` covers dictation,
overlay, whisper fine-tuning research, power/process. Remote is `github.com/SoumilB7/Slive.git`;
200 commits 2026-07-12 → 2026-08-23, last "Capture the mic directly". The 36k file count is
`Backend/.venv` (31,094) + `Frontend/.build` (4,921). The grep above finds **zero** references to
`model_unfolder`, `unfold-pkg` or `unfold(` anywhere in it. It is not the product shell for the
engine and reveals nothing about it, except a shared working method: the same `z-docs/` +
`project_context.md` (1,521 lines) + `.claude/` documentation planes the engine uses. Not read further.

## 2. unfold-npm — a JavaScript port of the engine, June 2026, never shipped

### What it is
`package.json:2-4` — name `model-unfolder`, version `0.2.11`, "Unfold any HuggingFace transformer
into an interactive architecture diagram **in JS**"; `type: module`, entry `src/index.js`, only
devDependency `jest`. No README, no git, not on npm (`registry.npmjs.org/model-unfolder` → 404
**[fetched 2026-09-02]**). The GitHub README of the Python repo does not mention a JS port
**[fetched 2026-09-02]**.

`src/index.js:14-26` mirrors the Python API exactly:
`unfold(cfg_or_id, {token, inspect_code, code_source, return_json})` → `config_to_ir` → `new Diagram(ir)`
→ `toJSON()` or the diagram; `version = "0.2.11"` (`:29`). The port is a one-to-one mirror of the
Python package layout as of v0.2.11–v0.2.15: `adapters/{custom,diffusor,transformer}`, `expanded/`,
`evidence/`, `everchanging/`, `renderers/html/{blockViews,modalityViews,…}`, `opgraph.js`, `labels.js`,
`blockSchema.js`, `params.js`, `ir.js`. 122 modules; ~20.6k LOC (`renderers` 10,779, `adapters`
5,003, `evidence` 1,520, `expanded` 969, `everchanging` 641, top-level 1,677).

What makes it more than a renderer port:
- **Browser-targeted loading.** `src/parser.js:79` branches on `typeof window === "undefined"`;
  `:94` fetches `https://huggingface.co/${model_id}/raw/main/config.json`; `evidence/sources.js:187,210`
  fetch the model-info API and raw `modeling_*.py` files from the Hub.
- **A hand-written Python tokenizer/AST scanner in JS** (`src/evidence/astScanner.js:8` `tokenize`,
  triple-quote handling `:47+`) so `inspect_code` works without Python — the same "read the modeling
  file as text, never execute it" stance as the engine.
- **The expanded JSON** (`src/expanded/index.js:10` `SCHEMA_VERSION = "3.0"`; `tests/expandedJson.test.js:150-151`
  asserts `"3.0"` and `format === "model_unfolder.expanded"`).

### How it was ported (the tooling left in the directory)
- `convert_views.py:5-28` — a first regex Python→JS pass, abandoned in-line: `:25` "Wait, this might
  break strings. Skip."; `:27` "I'll just write a basic skeleton and manually do the rest".
- `port_blocks.cjs:4-16` — regex-ports eight `block_views/*.py` (`unet, vae, text_encoder,
  scheduler_step, self_conditioning, moe_router, dsa_indexer, block_facts`) from
  `../unfold-pkg/model_unfolder/renderers/html/block_views` into `src/renderers/html/blockViews`
  ("Ported skeleton to …", `:77`).
- `port_styles.cjs:3,27-38` — lifts the `_style(mount_id)` f-string CSS out of `styles.py` into
  `styles.js` template literals.
- `fix_svg.cjs:8-16` — wraps every `${expr}` inside SVG path templates in `_num()` so JS prints
  `615` where Python printed `615.0` (applied to `viewsDiffusion.js` and `blockViews/unet.js`, `:20-21`).
- **Parity harness.** `dump_diffusion.py:6-21` runs the *Python* `unfold(FLUX)` (the FLUX fixture from
  `tests/test_diffusion`) and writes `test_diffusion_ir.json` (654 KB) plus
  `test_diffusion_python.html` via `render_fragment`; `testDiffusionParity.js:2-21` feeds the same IR
  to the *JS* `render_fragment` and compares after normalising `N.0`→`N`; `diff.js` writes
  `norm_py.html`/`norm_js.html`.

### How close parity got (June 21)
- `diff.txt` (normalised HTML, 214 KB file) has **2 hunks, 6 deletions, 19 additions** — the only
  structural differences are a blank line and whitespace inside the SAMPLING LOOP `<details>`.
  `fmt_diff.txt` (prettified) has 49 hunks, dominated by `d="M 615.0 336 …"` vs `d="M 615 336 …"`
  and tag-wrapping — cosmetic.
- `py_ids.txt` vs `js_ids.txt` (the `data-id` node sets): identical except the attention drill —
  Python emits `q_rope` and `k_rope` nodes, JS does not, and orders `q_proj/k_proj/v_proj` differently.
  So the JS attention drill silently dropped RoPE. `testDiffusionParity.js` prints SUCCESS only on
  byte-identity; the artefacts show it never reached it.
- Important scoping fact: the parity script imports **only** `src/renderers/html/document.js`, so it
  exercised the renderer on a Python-produced IR. The JS *parser* path was never part of parity.

### Current state: it does not load
`src/adapters/diffusor/parser.js:1` and `unet.js:1` import `'../../../ir.js'`, `blocks.js` imports
`'../../../labels.js'` — three levels up from `src/adapters/diffusor/` is the repo root, where no
`ir.js` exists (the correct depth is `../../`). Because `src/adapters/index.js` imports the diffusor
adapter and `src/parser.js`/`src/index.js` import that, **every entry point fails with
`ERR_MODULE_NOT_FOUND`** (verified by importing each of the 122 modules with node 20.20.1: 7 fail,
all on this chain). The jest suites (`tests/smoke.test.js`, `expandedJson.test.js`,
`codeEvidence.test.js`) therefore run **0 tests** ("2 failed, 2 total"). The six diffusor files are
all dated 2026-06-20 — the diffusor adapter was regex-ported that day with the wrong relative depth
and the package was never run end-to-end afterwards.

### Currency with the engine at HEAD
- The port mirrors Python **v0.2.11–v0.2.15** (113 `.py` files at tag `v0.2.15`, 2026-06-20). HEAD has
  **267** `.py` files; **154** are new since v0.2.15 (dominated by `evidence/*` readers —
  `unet_stage_*`, `weight_tying`, `vision`, `wrapper_features` — plus `submodel.py`, `sable.py`,
  `lint.py`, `input_formats.py`, `everchanging/{conformance,evidence}`). None of that exists in JS.
- The JS keeps `renderers/html/blockViews/attentionTypes/` (9 files: groupedQuery, latent, linear,
  multiHead, multiQuery, rwkv, slidingWindow, stateSpace) which Python deleted when attention moved
  onto the op-graph (`git log --diff-filter=D` → `f9c75c7` "attention onto the canonical op-graph:
  one region, two projections").
- Schema: JS `3.0` vs Python HEAD `expanded/__init__.py:43` `SCHEMA_VERSION = "3.2"` (and
  `done/model.json` is `3.1`).
- HTML vocabulary: HEAD's renderer emits classes absent from the June exports (`uf-evidence-section`,
  `uf-msg-bar`, `uf-external-tensor`, `uf-arch-variant-*`, `uf-nested-variant-*`) and has dropped the
  per-card classes the June output used (`uf-card-embed`, `uf-card-ffn`, `uf-card-vae`, …).

### Verdict
Downstream port, **dead**: last touched 2026-06-21, never published, un-importable, three engine
schema/HTML generations behind. What it reveals: in June the author intended a **browser-side
product** — fetch a config from the Hub in the page, parse and render entirely in JS, no Python,
no server. That is the only artefact in the umbrella pointing at a web surface beyond
Jupyter/Gradio, and it was abandoned the week the Sable/Dable honesty campaign started (memory:
"sable dable everywhere" 2026-06-27).

## 3. The Next.js explainer lineage (Feb → Jun 2026)

All three share the same scaffold: Next 15/16 App Router, React 19, Tailwind, framer-motion,
Inter + JetBrains Mono, a strict monochrome `.agents/design-philosophy.md` (Concepts and Neurons
copies differ only in one line about accent colours), and a `.agents/AGENTS.md` project guide.
They are the author's "Lab" product family; `Understand/` is literally their umbrella.

### 3.1 Concepts — "physics-lab" (Feb 10 → Apr 26)
`package.json` name `physics-lab`; home copy "Concepts Lab — Interactive simulations to explore
fundamental concepts. Click on any experiment to start." (`src/app/page.tsx:22-27`). Nine experiments
in `src/lib/experiments.ts:11-80` (pendulums, projectile motion, orbital mechanics, fractals, atoms,
quantum fields, spring-mass live; wave interference, EM field "coming-soon"). Each is a fixed
four-file unit — `simulation.tsx` (canvas + rAF physics), `toolbar.tsx`, `controls.tsx`,
`info-panel.tsx` (`.agents/AGENTS.md` structure block). Deployed at `https://conceptslab.vercel.app`
(`AGENTS.md:3`, `layout.tsx:15`) → **404 today [fetched 2026-09-02]**. Domain unrelated to the
engine. What it contributes: the product *form* — a catalogue grid → one interactive page per item
with play/pause, parameter sliders and a live info panel — and the design language later reused.

### 3.2 Neurons — "neurons-lab" / "Cognition" (Feb 21 → Jun 10)
Home: "Cognition — Explorations and interactive visualizations decoding the mechanics of
intelligence, neural networks, and conceptual representations", section label "Understand"
(`src/app/page.tsx:22-32`). Four scroll-narratives:
- `/neurons` (`concept-flow.tsx`, 783 lines): neurons as concepts — "Click any digit on the right
  to see which concepts in earlier layers it relies on. Watch how specificity builds layer by layer."
- `/compartmentalization` (396 lines): *why* architectures are split — "Attention was the first split
  we made by hand … The MLP stays behind and does what it is actually good at"; "This is what Mixture
  of Experts is. Take the one fat MLP block. Replace it with N smaller MLPs, called experts. For each
  token, only a few of them fire."
- `/compression` (685 lines): "Compression & Intelligence — why making data smaller means predicting
  it … why that model is what we mean by intelligence" (`compression/page.tsx:6-10`).
- `/human-learning` (261 lines): prediction-error learning by analogy to physical intuition.
`src/lib/tree-data.ts:9-56` is a planned knowledge tree — Brain → Neural Networks → **Transformers
(attention, self-attention, positional encoding) → Language Models (tokenization, embeddings,
generation)** — i.e. the syllabus the engine's diagrams would slot into. `AGENTS.md`: "Keep the home
route intentionally blank until design direction is finalized … Avoid introducing structures that
imply card grids or catalog flows". Deployed at `neuronslab.vercel.app` → **404 today [fetched]**.
Last commit 2026-06-10 overlaps the engine's diffusion month, so it was worked in parallel, then
dropped. No unfold embed anywhere.

### 3.3 Transformers — "Story Cards" (Mar 13 → 15; remote `SoumilB7/AI.git`)
Three days of work, 16 commits, abandoned. Home: "Story Cards — We will explain the model one idea
at a time." Chapter 01 Attention, Chapter 02 Tokenization (`app/page.tsx:12-37`); tokenization is a
blank placeholder (`app/tokenization/page.tsx:9-14`). The attention chapter is the richest single
explainer in the umbrella (`app/attention/page.tsx:40-48`: "Context becoming *you*. Not a math
formula. Not arrows flying everywhere. Just what actually happens to a word when a transformer reads
it."):
- a worked sentence, "The old bank by the river collapsed …" (`_lib/content.ts:19-33`), hover a token
  → its vector (`components/elements/diagram.tsx:16-49` `TokenRow`, `MeaningVector`);
- hand-set attention weights and a causal-mask walk-through (`content.ts:35-118`, `CAUSAL_STEP_LABELS`);
- a STAGES story Embedding → Layer 1 → Layer 2–3 → Layer 4+ where "bank" disambiguates
  (`content.ts:120-144`);
- **two registers** for every idea: `FeelReveal` ("feel it", `feel-reveal.tsx:8-13`) opens a
  freeform canvas; `TechnicalReveal` ("see technically", `technical-reveal.tsx:6-9`) discloses the math;
- `components/drawboard/guide.md`: a declarative step-animation engine on a freeform canvas — ten
  action types (`highlight, fade-in, fade-out, move, merge, multiply, split, flow, transform,
  annotate`), play/step timeline, keyboard stepping; `freeform-canvas.tsx` + attention components
  total 2,821 lines.
Not deployed (no URL in tree). Predates the engine's first commit (2026-05-06) by seven weeks.

### 3.4 Gemma — "Understanding Gemma 4 E2B" (Apr 13 → May 6; remote `SoumilB7/Gemma4-barebones.git`)
`README.md:1-3`: "Taking Gemma 4 E2B apart to understand it — not just use it. Two parallel tracks:
rebuilding the architecture from scratch in PyTorch (verified tensor-by-tensor against
HuggingFace), and running experiments." `architecture/*.py` (RMSnorm, attention, ffn, ple, rope,
embedding, transformer_block, text_model, tokenization) with a matching `architecture/docs/*.md`
per brick; `docs/oversimplified.md` gives one or two lines per step ("String → list of integers …",
"stream += attn(norm(stream)) then stream += ffn(norm(stream))"). `experiments/AttentionSinks`
has real results (`README.md:40-60`: BOS carries 16.76 units of residual shift per unit attention;
window-edge token 134× weaker; structural tokens absorb the role). `frontend/` exists but is
**empty** (created Apr 19). `model_weights/model.safetensors` is 10.2 GB on disk; `.gitattributes`
LFS. Commit log: tokenization → embedding → rmsnorm/rope "verified" → attention → ffn → "complete
the architecture" → "full text model" → "model done" (Apr 21) → experiments → "token differences"
(May 6).

Why this repo matters: **it is the engine's direct predecessor.** Its last commit is 2026-05-06,
the engine's first commit (`b8f0c73 initial commit`, `f03c0e1 lisencing`) is 2026-05-06. Its
`docs/overview.md:7-11` — "Numbers below come from Maarten Grootendorst's visual guide … but every
one is double-checked against the actual `config.json` and the safetensors shapes … Where the visual
guide and the real config disagree, we go with the config — and call it out" — is the seed of the
engine's "detect from evidence, never identity" law. Every Gemma mechanism it documents by hand
(5:1 sliding/full interleave, 8:1 GQA, KV sharing across layers 15–34, per-layer embeddings, p-RoPE,
logit softcap, 2× FFN in shared-KV layers; `README.md:16-33`) is a mechanism the engine later
detects generically (`per_layer_embedding` block view, partial rotary, sliding window, cross-layer
KV edges — `ir.js` `CrossLayerEdge` in the JS mirror). Gemma is one model done by hand with
tensor-level verification; the engine is "any model" done from config + source with no tensors.
What the engine dropped from Gemma: activation-level *behaviour* (the experiments) and the
"oversimplified" plain-language register.

### 3.5 Architecture/ and Dynamic-island/
`Architecture/` (not git, Apr 4): `Ovasense.ipynb` (ResNet continual training on an ovarian-cancer
biomarker dataset), `Decision_Tree_vs_XGBoost_Complete_Evaluation.ipynb` (telecom churn),
`TrainAnything/` (ANN/CNN/RNN/Transformers/text training notebooks; README "your launchpad to get
you cracking with deep neural networks"). Generic teaching notebooks; no relation to the engine.
`Dynamic-island/` is an empty directory (Jun 5).

### 3.6 What the lineage says the engine absorbed vs dropped
| explainer idea | where it lived | in the engine today? |
|---|---|---|
| card grid → per-item interactive page | Concepts, Neurons | no shell; the HF Space is one textbox |
| scroll narrative, one idea at a time ("Story Cards", "Chapter 01") | Transformers, Neurons | dropped — no journey, no chapters |
| worked example flowing through the model (a sentence, a token's vector) | Transformers | dropped — structure only, no data |
| two registers: "feel it" vs "see technically" | Transformers | half — the drill (click a block → detail card) is "see technically"; there is no "feel it" |
| step animation (merge, multiply, flow) | Transformers drawboard | dropped — the export's script only toggles panels (`sdxl-base.html` inline script: `showPanel`, `setPanelSize`) |
| the *why* of a design (why attention/MLP split, why MoE) | Neurons compartmentalization | dropped — cards state *what*, not *why* |
| one line per brick in plain words | Gemma `oversimplified.md` | partly — `uf-card-desc` per card (89 on the Gemma export), but auditor-register |
| numbers checked against config/safetensors, never trusted from a guide | Gemma `overview.md:7-11` | **absorbed as the core law** |
| per-model hand-written mechanism docs | Gemma `architecture/docs` | absorbed as generic detection; per-model docs gone |
| sketch/hand-drawn aesthetic (SketchBox, CanvasSheet) | Transformers | kept — `uf-name` in Caveat cursive, hand-drawn card borders |
| activation-level experiments | Gemma | dropped — static structure only |

## 4. The engine's own surfaces inside `llmvisualizer/`

### 4.1 `hf/` — the Gradio Space (the only public UI)
Remote `https://huggingface.co/spaces/SoumilB7/Unfold`; 10 commits 05-31 "initial commit" → 07-07
"version update". `README.md:1-12` front-matter: `sdk: gradio 6.15.2`, `python 3.13`, short
description "View any model's internal architecture just by HF tag"; `:14` "Library that runs this
space: github.com/SoumilB7/unfold". `requirements.txt:1` pins **`model-unfolder[hf]==0.2.17`**.
`app.py`: one textbox (default `Qwen/Qwen2.5-0.5B-Instruct`, `:9`), optional token, button
"Unfold"; `render_model` calls `unfold(model_id, token=token)` and drops
`diagram.to_html(standalone=True)` into a sandboxed `<iframe srcdoc>` (`:80-86`, `:114-120`); a status
line from `to_ir()['params']` and `diagram.warnings` (`:89-107`); **every exception collapses to
"That model card isn't accessible."** (`:122-123`). Tagline in-page: "Generated purely from the
Hugging Face config map. No model weights are downloaded." (`:129`).
Status: the Space page reports **"Sleeping (due to inactivity)"** **[fetched 2026-09-02]**. (The
fetch summary also claimed the Space is built on an "Agent SDK"; that contradicts `hf/README.md:6`
`sdk: gradio` and is treated as summariser noise.) PyPI latest is 0.2.17 uploaded July 6
**[fetched]**; local HEAD is **254 commits past `v0.2.17`**, and the public `origin/main`
(`a6c405f` 2026-07-07 "polishing like a diamond", 234 commits) is **253 commits behind** the working
branch. So the only public UI runs the pre-honesty-campaign engine, and the API it uses
(`to_html(standalone=True)`, `.warnings`, `.to_ir()`) still exists at HEAD (`diagram.py:70,29,126`).

### 4.2 The two root exports
`diffusiongemma-26B-A4B-it.html` and `sdxl-base.html` (both 2026-06-14, ~174 KB) are
`to_html(standalone=True)` documents: `<meta name="generator" content="Unfold">`, `.uf-frame`
wrapper — identical to what HEAD's `renderers/html/document.py:227-238` still emits; the only
external dependency is Google Fonts *Caveat*. Content: 134 / 85 `uf-node`s, 89 / 54 card
title+desc pairs, 112 / 63 `uf-fact` chips, 16 / 25 card SVGs, one collapsible `<details>` section
each. The inline script wires node clicks to nested inspect panels (`sourcePanelFor`, `showPanel`,
`clearPanelsFrom`, six `uf-panel-*` sizes) — open/close only, nothing computed client-side. They
correspond to `finalize.ipynb` cells 5–6. These files are what "the product" looked like the day
before the JS port and the Sable campaign began; they are the artefact a person would actually
share.

### 4.3 `done/` — the author's driving seat (stale since Jul 14)
- `TO_SERVE.md` (447 lines): "The Big Grill — Open-Weight LLMs & Diffusion Models by Family",
  five parts (text LLMs, VLMs, image, video, audio) + an exclusions section (SSM, RWKV, JEPA,
  CNN/LSTM, encoder-only — `:6`), 38 family subsections. This is the target universe. It differs
  from `unfold-pkg/previews/old/toserve_model.md` (299 lines, which opens with a note wanting "a
  verification skill … at the architecture creation part").
- `forced/CLAUDE.md` (318 lines): the 2026-06-20-era working nature — "honest, interactive
  architecture diagram", "The generated HTML is the ground truth … the JS only opens/closes", the
  Gate 0 / Gate A "Sable" procedure (render preview first, decompose, map to existing, is it truly
  dynamic, extract shared, end-to-end accuracy). `forced/previews/u2_readers_2026-07-11`,
  `u2_default_kill_2026-07-11` (67 files).
- `previews/previews/`: `new/` (7 phone screenshots), `individual_images/` (14 model dirs),
  `v1-done-ignore/` (June HTML previews — DeepSeek-V3/V3.2-Exp, Llama-3.2-11B-Vision, flux1-dev,
  gemma-3-27b, gpt-oss-20b, pixart-sigma, sd35-large, sdxl; `sable_agents/` reports;
  `sable_audit_2026-06-25/` with `_MASTER_FINDINGS.md`; **`serve/` = 88 rendered HTML files named
  `org__model.html`** — the largest gallery of engine output outside `unfold-pkg`).
- `model.json` (Jun 10): gpt-oss-120b expanded JSON, `schema_version 3.1`.
- `tryrun.ipynb` (May 31): cells 0–4 use `from unfold import unfold`, cells 7+ `from model_unfolder
  import unfold` — the rename is recorded here. 982 PNGs in `done/` are pixel galleries.

### 4.4 `finalize.ipynb` and `push.txt`
`finalize.ipynb` (3.7 MB, Jun 29): cell 0 `pip install -e unfold-pkg`, then 18 distinct
`unfold("…")` targets (Wan2.2-TI2V-5B, HunyuanVideo, gemma-2-9b-it, recurrentgemma-9b,
diffusiongemma-26B, SDXL, gemma-4-e4b, ideogram-4, DeepSeek-V3.1/V4-Flash/R1, sarvam-30b,
gpt-neox-20b, CodeLlama, Llama-3.2-11B-Vision, Qwen2-VL-7B, gpt-oss-120b, pixtral-12b) and a
`return_json`/`to_json`/`save` demo (cell 19). `push.txt`: the 0.2.17 release recipe and a
trailing "# Vonxtral left". Together they are the author's manual acceptance test; nothing in the
tree automates them.

## 5. Synthesis

### 5.1 The topology as it actually exists
```
                       Understand/  (the umbrella; name = the intent)
                             │
   ┌─────────────────────────┼───────────────────────────────┐
   │ explainer lineage       │ THE ENGINE                     │ unrelated
   │ (dead / dormant)        │ llmvisualizer/unfold-pkg       │ Flowy = Slive (live, voice)
   │ Concepts  (physics)     │ Python, live, 254 commits      │ Architecture/ (notebooks)
   │ Neurons   (cognition)   │ past last release              │ Dynamic-island/ (empty)
   │ Transformers (attention)│        │                       │
   │ Gemma (one model, hand) │        ▼  surfaces             │
   └────────────┬────────────┤  Jupyter _repr_html_  (README promise)
                │            │  standalone HTML  → root exports, examples/, done/serve (88)
   nothing links these       │  JSON schema 3.2  → done/model.json (3.1)
   to the engine             │  Gradio Space     → pinned 0.2.17, sleeping
                             │  JS port          → unfold-npm 0.2.11, unimportable, unpublished
                             └──────────────────────────────────────────────────
```
Every arrow that leaves the engine ends in a file, an iframe, or a dead directory. No application
in the umbrella consumes engine output; no engine surface carries narrative.

### 5.2 Live vs dead
- **Live**: the engine (commits through 2026-09-01); Slive (2026-08-23, different product).
- **Frozen but standing**: PyPI 0.2.17 (Jul 6), GitHub `main` (Jul 7), the Space (Jul 7, sleeping).
- **Dead**: `unfold-npm` (Jun 21, cannot import), `done/` (Jul 14), the two root exports (Jun 14),
  Transformers (Mar 15), Concepts (Apr 26; site 404), Neurons (Jun 10; site 404), Gemma (May 6, its
  purpose fulfilled), Architecture (Apr 4), Dynamic-island (empty).

### 5.3 The intended user experience, reconstructed across the pieces
Read chronologically the umbrella tells one story. Feb–Apr: "Lab" sites where a visitor picks a
card and *plays* with a simulation (Concepts), then scroll-narratives that explain *why* networks
are shaped as they are (Neurons: attention/MLP split, MoE, compression). March: a three-day attempt
to give the attention block the same treatment — a sentence, a token that "has no idea what it is",
"feel it" / "see technically" (Transformers). April: doing one real model by hand with every number
checked against `config.json` and tensor shapes (Gemma). May 6: the same day Gemma ends, the engine
starts — "any model, from config, one click". June: two attempts to *surface* it (the Gradio Space
May 31; the JS browser port Jun 1–21) and then the turn to honesty (Sable, Jun 20–27), after which
every surface froze while the engine grew 254 commits and 154 files.

The user the lineage keeps describing is the **learner**: someone who scrolls, plays, and is told
why. The engine kept the one register that lineage called "see technically" — structure, drills,
facts — and dropped every device the lineage had built for "feel it": the journey, the worked
example, the animation, the why. `00-goal-and-intent.md` reaches the same conclusion from inside the
engine ("The code serves U3; the README and the Space sell U2; the user is asking for U1"); the
siblings show U1 was not an aspiration but a **built and then abandoned** body of work — Neurons'
compartmentalization page already explains MoE in the plain register an unfold MoE card lacks.

### 5.4 Implications for the goal statement
"An interactive diagram a person can learn from — never guessing, never silently omitting."
- The **diagram** half lives only in the engine; its honesty half is the one thing the whole
  lineage converged on (Gemma's "go with the config" → evidence-never-identity).
- The **learn-from** half has no home. The only public surface is a sleeping Gradio textbox that
  flattens every error and runs a two-month-old engine; the browser-native surface the author
  actually started (unfold-npm) is dead at an import path; the narrative/"why" layer exists only in
  Next.js repos nobody linked to the engine.
- Product-shell question answered: **there is no product shell for the engine in the umbrella.**
  Flowy is not it. The closest existing form is the "Lab" scaffold (card grid → interactive page
  with toolbar/controls/info-panel, `Concepts/.agents/AGENTS.md`), which was designed for exactly
  this kind of content and could host `to_html()` output, but was never pointed at it.
- The JS port proves the author's intended endpoint was **in-browser, no Python, config fetched
  from the Hub**; and it proves the renderer can be ported nearly byte-for-byte (2 diff hunks) — the
  cost is in keeping a parser/evidence mirror current, which at 267 files and a 92k-line
  `evidence/` package is now implausible. A JS *renderer* over engine-produced JSON/IR is the viable
  shape; a JS *engine* is not.
- Two catalogues (`done/TO_SERVE.md` and `previews/old/toserve_model.md`) and one 88-file gallery
  (`done/…/serve/`) show the intended coverage — every named open-weight family — while the sable
  corpus locks ~29–42 witnesses; the gap between the served catalogue and the locked corpus is the
  coverage the author calls "absolute".

## Loose ends
1. **Secrets.** A Hugging Face token literal is present in plaintext in `finalize.ipynb` cell 1 and
   `done/tryrun.ipynb` cell 18 (and therefore in any tool logs that printed those cells). Not copied
   here; it should be revoked/rotated and the cells scrubbed.
2. `unfold-npm`'s broken import chain (`src/adapters/diffusor/{parser,unet,blocks}.js:1`) means the
   parser/adapters path was never executed after 2026-06-20; whether the jest suites ever passed after
   June 1 is **[unverified]** — the parity artefacts only prove the renderer.
3. Whether the Space actually boots when woken (gradio 6.15.2 + `model-unfolder[hf]==0.2.17` on
   Python 3.13) — **[unverified]**; only its "Sleeping" status was observed.
4. `conceptslab.vercel.app` / `neuronslab.vercel.app` returned 404 — could be Vercel bot-blocking
   rather than deleted deployments **[unverified]**; the repos still declare those URLs.
5. `Flowy/project_context.md` (1,521 lines), `z-docs/`, Backend/Frontend internals not read beyond
   headers, per the coordinator's scope correction.
6. Gemma `README.md` links `experiments/AttentionSinks/result.md`, which does not exist in the tree
   (only `outputs/sliding_attention_report.md` and `outputs/smoke/*`); minor.
7. The `unfold` → `model_unfolder` package rename date (visible only as `tryrun.ipynb` cells 0–4 vs
   7+) is not pinned to a commit here.
8. Remote repos may hold more than the local clones (e.g. `SoumilB7/AI.git` for Transformers,
   `SoumilB7/Neurons.git` after 06-10); only local trees were read.
9. `Frontend/` in `Gemma/` is empty — whether a Gemma-specific web explainer was planned there is
   **[unverified]**; nothing in the Gemma README mentions a frontend.
10. The `done/` PNG galleries (982 files) and `individual_images/` were not opened; their relation
    to `tests/sable_test_corpus/galleries/` (the blessed galleries) is assumed superseded, not checked.
