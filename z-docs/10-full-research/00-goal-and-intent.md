# 00 — Goal and intent: what this is supposed to be, for whom, with what promises

Written 2026-09-02 against the working tree at `a00ae48`+ (branch
`audio-composite-support`; `git status` shows 4 modified/deleted files under
`unfold-pkg/`). Read-only pass; the only file written is this one. Every claim
cites `file:line`; anything I could not verify is marked **[unverified]**. One
timing/param probe was run from the scratchpad against three corpus fixtures
(§3, promise 9) and nothing else was executed.

Sources read in full: `unfold-pkg/README.md`, `pyproject.toml`,
`model_unfolder/__init__.py`, `z-docs/00-start-here/*`, `z-docs/01-product/*`,
`z-docs/07-current-state/product-direction.md` (+ README, delivery-path,
implementation-state), `z-docs/08-reference/glossary.md`, `.claude/PROTOCOL.md`,
`unfold-pkg/docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` §1/§3/§14/§21/§22,
`unfold-pkg/docs/TRUE_CONFIG_PLAN.md`, the four catalogues
(`unfold-pkg/previews/old/toserve{,_model,_layers}.md`, `done/TO_SERVE.md`,
`z-docs/stale/TO_SERVE2.md`), `z-docs/stale/PROJECT_CONTEXT.md` Parts 0–3/8/9/14
+ all `z-docs/stale/*` headers, `z-docs/09-unit-verdicts/{README,first-principles-judgment,method-and-plan-judgment,systemic-findings,experimental-confirmation,recall-trajectory,cross-unit-notes,u00}.md`,
`hf/README.md`, `hf/app.py`, `hf/requirements.txt`, `finalize.ipynb`,
`push.txt`, `done/model.json`, `unfold-pkg/examples/`, and the user's own
messages in the session transcript (`~/.claude/projects/…/a92a5cee-….jsonl`).

---

## 1. The one-paragraph goal

**In the project's own words** (chronological, each still live in the tree):

- README, the public tagline: *"your one click model unfolder"* — `unfold("meta-llama/Meta-Llama-3-8B")` (`unfold-pkg/README.md:3,8-10`).
- pyproject: *"Unfold any HuggingFace transformer into an interactive architecture diagram, inline in Jupyter."* (`unfold-pkg/pyproject.toml:8`).
- Package docstring: *"turn any HuggingFace transformer into a clear architecture diagram"* (`unfold-pkg/model_unfolder/__init__.py:1`).
- PROTOCOL: *"it turns a HuggingFace config into an honest, interactive architecture diagram"* (`.claude/PROTOCOL.md:3-4`); Her Eyes' charter: *"honest diagrams a NEWCOMER can genuinely learn an architecture from"* (`.claude/PROTOCOL.md:365-366`).
- PROJECT_CONTEXT Part 1 (July 5, stale but the fullest statement): *"hand it any HuggingFace model (an id, a config dict, a pipeline repo) and it produces an honest, interactive architecture diagram — every block, arrow, repeat, connector and split the model actually computes, drillable to the leaves, understandable by a newcomer, and never fabricated."* (`z-docs/stale/PROJECT_CONTEXT.md:155-159`).
- Master plan §1: *"The project is a truth machine about architecture."* (`unfold-pkg/docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:26`).
- z-docs (July 14, current authority): *"Model Unfolder is a source-grounded architecture explainer. It supports a bounded, measured set of common mechanisms and renders unresolved mechanisms honestly instead of guessing a familiar structure."* (`z-docs/README.md:9-11`); *"an evidence compiler whose human-facing output is an interactive architecture explanation"* (`z-docs/00-start-here/README.md:8-9`).
- Hugging Face Space: *"View any model's internal architecture just by HF tag"* (`hf/README.md:12`).

**In plain words.** Give it the name of a model on Hugging Face and it draws, without downloading weights or running the model, a clickable diagram of everything that model computes — every block, wire, loop and merge, down to the leaves — such that a person who has never read the code can learn the architecture from the picture, and such that nothing in the picture is invented: what the evidence does not prove is shown as unknown, not filled in from habit.

The sentence the README sells ("one click", "any transformer") and the sentence the internals enforce ("truth machine", "bounded, measured set") are the two poles every conflict below sits between.

---

## 2. Who the user is

Nobody wrote a user down. The audit says so twice (*"Nobody wrote down who the user is"* `z-docs/09-unit-verdicts/first-principles-judgment.md:135`; *"He never wrote the product spec"* `method-and-plan-judgment.md:207`). The sources imply six candidates:

| # | candidate | evidence | served by the code today? |
|---|---|---|---|
| U1 | **The newcomer / learner** who wants to understand an architecture without reading code | PROTOCOL Her Eyes: *"a NEWCOMER can genuinely learn an architecture from"* (`.claude/PROTOCOL.md:365-366`), JOURNEY question *"Name where a newcomer's journey should END"* (`:389-391`), design habit *"someone who is trying to understand the architecture … be as intuitive as possible"* (`:447`); PROJECT_CONTEXT *"A newcomer will trust this diagram — that is the entire point of it"* (`z-docs/stale/PROJECT_CONTEXT.md:141-142`), *"understandable by a newcomer"* (`:158`) | Partly. The HTML is drillable and clickable (`unfold-pkg/model_unfolder/diagram.py:3-5,122`), but the audit measures that flagship diffusion models render as blank boxes and provable negatives render as "unknown" chips (`first-principles-judgment.md:78-82,94-100`). Her Eyes — the only learner-facing review — lapsed 2026-08-06 (`method-and-plan-judgment.md:184-186`). |
| U2 | **The notebook / Space user** who wants a picture in one call | README `unfold("…")` (`README.md:8-10`); pyproject *"inline in Jupyter"* (`pyproject.toml:8`); `_repr_html_` (`diagram.py:122-124`); Space: *"Paste a Hugging Face model repo ID … and render"* (`hf/README.md:16`), default model `Qwen/Qwen2.5-0.5B-Instruct` (`hf/app.py:9`) | Yes, mechanically. But the Space swallows every typed refusal into one string, *"That model card isn't accessible."* (`hf/app.py:122-123`), discarding the ten-category actionable errors the loader produces (`z-docs/stale/PROJECT_CONTEXT.md:207-210`); and cold latency is 6–38 s from a local config dict (§3 promise 9), not "one click". |
| U3 | **The auditor / verifier** who wants to know *why* a box is there | Product contract *"Every visible structural claim should be traceable to an exact fact or geometry target"* (`z-docs/01-product/product-contract.md:33-34`); master plan §22.8 *"How Soumil can identify mistakes after the transition … `diagram.explain(node_or_fact_path)` … a 'Why is this here?' evidence drawer from every structural block/chip"* (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:2879-2906`); `inspect_code=True` attaches source evidence (`__init__.py:74-77`) | This is who the internals serve best: typed facts, provenance chips, receipts, 26+ blocking nets. `diagram.explain()` / `audit()` themselves are **not built** (grep of `diagram.py` public methods: `to_ir, to_json, param_count, save_images, _repr_html_, save` — `diagram.py:29-137`) **[explain/audit absence verified by grep only]**. |
| U4 | **The researcher comparing models** ("what does Kimi do that DeepSeek doesn't") | Catalogue arch-signature columns *"the dispatch axes: attention · FFN · positional · specials"* (`unfold-pkg/previews/old/toserve.md:16-17`); the user: *"the newest Kimi and Quen models have multi-layer communication … Will it be able to handle those things"* (transcript, user msg 2026-08-16T20:21) | Not addressed: no cross-model view, no diff, no side-by-side exists in the public API (`__init__.py:29-51`). |
| U5 | **A downstream program** consuming the structure | `return_json=True` *"stable structural fields … instead of renderer labels"* (`__init__.py:80-84`), `save("model.json")` (`README.md:74`), `done/model.json` schema 3.1 (`done/model.json` keys `schema_version, format, model, dimensions, parameters, io, stack, layer_groups`), `true_config()` planned (`z-docs/01-product/true-config.md:8-10`) | JSON: yes. true_config: `#TODO — not implemented` (`true-config.md:3-4`; `TRUE_CONFIG_PLAN.md:3-4`). |
| U6 | **The author himself**, whose "design" the picture is | *"as long as I get the structure out … it's my work it's my design on how I want to show it I don't care about the other part"* (transcript, user msg 2026-08-16T20:24); `finalize.ipynb` is his hand try-run of 18 distinct model IDs (`finalize.ipynb` cells; list in §5) | The whole verification apparatus (Sable/Dable/Her Eyes, bless galleries) is built for this reviewer: *"The one thing the HTML can't show me is rendered pixels … those I surface to Soumil"* (`.claude/PROTOCOL.md:41-43`). |

**Where the sources conflict.** U1 and U3 pull in opposite directions and both are declared. PROTOCOL Gate C says over-blocking is *"the disease"* and a block must be *"a thing a researcher would draw on a whiteboard"* (`.claude/PROTOCOL.md:166-173`); the doctrine's priority order puts *Completeness* fifth and *Visual polish* sixth, after four honesty/consistency items (`z-docs/00-start-here/project-doctrine.md:82-91`). The user's own re-centering on 2026-08-16 — *"or else I'll just be showing blocks and blocks and connections and connections and nobody would actually give a fuck on why exactly do I want to see this why don't read the code"* (transcript 2026-08-16T20:24) — is a U1 statement. His 2026-09-01 statements — *"i want absolute coverage"* (07:56), *"but atleast the main structure which my library promises will be rendered right ?"* (11:08), *"by main structure i mean whats included"* (11:53), and, on being told an FFN would show a grey "mechanism unresolved" chip, *"but we need that we cant not have that"* (12:03) — are U6/U1 statements that reject the honesty-doctrine's tolerance for grey boxes on structure. The current z-docs never mention a learner, a newcomer, a notebook, or a journey (`z-docs/01-product/*`, `z-docs/07-current-state/product-direction.md` — verified by reading; the words do not occur). **The code serves U3; the README and the Space sell U2; the user is asking for U1 with U6's authority.**

---

## 3. The promises

Status key: **kept** / **partly** / **not yet** / **contradicted** (by the tree).

| # | promise | source | status | evidence |
|---|---|---|---|---|
| 1 | "one click" from a model ID | `unfold-pkg/README.md:3,8-10`; `hf/README.md:12` | **partly** | The call is one line; the wait is not (promise 9). Space is one click but hides all error detail (`hf/app.py:122-123`). |
| 2 | "only config.json is downloaded, never weights" | `README.md:35`; Space: *"No model weights are downloaded"* (`hf/app.py:129`) | **kept by default; conditional otherwise** | Default path downloads `config.json` (`unfold-pkg/model_unfolder/parser.py:754`) or falls back to `params.json` (`parser.py:823`); default `code_source="local"` reads *installed* transformers/diffusers source (`__init__.py:59,78-79`; `parser.py:55-57`). `code_source="hub"` downloads `*.py` + `config.json` with weights explicitly excluded (`evidence/sources.py:737-741`). No path loads tensors. |
| 3 | "no transformers install needed" for a raw dict; `dependencies = []` | `README.md:43`; `pyproject.toml:20` | **kept (dict path) [partly unverified]** | All third-party imports are lazy and inside functions (`parser.py:511,748`; `evidence/sources.py:112,436,729`; `everchanging/__init__.py:36` *"import yaml  # optional; not a hard dependency"*). My probe ran from a dict, but on a machine with transformers 5.12.1 installed, so "no transformers" was not exercised. Whether the source-evidence stream degrades honestly with no transformers installed is **[unverified]**. |
| 4 | "inline in Jupyter" | `pyproject.toml:8`; `__init__.py:3-6` | **kept** | `Diagram._repr_html_` (`diagram.py:3-5,122-124`). |
| 5 | Built on `transformers.AutoConfig`, retry with `trust_remote_code=True` only when required; remote code is read, never executed | `README.md:48-50`; `product-contract.md:8` *"model code is not executed"* | **kept** | `parser.py:511-528,637-651` (explicit `trust_remote_code=False` first); `__init__.py:75-77` *"parses modeling files as text/AST and does not execute model code"*. The audit argues this taboo is the wrong physics for the *structure* question (`first-principles-judgment.md:37-70`) and the master plan itself allows an opt-in runtime probe (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:2822-2838`). |
| 6 | Save `.html` / `.json`, `param_count()`, `to_ir()` | `README.md:71-77` | **kept** | `diagram.py:29,49,65,137`. |
| 7 | Param estimates "close to published": DeepSeek-V3 `~675B (~41B active)`, Llama-3-8B `8.03B` | `README.md:79` | **partly / README numbers stale** | Probe on the corpus config: DeepSeek-V3 total **669.8B / active 36.4B**; llama-7b **6.738B** (published 6.74B). Published DeepSeek-V3 is 671B/37B; the README's own 675B/41B matches neither the tool nor the paper. PixArt-Σ: `total=None` — no estimate at all for that diffusion fixture. |
| 8 | Supported model table: DeepSeek … MiniMax, **Jamba, Zamba, Mamba, Falcon-H1, RWKV** | `README.md:85-98` | **contradicted (scope)** | PROTOCOL: *"Scope: LLMs and diffusion only (no SSM/Mamba)"* (`.claude/PROTOCOL.md:438`); catalogue: *"Out of scope: pure SSM/Mamba and RWKV token mixers"* (`toserve.md:14-15`); PROJECT_CONTEXT: *"the biggest remaining honesty debt"* (`PROJECT_CONTEXT.md:189-193`). Yet the typing vocabulary still names `ssm`, `recurrent`, `rwkv` mixers (`everchanging/transformer/typing.yaml:55-57`) and the author tries `google/recurrentgemma-9b` in `finalize.ipynb`. Whether these draw today is **[unverified]**. |
| 9 | Latency implied by "one click" | `README.md:3` | **contradicted** | Probe (scratchpad, local dict, transformers 5.12.1 / diffusers 0.38.0, no network): llama-7b `unfold` **21.8 s**, deepseek-v3 **6.1 s**, pixart-sigma **37.5 s**; `to_html` ≤ 0.05 s. July's own number was *"unfold ~1.3s"* for deepseek-v3 (`PROJECT_CONTEXT.md:1964`). The audit's *"30–100 s cold"* (`first-principles-judgment.md:66`) is the same order. Nothing is cached across processes (`first-principles-judgment.md:141-142`) **[cache absence not independently verified]**. |
| 10 | "Diffusors — Coming soon." | `README.md:100-102` (unchanged since `87cf965` 2026-05-10; README last touched `56ca2fc` 2026-05-17) | **contradicted (stale, in the good direction)** | A full diffusor adapter exists (`model_unfolder/adapters/diffusor/{parser,unet,blocks,compound,…}.py`), seven diffusion examples are published (`unfold-pkg/examples/flux-1-dev.html`, `unet-*.html`, dated 2026-06-07/10), and 15 diffusion witnesses are in the blessed corpus (`tests/sable_test_corpus/`). |
| 11 | Version consistency | `pyproject.toml:7` = `0.2.17`; `__init__.py:27` = `0.2.15`; Space pins `model-unfolder[hf]==0.2.17` (`hf/requirements.txt:1`); tags end at `v0.2.17` | **contradicted** | Admitted in `z-docs/08-reference/verification-ledger.md:40` (*"incorrect; separate packaging fix required"*) and `PROJECT_CONTEXT.md:1965` (which already listed a three-way mismatch in July). Last version bump `55513f6` 2026-07-07; ~280 commits since with no release. |
| 12 | Honesty: "never fabricated"; unknown stays unknown; no identity branching | `PROJECT_CONTEXT.md:159,161-166`; `project-doctrine.md:77-80`; `.claude/PROTOCOL.md:10-20`; master plan invariant (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:32-36`) | **kept (the audit's strongest finding)** | *"42 rendered models, one wrong proven fact. No family branch anywhere."* (`first-principles-judgment.md:146-149`); the one wrong fact is DeepSeek's yarn `mscale²` scale (`systemic-findings.md:97-99`). |
| 13 | Completeness: "every block, arrow, repeat, connector and split the model actually computes, drillable to the leaves" | `PROJECT_CONTEXT.md:156-158`; Gate A.5 *"complete to the very end"* (`.claude/PROTOCOL.md:85-90`) | **partly — regressed** | Measured: transformer decoder core broad and correct; out-of-corpus transformers 162/216 ≈ 75 % of provable facts proven; diffusion denoisers 37/111 ≈ 33 %; PixArt 0/9, SD3.5 0/7 (`experimental-confirmation.md:70-78`). Drawn-view counts fell in U6/U8/U9/U10 (`recall-trajectory.md:15-40`). |
| 14 | Generality: "a new model never requires a family table or renderer branch"; "solve once → correct for every future model" | `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:35-36,98-99`; `.claude/PROTOCOL.md:13-15` | **kept as a property; not as a measurement** | Identity guard and debt register block per-model rows (`systemic-findings.md:18-22`). No "cost of an unseen family" number has been recorded since July's *"EXAONE = 2 findings (was ~30) · target 0"* (`PROJECT_CONTEXT.md:1963`). |
| 15 | Symmetric trust: "every evidenced fact owes a projection, and every projected claim owes evidence" | `z-docs/01-product/next-product-architecture.md:67-71` | **half** | The projected→evidence half is gated (receipts, conformance); the evidenced→projected half is not: *"Zero [nets] ask 'did we lose something we had proven?'"* (`first-principles-judgment.md:91-93`); proven facts not drawn (Qwen2-VL FFN, MusicGen T5) (`systemic-findings.md:100-103`). |
| 16 | "Custom — Drop a request in issues." | `README.md:104-106` | **not yet** | `adapters/custom/` is an empty stub (`adapters/custom/__init__.py` only). |
| 17 | `true_config()` — "the config.json this checkpoint should have shipped" | `z-docs/01-product/true-config.md:8-10`; `TRUE_CONFIG_PLAN.md:19-21` | **not yet** | Both documents open with `#TODO — not implemented` (`true-config.md:3-4`; `TRUE_CONFIG_PLAN.md:3-4`). |
| 18 | "Soumil commits; Claude never runs `git commit`" | `.claude/PROTOCOL.md:437`; `PROJECT_CONTEXT.md:181` | **contradicted (by explicit later ruling)** | 487 commits, one author identity (`first-principles-judgment.md:112`); memory `feedback-user-commits-himself` records the 2026-07-14 flip to "commit per reviewed unit and push". |

---

## 4. Non-goals and boundaries

Declared, with source:

1. **Bounded-product law.** *"The product is not a universal interpreter for arbitrary Python. Completion is measured against an explicit set of supported upstream versions and mechanism classes. A genuinely new mechanism may remain opaque with an actionable explanation."* (`z-docs/00-start-here/project-doctrine.md:93-98`). Repeated as *"It does not promise to understand every arbitrary repository"* (`next-product-architecture.md:13-16`) and *"Completion is not defined as understanding every repository on the Hub"* (`finish-line-and-scaling.md:19-22`).
2. **Scope line: decoder-only LLMs (+ multimodal towers) and diffusion (DiT/MMDiT/UNet/VAE); pure SSM/Mamba and RWKV out** (`PROJECT_CONTEXT.md:189-193`; `toserve.md:13-15`; `.claude/PROTOCOL.md:438`; `done/TO_SERVE.md:6,443`). Audio-gen was added to scope on 2026-07-07 (`PROJECT_CONTEXT.md:2236`; commit `55bd81a` 2026-07-08).
3. **Explicit non-goals list** (`z-docs/07-current-state/product-direction.md:42-48`): *"perfect interpretation of arbitrary remote Python; zero config usage; a diagram for every repository at any cost; matching a familiar architecture when evidence is missing; keeping every historical check, ledger, or compatibility API permanently."*
4. **Remote code is read, never executed** (`product-contract.md:8`; `__init__.py:75-77`); a runtime probe is allowed only opt-in, sandboxed, non-authoritative (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:2822-2838`).
5. **Identity is an address, never a fact** (`project-doctrine.md:69-73`; `.claude/PROTOCOL.md:17-20`; master plan §1.2 roles table `:59-66`).
6. **Config is checkpoint value, code is mechanism; neither is eliminated** (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:38-44`; `PROJECT_CONTEXT.md:265-274`).
7. **Deletion law / growth control**: *"Sustained production growth without deletion is a stop signal"* (`finish-line-and-scaling.md:78-80`; `project-doctrine.md:104-109`). The audit records `evidence/` at 14.3k → 92.6k lines (`systemic-findings.md:117-120`).
8. **Presentation limits** (design habits): no numbers on blocks, no dotted arrows, never an unclickable block, connectors are glyphs not boxes (`.claude/PROTOCOL.md:442-452,175-189`).
9. **Not a Hub-wide catalogue**: fine-tunes, merges, quantizations, encoder-only embedders, JEPA/CNN/LSTM excluded from the target list (`done/TO_SERVE.md:443-445`; `TO_SERVE2.md:7`).

Undeclared boundary the tree enforces anyway: the source index reads the model's *own* modeling file(s); shared Diffusers/Transformers files were meant to enter as *"explicit external source nodes"* (U3 boundary 8) and never did until U11, for UNets only (`systemic-findings.md:30-38,58-72`). This is the boundary that blanked PixArt/SD3.5.

---

## 5. The declared target models (the catalogues)

Five catalogues exist; none is referenced from the current z-docs, and the memory file still points at a path (`repo root toserve_model.md`) that no longer exists.

| catalogue | date | count (unique HF IDs, counted by regex on backticked/bare `org/name`) | scope statement | notes |
|---|---|---|---|---|
| `unfold-pkg/previews/old/toserve_model.md` | ≤ June 2026 | **117** "should support" + **48** "nice to have" = 165, across 15 + 19 family headers | text LLMs only | Opens with a half-written *"verification skill"* note (`:1-10`). "Nice to have" includes RWKV (`:232-235`), Jamba (`:237-238`) and FLAN-T5 (`:222-225`) — later ruled out of scope or never handled (enc-dec). |
| `unfold-pkg/previews/old/toserve.md` | June 2026 | **118** IDs, 99 table rows, 18 LLM + 6 diffusion sections | *"decoder-only LLMs (incl. multimodal text backbones) + diffusion (DiT / MMDiT / UNet / VAE). Out of scope: pure SSM/Mamba and RWKV"* (`:13-15`) | Self-described *"canonical list … the coverage standard: a family belongs here when it is popular or architecturally distinct"* (`:3-7`). Includes B5 audio (Stable Audio) and B6 conditioning add-ons (ControlNet/IP-Adapter/schedulers as "Tier-3") (`:164-177`). Names the verification scripts (`:181-187`). This is the `~118 ids / 18 families` the July numbers cite (`PROJECT_CONTEXT.md:1961`). |
| `done/TO_SERVE.md` ("The Big Grill") | 2026-07-07 | **299** IDs; 38 family headers; by part: 161 text LLM · 32 VLM · 49 image · 26 video · 32 audio/speech/music | in scope: AR LLMs, MoE, VLMs, diffusion/flow/AR image, video, audio (`:5`); excluded: SSM, RWKV, JEPA, CNN, LSTM, encoder-only (`:6,443-445`) | *"HF hosts ~2M+ repos. 'Every single one on earth' is asymptotic"* (`:7`). Broadest and latest. |
| `z-docs/stale/TO_SERVE2.md` | 2026-07-07 (10 min later) | **198** IDs; 31 LLM + 5 diffusion sections | same exclusions *"per your request"* (`:7`) | A trimmed sibling of the Big Grill; quarantined as stale. |
| `unfold-pkg/previews/old/toserve_layers.md` | 2026-06-27 | 15 model rows | not a catalogue: an audit of IMP findings (Qwen3-Coder-Next, Gemma-4, DeepSeek-V4 …) (`:1-12`) | Shows the frontier the author was already probing in June (DeepSeek-V4 mHC, Gemma-4 PLE). |

**Characterisation.** The target is *every notable open-weight generative family*, weighted toward the 2025–26 frontier (Llama 4, Qwen3/3.5/3.6, DeepSeek V3.1–V4, GLM-4.5→5, Kimi K2.x, gpt-oss, FLUX.2, Wan 2.2, HunyuanVideo 1.5) plus the historical positional-variety set (GPT-2/NeoX, BLOOM ALiBi, OPT learned, MPT) kept because it is *"architecturally instructive"* (`toserve.md:111-118`). Domains: text decoders (dense, MoE, MLA, hybrid-attention), VLMs, image DiT/MMDiT/UNet, video DiT, audio DiT and codec LMs. The blessed corpus that actually gates releases is **29 witnesses** (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:2669` U8 DONE row; `tests/sable_test_corpus/*.json`) — roughly a tenth of the Big Grill.

**Revealed intent — what the author actually tries.** `finalize.ipynb` (workspace root, 24 cells) calls `unfold()` on 18 distinct IDs: Wan2.2-TI2V-5B, HunyuanVideo, gemma-2-9b-it, recurrentgemma-9b, diffusiongemma-26B-A4B-it, SDXL-base, gemma-4-e4b, ideogram-4-nf4, DeepSeek-V3.1-Base, sarvam-30b, DeepSeek-V4-Flash, gpt-neox-20b, CodeLlama-7b, Llama-3.2-11B-Vision, Qwen2-VL-7B, gpt-oss-120b (×3), pixtral-12b, DeepSeek-R1. Eight of eighteen are diffusion/video/multimodal or frontier-2026 text; one (recurrentgemma) is the out-of-scope class. `push.txt` ends with the note *"# Vonxtral left"* — an unfinished target. **Published showcase** (`unfold-pkg/examples/`, 13 files): six LLMs dated 2026-05-10 (deepseek-v3, gemma-4-31b, gemma-4-e4b, kimi-k2, llama-3-8b, mistral-7b-v0.3) and seven diffusion (flux-1-dev + six UNets) dated 2026-06-07/10; the README links only `llama-3-8b` (`README.md:13-14`). No example has been re-rendered since June; they predate the evidence campaign entirely. (Note: `finalize.ipynb` contains a plaintext HF token in a cell — see Loose ends.)

---

## 6. Success metrics the sources define — and do not

**Defined (all soundness/process):**
- *"No known confidently wrong structural claim remains in the release corpus"* + five more finish conditions (`finish-line-and-scaling.md:5-17`); release boundary (`delivery-path.md:76-85`).
- Master plan definition of done, ten points, all about authority, typing, deletion, generality (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:85-101`).
- Blocking health counters, all `== 0`: `unreceipted_structural_claims`, `identity_tainted_claims`, … `semantic_family_table_hits` (`:2925-2937`); *"A falling unknown count is coverage progress; a zero unknown count achieved through defaults is a regression"* (`:2942-2944`).
- Tree fingerprint identical before/after every receipt (§21 rows); Sable 29/29; zero-drift preservation (`:2669`).
- Per-model: Sable's mechanical nets + the 7/8-item visual rubric; Her Eyes' LOVE/FINE/DISLIKE + APPROVE/SUGGEST per image (`.claude/PROTOCOL.md:378-391,408-426`).
- Generality cost: *"Cost of an unseen family — EXAONE = 2 findings (was ~30) · target 0"* (`PROJECT_CONTEXT.md:1963`, July, never re-measured).

**Not defined anywhere:**
- **No user metric.** Nothing measures whether a learner understood, how long a journey takes, or what a first-time visitor sees. The audit: *"no clear answer to the question 'who is this for, and what happens when they click'"* (`first-principles-judgment.md:30-31`).
- **No coverage denominator.** The finish line says *"The coverage denominator must be measurable: official architecture classes in explicitly supported Transformers and Diffusers versions, plus a published representative corpus"* (`finish-line-and-scaling.md:19-22`) and supported-scope says coverage *"belongs in tested corpus manifests and release receipts"* (`supported-scope.md:50-52`) — but no such number is published; the last catalogue-wide run is `docs/serve_audit.md` (2026-07-04: 105 audited, 94 rendered clean) and `docs/coverage_audit.md` (49 models, 0 unparsed fields), both pre-campaign. `docs/models_supported.md` is an empty file.
- **No recall metric** — the audit's central finding: *"Every gate asks 'did we claim something false?' None asks 'did we lose something we had proven?'"* (`systemic-findings.md:87-89`). The first recall table is the audit's own reconstruction (`recall-trajectory.md:1-11`).
- **No latency budget**, despite "one click" being the tagline.
- **No definition of "complete structure"** — which the user has now asked for twice (*"whats included"*, transcript 2026-09-01T11:53).

---

## 7. The intent timeline

Dates from `git log` in `unfold-pkg/` (first commit `b8f0c73` 2026-05-06; 487 commits: May 99 · June 111 · July 163 · Aug 111 · Sep 3).

| period | turn | gained | lost |
|---|---|---|---|
| **May 6–17: "one click"** | Config-only parser → HTML; tagline `bb74675` 2026-05-10; models table + *"Diffusors — Coming soon"* `87cf965`; README frozen `56ca2fc` 2026-05-17; JSON schema `cfd0191`; versions 0.2.0→0.2.7. Memory of that era: *"NO per-family adapters and no 'supported model' gate — everything is config-field-driven"* (`memory/project-model-unfolder.md`). | A public, pip-installable, notebook-inline product with a clear promise; six LLM examples. | Nothing yet — but "config-derived facts" silently meant *defaults for what config doesn't say* (Phi drawn gated: `PROJECT_CONTEXT.md:280-286`). |
| **June 6–27: diffusion + Sable/Dable** | `8d5ba98` diffusion v1; UNet `919f0a1`; seven diffusion examples; catalogue `toserve.md` consolidated; CLAUDE.md (now `done/forced/CLAUDE.md`, 2026-06-20) introduces *"honest, interactive"* (`:3-4`); *"sable dable everywhere"* `6b2c15a` 2026-06-27; op-conformance against `forward()` AST. Last release 0.2.17 `55513f6` 2026-07-07. | Diffusion scope; the first mechanical honesty net (diagram↔code both directions); the "pixels are the oracle" discipline. | README/Space never updated — the public surface froze here. |
| **July 5–14: the doctrine** | PROJECT_CONTEXT Part 1 laws + product decision *"public library/product … consolidation aggressive … unit-based roadmap"* (`PROJECT_CONTEXT.md:184-187`); scope line excludes SSM; run_77 pixel↔HF-truth review of 425 models (`method-and-plan-judgment.md:155-160`); Her Eyes persona; audio scope blessed 07-07; Part 3 *"config-based → code-parsing-based … the one to never walk back"* (`:260-263`); Part 0 *"truth machine … the map cannot lie about the territory"* (`:24-38`); master plan 2026-07-13; z-docs restructure 07-14 → *"source-grounded architecture explainer"* / *"evidence compiler"*. | Evidence-never-identity as law; honesty > completeness > beauty explicitly ranked (`project-doctrine.md:82-91`); typed unknown; one-way evidence; the vet-and-receipt court. | The learner vanished from the authoritative docs (z-docs contain no "newcomer"); completeness demoted to 5th; the recall-capable run_77 procedure was never repeated (`method-and-plan-judgment.md:176-183`). |
| **July 15 → Sep 1: the evidence compiler (U0–U11)** | Exact occurrences, ProgramIndex, unknown-safety, firewall, attention/FFN/position/modality/diffusion cutovers; `evidence/` 14k→93k lines; 3,920 tests; 26+ blocking nets. | Real soundness: 42 models, one wrong proven fact; no family branch anywhere; reproducible receipts. | Verified detail in U6/U8/U9/U10 (`recall-trajectory.md:15-40`); PixArt/SD3.5 to blank boxes; latency 1.3 s → 6–38 s (§3 promise 9); the visible product regressed while every gate stayed green (`09-unit-verdicts/README.md:52-62`). |
| **Aug 16 / Sep 1: the user re-centers** | *"as long as I get the structure out … it's my design on how I want to show it"* (08-16); *"i want absolute coverage"*, *"the main structure which my library promises"*, *"by main structure i mean whats included"*, *"we cant not have that"* (09-01). | A restated priority: **structure first, then design, honesty as a property of both** — not honesty as a licence for grey boxes. | Not yet reconciled with `project-doctrine.md:82-91` or with any gate. |

What each turn kept from the previous is asymmetric: every turn kept the *honesty* of the one before and dropped a piece of the *product* (README ↔ docs, examples ↔ tree, learner ↔ auditor). The one-line description drifted from *"one click model unfolder"* → *"honest, interactive architecture diagram"* → *"truth machine"* → *"evidence compiler whose human-facing output is an interactive architecture explanation"* — the human moved from subject to appositive.

---

## 8. Reconciled goal statement for the rest of this research

The honesty doctrine and the "worth looking at, complete structure" intent are not in conflict once *what must be complete* and *what may be unknown* are separated by tier. The audit already proposed the split (`first-principles-judgment.md:165-170`) and the coordinator's brief states the user has agreed to three tiers **[the user's agreement is asserted by the brief; I did not find it in the transcript or docs I read]**:

> **Model Unfolder turns a Hugging Face model ID into an interactive diagram of the model's true, complete structure that a newcomer can learn from — never guessing, never silently omitting.**
>
> **Tier 1 — Structure (what is included): must be complete.** Every module the checkpoint constructs and reaches in `forward()` — towers, stacks, ×N, block classes, sub-blocks, parameter shapes, loops, merges, splits — is drawn, with its real name, for every model in the declared support set. A structural element that cannot be resolved is a **visible finding that blocks bless**, never a pale box the gates accept. This tier is the *"main structure which my library promises"*; the user's *"absolute coverage"* applies here. Recall is ratcheted here exactly as soundness is.
>
> **Tier 2 — Mechanism (how each block computes): must be honest.** Op order, residual taps, position application, gating algebra, routing, masks — proven from the reachable modeling source and drawn as drills; where source is missing or genuinely ambiguous, the block stays drawn (Tier 1) and carries an explicit unknown chip with the reason. This is where the evidence doctrine's *"unknown is a successful honest result"* (`project-doctrine.md:77-80`) applies — and only here. A provable negative (no QK-norm, plain LayerNorm) is proven negative, not "unknown".
>
> **Tier 3 — Values (numbers): must be exact and sourced.** Dimensions, counts, θ, scales, thresholds come from the checkpoint config (or an exact class default with its span), shown on cards/chips, never on blocks; the config-value/code-meaning law (`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md:43-44`) governs.
>
> **Presentation is the author's design**, judged by the learner's journey (Her Eyes' five questions, `.claude/PROTOCOL.md:378-391`): bundling and drills may calm the page but never hide a Tier-1 element or a Tier-2 unknown.
>
> **Boundaries stay:** declared support set with a published, measurable denominator; no model execution by default; identity as address only; SSM/RWKV out unless re-scoped.

Restated priority order under this statement (replacing `project-doctrine.md:82-91`): (1) no false claim at any tier; (2) Tier-1 structure complete for the support set; (3) Tier-2 mechanism proven where source exists, provable negatives proven; (4) one canonical fact per projection; (5) exact Tier-3 values; (6) the learner's journey. Honesty stays first; completeness of *structure* moves from fifth to second, and "unknown" is confined to mechanism.

This is the goal the remaining passes (01–10) should judge against.

---

## Loose ends

1. **The user's tier agreement is unsourced in the tree.** The brief says the three tiers are agreed; the transcript I read shows the user asking for *"absolute coverage"* and *"whats included"* but not an explicit assent to "structure / mechanism / values". A one-page product spec stating it (the audit's obligation 3, `method-and-plan-judgment.md:231-233`) does not exist.
2. **`finalize.ipynb` contains a plaintext Hugging Face token** in an `unfold("Wan-AI/Wan2.2-TI2V-5B", token="hf_…")` cell. Not copied here. It should be revoked and the cell scrubbed; the workspace root is not a git repo, but the notebook has been in this form since at least the May memory note.
3. **Promise 3 (no transformers needed) was not exercised** on a machine without transformers; the behaviour of the code-evidence stream and the honesty of the resulting diagram in that configuration is unverified.
4. **Promise 8 (SSM/RWKV rows in the README table)**: whether Mamba/Jamba/Zamba/RWKV configs still render anything today, and what, is unverified; `typing.yaml:55-57` suggests the vocabulary survived the scope ruling.
5. **Latency probe is single-run, single-machine** (first `unfold` includes import/index warm-up; llama-7b at 21.8 s was the first model). Pass 02 should profile where the 6–38 s goes and whether anything is cached across processes.
6. **README param claim** — the README's 675B/41B for DeepSeek-V3 matches neither the tool (669.8B/36.4B) nor the paper (671B/37B); Llama-3-8B's 8.03B was not re-checked (no corpus fixture; gated).
7. **`docs/models_supported.md` is empty**, `docs/coverage_audit.md` / `serve_audit.md` are dated 2026-07-04, and no catalogue is referenced from z-docs: the "declared support set" the doctrine relies on has no current, published instance. Which of the five catalogues is canonical is undecided (the memory file points at a deleted path).
8. **`diagram.explain()` / `diagram.audit()`** (master plan §22.8) — absence verified only by grepping `diagram.py`'s method list; a differently named equivalent may exist elsewhere.
9. **The Space (`hf/app.py`) mis-describes the product** (*"Generated purely from the Hugging Face config map"*, `:129`) now that the modeling source is a primary evidence stream, and collapses all refusals to one message (`:122-123`). Whether the Space is deployed and at which package version is unknown (requirements pin 0.2.17).
10. **Version triple** (0.2.17 / 0.2.15 / tags) is a two-line fix nobody has made in seven weeks; the audit and the ledger both record it.
11. **Audio scope** (blessed 2026-07-07, `PROJECT_CONTEXT.md:2236`) is in the code (MusicGen witness) but absent from `z-docs/01-product/supported-scope.md`'s adapter list except as "audio … paths" (`:9`); whether TTS/codec LMs are a promise or an experiment is undecided.
12. **Her Eyes reviews lapsed 2026-08-06** (`method-and-plan-judgment.md:184-186`); no learner-facing judgment exists for anything blessed in U8–U11.
13. **The examples directory is pre-campaign** (May/June renders). The README image is the only public picture of the product and predates every honesty change.
