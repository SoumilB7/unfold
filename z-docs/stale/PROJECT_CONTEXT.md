# PROJECT CONTEXT — DEPRECATED HISTORICAL SOURCE

> **Do not use this file as current implementation or product direction.** The
> authoritative chaptered context is [z-docs/README.md](../README.md). This
> legacy body is retained only to avoid destructive loss of historical reasoning.
> Current status, plan ratings, remaining config authority, and forward product
> direction live under `z-docs/07-current-state/`. See
> [the legacy-removal map](../08-reference/legacy-removal.md).

### Historical working context retained for traceability
*(written 2026-07-05, from the sessions that built the recent waves; companion to MASTER_PLAN.md and RESTRUCTURE_PLAN.md)*

---

## PART 0 — THE ONE IDEA (the deepest intent; every part below is this same law at a different altitude)

*Written 2026-07-12, as the keystone. Everything in this document — the product
laws (Part 1), the config→code transition (Part 3), the evidence-hardening
campaign (Part 7), the process meta-principle (Part 14) — is not four disciplines.
It is ONE idea stated at four altitudes. This part is that idea, in its most
compressed and most general form, so that a fresh session reads every part below
through it. If you internalize only one thing, internalize this.*

### 0.1 What "understanding a model" actually means here

The product is not a diagram generator. It is a **truth machine about
architecture.** Its job is to reconstruct a system *from the system's own
declared evidence* so faithfully that **the map cannot lie about the territory** —
and where the evidence runs out, to say so, visibly, rather than fill the silence.

"Understand" therefore has a precise, unromantic definition here: a claim about a
model is understood only if it can be **traced to a declaration the model itself
ships** — a value in its checkpoint config, or a branch in its modeling source
read as text. A claim that traces to anything else — the model's *name*, a
*convention* about "models like this", a *default* that felt reasonable, the
implementer's *memory* — is not understanding. It is fabrication wearing
understanding's clothes, and it is the single thing this entire architecture
exists to make **impossible by construction**, not merely forbidden by doctrine.

### 0.2 The one law

> **Replace every act of authority with an act of evidence.**
> Wherever the system would otherwise decide something by *identity, convention,
> vigilance, or judgment*, force it to decide by *traceable evidence, typed
> provenance, mechanism, and a gate that fails the build.*

That is the whole project in one sentence. Every mechanism we have ever
built — the op-conformance net, the tower cell, the identity guard, the
positional tri-state, the `ConfigAccessEvent` ledger, the `StructuralWrite`
census, the `H<n>_EXIT` checklist — is one instantiation of that single move.
Naming them separately is useful for work; believing they are separate is the
error the campaign keeps catching.

### 0.3 The self-similarity — the same law at three altitudes

The reason this document has to be read as a whole is that the one law is
**fractal**: it recurs, structurally identical, at three scales. Each level is
the previous level's discipline turned inward.

**Altitude 1 — the MODEL (Part 1, Part 3).** A fact about the drawn architecture
must trace to the model's own declared evidence: the checkpoint config supplies a
*value*, the modeling source proves what that value *controls*. Identity may be
used exactly once — as an **address** to locate the source — never as a fact. The
test of success is generality: **solve once → correct for every future model.** A
fix that needs a new per-model row to work on the next model is, by definition,
not a fix. This altitude is where fabrication about *the model* is killed.

**Altitude 2 — the PARSER'S OWN MIND (Part 7, the evidence-hardening campaign).**
Turn the same suspicion on the parser's own epistemics. Every fact the parser
holds must be **typed with how it is known** — `code_proven`, `code_and_config`,
`config_declared`, `derived`, `ambiguous`, `oracle_missing`, `unknown` — and
those weak statuses are **first-class results, not failures**: an unknown is
drawn as an honest pale unknown, an ambiguity is a blocking finding, a missing
oracle is an exempt honest absence. A `registry` defines what may be claimed at
all; a claim with no evidence is **refused, not defaulted.** This altitude is
where fabrication about *what we know* is killed — where the parser stops being
able to quietly upgrade a guess into an asserted fact.

**Altitude 3 — the IMPLEMENTER'S WORK (Part 14).** Turn the same suspicion on the
person executing the plan. A rule that lives only as **prose** is honored by
memory and judgment — which fail exactly when the implementer is tired,
optimistic, or rushing. So the same move again: **`DONE` is not a sentence I
write; it is a checklist of passing gates plus an unchanged tree fingerprint.**
Measure before you design (the probe is the first artifact). The blast-radius is a
tool that enumerates production + tests + nets + cross-scope, not a grep. This
altitude is where fabrication about *the work itself* — "I completed H2" when a
gate was unbuilt — is killed.

The three altitudes are not analogies. They are the **same defect** (an authority
substituting for evidence) and the **same cure** (a gate substituting for the
authority), applied to the model, then to the model of the model, then to the
maker of the model of the model. This is why a fix at one altitude so often
reveals a gap at another, and why the campaign is worth doing as one campaign.

### 0.4 Why §16 correcting my own "DONE"s is the law WORKING, not failing

The most recent turn of the campaign is an independent audit (plan §16) that
**overturned several units I had marked DONE** — H0/H1/H2 back to ACTIVE, H3 to
RESTART. This is not a setback and it is not embarrassing to record plainly: it is
the doctrine **eating its own tail correctly.** An independent evidence-check
caught an over-claim *exactly as a conformance net catches a fabricated op* — by
demanding the receipt and finding none. A campaign whose whole thesis is "detect
the breach from evidence, never from vigilance" cannot then exempt *itself* from
evidence. The audit applying the law to the campaign's own status claims is the
law reaching its highest altitude. **Recording the correction faithfully is the
deliverable; hiding it would be the one truly disqualifying act.**

### 0.5 The two distinctions that make honesty mechanical (never lose these)

Two separations do most of the load-bearing work; collapsing either one is how
honesty silently dies.

1. **Config is checkpoint truth; code is mechanism. Neither is eliminated.** The
   transition to code-parsing (Part 3) is endlessly misread as "eradicate config."
   It is the opposite. Config supplies the **value this checkpoint selected**
   (geometry, declared enums, source-bound branch values); code proves **what that
   value means** — which structural branch or expression it controls. The value
   ladder is: *config value → aliased config value → code relational/`__init__`
   default → honest unknown*, and it never skips a rung. Critically, `__init__`
   defaults are **family fallbacks, not checkpoint truth** — reading them as truth
   would collapse a 7B and a 70B into the same picture. What *dies* in the
   transition is never config's values; it is letting field-*presence*, *spelling*,
   *identity*, or a *convention* DECIDE architecture. Both streams are evidence,
   with strictly distinct roles.

2. **Every claim is owner-scoped; the flat global namespace is the root defect.**
   A fact, a config access, an ignore rule, a structural write — each carries its
   **owner path** (root / text / vision / audio / VAE / denoiser / …). A
   transformer `hidden_size` and a `_vae_config` `hidden_size` are **different
   entries**, not one. The single deepest architectural bug the campaign has named
   (I-10 / H3.6, and the reason H3 needs a full RESTART) is that `accessed` /
   `consumed` / `ignored` / `unread` were matched by **bare key name across all
   scopes** — so one scope's reads papered over another scope's gaps, and any
   global change had unpredictable cross-scope blast radius. The cure is not to
   tiptoe around the leak; it is to **close it** — owner-qualify every ledger so no
   scope can mask or clear another's debt.

### 0.6 The felt intent (what I am actually trying to protect)

Beneath the mechanism is a stance about honesty that outranks every other value
in this project. A newcomer will *trust* this diagram — that is the entire point
of it — and so **a beautiful lie is far worse than an honest gap.** Honesty
outranks completeness and beauty (Part 1, Law 5): a pale static unknown that
admits "the source didn't say" is a *success*; a confident SwiGLU drawn over a
model that is genuinely dense is a *failure*, even though it is prettier and more
complete. Everything hard we do — reading AST instead of running code, typing
provenance instead of asserting facts, gating `DONE` instead of declaring it —
buys exactly one thing: **the right to be believed.** Protect that, and the rest
of the document is just the engineering of how.

---

## PART 1 — WHAT WE ARE TRYING TO ACHIEVE

**The product in one sentence:** hand it any HuggingFace model (an id, a
config dict, a pipeline repo) and it produces an *honest*, interactive
architecture diagram — every block, arrow, repeat, connector and split the
model actually computes, drillable to the leaves, understandable by a
newcomer, and never fabricated.

**The laws (non-negotiable, in priority order):**

1. **Detect from EVIDENCE, never from identity.** No class-name matching, no
   repo-id branches, no per-model tables. A fact comes from (a) the config's
   own declarations, resolved through *general* YAML vocabulary in
   `everchanging/`, or (b) the modeling source itself, read as AST — never
   executed. If a fix needs a per-model row to work on the next model, it is
   not a fix. When something is solved, it is solved for every future model.
2. **One declaration, three projections.** A structural fact is declared once
   (op-graph region / tower cell / IR spec) and projected to SVG, JSON, and
   cards. Divergence between projections is a bug class we have been burned
   by repeatedly (see Part 5).
3. **The generated HTML is ground truth.** Everything is baked statically;
   JS only toggles. Work is verified by rendering and *inspecting the
   output*, never by reading code and imagining the result.
4. **Pixels are the final oracle.** Mechanical nets check structure; only
   eyes catch a floated ⊕, a collapsed elbow, a lying upsample. Hence
   Dable (exhaustive image pass) and now Her Eyes (design/UX judgment).
5. **Honesty outranks completeness and beauty.** An unknown is drawn as an
   honest-unknown (pale, static), never guessed. A refusal is typed and
   actionable. A visual bundle may never hide a real op.
6. **Soumil commits; Claude never runs `git commit`.**

**Product decision (2026-07-05):** the end state is a **public
library/product** (pip-installable, versioned, documented), consolidation is
**aggressive** (twins get deleted, mass re-bless is a planned event), and the
roadmap is **unit-based** (pull the next unit when ready; no dates).

**Explicit scope line:** decoder-only LLMs (+ multimodal towers) and
diffusion (DiT/MMDiT/UNet/VAE). Pure SSM/Mamba and RWKV are out of scope;
hybrid-SSM stacks currently draw their attention layers and honestly warn
that mamba/conv layers are collapsed (this is the biggest remaining honesty
debt — MASTER_PLAN unit C1 upgrades them to honest first-class cells).

---

## PART 2 — THE SYSTEM, END TO END (the black box contract)

**INPUT — `unfold(x)` accepts, in a ladder:**
dict → PretrainedConfig → HF id via AutoConfig (auth retry; trust_remote_code
always False — remote code is *read*, never executed) → diffusers pipeline
index (model_index.json or pipeline-index-in-root-config dialect) → raw
config.json (tolerates comments/trailing commas) → Mistral `params.json`
format normalizer (mapping lives in `everchanging/transformer/
mistral_params.yaml`; "pixtral" decided by STRUCTURE — vision_encoder
present — never repo name) → unfoldable-signal validation → repo-layout
hints via the HF file-listing API (adapter-only repos, variant-nested
pipelines → typed honest refusals). Ten refusal categories, all typed
(`ModelNotFoundError` / `ModelAccessError` / `ConfigParseError`), each with
an actionable message.

**EVIDENCE — two streams feed every fact:**
- *Config stream:* alias resolution (`aliases.yaml` spellings → canonical
  fields), ignore vocabulary (`ignored_fields.yaml` — every unread owned
  field is a blocking audit finding until parsed or classified), config
  facts chips, typing/stage vocabularies. All data, no code.
- *Code stream:* the modeling source located ONCE by address (installed
  transformers/diffusers; the repo's own `.py` files for remote-code repos;
  diffusers `pipelines/` for vendored towers), then read by AST extractors:
  class registries, forward-op presence sets, transitive closures,
  construction graphs, positional mechanisms, FFN structure/storage, norm
  math, secondary stacks. Never executed.

**FACTS — `ModelIR`:** typed `LayerSpec`/`AttentionSpec`/`FFNSpec` +
extras (modalities, render specs, sub_models, provenance, config audit).

**STRUCTURE — one structural author per concept:** `opgraph.py` Regions
(attention/FFN/router/scheduler/…); `tower.py` cells (the ONE cell
projector for every supporting tower); declared-ops floor for anything else.

**PROJECTION — outputs:**
- `.to_html()/.save()` — the interactive doc (all views baked).
- `.to_json()` — the expanded machine schema (schema_version 3.2).
- `.save_images()` — exhaustive, deduped PNG gallery + MANIFEST (the Dable
  input; amber border = clickable, an image-only debug overlay).
- `.wiring_problems()` — dangling-connector flag; non-empty = P0.
- `.to_ir()`, `.param_count()`, `.warnings()`.

**QUALITY HARNESS:**
- `sable(model)` — 12 blocking mechanical nets in one pass: click-coupling,
  dangling connectors, unique ref-ids, no dotted arrows/boundaries,
  config-field audit, op-conformance (diagram↔code both directions),
  wiring-conformance, fact-conformance (positional scheme, attention
  algorithm), nested conformance (every leaf drill vs the transitive
  forward() closure), label lint, evidence-ambiguity. Plus the 8-item
  VISUAL_RUBRIC for the eye pass.
- `bless(report, model)` — freezes config + per-view SVG hash signature +
  durable gallery copy; refuses without artifacts; records
  `superseded_hash_signature` on re-bless. `check_regression` = the CI lock.
- **Dable** (procedure): render EVERY distinct view to PNG and look at every
  one. Sampling is forbidden — this rule exists because sampling missed 9
  real findings in one sweep (Part 6).
- **Her Eyes** (procedure, new): the design/UX persona — judges delight,
  ceiling, bundling, newcomer-readability, journey pacing, from images only;
  approves/suggests, never edits. (Her review-file home is an OPEN DECISION —
  see Part 8.)

---

## PART 3 — THE BIG TRANSITION: CONFIG-BASED → CODE-PARSING-BASED

This is the deepest architectural decision in the project and the one to
never walk back.

**REFINEMENT (2026-07-12, binding — supersedes any "eradicate config" reading):**
this transition ADDS code as evidence; it does NOT eliminate configuration.
Config is not the enemy — it is checkpoint truth. The exact law, from
EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md §1.2, is: **code proves that a field
controls a structural branch or expression; the checkpoint config supplies the
value selected for this checkpoint.** So geometry, declared enums, and
source-bound branch VALUES stay config-read (marked `config_declared`); what
dies is letting field-presence, spelling, identity, or a conventional default
DECIDE architecture. Read every "config → code" statement below through this
law: both are evidence, with distinct roles.

**Where we started:** pure config parsing. Read config.json fields, map them
to structure. It works for geometry (hidden sizes, head counts, layer
counts) because configs declare geometry.

**Why it broke:** configs are SILENT about structure. Nothing in a config
says whether the FFN is gated, where the norms sit, whether RoPE is applied,
whether Q/K/V are stored fused, whether the residual is scaled. The early
codebase filled those silences three bad ways:
1. **Defaults** — "modern LLMs are gated SwiGLU" → Phi (genuinely dense) was
   drawn gated. A default silently becomes an asserted fact. (The last such
   hazard, `AttentionSpec.mask = "causal"` as a dataclass default, was killed
   by the evidence-hardening campaign's default-kill work — mask is now
   tri-state, decoderness-gated; see EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md.)
2. **Per-model tables** — `class_defaults.yaml`, `layer_topology.yaml` keyed
   by model_type, conformance_map overrides. Identity-based. Each new model
   needed a new row = the antithesis of generality.
3. **Vibes** — "this family usually does X". Fabrication with extra steps.

**The decisive realization:** the modeling source is ALSO a shipped,
declared artifact. `forward()` and `__init__` state the structure
explicitly, and reading them via AST is evidence — exactly as legitimate as
reading config.json, and strictly more complete. And crucially: **the same
code-evidence a conformance net reads to CATCH a wrong drawing must FEED the
parser so the drawing is right in the first place.** One source of truth; a
net that can see a fact the parser guesses is an architecture bug.

**The transition, in the order it actually happened:**
1. Op-conformance net first (born from the FLUX single-stream lie — Part 5):
   diff the drawn op-set against the class's forward() op kinds, both
   directions, with a declared abstractions allow-list.
2. Code-derived FFN facts: activation/gating read from the FFN class's
   forward (killed `class_defaults`; fixed CogView4 pale-FFN, Phi
   dense-drawn-as-gated, Allegro/Lumina fabricated NoPE).
3. Norm facts from math: RMS vs LayerNorm decided by the norm class's actual
   arithmetic (fixed T5), eps-spelling fallback retained as config evidence.
4. Norm PLACEMENT from forward dataflow (pre/post/sandwich/parallel) —
   killed the `layer_topology.yaml` model_type table (Gemma sandwich,
   OLMo-2 post-norm, Cohere parallel all derived, not tabled).
5. Construction evidence: factories (`X._from_config`), string-dispatch
   factories (diffusers `get_down_block`), ModuleList element classes,
   init-helper folding — the walk from an architecture class to its real
   block classes with no name guessing.
6. Transitive closure (the nested-conformance net): every leaf drill diffed
   against the *transitive* forward() closure (follows sdpa, rotary,
   diffusers processors, fused experts, self-methods).
7. Storage facts: fused QKV (`query_key_value`/`c_attn`), fused gate_up
   (name-based AND structurally: gated + 2 linears + chunk/split in
   forward), expert storage — because a whiteboard-equivalent split drawing
   is not code-faithful when the source stores fused.
8. Positional evidence tri-state: **proven** (mechanism found and applied) /
   **ambiguous** (source present but unresolved — an extractor gap, BLOCKING)
   / **oracle_missing** (no source — honest absence, exempt). This tri-state
   is the discipline that keeps code-reading honest: a detector failure is
   never permission to assert a fact.
9. Identity guard (Unit 9): a net that name-blinds the config (scrubs
   model_type/architectures) and asserts the parsed STRUCTURE is unchanged —
   identity may be used exactly once, as an ADDRESS to locate source, never
   as a fact. Identity debt reached zero and the guard blocks.
10. The remote-code rail (2026-07-05): config-declared `auto_map` = "my code
    ships with me" → fetch the repo's own `.py` into the SAME AST rail.
    Plus: vendored towers found inside diffusers `pipelines/` by declared
    `architectures`; the loader's `_repo_id` stamp outranking exporter junk
    `_name_or_path`; class-default hydration through the installed config
    class at LOAD time (address resolution) so parses stay name-blind.

**Why this design and not alternatives we considered:**
- *Executing model code* would be simpler and is banned: security, weights,
  and honesty (we would be describing runtime behavior of one input, not
  declared architecture).
- *Pure config + bigger vocabulary* cannot ever express structure configs
  don't carry; it converges to per-model tables.
- *LLM-guessing the architecture* violates the entire premise; every fact
  must be traceable to a declaration.

**The current frontier of this transition (the "acquisition" thesis):**
detection is mechanical, but *acquisition* of a NEW FACT FAMILY is still a
hand-written reader. Soumil's standing critique — "why is there still a code
change for every model?" — is answered by: (a) an **unclaimed-signal net**
(forward-ops gated/parameterized by config fields no fact reader consumes →
loud CI finding), and (b) **family mechanisms over field readers**
(stream-scale, width-bridge, rope-scaling dialects, storage modes,
layer-schedules): a new spelling becomes one YAML row, a new family one
mechanism. Measured baseline: a never-seen remote-code family (EXAONE)
today costs exactly **2 findings** (was ~30 pre-rails). North star: 0.

---

## PART 4 — THE HARDEST PARTS OF THE TRANSITION (honest engineering notes)

1. **Twin AST scanners.** `evidence/transitive.py` (CallableInfo registry +
   closures) and `evidence/forward_ops.py` (ForwardOps presence sets) each
   re-implement class scanning and init walking. The tax is real and was
   paid twice IN ONE WEEK: the "init-local function bound to self.attr"
   fold (ChatGLM's `swiglu`) had to be implemented in BOTH. MASTER_PLAN A4
   merges them into one pass emitting both dataclasses.
2. **Blanket `except Exception` wrappers swallowing real bugs.** The
   `_code_*` wrappers (28 across both parsers) return None on ANY failure.
   Lived failure: an AttributeError typo (`info.call_tokens` vs
   `signature_tokens`) was silently swallowed; glm-4 looked correct only
   because the fallback heuristic happened to agree; the bug surfaced only
   when a test called the reader directly. The consolidation (one
   `_code_evidence()` helper, narrow exceptions) IS the fix.
3. **Serializer fan-out.** `AttentionSpec` is projected to dicts in THREE
   places (ir.py `_attention_to_dict`, blocks/attention.py
   `attention_detail`, opgraph meta). Adding `scores_scale` required all
   three (the drill reads the BLOCK's fact dict, not the dominant spec —
   found only because the pixel still said sqrt(dim) after two of three).
   `rope_3d` currently exists in the spec and one projection but NOT in
   `_attention_to_dict` — live serializer drift, same class.
4. **Anchoring the walks.** The code-evidence walks need a start class, and
   every assumption about it broke eventually: the declared architecture
   can be a stale upstream-renamed name (Qwen2.5-Omni configs say
   `Qwen2_5OmniModel`; the source defines `...ForConditionalGeneration`);
   the entry class can be a generation-only CONTAINER with no forward
   (invisible to scanners until container nodes were added, with
   init-helper folding for `enable_talker()`); a component-only model_type
   (`qwen2_5_omni_text`) has no AutoModel row — ownership had to be read
   from the source's own declaration (`config_class = X` legacy spelling
   AND the modern class-body annotation `config: XConfig`), picking the
   CONSTRUCTED declarer. Recovery rule everywhere: structural (unique
   construction root), and on ambiguity keep the declared name and fail
   honestly.
5. **Attention impl-dispatch narrowing.** transformers dispatches attention
   classes by `_attn_implementation`. The narrowing logic assumed dispatch
   at the LAYER level (Llama). ChatGLM dispatches INSIDE the attention
   class (`SelfAttention.core_attention = CORE_ATTENTION_CLASSES[impl]`) —
   selecting the inner kernel dropped the rope-applying owner, and the
   "candidates disagree on rotary" ambiguity rule then killed the fact. Two
   general fixes: keep the OWNER when the dispatch owner is itself an
   attention-role class; and `_outermost_candidates` — a candidate
   constructed as a FIELD of a fellow candidate is a sub-op, not a rival.
6. **Defaults-as-facts.** The two live examples: `mask="causal"` (a
   bidirectional block that forgets to override asserts causality) and the
   scores denominator (`sqrt(dim)` drawn while Granite's code multiplies by
   `attention_multiplier` = 1/128 — fixed by `scores_scale`, emitted ONLY
   when declared so undeclared models stay byte-stable). The general cure
   (bug-triage consensus): defaults must be distinguishable from declared
   facts in the spec itself.
7. **The pixel gap.** Mechanical nets verify structure against itself and
   against code, but only pixels catch rendering lies: the VAE decoder
   upsample REVERSAL (upsampler drawn on the wrong end of the stack)
   survived in FOUR galleries *including blessed, pixel-reviewed SDXL*; the
   scheduler ⊕ once floated unattached; DSA's third lane collapsed the V→⊙
   elbow invisibly (edge existed in the model, invisible in render). Answer:
   Dable exhaustiveness is law — and even then, an *aesthetic* review axis
   was missing, which is why Her Eyes now exists.
8. **The wrapper-vocabulary unification.** Multimodal wrappers hide the LM
   (`text_config`, `thinker_config`, …) — and Omni proved they hide the
   WHOLE MULTIMODAL HOST (vision/audio configs + token ids live one level
   down). One shared `TEXT_WRAPPER_KEYS` now feeds both the LM unwrap and
   the modality-host walk. Depth guard, object-vs-dict carriers
   (`to_dict()`), double nesting (`thinker_config.text_config`).
9. **Class defaults invisible in raw JSON.** Gemma-2's sliding/global
   alternation lives in `sliding_window_pattern=2` — a config-CLASS default
   never serialized. Raw component JSON therefore drew all-sliding towers.
   Fix: hydrate fetched component configs through `AutoConfig.for_model` at
   LOAD time (identity-as-address; the frozen fixture then carries the
   facts as data, keeping parses name-blind). Trap: `model_type` must be
   stripped from kwargs; loader stamps survive on `_`-prefixed keys.

---

## PART 5 — THE MODELS THAT FOUGHT HARDEST (and what each one taught)

- **FLUX (single-stream block)** — was drawn as a GPT-J-style parallel-sum
  (⊕) while the code does `cat([attn, mlp]) → proj_out → gate× → +`. The
  founding lie of the op-conformance net: internally-perfect diagrams can
  diverge from forward(). Its negative control (the old wrong rendering
  MUST fail) is pinned forever.
- **ChatGLM / GLM-4 family** — the single richest source of general fixes:
  remote code (auto_map rail); `swiglu` as an init-LOCAL function bound to
  `self.activation_func` (both scanners learned init-local folding); fused
  `dense_h_to_4h` with only 2 linears (structural fused signature:
  gated + 2 linears + chunk/split; BLOOM's dormant tensor-parallel path is
  the negative control); `CoreAttention` inner-kernel dispatch (owner
  rule + outermost filter); `multi_query_attention: true` **plus**
  `multi_query_group_num: 2` (the flag means "shared", the count says how
  many — clobbering to 1 fabricated MQA; Falcon-7B stays true MQA);
  `rope_ratio`/`original_rope` initially misclassified as ignorable (a
  self-inflicted fabrication-by-omission). Kolors' encoder is the SAME
  family vendored INSIDE diffusers `pipelines/kolors/text_encoder.py` —
  proving the rail works when source is findable, and forcing the
  in-tree-arch-search + `_repo_id` stamp.
- **Qwen2.5-Omni** — the last red model of the fresh sweep and the hardest
  multi-mechanism unit: modality host behind `thinker_config`; entry class
  renamed upstream (stale `architectures`); talker built in an init-helper
  behind `enable_audio_output` (container nodes + helper folding);
  `qwen2_5_omni_text` with no AutoModel row (config-class ownership
  annotation); audio lane missing from the fusion panel (added: AUD slot,
  a0/a1 stream, time-aligned positions with declared
  `position_id_per_seconds`/`seconds_per_chunk` constants). Result: gallery
  went 4 → 18 views; Qwen3-Omni (MoE thinker) worked on the same
  mechanisms untouched — the generality test passing.
- **Granite 3.x** — the scale family: `residual_multiplier` (drawn as ×
  connectors with the constant beside the glyph — and the chain renderer
  had never learned constant captions, only tower cells had: twin tax),
  `attention_multiplier` (the scores block honestly says `Q K^T / 128`
  now), embedding/logits multipliers (read; drawing pending the
  stream-scale family mechanism, MASTER_PLAN B2).
- **Gemma-2** — sandwich norms (the one-tower cell projector's `double`
  placement; Sana/Lumina encoders had been WRONGLY drawn pre-norm from a
  hardcode until the fold); `query_pre_attn_scalar` with ^-0.5 semantics
  (a second scores-scale dialect; equal-to-default stays sqrt so blessed
  galleries don't drift); the invisible class-default alternation (Part 4
  §9).
- **T5 / UMT5** — unscaled scores (drawing sqrt(dim) would fabricate an op
  the forward never performs — the `scores_scaled=False` code-proven
  variant), relative position bias added pre-softmax, norm-from-math.
- **Phi-3 / phi-4** — fused QKV and fused gate_up storage (dead clicks
  until fused child cards); the rmsnorm→gated heuristic mis-gating dense
  MLPs (first catch of the nested-conformance net).
- **Sana / Sana-Sprint** — linear attention (kernel feature map, no
  softmax — drawn as its own honest shape), conv-GLU Mix-FFN (was a single
  opaque leaf whose own description named five drawable ops; now the real
  chain with per-op cards — sable's panel-walking click-coupling is
  stricter than the flat validator), DC-AE VAE (no fabricated ResNet
  claims), Gemma-2-as-encoder alternation.
- **HunyuanVideo** — token-refiner secondary stack (the general
  config-count-bound ModuleList detector in `evidence/stacks.py`),
  heterogeneous dual+single stream stacks (every layer-type VARIANT renders
  and enters the conformance surface, not just the dominant).
- **Lumina Image 2.0** — context/noise refiners REUSING the root's block
  class (secondary-stack exclusion must key on count_field, not class),
  joined-text ‖ entry (strict-two-input ‖ law; the all_text_joined gate),
  Gemma-2 encoder alternation.
- **BLOOM** — embedding-stage LayerNorm (a real drawn bookend read from
  source), and its dormant tensor-parallel slow path is the reason
  "a multiply exists in forward" is NOT sufficient evidence of gating
  (constructor shape + chunk/split signals required).
- **Mistral/Pixtral original releases** — the `params.json` FORMAT ladder
  (normalizer vocabulary; pixtral identified by structure not name);
  adapter-only and variant-nested repos → honest typed refusals via the
  repo-layout hint.
- **EXAONE** — never touched until it became the measured acquisition
  baseline: 2 findings (a rope_scaling sub-field + remote-code component
  anchor), both loudly flagged. The number the roadmap tracks.

---

## PART 6 — WHAT CHANGED IN THE RECENT WAVES (newest work, in order)

1. **One-tower capstone** (pre-context): one fact dialect (`sub_model`
   groups) + ONE cell projector (`tower_cell`) + one frame + namespaced
   canonical drills; secondary-stack detector; hero-altitude lossless lock.
2. **Fresh 12-model sweep**: 11/12 clean after fixes (granite scale glyphs,
   glm-4 dialect, chips for Kolors/CogView4/Sana-Sprint); Omni honestly red.
3. **Soumil's process challenge** ("did you actually look at the images?")
   → the EXHAUSTIVE pixel pass over all 92 gallery images → **9 findings
   the sampled pass missed**, none caught by any mechanical net: VAE
   upsample reversal (systemic, incl. blessed SDXL), granite scores lie,
   glm-4 missing RoPE + split-drawn fused FFN, Kolors MQA claim + pale
   encoder FFN, Sana Mix-FFN stub + strip label dupe, gemma-encoder
   flattening, cache badges on diffusion attention, (CogView4 mid-block —
   retracted: config declares no attention; the drawing was honest).
   Lesson locked into CLAUDE.md: sampling ≠ Dable.
4. **The fix batch** (all general, none per-model): VAE upsample flag +
   diffusers-placement pin (11 fixtures re-blessed); `scores_scale`
   declared-scale family (2 dialects, 3 projections); MQA-flag-defers-to-
   group-count + chatglm spellings → YAML; **the remote-code evidence
   rail** (auto_map fallthrough + diffusers-pipelines arch search +
   `_repo_id` stamp) with five extractor gaps closed en route; conv-GLU
   chain + cards; strip tag dedupe; cached=False on diffusion attention
   (SDXL's self-attn drill then honestly DEDUPED with CLIP's — 30→29
   views); vae mid-block dims. ~20 re-blesses total, every changed view
   pixel-reviewed, supersede recorded.
5. **Omni thinker towers unit**: modality-host walk through shared
   TEXT_WRAPPER_KEYS; container classes in both registries; stale-anchor
   recovery (`resolve_architecture_anchor` = unique construction root);
   config-class ownership (annotation spelling); audio lane in the unified
   fusion view with time-aligned position cards. 4 → 18 views, sable green,
   Qwen3-Omni as second witness. 450 tests green, 21/21 corpus locked (at
   that time — see Part 7 for current state).
6. **The acquisition conversation**: Soumil's critique ("a code change per
   model, and you keep saying it's general") → the honest decomposition
   (zero-change vs data-only vs latent-bug vs new-family models) → the
   family-over-field thesis and the unclaimed-signal net design → measured
   EXAONE baseline = 2.
7. **Her Eyes procedure** (new flow): the design/UX persona ingrained in
   `.claude/CLAUDE.md` beside Sable/Dable — five questions (delight,
   ceiling, bundling, newcomer, journey), images-only input,
   DISLIKE-must-suggest, lawfulness (bundles keep drills), auto-run after
   every bless, staleness by manifest diff. First mistake corrected: it is
   a PROCEDURE, not package code (a mistakenly created
   `model_unfolder/her_eyes.py` was deleted). Try-out reviews written for
   llama-7b (LOVE 2 / FINE 1 / DISLIKE 1) and sana-1600m (LOVE 5 / FINE 7 /
   DISLIKE 1). Her recurring themes: the layer-strip panel's font clash
   with the diagram family; glyph legibility for newcomers (cache badges,
   ×/⊕ grammar); repeat-idiom consistency (six literal VAE up-stage boxes
   vs the ×N frame); journey pacing verdicts ("the first five views carry
   Sana's story; encoder drills are appendix depth").
   A SECOND sibling procedure was announced by Soumil but not yet described
   — a slot is reserved in CLAUDE.md.
8. **MASTER_PLAN.md** (at workspace root, mirrored to MASTER_PLAN.docx):
   the full audit (35,783 package LOC / 131 files; evidence/ largest at
   9.6k; ranked 15-item consolidation table; ~700–1,000 LOC twin debt;
   support scorecard) + the 27-unit roadmap: Phase A shrink/single-source
   (A1 trivia → A2 one-grouping → A3 CodeModel → A4 scanner merge → A5
   data-as-code → A6 fusion views → A7 THE FOLD → A8 prose), Phase B
   acquisition engine (B1 unclaimed-signal net, B2 stream-scale, B3
   rope-scaling, B4 remote-code anchors, B5 width-bridge, B6 acquisition
   drill metric), Phase C coverage/honesty (C1 hybrid-SSM honest cells, C2
   MMDiT second column, C3 toserve_layers IR extensions, C4 audit regen,
   C5 corpus to ~35 witnesses), Phase D product hardening (D0 source-of-
   truth repair FIRST — corpus not in git!, D1 packaging, D2 API freeze +
   wiring_problems cache-pop bugfix, D3 test split, D4 CI, D5 JSON schema
   contract, D6 user docs generated-never-hand-written, D7 perf budgets +
   release gate), Phase E steady state. Deletion ledger; size governance
   rules; LOC target ≤ ~30k.
9. **RESTRUCTURE_PLAN.md** (companion, from a parallel session): the
   8-kind file taxonomy; workspace/repo/package/tests target trees (src/
   layout, var/ for ALL generated output, tools/audit.py unification,
   core//qa/ regrouping, everchanging loader split); R0–R6 migration
   sequence interleaved with MASTER_PLAN units; urgent defects: **live HF
   token in notebooks + .claude/settings.local.json (revoke FIRST)**, the
   red suite (below), corpus not in git, version triple desync
   (pyproject 0.2.16 / __init__ 0.2.15 / last tag v0.2.9 — releases
   0.2.10–0.2.16 shipped tagless), diverged protocol docs. My review:
   endorsed ~95%; corrections: the corpus-root rename fear is overstated
   (fixtures store gallery_dir RELATIVE, and the rename already happened);
   the her_eyes_review eviction from the corpus conflicts with Soumil's
   original spec — resolution proposal: reviews live in
   `docs/dev/reviews/<slug>.md` pinned to the reviewed hash_signature
   (bless's rmtree destroys in-folder reviews) — **Soumil must decide**.
10. **The 21-hardest-LLM sweep (2026-07-05, parallel session; AUDIT-ONLY —
    zero source changes):** full Sable + Dable + Her Eyes on the 21 hardest
    oracle-present LLMs (`previews/llm_sable_sweep_2026-07-05/` —
    _SWEEP_LOG.md, SLACK.md, 21 reports, 21 her-eyes reviews, 179 PNGs).
    Result: **6 fully clean** (DeepSeek-V3 gold-standard, DSV2-Lite,
    Qwen2-VL non-regressed, Mixtral, Bloom, Phi-3-mini), 15 with findings,
    **all 21 Dable-clean** (zero wiring/pixel defects). Every finding is one
    family: **evidence that exists (in code, or already in the IR) but is
    not consumed on the ship path** — i.e., the sweep is a Law-1 enforcement
    audit, and it found the remaining violations. 12 ranked systemic
    findings; top two: (1) config-silent QK-norm dropped because detection
    lives only behind `inspect_code=True` (Qwen3/Gemma-3/OLMo-2/OLMoE miss;
    Llama-4 fabricates on NoPE layers by skipping the per-layer `use_rope`
    gate; StableLM proves the config-declared path healthy); (2) partial
    rotary drawn as full on 5 models (`rope_dim` in the IR, no render
    rule). Also: router facts from identity strings (GLM-4.5 sigmoid drawn
    softmax + dropped noaux bias; Phi-3.5-MoE sparsemixer), LayerNorm drawn
    RMSNorm via the `rms_norm_eps` spelling (PhiMoE, Persimmon),
    two-norms-drawn-as-one in parallel-residual layers (Falcon-40B, NeoX;
    GPT-J genuinely single = the count-don't-assume control), Llama-4 MoE
    drawn dense (`moe_layers` unread — and op_conformance validated the
    WRONG class and passed), GPT-J param undercount (n_inner=None fallback
    not derived), Granite embed/logits multipliers unshown, ParallelExperts
    fused-drawn-split, ChatGLM closure-resolver gaps.

---

## PART 7 — CURRENT STATE, RIGHT NOW (2026-07-12, evolving — rewrite me, never append)

**The active work is the EVIDENCE ARCHITECTURE HARDENING CAMPAIGN.** The single
source of truth for it is `unfold-pkg/docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md`
— read its §6 tracker and §15 (current state for judgment) for exact unit status;
do not duplicate it here. In one breath:

- **Goal:** make architectural fabrication difficult BY CONSTRUCTION (typed
  evidence → registry → projection → symmetric conformance), not just forbidden by
  doctrine. Units H0–H10.
- **Status (AUTHORITATIVE = plan §16, the independent-audit recovery plan; it
  CORRECTED the earlier optimistic DONE claims):** no H0–H4 unit is DONE. NAS
  quarantine = DONE sub-unit. **H0/H1/H2 = ACTIVE** (real gaps: H0 display-pin +
  exemption evasions; H1 public `migrated_legacy` bypass + provenance loss; H2 sees
  only top-level extras). **H3 = RESTART REQUIRED** — the global bare-name
  `_touched/_bound/_consumed` accounting is the wrong abstraction (records absent
  fields as consumed, loses the alias, unions names across components); it is being
  replaced by an owner-scoped `ConfigAccessEvent` ledger, and the 3 audit-clearing
  diffusion reads are being removed. **H4 = early slice only.** Recovery execution
  order (plan §16): fixture isolation → H0 → H1 → H2 → H3 restart → H4 → H5/H6/H9-core
  → THEN H7/H8. Do NOT jump to H7 (decision D2).
- **In progress right now:** §16 recovery, in its mandated order. **Items 1–4 DONE
  and verified; item 5 (H1) active.**
  - **Item 1** (Section 16 authoritative + §6 tracker correction + Part 0 keystone +
    Part 7 versioned) DONE.
  - **Item 2** (fixture isolation) DONE: shared test configs in an importable
    top-level `test_support` package; every cross-test import removed (incl. the
    `import test_tower` hack); blocking static gate (`tests/test_isolation.py`) with a
    firing poison; `pyproject` `pythonpath=["."]`; affected-files suite 240 passed.
  - **Item 3** (commit NAS quarantine separately) — Soumil's act.
  - **Item 4 (repair H0) DONE** — the filename/table-name exemptions are replaced by
    a **lawful-resource manifest** (`_LAWFUL_TABLES`: 21 tables, each carrying
    path·table·category·permitted-consumers·content-fingerprint). The blanket
    `conformance/` exemption is gone (a table is lawful only if REGISTERED). Detection
    is single-entry + single-capital + dict-comprehension aware. The function-name
    exemption set is replaced by typed `@identity_address`/`@identity_display`
    wrappers (`evidence/identity_roles.py`, 3 sites). 12 E-criteria poison controls
    (`docs/H0_EXIT.md`). Measuring first (G2) caught a would-be regression — the
    `role=Class` marker tables — before it shipped. Verified: 25 alone + 145
    blast-radius, tree quiescent. Awaiting Soumil's commit.
  - **Item 5 (make H1 sound) DONE** — `migrated_legacy` is now `init=False`
    internal provenance set only by the private `_lift` path (via `from_record`), so
    a native caller can no longer opt out of the negative-proof law; a derived
    NEGATIVE requires complete effective completeness; `reason` (human) is separated
    from a stable `legacy_source` label — a reason is never serialized as source;
    structured `SourceSpan`/config paths survive the typed channel. Verified: 330
    passed alone (incl. the `context.py` FactLedger blast-radius + corpus round-trip),
    tree quiescent. `docs/H1_EXIT.md`.
  - **Item 6 (complete H2) DONE — both parts, verified.** *Part A (write-side):*
    `FactLedger.record_typed` validates every typed write against the registry
    (key/owner/status/value-type/negative-completeness) — a typed author can no
    longer bypass the closed world; typed `legacy_asserted` is REPRESENTED, not
    laundered into `asserted` (still serializes `asserted` so the 641 baseline holds).
    *Part B (`evidence/structural_writes.py`):* a line-insensitive `StructuralWrite`
    census keyed by sink·target over a **202-entry surface** (`ledger`, `spec`,
    `spec_field` (78), `extras` incl. nested leaves, `opgraph` (88 kinds), `card`,
    `params`) — a static no-growth + no-stale gate, a **structured legacy register**
    (`LegacyExtrasWrite`, owner·reason·unit·deletion — replaces the bare
    `RAW_EXTRAS_BASELINE`), a runtime top-level gate over the corpus, and **5 poisons**
    (one per sink). A new structural author cannot bypass the registry through a
    different representation. Measuring first (G2) caught the static-vs-runtime gap
    (dynamic `unet.*` leaves) and a would-be `role=Class`-style miss before shipping.
    Verified: 359 + 41 passed, corpus `asserted` baseline 641 intact, tree quiescent.
  - **⚖️ CHAIN OF AUTHORITY (evolved 2026-07-14). Three successive Soumil vets
    govern; each superseded the last as TRACKER while ratifying retained work:**
    1. *(retired as tracker)* `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` — the
       §16-ledger/§20-runbook vet of the 9-procedure marathon. Its U0/U1 ran as
       commits `e3e8b85` (U0, Claude) + `b2316f5` (U1, Soumil committed).
    2. *(retired as tracker; REC log = history)*
       `U0_U1_RECOVERY_EXECUTION_PLAN.md` — found R-01..R-11 (bind-as-consumed,
       owner+leaf paths, ambiguity→0, null=missing, repr equality, re-entrant
       dup, vacuous nets, MY config-authored facts incl. the vision
       width-comparison heuristic, 15/25 diffusion IR witnesses drifted with my
       blast using the wrong comparator). Executed as **REC-0..REC-7 =
       `50d2408, a14341b, 33ca12e, b7b859c, 4677c95, d720a32, 1f3316c,
       cc5e328`** — all receipted; final full suite 1152p+5xf, 25/25 witnesses
       at U0 parity, fingerprint-bracket VALID.
    3. **ACTIVE + BINDING: `U0_U1_FINAL_RECOVERY_CORRECTION_PLAN.md`** —
       defects C0–C4 (clean-checkout preservation not self-contained;
       primitive/null/legacy semantics; approximate joins; ambiguity still
       reaches `dim 0`/SiLU retries; flattened modality scopes + generic
       projector width). Execute **COR-0→COR-5**, one reviewed commit each,
       reproduce-defect-first; §13 forbidden shortcuts (no or-0, no
       first-alias, no retry-after-ambiguity, no skips, no family branches);
       U0/U1 not DONE and **U2 locked** until the §12 receipt is factual and
       Soumil approves the §11 visual matrix.
    **⚡ PUSH RULE FLIPPED (2026-07-14, explicit):** Soumil: "Why aren't you
    pushing any commits… Please do that." → push origin after every unit
    commit from now on (first push `7aeb815..7cd9066`; blessing stays
    Soumil-only). Branch `audio-composite-support` on `SoumilB7/unfold`.
    **COR state (live, 2026-07-14):**
    - **COR-0 DONE `602dcb0` (pushed):** C0 reproduced on `git archive HEAD`
      (FileNotFoundError + skip), then closed — 25 corpus inputs COMMITTED
      (galleries ignored at exact path), authoritative
      `preservation_expected_manifest.json` (25 witnesses × surfaces + 321
      view hashes via production `svg_views`/`_visual_hash`, none None),
      exact-ROOT-only mount normalization, zero-skip verifier (regenerates
      fresh, validates EVERY hash), 6 poisons @limit=1, clean-export
      isolation test green POST-commit.
    - **COR-1 DONE `b8cea8c` (pushed):** event `value_state` law
      (missing|explicit_null|value; consume/ignore of missing = constructor
      error); big-int exact equality (2⁵³ trap reproduced→killed);
      consume-requires-target; ignore-refuses-absent/ambiguous; null-beside-
      value = AMBIGUOUS by default w/ named-policy escape (corpus-measured:
      ZERO rival pairs); `resolve_aliases` DELETED (10 callers migrated);
      funnel carries value_state; unparsed null-skip removed → unmasked 7
      explicit-null declarations → `PendingConfigClassification` (COR-2 type
      forward-pulled per §3.2) w/ exact owner+path+deletion-unit entries,
      visible in `config_audit.pending_classification`.
    - **COR-2 DONE `7cd9066` (pushed):** `ConfigOccurrenceKey` = PRIMARY join
      (owner_field demoted to debug); `ProjectionTarget`+`ProjectionObligation`
      structured rows (projected|pending|scoped_ignored|unreceipted);
      exact `config_path` on all 9 projection-debt entries + exact-path
      parser join; `projection_receipts_available=False` PUBLISHED (empty ≠
      proof); sable net-2 reads structured state; 8 §7 counterexamples green.
    - **COR-3 DONE `ff92b4a` (pushed):** all 3 counterexamples reproduced at
      depth then killed — IR `hidden_size: int|None`; params return
      `{"total": None, "incomplete": reason}` never zero-math; geometry
      None-preserving in BOTH parsers (surviving `or 0` = annotated iteration
      counts; 6 range sites guarded); activation resolves ONCE as a family
      (gelu/silu conflict renders BYTE-IDENTICAL to control + Sable blocks);
      ambiguity emission idempotent. 4 permanent depth-guards green.
    - **COR-4 IN FLIGHT (Part A landed):** `current_container` ContextVar +
      `config_container`/`container_scoped` — legacy `owner:leaf` labels
      RETIRED (qwen2-vl vision: 46/46 events exact `vision_config.*`);
      `path_exact` event field (resolver/container-scoped = exact; bare
      funnel leaf = transitional fallback, preserving R-04); container
      scopes at modality builder (matched registry key), encoder slots
      (`_text_encoder_configs.<slot>`), `_vae_config` (geom + chips),
      `_scheduler_config` (+ top-level slot-key escape), rope containers
      w/ wrapper prefix (`text_config.rope_parameters.*`). All-25 regression
      sweep RUNNING. **Part B designed, next:** projector out-width becomes
      SOURCE-authoritative — extend `evidence/projector.py` (ProjectorEvidence
      gains out-width binding from the resolved PatchMerger-style constructor
      call-site kwargs → config attr), `apply_projector_evidence` sets
      `out_features` from the bound field, DELETE the generic
      `text_hidden_size` fallback arms in `vision_projector_out` (qwen2-vl
      measured: in=5120/out=3584, evidence PROVEN PatchMerger 5-ops — the
      3584 must flow from the source binding, keeping the pixel identical);
      then 7 §9 counterexamples, gate, commit+push.
    - **COR-5 PENDING:** gate cutover (Net-1 blocking for migrated sources;
      Net-2 where receipts declared) + §12 receipt + §11 FRESH visual matrix
      → Soumil approves → U0/U1 DONE, U2 UNLOCKED.
  - **THE OVERNIGHT MARATHON (2026-07-13, history — deliverables real, statuses
    superseded above): eradicate ALL H's, each as a numbered
    `procedure N` commit (Soumil's standing instruction — see
    [[feedback-user-commits-himself]]). Per unit: implement → VET → foresee the flaw a
    §16 audit would find → fix that too → verify → commit → evolve THIS doc.**
    Procedures landed:
    - **`procedure 1` (daf056f)** — recovery items 1-6 (fixture isolation + H0 + H1 +
      H2 both parts) + docs cleanup + H3 steps 1-2 substrate.
    - **`procedure 2` (22ddaf0) — H3 RESTART cutover DONE, verified 505 passed** (a
      fast subset — the re-vet later found this smoke EXCLUDED the slow `test_sable.py`,
      which held one un-ported `capture_accesses` call; fixed in `procedure 9`). The
      owner-scoped `ConfigAccessEvent` ledger is now the SINGLE source: `emit` funnels
      through `note_access`; the global `_touched/_bound/_consumed` + `capture_accesses`
      are DELETED; `config_audit`/`config_consumed` DERIVED from the ledger
      (behavior-neutral, compat==old exactly). Sibling scoping: `owner_scope`/
      `owner_scoped` → `root.vision`/`root.audio` (modality-registry loop) + `root.vae`
      (diffusor). The live `config_accessed_unprojected` net reads the owner-scoped
      ledger, per-owner gated (diffusion inert), with address/identity reads
      scoped-`ignored`. **Honest end-states:** the BLOCKING flip needs every decision
      site on `consume` (real fields like `rms_norm_eps` stay read-not-consumed until
      then — H8); net-2 (consumed-but-unprojected) is substrate-ready+tested, its live
      wiring needs accurate fact-targets from the H8 migration. Both documented, not
      hidden.
    - **`procedure 3` (ea15aff) — H4 taint net DONE.** `scan_identity_source` now
      catches a class-name value compared to ANY string whose branch WRITES A
      STRUCTURAL SINK (spec/opgraph ctor, or a dict keyed by a structural term) — the
      fabrication the domain-marker predicate missed. It distinguishes UNLAWFUL
      identity→structure from LAWFUL code-shape (returns a role string/bool), so the
      already-clean production tree stays at **0 identity debt** — the net is
      PREVENTIVE and closes the pinned substring-nonmarker boundary (its `xfail`-style
      test FLIPPED). Added the renderer/parser **dependency firewall** gate (no
      `renderers/` module may import the parser/adapter layer or a config accessor —
      clean, pinned). VET note: interprocedural + mapping-lookup taint are
      preventive-future (0 real flows on the clean tree), honestly deferred.
    - **`procedure 4` (706818b) — H5.** No-broad-reader-exception RATCHET
      (`tests/test_reader_exceptions.py`): a new broad `except` anywhere fails the
      build, a shrunk count forces its baseline down (monotone shrink); 2 evidence
      readers converted to typed (conformance/decoderness config-coercion → 8→6).
    - **`procedure 5` (a969b0f) — H6.** Registry-driven **reverse-fabrication audit**:
      every leaf the renderer DRAWS must be a REGISTERED fact or structured pinned
      debt (`DRAWN_UNLEDGERED_DEBT`, the 6 drawn-but-unledgered leaves w/
      owner/reason/unit); **projection obligation** (registered-drawable ⊆ drawn).
    - **`procedure 6` (6447e42) — H9-core.** The reusable **metamorphic harness**
      (`test_support/metamorphic.py`): five relations (rename / collision /
      missing-source / partial-source / equivalent-control), proven on decoder +
      multimodal + diffusion; missing-source FIRES non-vacuously.
    - **`procedure 7` (6ed88a3) — H7 (diffusion).** The 3 removed reads return as
      REGISTERED typed facts with declared **pending-projection debt**
      (`PENDING_PROJECTION_DEBT`); metamorphic harness holds on FLUX.
    - **`procedure 8` (10f6fa5) — H8 (transformer).** The migration RAIL + one full
      migration: **`sinks` : drawn-but-unledgered → a REGISTERED code-proven fact**
      (parser records it from the attention forward, gpt-oss witnesses it, removed
      from the drawn-debt). 63 passed (corpus census + projection-audit intact).
    - **`procedure 9` (d475d87) — H9/H10 + the FORESEER RE-VET. DONE, gate green:
      FINAL full suite 1093 passed / 0 failed (no `-x`).** H9 frontier metamorphic
      matrix (5 archetypes, 6 passed). Then a full adversarial re-vet of the whole
      marathon ("check everything"), which surfaced **TWO REAL defects the fast-test
      strategy had hidden** — both in `test_sable.py`, the slow file the procedure
      smokes skipped for throughput.
      **DEFECT 1 (test-only):** `procedure 2`'s H3 cutover deleted
      `debug.capture_accesses` but left
      `test_sable.py::test_config_access_capture_survives_nested_reset...` still
      calling it — proc-2's "505 passed" smoke had EXCLUDED `test_sable.py`, so the
      miss slipped through (the pre-fix release run confirmed it: 0 failures through
      79%, then it would have hit that test in the final 21%).  **Fixed** by porting
      the test to the owner-scoped `capture_events()` ledger + `ledger.touched_names()`
      (now a regression guard that `reset()` stays a no-op).  A suite-wide sweep
      (every `debug.<attr>` call + every `config_access` import re-resolved against the
      live modules) found **NO sibling deleted-API references**.
      **DEFECT 2 (production — an H7 root-cause incompleteness):** `test_sable_regression_corpus`
      regressed on **10 items across 8 models** — `_vae_config.act_fn` (8×),
      `_vae_config.temporal_compression_ratio` (hunyuanvideo), `max_sequence_length`
      (mochi).  Root cause: `procedure 2` removed those 3 diffusion reads *believing
      `config_field_audit` was ADVISORY* (the parser comments said so); it is
      **BLOCKING** (since 2026-07-04).  `procedure 7` registered the 3 fields as
      `PENDING_PROJECTION_DEBT` but never resolved the blocking failure — and no
      procedure smoke ran `test_sable_regression_corpus`, so it stayed hidden until
      the re-vet's full run.  Owner-scoping (proc 2) also correctly UNMASKED these:
      the denoiser's `act_fn` no longer name-collides with the VAE's, so the VAE's is
      honestly flagged.  **Fixed** the honest way: not by re-adding the reads (proc-7
      pins a `test_the_three_reads_are_still_removed` guard, and a bare re-read is a
      silent audit-clearing read), but by teaching `config_field_audit` that a field
      **registered as pending-projection debt is a DECLARED classification** — a
      fourth resolution beside parse / chip / ignore.  The parser excuses exactly the
      registry's pending canonicals (bounded; a NEW unread field still blocks), so the
      reads stay removed (proc-7's guard holds) AND the honest "removed until the
      H7-full reader draws it" state clears the blocking net.  Added
      `test_config_field_audit_excuses_the_pending_projection_fields`.  Verified:
      test_h7 5-passed, projection/config/registry 50-passed, config_field_audit CLEAN
      on all affected models, blast-radius 0 drift (no re-bless).
      Re-vet evidence that HELD: config accesses are owner-partitioned
      (FLUX 159 root / 42 vision / 13 vae, the VAE's structural fields owned by
      `root.vae` not smeared into root); every campaign net BLOCKS (H4/H5/H6/H7/H8 +
      metamorphic) while `config_accessed_unprojected` is advisory-BY-DESIGN (its
      non-blocking status is itself pinned by a test); 0 blessed artifacts changed by
      the whole marathon (sinks draws from spec → byte-stable bless); token-safe (0
      secret hits across all diffs). H10 = the full-suite release run: its FIRST pass
      (986 passed / 1 failed) is what CAUGHT DEFECT 2 at `test_sable_regression_corpus`
      — proof the full corpus render is the real gate; the FINAL re-run with both
      fixes landed **1093 passed / 0 failed (27:57, no `-x`)** — the first complete
      green full-suite pass of the campaign tree, covering the alphabetical tail
      (`test_smoke`→`test_vision_evidence`) no earlier pass reached. Committed as
      `procedure 9` (d475d87) only after that green, per "fix it first".
      **Process lesson (Part 14):** a fast-subset smoke is not a release gate — BOTH
      defects lived in `test_sable.py`, the one file every procedure smoke skipped for
      speed; the "audit is advisory" premise in DEFECT 2 was PROSE in a comment, never
      a mechanical assertion, so nothing forced it true (the §16 anti-pattern again).
    - **HONEST SCOPE (Part 0.4 — never fake completion):** the campaign's SUBSTRATE
      and GATES are complete and verified (H0-H6 + H9-core + the H3 restart). H7/H8
      are inherently per-fact-family / per-mechanism migrations that §16 itself says
      to do "one at a time" — each committed procedure lands the RAIL + one complete
      migration + the metamorphic contract; the remaining families/mechanisms follow
      that exact rail incrementally. That is stated plainly, not hidden as "done".
    Steps 1-2 recap: *Step 1* removed the 3 audit-clearing diffusion reads (verified
    no structural consumer); *Step 2* built the `evidence/config_access.py` substrate
    (`ConfigAccessEvent` 10 fields, owner-qualified joins, `resolve_aliases`,
    `current_owner`) with 16 counterexamples — behavior-neutral first, as H1/H2 were.
    Then items 9–13 (H4 → H5/H6/H9-core → H7/H8; not H7 first, D2).
- **The process lesson (Part 14, binding):** every implementation error this
  campaign came from a rule that was PROSE, not a mechanical gate — which is exactly
  what the §16 audit re-proved by finding the over-claimed DONEs. Standing rule:
  re-sync with the plan before each unit, MEASURE before designing, run the §8.1
  pre-change blast-radius (incl. tests + nets) before any edit, and treat DONE as a
  checklist of passing exit-tests plus an unchanged tree fingerprint, never a
  judgment.
- **Prior arc, LANDED (history, not current work):** the honesty-wave units + the
  audio-gen support (PR #14) — see Parts 6/12/13 and their SURGICAL_PLAN docs.

**Standing invariant facts (still true):**
- **HF token** is plaintext at the WORKSPACE ROOT (finalize.ipynb,
  done/tryrun.ipynb, .claude/settings.local.json) — Soumil decided it's OK (purely
  local, never in git). Never print it.
- **Corpus/galleries live on ONE DISK only** (gitignored) — backup is the standing
  safety item; fresh clones run a corpus-coupled suite.
- **Soumil commits and blesses; Claude never runs `git commit` and never
  refreshes a blessing.** Everything in the campaign is uncommitted for his review.
- **Open decisions** for the campaign live in the plan's §15.3/§15.6 (D1–D4), not
  here.

---

## PART 7.5 — THE SWEEP-FIX MAP (each fix CONFIRMED at its code site)

Every entry below was re-verified against the live source on 2026-07-05 —
file:line checked, not taken from the sweep report on faith. None of this
is new scope: **all of it is Law 1 (evidence-never-identity) enforcement on
the ship path**, the exact frontier Part 3 ends on.

---

### ✅ UNIT (COMPLETED 2026-07-05/06 — kept for its principles/gate-taxonomy):
### MoE-vs-DENSE LAYER SCHEDULE FROM CONSTRUCTION EVIDENCE

**Why / mandate.** After S2, the DISTRIBUTION register (§5) has NO active
wrong-drawing config *lie* left, but several "sometimes-there, sometimes-not"
config dependencies remain that silently mis-draw STRUCTURE when a checkpoint
omits/renames the field. Soumil wants to "eradicate config-based uncertainty."
The #1 target (biggest structural damage) is **which layers are MoE vs dense**.
Soumil's hard constraint for this unit: **"we should never lose the ability
that a second tower CAN open an MoE"** — the fix must be tower-aware; a
SECONDARY tower / sub-model that is itself MoE (a MoE thinker, a MoE encoder)
must still detect + render + drill its MoE. Do NOT make MoE detection
root-decoder-only.

**THE EXACT BUG (verified in live source 2026-07-05).** Llama-4-Scout's whole
MoE is drawn DENSE. Precise mechanism:
- Config (`llama4_text`): `moe_layers = [0..47]` (the LIST of MoE layer
  indices), `interleave_moe_layer_step = 1`, `num_local_experts = 16`,
  `num_experts_per_tok = 1`. (Real 16E/128E checkpoints make `moe_layers` a
  SUBSET — interleaved dense/sparse.)
- Code (`modeling_llama4.py`): `self.is_moe_layer = layer_idx in
  config.moe_layers`; `if self.is_moe_layer: self.feed_forward =
  Llama4TextMoe(config)` else `Llama4TextMLP(...)`.
- Our two failures: (a) we NEVER read `moe_layers`; (b) `_is_dense_at_layer`
  (`parser.py:1266`) has the rule `if interleave_moe_step and (i %
  interleave_moe_step == 0): return True`, and Llama-4's
  `interleave_moe_layer_step = 1` makes `i % 1 == 0` true for EVERY layer →
  ALL layers marked dense. So the entire 48-layer MoE renders as dense SwiGLU.
  (op_conformance passed anyway because the mis-tagged dense region got
  validated against `Llama4TextMLP` — the wrong class.)

**THE FIX SHAPE = the QK-norm template, applied to the FFN field's CLASS
ROLE.** This is byte-for-byte the same discovery as QK-norm: *the code gates
the construction on a config field it NAMES, indexed per layer.* Llama-4:
`layer_idx in config.moe_layers` is the exact analogue of QK-norm's
`config.no_rope_layers[layer_idx]` subscript gate (I already built per-layer
gate resolution in `_resolve_qk_norm_layers` + `_ConfigExprEvaluator`). The
general reader analyzes the DECODER LAYER's `__init__` for how the FFN-role
field is constructed:
1. **Unconditional MoE-role class** (Mixtral `self.block_sparse_moe =
   MixtralSparseMoeBlock`) → all layers MoE.
2. **Conditional** `if <gate>: feed_forward = <MoE-role> else <MLP-role>` →
   extract the gate, evaluate per-layer against config. Gate forms seen:
   - membership: `layer_idx in config.moe_layers` (Llama-4) — NEW form, add
     to the evaluator (an `ast.Compare` with `In` op over a config list).
   - threshold: `layer_idx >= config.first_k_dense_replace` (DeepSeek dense
     prefix).
   - modulo: `(layer_idx + 1) % config.decoder_sparse_step == 0` (Qwen3-MoE),
     `layer_idx % config.moe_layer_freq` — the `_ConfigExprEvaluator` already
     does `%`, `+`, comparisons need adding.
   - exclusion: `layer_idx not in config.mlp_only_layers`.
3. **Unconditional MLP-role** → all dense (no MoE).
Role of a constructed class = `_role_of` (ffn/route/etc.); an MoE block is the
class the layer builds that `_has_moe(...)`. Config supplies the VALUES
(the lists/ints); code supplies the SHAPE + names the fields.

**CURRENT PLUMBING MAP (root decoder path, file:line):**
- `moe_active = bool(num_experts) and enable_moe_block is not False`
  — `parser.py:652`.
- schedule config reads — `parser.py:653-665` (`first_k_dense_replace`,
  `moe_layer_freq`, `interleave_moe_layer_step`, `decoder_sparse_step`,
  `mlp_only_layers`); `moe_every_layer` derived `:660`.
- per-layer decision — `is_dense_at_layer = _is_dense_at_layer(i, ...)`
  `parser.py:791-799`; reader `_is_dense_at_layer` `parser.py:1266`.
- FFN spec built MoE iff `moe_active and not is_dense_at_layer`
  `parser.py:801-816` (else dense `:817`).
- WIRING PLAN: add `_code_moe_schedule(text_cfg, context, num_layers)` (11th…
  well, Nth `_code_*` reader) returning a per-layer `list[bool]` (MoE?) or
  None; reader `decoder_moe_schedule_from_files` in `evidence/patterns.py`
  (mirrors `decoder_qk_norm_from_files` + reuses `_find_decoder_layer` and
  `_ConfigExprEvaluator`). Resolution: `code list` wins when present; else the
  existing `_is_dense_at_layer` config path (fallback, unchanged — so no
  drift for models it already gets right). Fix the `interleave_moe_step==1`
  inversion regardless (a step of 1 means every-layer, not every-layer-dense).

**THE SECOND-TOWER CONSTRAINT — RESOLVED (how to satisfy it safely).**
There are TWO tower paths, and they treat MoE differently:
1. **Recursive sub-model** (`submodel.py:52 submodel_spec`; MoE drill built at
   `submodel.py:298` when `group["ffn"]["kind"]=="moe"`, children at `:310
   _moe_children_from_fact`). These groups come from a RECURSIVE PARSE of the
   sub-config through the SAME transformer parser — so a MoE sub-model (an
   embedded MoE LM, Qwen3-Omni-style MoE thinker) gets `ffn.kind=="moe"` from
   the SAME root `moe_active`/`_is_dense_at_layer` path applied to ITS config.
   **THIS is the "second tower can open an MoE" ability** — it must survive.
2. **Supporting tower** (vision/audio/refiner/encoder via
   `layer_facts_from_block` → `tower_submodel_spec`, `modalities/schema.py`)
   **HARDCODES `"ffn": {"kind": "dense"}` at schema.py:132** — these towers
   CANNOT open an MoE today (a separate pre-existing limitation; NOT something
   this unit breaks, and a possible future enhancement — a MoE encoder would
   currently draw dense).

**⇒ The design that SATISFIES the constraint by construction: fallback-safety,
exactly like QK-norm.** `_code_moe_schedule` OVERRIDES the schedule only when it
POSITIVELY resolves the FFN-field construction from the CURRENT parse's context
source (`_source_files(cfg, context)` — already per-parse, so a recursive
sub-model with its own context reads its OWN layer class; the existing 11
`_code_*` readers already work this way). When it can't resolve (returns None) →
fall back to the existing config `_is_dense_at_layer` path UNCHANGED. So:
- root decoder w/ source → code-first (fixes Llama-4);
- recursive MoE sub-model w/ its own source ctx → code-first on the sub-model
  (works, better);
- recursive MoE sub-model w/o source ctx → config fallback → the MoE STILL
  renders because `moe_active` (num_experts on the sub-config) + the config
  schedule already produce it TODAY — **the second-tower MoE ability is
  preserved verbatim**;
- supporting tower (schema.py hardcoded dense) → my change never touches that
  code path → untouched.
Net: the reader is a fallback-safe OVERRIDE on the root path only; it can only
ADD correctness (Llama-4) and never removes the recursive-tower MoE. A test
MUST assert a recursive MoE sub-model still opens its MoE drill (Qwen3-Omni
thinker, or an embedded MoE LM) — the explicit guard for Soumil's constraint.

**NEGATIVE CONTROLS / must-not-drift:** Mixtral (unconditional MoE → all MoE),
DeepSeek-V3 (dense prefix `first_k_dense_replace`), Qwen3-MoE
(decoder_sparse_step schedule), gpt-oss (blessed — its schedule), a dense
model (llama → no MoE reader fires). Blessed MoE fixtures = deepseek-v3,
gpt-oss-20b, glm-4-5 — all must stay byte-identical (their config path already
draws right; the code reader must AGREE or the fallback preserves them).
Llama-4 is NOT blessed, so fixing it adds no drift (it's a fresh-render win).

**VERIFICATION PLAN:** direct-call tests on synthetic layer shapes (each gate
form: membership/threshold/modulo/exclusion/unconditional-MoE/unconditional-
MLP); real integration (Llama-4 → interleaved/all MoE per config; Mixtral all
MoE; DSV3 dense-prefix; Qwen3-MoE schedule; a MoE tower witness renders MoE);
negatives (dense model, and the blessed MoEs byte-stable); full suite +
25-fixture regression zero-drift; render Llama-4 + the MoE-tower witness,
Sable + Dable pixel pass; bless Llama-4 (+ the tower witness) → corpus grows.

**EMPIRICAL DIAGNOSTIC (2026-07-05, drawn-vs-code-truth per layer across 5
MoE families) — NARROWS THE FIX:**
- **ONLY Llama-4 is broken.** Its drawn pattern = 48 dense (`.`); truth =
  all/interleaved MoE. The config path draws CORRECTLY today for Mixtral
  (all MoE), DeepSeek-V3 (`...MMM…` — 3 dense-prefix then MoE), Qwen3-MoE,
  Granite-MoE — all MATCH code truth. ⇒ the fix must be FALLBACK-SAFE to keep
  these 4 byte-identical; do NOT rewrite the working config path, ADD a
  code-first override that only fires when it resolves construction.
- **Llama-4's gate has a SELF-FLAG INDIRECTION** (why a naive gate-eval sees
  `?`): `self.is_moe_layer = layer_idx in config.moe_layers` then
  `if self.is_moe_layer: self.feed_forward = Llama4TextMoe else
  Llama4TextMLP`. The construction is gated on `self.is_moe_layer`, a flag
  assigned from a MEMBERSHIP test — the EXACT self-flag pattern already solved
  in `_attention_qk_norm` (`self.qk_layernorm = config.qk_layernorm`), PLUS a
  new membership (`in`/`not in`) gate form. So the reader MUST: collect
  `self.<flag> = <gate>` assigns (the `flags` dict, mirror _attention_qk_norm),
  follow `if self.<flag>:` back to its gate, and evaluate the gate per-layer.
- **Gate forms to evaluate per-layer** (a small dedicated evaluator, NOT the
  numeric _ConfigExprEvaluator — these are BOOLEAN): `layer_idx in config.LIST`
  (Llama-4), `layer_idx not in config.LIST` (mlp_only_layers), `layer_idx >=
  config.INT` (dense prefix), `(layer_idx [+1]) % config.INT [== 0]` (sparse
  step), AND/OR BoolOp combinations. Follow self-flag indirection first.

**CRUCIAL GENERALITY FINDING (Soumil's question 2026-07-05: "is Llama the only
one using this format, or is everyone doing it and you used a different
indicator?"). Answer: EVERYONE uses the SAME code format — construct an
experts-building class as the FFN field, gated or unconditional. I detected the
4 "working" families via a DIFFERENT indicator (the CONFIG schedule fields),
which agreed by coincidence of our having readers for their spellings; Llama-4
fell through only because `moe_layers` isn't read + the interleave inversion.
The question forced the key insight: THE CLASS NAME IS NOT A RELIABLE MoE
SIGNAL — verified 2026-07-05:
- Qwen3-Moe / Granite-Moe / Phi-Moe put "moe" in EVERY class name
  (`Qwen3MoeAttention`, `Qwen3MoeRMSNorm`) → name-matching FALSE-POSITIVES;
- gpt-oss's MoE class is literally named `GptOssMLP` → name-matching
  FALSE-NEGATIVE.
⇒ MoE-vs-dense MUST be detected STRUCTURALLY: does the constructed FFN class
BUILD MULTIPLE EXPERTS (an `experts`/`num_experts`/router field). Verified
name-independent across ALL 7 classes: every real MoE class → True (incl.
GptOssMLP), every plain MLP → False (Llama4TextMLP, DeepseekV3MLP, Qwen3MoeMLP,
LlamaMLP). This makes the reader genuinely general, not Llama-4-specific.

**FIX PARTS (MEDIUM, not big — reported to Soumil):**
1. `decoder_moe_schedule_from_files(files, cfg)` in `evidence/patterns.py`
   (~70 lines): `_find_decoder_layer` → layer __init__ → find the FFN-field
   construction (unconditional OR gated if/ternary with two branches);
   classify each candidate class STRUCTURALLY (builds experts = MoE, else
   dense — NOT by name); collect self-flags for indirection; evaluate the
   per-layer gate (membership/threshold/modulo/exclusion + AND/OR). Returns
   per-layer bool list, or None if unresolvable. New small boolean gate
   evaluator; new structural `_class_builds_experts` helper (field/src based).
2. `_code_moe_schedule(cfg, context, num_layers)` wrapper (parser) + wire at
   the per-layer loop: `code_sched` (list) wins per layer; else existing
   `_is_dense_at_layer` path UNCHANGED (preserves the 4 working families +
   recursive-tower MoE via fallback-safety).
3. Config-path belt-and-suspenders: read `moe_layers` LIST directly (layer i
   MoE iff `i in moe_layers`) so a source-less Llama-4-like still resolves;
   this also sidesteps the `interleave_moe_layer_step==1` inversion for
   Llama-4 without rewriting the interleave rule (only Llama-4 uses interleave
   among the tested families).
4. Test: RECURSIVE MoE sub-model still opens its MoE drill (Soumil's
   second-tower guard).

**SCOPE UPGRADE (Soumil 2026-07-05): FULL CONVERSION to code-based, senior-
engineer surgical.** Shift AUTHORITY entirely to code; config demoted to
oracle-missing fallback ONLY (a source-less model must still draw something
honest — never delete the config path, just stop trusting it when code is
present). Prove correct across a BROAD family sweep BEFORE flipping, because
the 5 working families are blessed and a false code verdict breaks them. Tri-
state law: proven (confident per-layer list) / unresolvable (None → config
fallback). NEVER a wrong guess — return None on any doubt.

**PHASE-0 RECON — THE MoE FAMILY MAP (19 families, verified in installed
source 2026-07-05).** Construction shapes + gate forms:
- **Unconditional MoE** (`self.x = MoEClass(config)`) → all-MoE: mixtral,
  granitemoe, gpt_oss, phimoe, olmoe, dbrx, minimax, hunyuan_v1_moe.
- **If-gated / ternary** (MoE vs MLP by a per-layer gate):
  - THRESHOLD `layer_idx >= config.first_k_dense_replace`: deepseek_v2,
    deepseek_v3 (= DeepSeek-R1, same modeling), glm4_moe, dots1.
  - EXCLUSION+THRESHOLD `(layer_idx not in config.mlp_only_layers) and
    (config.num_experts > 0 ...)`: qwen2_moe, qwen3_moe, qwen3_next.
  - MODULO+THRESHOLD `((layer_idx+1) % config.moe_layer_interval == 0) and
    layer_idx >= ...`: ernie4_5_moe.
  - MEMBERSHIP via SELF-FLAG `self.is_moe_layer = layer_idx in
    config.moe_layers; if self.is_moe_layer:`: llama4.
- **HYBRID SSM-MoE — OUT OF CLEAN SCOPE, reader returns None → config
  fallback (unchanged, no regression):** granitemoehybrid (per-layer
  `config.layers_block_type[layer_idx] == "mamba"` chooses mamba vs MoE, PLUS
  a `shared_mlp` field — multiple ffn fields, ambiguous), jamba
  (`ffn_layer_class = X if … else Y; self.feed_forward = ffn_layer_class(…)`
  — VARIABLE-class construction). These are C1 hybrid-SSM territory; do NOT
  handle their exotic forms now — returning None is the safe, honest call.
- **Named-by-Soumil coverage:** Qwen3-Next ✓ mapped (exclusion+threshold);
  DeepSeek-R1 = deepseek_v3 modeling ✓; DeepSeek-V2 ✓; **Kimi-K2** not a
  separate modeling file — uses `deepseek_v3` model_type ⇒ covered by the
  deepseek reader; **DeepSeek-V4** unreleased/not in transformers — if it
  follows the deepseek threshold lineage the reader handles it, else falls to
  config; add when it lands.

**GATE-FORM TAXONOMY the evaluator MUST cover (forms 1-8 = clean MoE; 9-10 =
return None):** (1) unconditional; (2) threshold `layer_idx >= config.INT`
(+ <,<=,>,==,!=); (3) membership `layer_idx in config.LIST`; (4) exclusion
`layer_idx not in config.LIST`; (5) modulo `(layer_idx [+k]) % config.INT
[== 0]`; (6) global `config.INT > 0` (no layer_idx → uniform all layers);
(7) AND/OR BoolOp of the above; (8) self-flag indirection `self.flag = <gate>;
if self.flag:`. (9) list-subscript string compare (mamba hybrid) → None;
(10) variable-class `cls = A if g else B` (jamba) → None.

**SURGICAL SAFETY RULES (senior-engineer, non-negotiable):**
- Identify the PRIMARY ffn field only; if the layer builds MULTIPLE ffn-role
  fields (shared_mlp + block_sparse_moe + mamba) → ambiguous → None → config.
- Structural MoE detection = the constructed class BUILDS EXPERTS (field/src:
  `experts`/`num_experts`), NAME-INDEPENDENT (gpt-oss MoE named `GptOssMLP`
  True; Qwen3Moe* attention/norm names → not the ffn field so irrelevant).
- Any unresolved gate atom / unknown form / missing source → None → config
  fallback. Never emit a wrong per-layer verdict.
- The sweep (Phase 2) must show code-reader ⊇ config-path correctness on
  EVERY installed family before wiring; blessed fixtures (deepseek-v3,
  gpt-oss-20b, glm-4-5) MUST stay byte-identical (code agrees with config).

**ROADMAP (6 phases, this is THE plan):**
- P0 RECON ✓ (family map + gate taxonomy above).
- P1 BUILD: `decoder_moe_schedule_from_files(files, cfg)` — structural experts
  detection + per-layer boolean gate evaluator (forms 1-8) + self-flag
  indirection + primary-ffn-field guard; None on any doubt.
- P2 SWEEP: code-reader vs config-path across ALL installed families; every
  divergence investigated; reader proven correct (config-path is the current
  baseline for the 5 that work; Llama-4 is where they diverge and code wins).
- P3 WIRE: `_code_moe_schedule` wrapper; code AUTHORITATIVE per-layer when
  present; config `_is_dense_at_layer` ONLY when code returns None; fallback-
  safe (recursive-tower MoE preserved). Fix the interleave==1 inversion is
  moot (code primary) but read `moe_layers` in config fallback anyway.
- P4 TEST+VERIFY: gate-form direct-call tests (each of 1-8 + hybrids→None),
  per-family integration, 2nd-tower guard (recursive MoE sub-model still opens
  MoE), negatives (dense model); full suite + 25-fixture zero-drift.
- P5 WITNESS+BLESS: render Llama-4 (now interleaved/all MoE) + a mapped-family
  witness if loadable; Sable + Dable; bless → corpus grows.
- P6 RECORD: PROJECT_CONTEXT + DISTRIBUTION (A2 MoE-schedule SHIFTED to code,
  the twin of A4 router; note hybrids consciously deferred to C1).

**P1 BUILD DONE + P2 SWEEP DONE (2026-07-05) — reader proven correct across
17 installed families:**
- Reader `decoder_moe_schedule_from_files(files, cfg)` +
  `_class_builds_experts` (AST-only, name-independent) + `_MoEGateEvaluator`
  (threshold/membership/exclusion/modulo/global/AND-OR + self-flag
  indirection) landed in `evidence/patterns.py`. Returns per-layer bool list
  or None-on-any-doubt.
- **SWEEP RESULT (code-reader vs config-path, all installed MoE families):**
  - **12 AGREE byte-for-byte** (stay identical): mixtral, qwen2_moe,
    qwen3_moe, qwen3_next, deepseek_v2, deepseek_v3, glm4_moe, granitemoe,
    gpt_oss, phimoe, olmoe, minimax.
  - **llama4 DIVERGE → code RIGHT** (all-MoE; config drew all-dense) = THE fix.
  - **ernie4_5_moe DIVERGE → code RIGHT** (layer 0 dense via `(i+1)%interval==0
    AND i>=moe_layer_start_index=1`; config drew all-MoE, blind to ernie's
    spelling) = a SECOND bug the sweep FOUND. Proof the code shift adds
    correctness, not just parity.
  - **dots1 DIVERGE → NOT REAL** (isolated-test artifact): its default config
    has `n_routed_experts=None` (no experts) ⇒ `moe_active=False` ⇒ all dense
    is correct; the reader tested alone lacks the moe_active gate.
  - **granitemoehybrid, jamba → code=None ✓** (hybrid SSM-MoE, consciously
    deferred) → config fallback unchanged.
- **⇒ THE WIRING RULE (confirmed by the dots1 case):** per-layer MoE verdict =
  `moe_active AND code_schedule[i]` when code present, else `moe_active AND not
  _is_dense_at_layer(i)` (config fallback). moe_active (num_experts>0) is the
  is-this-MoE-at-all geometry gate — ALWAYS applied; the code reader supplies
  only the per-layer SHAPE. This keeps the 12 byte-identical (code≡config for
  them) and fixes llama4 + ernie.

**✅ UNIT COMPLETE (2026-07-05) — MoE-vs-DENSE SCHEDULE SHIFTED TO CODE.**
- LANDED: `_code_moe_schedule` wrapper (`parser.py`) + `code_moe_schedule` at
  the MoE block + the per-layer rule at the layer loop: `is_dense = not
  code_schedule[i]` when code resolved, `elif moe_layers list present → i not
  in list` (config fallback + ownership), `else _is_dense_at_layer` (config).
  moe_active (num_experts>0) stays the shared is-MoE gate. `moe_layers` read
  for ownership. Reader + `_class_builds_experts` + `_MoEGateEvaluator` in
  `evidence/patterns.py`.
- VERIFIED: **499 tests green** (+14 gate-form/family/guard tests), **25/25
  corpus zero-drift** (the 3 blessed MoE fixtures + all others byte-identical),
  sweep correct across 17 families. **Llama-4 now draws MoE** (was all-dense —
  architecture pixel-confirmed: ×36 layers show MoE after GQA(QK-Norm)).
  **Ernie-4.5 layer-0-dense** fixed (bug the sweep found). Second-tower MoE
  ability preserved (test_submodel_parity passes; fallback-safety).
- NOT BLESSED: Llama-4 stays unblessed — it has OUT-OF-SCOPE backlog (5 unread
  config fields: attention_chunk_size/attn_scale/attn_temperature_tuning/
  floor_scale/intermediate_size_mlp — chunked-attn+iRoPE; and the expert-drill
  conformance below). The schedule fix is proven WITHOUT a new bless (suite +
  sweep + zero-drift + architecture pixel).
- **IMMEDIATE FOLLOW-UP the fix EXPOSED (record, do next):** `Llama4TextExperts`
  is a FUSED expert (batched `gate_up_proj` param + bmm), but the expert drill
  draws separate `Linear` ops → nested_conformance FAIL(8) "draws 'linear' but
  Llama4TextExperts's forward() never does it". This is EXPERT-STORAGE (fused
  vs split), a DISTINCT concern from the schedule — the `_code_expert_fused` /
  `expert_fused_gate_up_from_files` rail needs to detect Llama-4's fused expert
  form. Newly visible ONLY because the schedule fix made Llama-4's experts
  render for the first time. Next unit.
- DISTRIBUTION register: A2 MoE-schedule is now CODE-authoritative (config =
  oracle-missing fallback), the twin of A4 router. Hybrids (granitemoehybrid,
  jamba) consciously return None → C1 hybrid-SSM territory.

---

### ⏳ IN-FLIGHT: TOWER CLASS-NAME LEAK → TIER-2 STRUCTURAL PARSE (the
### "camelCase classes shown as blocks" problem)

**The leak (verified 2026-07-05).** Exotic SUPPORTING-TOWER classes render
their RAW class name as the block label (identity leak, forbidden by our
laws). Leak site: `evidence/vision.py:399` — the tower op-chain builder labels
a conv/unknown sublayer with the raw `field_type` (class name) as fallback.
Scope (swept multimodal models): plain-vision models CLEAN (Qwen2-VL/Gemma-3/
LLaVA/Pixtral — `LayerNorm` is a legit norm label, false alarm). Real leaks:
Llama-4 `Llama4UnfoldConvolution` (patch embed); **Phi-4-multimodal AUDIO
CONFORMER = the big one: 5 raw names** (`Phi4MultimodalAudioAttention`,
`…ConformerEncoderLayer`, `…ConvModule`, `…GluPointWiseConv`, `…MLP`). ALL are
CODE-VISIBLE (classes exist, forwards readable) — we just don't PARSE them, we
fall back to identity.

**Soumil's decision: do TIER 2 (structural parse) directly, NOT the Tier-1
humanizer.** His key constraint/worry: deconstructing arbitrary classes will
FLOOD the diagram with everyday plumbing (transpose/reshape/permute/view) — an
"explosive overhaul" he does NOT want. But he DOES want the class parsed. So we
need a FILTRATION mechanism (compute vs plumbing) and will SEE the raw output
first, then "modernise and filter it into a considerable output" + decide
naming.

**RECON — the raw forward() op sequences (2026-07-05), which make filtration
tractable:**
- `Llama4UnfoldConvolution.forward`: `self.unfold(x)` → `x.permute(0,2,1)` →
  `self.linear(x)`. = extract patches → (plumbing permute) → project. Semantic:
  "Patch embedding" [Unfold patches → Linear].
- `Phi4…ConformerEncoderLayer.forward`: `residual = x + 0.5*self.feed_forward_in(x)`
  → `self.layer_norm_att` → `residual + self.self_attn(...)` → `x + self.conv(x)`
  → `x + 0.5*self.feed_forward_out(x)` → `self.layer_norm`. = the CLASSIC
  CONFORMER cell FFN(½)→Attn→Conv→FFN(½)→Norm. **Every op here is a SUBMODULE
  CALL — ZERO plumbing at this altitude; a beautiful drawable cell.** The 0.5 =
  macaron half-step (a real caption-worthy fact).
- `Phi4…ConvModule.forward`: norm→GLU → (permute) → dw_sep_conv → (causal slice)
  → act → pw_conv → (permute) dropout. Compute: norm/GLU/dwconv/act/pwconv;
  plumbing: permutes, causal slice, dropout.
- `Phi4…GluPointWiseConv.forward`: (permute) → pw_conv → split+bias → GLU-mul →
  (permute). Compute: pw_conv, GLU gate-mul; plumbing: permutes, slices.
- `Phi4…MLP.forward`: norm → gate_up_proj → chunk(2) → up*act(gate) → down_proj.
  = a GATED (SwiGLU) MLP; chunk-as-gate-split is SEMANTIC, dropout plumbing.

**THE FILTRATION MECHANISM (senior-engineer insight — tractable, NOT
explosive):** a drawn op = a CALL TO A CONSTRUCTED SUBMODULE FIELD
(`self.<field>(...)` where field is a class the layer builds — linear, conv,
norm, glu, self_attn, feed_forward); PLUMBING (filtered) = a bare TENSOR METHOD
(`.permute/.reshape/.view/.transpose/.contiguous/.unsqueeze/.squeeze`), an
index/slice, a cast (`.to`), or `dropout` (inference no-op). Bounded by the
class's constructed fields (usually 3-8), reusing the existing `_field_types`
role machinery — so it CANNOT explode. Small SEMANTIC-tensor-op vocabulary for
the exceptions that ARE meaningful: `chunk(2)`+gate-mul = GLU split (draw it),
the `0.5*` macaron scale = a caption, a causal slice = optional caption. Draw
submodule calls (humanized/role-labeled), filter tensor rearrangements.

**PROPOSED PARSED OUTPUT (to show Soumil, calibrate, then build):**
- Llama-4 patch: "Patch embedding" → [Unfold patches, Linear projection].
- Phi-4 conformer cell: FFN-in (½) → LayerNorm → Self-Attention → Conv module →
  FFN-out (½) → LayerNorm — a standard-ish drillable cell (the conv module and
  the macaron FFNs each drill one level deeper via the same submodule-call
  rule).

**NAMING / ENVISIONING (open — Soumil decides after seeing output):** how to
present a parsed-but-nonstandard tower cell (a "conformer" archetype? a generic
"parsed cell"?), and where the filter vocabulary lives (an `everchanging/`
YAML: compute-roles + plumbing-tokens + semantic-tensor-ops).

**WHERE I AM (RESUME):** recon done (forwards dumped + filtration mechanism
identified). Tier-2 presentation layer DEFERRED per Soumil's pivot below.

**SOUMIL'S PIVOT (2026-07-06): FACTS-FIRST.** He does NOT want to design the
presentation/archetype layer yet. His architecture vision (for later): NOT a
100%-confined archetype set — general STRUCTURING LAWS for any diagram + a TYPE
SYSTEM distributing classes into types, curated by him ("all classes go through
me, we decide how their diagrams look"; the MoE view is the exemplar — a
DESIGNED fill-in template, not a state-the-facts code dump). But FIRST: "get the
facts right — parse everything in the code and out here for us to see, so we
know what to build on top of. If we don't have the correct picture, what are we
building on?" ⇒ Immediate work = complete + correct FACT EXTRACTION; the
diagrammatic representation gets "inputted" afterward, on top of a correct
picture.

**MoE PART COMPLETED (2026-07-06) — fused-expert storage fact fixed:**
`Llama4TextExperts` does `torch.bmm(x, self.gate_up_proj)` over a stacked weight
Parameter — a batched linear the op-classifier mapped to `dot_product` (bmm), so
the expert drill's "Linear" read as fabricated (nested_conformance FAIL). FIX
(`forward_ops.py:_call_op_kind`): a matmul/bmm/einsum CALL with a proj/weight
operand → `linear` (the call-form twin of the existing `x @ self.gate_up_proj`
BinOp rule). Surgical, general; `Q·K` attention (no proj operand) stays
dot_product. **Llama-4 expert drill now draws the fused SwiGLU correctly**
(Linear(gate+up)→Split→SiLU→×→Linear(down)→weighted sum) and CONFORMS. 499
tests green, 25/25 corpus zero-drift. ⇒ Llama-4 MoE is now FULLY correct:
schedule (48 MoE) + router + fused experts.

**LLAMA-4 REMAINING FACT GAPS (the "get everything parsed" worklist, surfaced
2026-07-06 config_field_audit):** text_config: `attention_chunk_size` (chunked/
local attention), `attn_scale`/`attn_temperature_tuning`/`floor_scale` (attn
temperature tuning for length-gen), `intermediate_size_mlp` (dense-MLP width
alias). vision_config: `pixel_shuffle_ratio`, `projector_input_dim`/
`projector_output_dim`/`projector_dropout`/`multi_modal_projector_bias`. Plus
`boi_token_index`/`eoi_token_index` (image markers). These are ATTENTION +
VISION-PROJECTOR facts (distinct from MoE) — the next facts-first mini-units;
mechanisms are code-visible, values config. NOT yet parsed.

---

### ⏳ IN-FLIGHT UNIT (2026-07-06 — RESUME HERE): GROUP 1 —
### PER-LAYER LAYER-TYPE SCHEDULES FROM CONSTRUCTION (the MoE-schedule family)

**THE UNIFYING INSIGHT (why this is one unit, not four).** MoE-vs-dense schedule
was ONE INSTANCE of a general pattern: *a heterogeneous decoder stack decides,
PER LAYER, which TYPE each layer is along some dimension — and the CODE
constructs that per-layer type, gated on a config field the code itself names.*
The register (DISTRIBUTION §3.2/§5) has several more of exactly this shape,
still config-guessed. Group 1 shifts them all to construction evidence, reusing
the MoE-schedule machinery (`_find_decoder_layer` + `_MoEGateEvaluator` +
structural class detection in `evidence/patterns.py`).

**THE PRINCIPLES WE ARE BUILDING THIS ON (the durable "why" — do not violate):**
1. **DETECT FROM EVIDENCE, NEVER IDENTITY (Law 1, the root law).** A layer's
   type comes from what the code CONSTRUCTS/ENACTS, resolved through general
   vocabulary — never a class name, repo id, or per-model table. Config
   spellings are one evidence channel, demoted to fallback.
2. **CODE DECIDES THE SHAPE; CONFIG SUPPLIES THE VALUE.** The recurring gate
   pattern (QK-norm → MoE-schedule → here): the code gates a per-layer
   construction on a config field IT NAMES (`if layer_idx in config.moe_layers`,
   `self.is_sliding = (layer_idx+1) % config.sliding_window_pattern`,
   `config.no_rope_layers[layer_idx]`). We read the GATE (shape) from code and
   the checkpoint supplies only the VALUE. We never need to know config
   spellings in advance — the code names its own fields.
3. **STRUCTURAL, NAME-INDEPENDENT.** A type is what the constructed class/branch
   STRUCTURALLY is (builds experts = MoE; applies a sliding mask = sliding),
   NEVER the class name (gpt-oss MoE named `GptOssMLP`; Qwen3Moe names
   everything `*Moe`).
4. **CODE-AUTHORITATIVE; CONFIG = ORACLE-MISSING FALLBACK ONLY.** Code decides
   when source is present; the config path runs ONLY when code returns None
   (no source / unresolvable). NEVER delete the config path — a source-less
   model must still draw something honest.
5. **TRI-STATE: proven / unresolvable(None→fallback) / NEVER-A-WRONG-GUESS.**
   Any ambiguity (unknown gate form, multiple candidates disagree, exotic
   construction) → None → config fallback. A FALSE type verdict BREAKS a model;
   an honest fallback never does. Return None on any doubt.
6. **FALLBACK-SAFETY PRESERVES EVERYTHING THAT WORKS.** Because code AGREES with
   config for the families config already gets right, the blessed fixtures +
   working families stay BYTE-IDENTICAL; only the broken ones (Llama-4-like)
   change. Prove agreement via a code-vs-config SWEEP before flipping.
7. **VALUES CAN'T SHIFT, SHAPES CAN.** Geometry VALUES (window SIZE, chunk SIZE,
   num layers, head counts) stay config — `nn.Linear(config.hidden_size,…)` has
   no number. Only the SHAPE (which layers are which type) shifts to code.
8. **ONE ENGINE, PARAMETERIZED BY ROLE-DIMENSION.** Not four readers — one
   per-layer-variant primitive parameterized by which dimension it resolves.
   The next schedule type becomes a parameter, not new code (this is the
   "build on this so the next model is free" north star).

**THE GENERAL PRIMITIVE (the key design nuance vs MoE).** A per-layer variant is
revealed in code THREE ways — the engine must handle all three:
  (a) **per-index CLASS choice** — `if <gate>: self.x = TypeAClass else TypeBClass`
      (MoE-vs-dense: structural class detection). DONE for MoE.
  (b) **per-index SELF-FLAG / attribute** — `self.is_sliding = (layer_idx+1) %
      config.sliding_window_pattern` then the flag drives the mask (SAME class,
      different per-layer behavior — Gemma-2/3 sliding). NEW: evaluate the flag
      expression per index via `_MoEGateEvaluator` (already handles
      modulo/membership/threshold/self-flag).
  (c) **per-index LIST lookup** — `config.layer_types[layer_idx] == "sliding"` /
      `config.no_rope_layers[layer_idx]` (explicit per-layer type list). Evaluate
      the subscript+compare per index.
So the shared reader: find the per-layer signal (class-choice / self-flag /
list-lookup) for a dimension, evaluate it per index (reusing the gate evaluator),
return `list[variant]` or None. `_class_builds_experts` generalizes to
`_classify_construction(role_dim)`; the gate evaluator is already general.

**GROUP 1 ITEMS + current state + detection method:**
1. **Sliding-vs-full attention schedule** — CURRENTLY CONFIG (`layer_types`,
   `sliding_window_pattern` via class-default HYDRATION band-aid, `use_sliding_
   window`, `max_window_layers`, `full_attention_interval`). Enacted in code via
   (b) self-flag `self.is_sliding=<index expr>` (Gemma-2/3) or (c) `layer_types`
   list. HIGHEST VALUE (most models; retires the hydration band-aid). The SIZE
   stays config (principle 7); only WHICH layers are sliding shifts.
2. **NoPE-layer schedule** — PARTLY code (S1b reads `no_rope_layers[i]` for
   placement + phase-corrected interval fallback). Complete the shift: the
   interval form still risks the same inversion the MoE `interleave==1` bug had.
   Enacted via (c) `config.no_rope_layers[layer_idx]` or (b)/interval.
3. **Cross-attention-layer schedule** — CONFIG (`cross_attention_layers` + a
   `vision_config`, mllama). Which layers are cross-attn vs self. Enacted via
   (a) class choice or (c) membership `layer_idx in config.cross_attention_layers`.
4. **Hybrid mixer-type schedule (in-scope part)** — CONFIG (`layer_types` with
   `linear_attention`/`gated_delta`). Which layers are full-attn vs linear/gated-
   delta mixer. Enacted via (c) list or (a) class. NOTE: pure-Mamba layers are
   OUT of scope (C1 hybrid-SSM); the attention-vs-linear-attention case is IN.

**PHASED PLAN (mirrors the MoE unit, proven surgical):**
- P0 RECON: dump, across families, HOW each schedule is enacted (self-flag vs
  list vs class) — Gemma-2/3 (sliding), Cohere2/Ministral/Phi3-small (sliding),
  Llama-4 (chunked+NoPE), mllama (cross-attn), Qwen3-Next (linear mixer). Build
  the enactment taxonomy (a/b/c forms per dimension).
- P1 BUILD: the shared per-layer-variant primitive + one reader per dimension
  (or a parameterized reader), None-on-doubt.
- P2 SWEEP: code-vs-config across ALL families per dimension; investigate every
  divergence (expect config-right agreements + a few code-right fixes like
  ernie/llama-4 were); reader must be correct everywhere.
- P3 WIRE: code-authoritative per dimension, config fallback only when None;
  the SIZE/geometry stays config; fallback-safe (recursive towers preserved).
- P4 TEST+VERIFY: gate/enactment direct tests, per-family integration, negatives;
  full suite + 25-fixture ZERO-DRIFT (blessed sliding models — gemma-2-2b is
  blessed! — MUST stay byte-identical).
- P5 WITNESS: render a fixed model (Gemma-3 sliding, or Llama-4 chunked); Sable
  + Dable; bless if clean.
- P6 RECORD: PROJECT_CONTEXT + DISTRIBUTION (§3.2 sliding-schedule + A2 shifted
  to code, the layer-type-schedule family alongside A4 router / MoE-schedule).

**SURGICAL SAFETY (non-negotiable):** gemma-2-2b-it IS blessed (sliding
schedule) — it MUST stay byte-identical, so the sliding reader must AGREE with
the current config path for it (or fall back). Every dimension: sweep-prove
agreement before wiring; None on doubt; SIZE stays config; hybrids/pure-Mamba →
None. No per-model rows.

**⚠ P0 RECON FINDING (2026-07-06) — CRITICAL CORRECTION: Group-1 schedules
ALREADY WORK; they are NOT broken like MoE was.** Verified by code-vs-drawn
sweeps:
- SLIDING: enacted in the ATTENTION class (takes layer_idx) as
  `self.layer_type = config.layer_types[layer_idx]` → `is_sliding = layer_type
  == "sliding_attention"`. The config CLASS builds `layer_types` from
  `sliding_window_pattern` via `["sliding_attention" if (i+1)%pattern else
  "full_attention"]`. Our drawn schedule AGREES EXACTLY with code `layer_types`
  (gemma2/gemma3/cohere2: full layers draw `global`, sliding draw `sliding`).
- NoPE: Llama-4 drawn NoPE (12 layers) AGREES with code `no_rope_layers`
  (S1b already shifted this).
- chunked attention: already partly surfaced ('chunk' in the diagram).
**THE LESSON:** a per-layer schedule is CORRECT when we read the AUTHORITATIVE
per-layer LIST (`layer_types` / `no_rope_layers`) that the config serializes OR
the config class builds (+ hydration materializes it). Sliding/NoPE already do
this. MoE was the OUTLIER — broken because it DIDN'T read `moe_layers` and
re-derived from an inverted interval. The MoE fix brought MoE to the SAME
standard the others were already at.

**⇒ REVISED VALUE OF GROUP 1 (honest):** NOT a broken-model fix. The only
remaining value is PRINCIPLED HARDENING — eliminating the RE-DERIVATION
FALLBACKS (`_layer_mask`'s `(i+1)%pattern`, the NoPE interval) that are a LATENT
inversion-bug source (they happen to agree today but are re-implementations of
the code's logic, and re-implementations drift — MoE proved it). Reading the
code's ACTUAL per-layer gate as authoritative, demoting re-derivation to
last-resort, future-proofs all schedules. Insurance, not urgent.

**⇒ THE HIGHER-VALUE REMAINING CONFIG→CODE WORK IS GROUP 2 (structural
attention facts with ACTUAL silent-wrong risk), not Group-1 schedules:**
- MLA/MQA attention-kind (A5): config `kv_lora_rank`/`multi_query` flag; a
  code-MLA with silent config draws GQA. Code CONSTRUCTS the MLA projections.
- attention/MLP bias: config-only; a `nn.Linear(…, bias=True)` in code with a
  silent config draws bias-less.
- norm multiplicity (Falcon/NeoX two-norms-as-one; last S3 piece).
These CAN be silently wrong TODAY (like MoE was); Group-1 schedules cannot.

**✅ GROUP 1 COMPLETE (2026-07-06).** Finding: the per-layer schedules were
ALREADY construction-faithful (sliding reads `layer_types`, NoPE reads
`no_rope_layers` — the authoritative lists the config serializes / the config
class builds; the `(i+1)%pattern` re-derivation provably matches the config
class's OWN construction, same polarity — verified across ~15 families, zero
divergence). So the honest completion was NOT a rewrite (that'd be a zero-drift
no-op) but a **REGRESSION LOCK**: 3 conformance tests
(`test_sliding_schedule_matches_code_layer_types`,
`test_nope_schedule_matches_code_no_rope_layers`,
`test_moe_schedule_matches_code_construction`) that assert the drawn per-layer
schedule EQUALS the code's authoritative gate for all three dimensions — the net
that would have caught the MoE interleave==1 inversion, locking sliding/NoPE/MoE
against future re-derivation drift. No source change (schedules already correct);
+3 tests. Soumil chose "complete Group 1, then Group 2."

**▶ NOW: GROUP 2 — structural attention facts with ACTUAL silent-wrong risk
(the true MoE-like targets).** Same principles (the 8 above), same method
(construction evidence, code-authoritative, tri-state None-on-doubt, sweep
before wiring, blessed fixtures byte-identical). Items:
1. **MLA/MQA attention-kind (A5).** Config: MLA iff `kv_lora_rank`; MQA iff
   `multi_query` flag. A code-MLA with silent config draws GQA. The code
   CONSTRUCTS the MLA projections (`q_a_proj`/`kv_a_proj`/`kv_a_layernorm`) — a
   detector exists off-ship-path (`_detect_attention_shape`, `patterns.py:110`,
   only in the optional inspect_code pass). Promote to code-authoritative like
   QK-norm/router.
2. **Attention/MLP bias.** Config-only (`attention_bias`/`mlp_bias`/
   `use_qkv_bias`), but it's `nn.Linear(…, bias=True/False)` in the constructor
   — a bias-in-code/silent-config model draws bias-less. Read the Linear's bias
   arg from construction.
3. **Norm multiplicity (last S3 piece).** Falcon/NeoX build TWO input norms
   (`ln_attn`+`ln_mlp` / `input_ln`+`post_attn_ln`), drawn as ONE shared;
   `new_decoder_architecture` unread. Count distinct norm modules the layer
   constructs; GPT-J (genuinely 1) is the pinned negative control.

**GROUP 2 RECON DONE (2026-07-06) — CONFIRMED BROKEN CASES:**
- **BIAS = the clear MoE-shaped bug (HIGH VALUE, confirmed in code).** The
  attention projections have bias in CODE but we draw bias-less because the
  config doesn't declare `attention_bias` in our spelling:
  - **Bloom**: `self.query_key_value = nn.Linear(..., bias=True)` (+ `dense`
    default bias) → HAS attention bias. **BLESSED bloom.json draws bias=False
    (attention.bias=False, no '+bias' chip) — a GENUINE BUG IN A BLESSED
    FIXTURE** (like the VAE-upsample-in-blessed-SDXL class). Fixing re-blesses
    bloom.
  - **Qwen2**: `q_proj/k_proj/v_proj = nn.Linear(..., bias=True)`, `o_proj =
    bias=False` → QKV bias, no O bias. Drawn bias-less. (Qwen2-VL text tower
    uses Qwen2 attention → the BLESSED qwen2-vl fixture likely also affected —
    CHECK before wiring.)
  - Contrast (correct): Llama `bias=config.attention_bias` (reads config,
    default False → drawn False ✓); Phi-3 `bias=False` ✓.
  - FIX: read the attention Linear's bias FROM CONSTRUCTION (code-authoritative,
    the QK-norm/MoE template). NUANCE: per-projection (Qwen2 has QKV bias but NOT
    O bias) — the reader must report which projections carry bias, and the
    drawing/`+bias` chip must reflect "QKV bias" honestly, not a blanket flag.
    Bias VALUE stays config when the code gates on `config.attention_bias`
    (Llama); code-True/False literal is authoritative (Bloom/Qwen2/Phi).
- **NORM MULTIPLICITY (Falcon/NeoX 2-drawn-as-1) — confirmed, nuanced.**
  Construction norm-module counts: falcon builds 4 fields (`ln_attn`+`ln_mlp`
  new-decoder AND `input_layernorm`+`post_attention_layernorm` old — only 2 used
  per `new_decoder_architecture`); gpt_neox 2 (`input`+`post_attn`); gptj 1
  (`ln_1` — the pinned negative control); llama 2 (standard pre-norm, correct).
  FIX: count the DISTINCT INPUT norms the parallel-residual layer actually uses;
  Falcon's conditional 4→2 needs the `new_decoder_architecture` gate resolved
  from construction. More nuanced than bias.
- **MLA/MQA attention-kind: NO broken witness found** in the sweep (would need a
  code-MLA with config-silent latents — rare). Lower priority; the off-ship-path
  detector promotion is cheap insurance, not an active fix.

**⇒ GROUP 2 BUILD ORDER:** (1) BIAS first — clearest, confirmed, real
blessed-fixture bug; per-projection construction reader; re-bless bloom (+
qwen2-vl if affected) with pixel review. (2) NORM MULTIPLICITY — construction
count with the Falcon new-decoder gate. (3) MLA/MQA cross-check — insurance.

**✅ GROUP 2 ITEM 1 — ATTENTION BIAS: DONE (2026-07-06).**
`decoder_attention_bias_from_files` reads the attention class's QKV Linear
constructions: literal `bias=True/False` wins; `bias=config.X` resolves the
checkpoint value (Llama); absent kwarg → nn.Linear default True; None on doubt
(→ config fallback). Wired `_code_attention_bias` code-authoritative at
`parser.py`. **Fixed a REAL BLESSED-FIXTURE BUG:** Bloom
(`query_key_value=nn.Linear(bias=True)`) + Qwen2/Qwen2-VL (QKV bias=True) were
drawn bias-less (config declared no `attention_bias`); now draw the `+bias`
chip. bloom + qwen2-vl RE-BLESSED with pixel review + supersede (drill unchanged,
+bias chip added). Drift was EXACTLY those two (the wrong ones); all else
byte-identical. 503 tests green, 25/25 locked. +2 tests
(`test_attention_bias_from_construction` + updated
`test_attention_bias_and_rope_theta` documenting "code wins over a dead config
flag": Qwen2 hardcodes bias so `attention_bias=False` is dead; Llama gates on it
so the flag is honored). Bias VALUE stays config only when the code gates on it.

**✅ GROUP 2 ITEM 2 — NORM MULTIPLICITY (context; resolution two blocks below).** Falcon/NeoX build
TWO input norms per parallel-residual layer, drawn as ONE shared. Construction
counts (verified 2026-07-06): falcon builds 4 norm FIELDS (`ln_attn`+`ln_mlp`
new-decoder AND `input_layernorm`+`post_attention_layernorm` old — only 2 USED
per `new_decoder_architecture`); gpt_neox 2 (`input_layernorm`+
`post_attention_layernorm`); **gptj 1 (`ln_1`) = the PINNED NEGATIVE CONTROL
(genuinely one shared norm — must NOT become two)**; llama 2 (standard pre-norm,
already drawn as 2, correct). THE FIX: count the DISTINCT INPUT norms the
parallel-residual layer actually CONSTRUCTS+USES (not from a `use_parallel_
residual` assumption); Falcon's conditional 4→2 needs the
`new_decoder_architecture` gate resolved from construction (which pair the
forward uses). This is a per-layer STRUCTURE count (like norm placement was) +
a RENDER change (draw 2 input norms fanning to the two branches vs 1 shared).
Blast radius: falcon/neox are NOT blessed (fresh-render wins); gpt-j IS the
control (must stay 1). Method: construction-count reader (tri-state None-on-
doubt) + render the counted norms.

**✅ GROUP 2 ITEM 2 — NORM MULTIPLICITY: FACT DONE, render deferred (2026-07-06,
honest split).** `decoder_parallel_norm_count_from_files` reads the norm
DATAFLOW: which norm field feeds the attention call vs the FFN call — SAME → 1
(GPT-J `ln_1` shared, the pinned negative control), DIFFERENT → 2 (GPT-NeoX
`input_layernorm`+`post_attention_layernorm`); >2 constructed norm fields → None
(Falcon's conditional `new_decoder_architecture` 4-field case → fallback).
Wired `_code_parallel_norm_count` → `parallel_decoder_layer(norm_count=…)` gated
on parallel-residual. Verified gptj=1, neox=2, falcon=None.
**RENDER HONESTLY SPLIT:** my first attempt drew 2 norm BOXES via a side-lane
tap chain (`rms2 tap_from rms1`, ffn tap_from rms2) — it PASSED Sable but the
naive wiring HID THE FFN in the hero view (GPT-J correctly shows Feed-Forward +
Attn from the shared norm; my NeoX lost the FFN). REVERTED to the working
layout; for count==2 the 2-norm FACT is now stated in the norm's CARD ("the code
applies a SECOND separate norm before the FFN — input_layernorm +
post_attention_layernorm"). The full two-BOX parallel render is a SCOPED
FOLLOW-UP (the lane/tap layout engine needs real work). Facts-first per Soumil:
the fact is captured + surfaced; the 2-box drawing is deferred. gptj/neox NOT
blessed (no re-bless); GPT-J stays 1; 504 tests green, 25/25 zero-drift. +1 test.

**GROUP 2 STATUS:** item 1 (bias) DONE + re-blessed; item 2 (norm multiplicity)
FACT DONE + honest card note, 2-box render deferred; item 3 (MLA/MQA cross-check)
NOT started (insurance, no broken witness).

**FOLLOW-UPS QUEUED (scoped, recorded so nothing is lost):**
- 2-BOX parallel-norm render (NeoX draws 2 separate norm boxes fanning to the
  two branches) — needs the lane/tap layout engine understood; the FACT
  (norm_count) is already wired and available.
- Llama4TextExperts / expert-storage is DONE; Llama-4 chunked-attention +
  attn-temperature-tuning + vision-projector facts still unread (recorded
  earlier).
- MLP bias code-shift (twin of the attention-bias fix, `mlp_bias`).
- MLA/MQA attention-kind cross-check promotion (Group 2 item 3, insurance).

**WHERE I AM:** superseded — Part 7 is the live state; this section is history + principles only.

---

**Wave S1 — QK-norm + partial rotary (clears ~9 of the 15 finding models):**
- *QK-norm:* the transformer adapter already runs an ALWAYS-ON `_code_*`
  reader family at parse time (ten readers, invoked at
  `adapters/transformer/parser.py:373-422` — ffn gating/storage, norm
  kind/math, fused qkv, expert storage, position…). QK-norm never got one —
  its detection sits only behind the optional `inspect_code=True` pass
  (root `parser.py:98-99`), which nothing on the ship path sets, and the
  config-field reader (`evidence/patterns.py:140-143`) only sees declared
  spellings (verified: the inspect_code tier writes a Code Evidence PANEL
  + warnings, never the spec field the chips read). Fix = `_code_qk_norm`,
  the 11th always-on reader, with **CODE-FIRST semantics (Soumil's
  refinement, verified against 6 installed modeling files 2026-07-05):**
  the code decides the SHAPE of the answer and even NAMES its own config
  gate —
  (1) q/k norm constructed UNCONDITIONALLY + applied in forward → TRUE,
      config not consulted (Qwen3, OLMo-2, Gemma-3);
  (2) constructed under `if <gate>:` → extract the gate expression; the
      code itself names the config field (`config.qk_layernorm`,
      `config.use_qk_norm`) — read THAT field's VALUE from the checkpoint
      config (no spelling vocabulary on the primary path), and resolve
      structural terms (Llama-4's `and self.use_rope`) per layer from
      already-derived facts (Persimmon, StableLM, GLM-4.5, Llama-4);
  (3) no construction → FALSE;
  (4) oracle_missing → fall back to the declared-spelling vocabulary
      (patterns.py:140) as honest config evidence — a declaration is
      still a declaration;
  (5) config declares a gate the code never reads → dead flag: code wins
      + ambiguity note (unclaimed-signal logic, forward direction).
  Application-in-forward required, not just construction (no dead modules
  drawn). Bonus fact for free: the normalized WIDTH differs (Qwen3 per
  head_dim vs OLMo-2 full projection) and is in the constructor args.
  Render side needs ZERO work —
  chips already gated on `attention.qk_norm` at labels.py:121/269/386/496.
- *Net side (refined from the sweep's suggestion):* transitive.yaml's own
  omission-scope NOTE documents why FFN salients were excluded — the
  role-UNION conflates text-vs-vision sub-modules, so omission checks are
  only sound per-role when un-conflatable; the fact must be fixed AT THE
  SOURCE by a parser rail (the documented FFN-gating precedent). The
  QK-norm net check must follow that precedent: source-rail primary;
  add an `attention` salient ONLY in a union-safe form (flag only when
  EVERY same-role closure carries the norm).
- *Partial rotary:* `rope_dim` is already in the IR AND serialized
  (`ir.py:251`), and the drawing idiom already exists — `_mla_child_blocks`
  (`blocks/attention.py:393-399`) draws per-head rope/nope split widths for
  MLA. Fix = generalize that same idiom to the standard attention drill
  when `rope_dim < head_dim` + the labels chip. ChatGLM is config-blind
  (fraction is code-only) → extend the position rail to derive it from the
  rotary slicing width. Clears GLM-4.5, NeoX, Persimmon, StableLM (+GPT-J
  chip). Serializer-fan-out reminder (Part 4 §3): touch ALL THREE
  AttentionSpec projections, and fix the known `rope_3d` drift in
  `_attention_to_dict` while there.

**Wave S2 — router facts from code + stream-scale (the Law-1 repairs):**
- *Router:* confirmed identity-string sites: `blocks/feed_forward.py:333`
  is literally `r.get("scoring_func") or "softmax"` (default-becomes-fact)
  and the noaux bias/gather drawing keys on the `topk_method == "noaux_tc"`
  string (`renderers/html/block_views/moe_router.py:98`,
  `feed_forward.py:278`) — strings GLM-4.5's config doesn't carry while its
  code says `.sigmoid()` + bias-buffer + gather. Fix = code-derive scoring
  fn + bias-correction + sparsemixer from the router closure's forward ops
  (the exact `decoder_ffn_gated_from_files` precedent named in
  transitive.yaml's NOTE); config strings stay as one evidence channel.
- *Stage multipliers:* Granite `embedding_multiplier`/`logits_scaling` —
  this IS MASTER_PLAN B2 (stream-scale family), pulled forward.
- *Unclaimed signals:* GPT-OSS sinks, Llama-4 chunked attn + `moe_layers`,
  Gemma-3 dual RoPE-θ, GPT-J `n_inner=None → 4×hidden` — all instances of
  B1 (unclaimed-signal net); the sweep hands B1 its test corpus.

**Wave S3 — evidence-priority + net hardening:**
- *Norm-kind:* confirmed at `adapters/transformer/parser.py:1159-1193` —
  channel order puts the `rms_norm_eps` spelling (1189) ABOVE
  `_code_norm_math` (1191); the docstring's premise ("a spelling only RMS
  implementations carry") is falsified by PhiMoE/Persimmon (nn.LayerNorm +
  rms_norm_eps). Fix = math outranks BOTH eps spellings (the project's own
  T5 lesson applied consistently) + recognize torch-builtin norm
  constructors (`nn.LayerNorm`/`nn.RMSNorm`) as construction evidence,
  since `_code_norm_math` can't read torch internals. Library API symbols
  are vocabulary, not model identity — lawful.
- *Norm multiplicity:* count distinct norm modules from the construction
  registry (Falcon-40B `ln_attn`+`ln_mlp`, NeoX 2 norms drawn as 1);
  GPT-J single-norm is the pinned negative control.
- *Net upgrades:* op_conformance must resolve the backing class from
  CONSTRUCTION, not the region's own claimed kind (Llama-4 MoE-as-dense
  passed by validating the wrong class); fact_conformance gains a
  param-count oracle (GPT-J 6.05B reported 2.29B and passed).
- *Resolver gaps:* dict-indexed dispatch + local-fn ModuleList in the
  closure resolver (ChatGLM escaped nested_conformance) — write ONCE in
  the shared extractor so A4's twin-scanner merge doesn't unpick it.

**Standing discipline for every wave:** each ends with re-render + Sable +
exhaustive Dable + Her Eyes + bless-with-supersede for every affected
model (which also grows the corpus toward C5's ~35 witnesses). No commits
(Law 6). Fixes general only — if it needs a per-model row, it's not a fix.

**THE REGRESSION LEDGER (verified 2026-07-05 — the "don't break other
examples" contract).** The 21 blessed fixtures are: auraflow, bloom,
cogvideox, deepseek-v3, flux-2-dev, fluxtransformer2d, gemma-2-2b-it,
gpt-oss-20b, hunyuanvideo, llama-7b, ltx-video, lumina, mochi, pixart,
prxpixel, qwen-image, qwen2-vl, sana, sd3.5-large, sdxl, wan2.2. Expected
drift per wave, declared IN ADVANCE — any hash change outside this list is
a STOP-AND-INVESTIGATE, never a bless-over:
- *S1 QK-norm:* corpus drift allowed on **hunyuanvideo only** (the
  diffusion attribution bug: main denoiser gains its real QK-norm, the
  token-refiner loses its fabricated one). Every transformer-side witness
  (Qwen3, Gemma-3, OLMo-2, OLMoE, Llama-4) is OUTSIDE the corpus — the
  other 20 fixtures must stay byte-identical.
- *S1 partial rotary:* **zero corpus drift** (no partial-rotary model is
  blessed; DSV3's MLA drill draws its own split and is excluded from the
  new rule).
- *S2 router:* **zero corpus drift** (DSV3's declared `noaux_tc` channel
  keeps working and the code channel must AGREE with it — that agreement
  is itself a test; gpt-oss must reproduce its current drawing).
- *S3 norm-kind/multiplicity:* **zero corpus drift** (no parallel-residual
  or eps-lying model is blessed; bloom and qwen2-vl are the fixtures that
  must NOT move — bloom's sequential 2 norms and qwen2-vl's per-tower
  RMS-vs-LayerNorm are the guards).

**S1 BUILD OUTCOME (2026-07-05 — SHIPPED, suite 461 green, corpus 21/21
locked):** `decoder_qk_norm_from_files` + `_attention_qk_norm` landed in
evidence/patterns.py (code-first: unconditional/gated/absent, gate atoms =
the config fields the code names, per-layer subscript atoms, MLA-latent
dataflow exclusion, ≥2-site rule); `_code_qk_norm` + `_resolve_qk_norm_layers`
wired as the 11th always-on reader in the transformer parser (per-layer
spec facts; dead-flag = code wins). Smoke: 16/16 real modeling files
correct first run (Qwen3/OLMo-2/Gemma-3 True; StableLM/Persimmon/GLM-4.5/
Cohere gated; Llama-4 composite per-layer; DSV3-MLA/plain/GPT-OSS-sinks all
False). **Bonus bug the composite gate exposed and fixed:** the iRoPE NoPE
PLACEMENT was off-phase — `is_nope` used `i % interval == 0` (layers 0,4,8…)
while the code's own `no_rope_layers` puts NoPE at 3,7,11…; now derived
from the materialized list first, phase-corrected interval fallback.
Diffusion half: tower lane-norm facts are now CONSTRUCTION-SITE aware
(`_lane_norm_fact` in evidence/vision.py — the shared diffusers Attention's
`norm_q` field no longer fabricates on callers that pass no `qk_norm`;
only a PROVEN falsy site turns a lane off). HunyuanVideo refiner
de-fabricated → the ONE in-ledger drift → pixel-reviewed → re-blessed with
superseded signature (22 views). Witnesses pixel-verified: Qwen3-8B
("Grouped-Query (QK-Norm)" + strip row, 12/12 nets), OLMo-2 ("(QK-Norm)" +
post-norm placement), StableLM-2 (chip correctly ABSENT; its pre-existing
`use_qkv_bias` audit FAIL fixed with one aliases.yaml row → 12/12).
Tests +17: five synthetic shapes direct-call, 5-state resolution unit,
4 real-oracle integrations, construction-site tower test, Llama-4
NoPE-position pin. Also done en route: R0's conftest corpus fixture
(tests/conftest.py + 3 sites through sable.DEFAULT_CORPUS — the red suite
is green). NOT done: HF token revocation (Soumil-only), Her Eyes refresh
on the re-blessed hunyuan gallery + witness blessing into the corpus
(candidates: Qwen3-8B, OLMo-2, StableLM-2 — all sable-green, galleries in
scratchpad), the S1 partial-rotary half, S2/S3.

**S1b BUILD OUTCOME (2026-07-05 — partial rotary + drill-altitude QK-norm):**
- *Partial rotary SHIPPED, all five sweep models + two more:* the drill's
  RoPE ops now state the real split ("rot 16 · pass 48 dims"), "Partial
  RoPE" chips at the label sites (`_partial_rope_dims`, MLA excluded), and
  the rope cards carry the fraction. Three evidence dialects: top-level
  fields (StableLM/Persimmon/Phi-1/2, GPT-J's rotary_dim), the MODERN
  NESTED dialect (`rope_scaling.partial_rotary_factor` — GPT-NeoX's config
  class dropped top-level rotary_pct entirely; parser now reads the nested
  key), and CODE-ONLY arithmetic (`decoder_rope_dim_from_files` evaluates
  ChatGLM's `RotaryEmbedding(rotary_dim // 2)` with the kv_channels ternary
  — the fraction exists nowhere in config). Verified live: NeoX-20B
  rot 24/96, glm-4-9b rot 64/128, Qwen3 negative clean.
- *Her Eyes forced the QK-norm boxes into the drills (lawfulness line):*
  two independent reviews (qwen3-8b, olmo-2) DISLIKED chip-only QK-norm —
  "a real op invisible at the altitude where it lives." Fix: per-lane
  q_norm/k_norm facts in `attention_detail` (SELF-attention only — a cross
  sublayer inherits qk_norm by _replace, which is NOT per-sublayer
  evidence: drawing there would repeat the refiner attribution bug) + the
  two cards (click-coupling law; 15 of 17 initial test failures were
  card-less clickable nodes).
- *LEDGER EXTENSION (deliberate, out-of-original-ledger):* the shared drill
  builder means 12 blessed DIFFUSION mains (SD3.5, FLUX×2, hunyuan,
  auraflow, cogvideox, ltx, lumina, mochi, prxpixel, qwen-image, wan) +
  qwen3-8b + olmo-2 gained the same true boxes. Decision: keep — the
  qk_norm=True FACT was already blessed on every one of them (chips); the
  boxes are the same fact at drill altitude. All 14 re-blessed with
  supersede after pixel review (12 diffusion via a vision-fleet agent per
  sable's own fleet provision).
- *Corpus grew to 24 — FINAL STATE: 466 tests green, 24/24 locked, zero
  drift.* + qwen3-8b, olmo-2-1124-7b, stablelm-2-1-6b (the declared-False
  QK-norm + partial-rotary witness; its overview cell carries the
  "(Partial RoPE)" variant tag per Her Eyes' consistency ask — GQA and MHA
  spine branches both tag it now). The 12-model diffusion re-bless ran as
  a vision-fleet agent with hash-diff discipline: 11/12 exactly ONE view
  changed (the self-attn drill), the 12th (prxpixel-t2i) legitimately two
  — its Qwen3-VL text-encoder tower gained the same true norms. All
  supersedes recorded. Her Eyes refreshed post-change: the qwen3/olmo-2
  DISLIKEs cleared (drills now LOVE — "the promise is kept"); recurring
  cross-gallery theme now THREE-time confirmed: the KV-cache corner
  badges are undecodable to newcomers (standing Gate-C design item).
- *Recorded follow-ups:* cross-attention sublayer QK-norm needs
  PER-SUBLAYER code evidence (currently honestly undrawn); hunyuan refiner
  FFN "Activation" pill should name its function (Her Eyes DISLIKE);
  diffusion tower FFN activation naming; S2 router unit NOT started.

**S3 + hydration + FFN-expr WAVE (2026-07-05 — the three "just-execute"
config-fragility fixes, surgical, ZERO corpus drift, suite 475 green):**
- *Norm-kind math-above-eps [S3a]:* one-line ladder reorder in
  `_norm_kind_evidence` — `_code_norm_math` now outranks BOTH eps spellings.
  PhiMoE/Persimmon (construct `nn.LayerNorm` but carry `rms_norm_eps`) draw
  LayerNorm (pixel-confirmed); T5 (name lies, math=RMS) stays RMS; 11 controls
  unchanged. No separate torch-builtin reader needed — `_norm_math_verdict`
  already maps `nn.LayerNorm`/`nn.RMSNorm` as fixed library math.
- *Raw-JSON rung hydration:* `_hydrate_config_class_defaults` routes the
  tolerant-JSON DOWNLOAD rung through `AutoConfig.for_model` when the type is
  known — injects the 34 gemma-2 class defaults (`query_pre_attn_scalar`,
  `sliding_window`, softcapping…) a bare config.json omits. Raw wins, `_`-stamps
  survive, unknown untouched. Scoped to the download rung ONLY — user dicts and
  the frozen corpus never hydrate (why drift is zero).
- *GPT-J FFN-width expression:* extracted the ChatGLM AST arithmetic evaluator
  into a shared `_ConfigExprEvaluator`; `decoder_intermediate_size_from_files`
  reads the layer's own default ternary (`n_inner=None → 4×n_embd`) keyed on the
  intermediate_size vocabulary. GPT-J/GPT-2/CodeGen FFN width 0→correct; param
  count 2.29B→**6.05B** (pixel + header confirmed). Fires only when the config
  field is absent (declared models untouched).
- *Scope discipline:* zero corpus re-bless — each fix triggers only on its exact
  condition (math contradicts eps / download rung / absent intermediate_size),
  and no blessed model meets any. +9 tests (direct-call + real-config +
  negative controls). Two register candidates DROPPED after scouting real
  source: `tie_word_embeddings` (config consistent everywhere) and
  `mask="causal"` (no bidirectional victim) — they belong to B1's net, not a
  hand unit. Next: S2 (router facts from code) is the only remaining active-lie.

**S2 ROUTER-FROM-CODE WAVE (2026-07-05 — the last active-lie config
dependency, SHIPPED; suite 482 green, corpus 25/25 zero-drift):**
- *The lie:* modern MoE checkpoints omit `scoring_func`/`topk_method` from
  config (verified: NO family declares them as class defaults; DSV3 works only
  because its checkpoint serializes them). GLM-4.5 copied DeepSeek-V3's routing
  CODE (`.sigmoid()` + `e_score_correction_bias` + group + gather) without the
  strings, so the `or "softmax"` default drew it as a plain softmax router and
  dropped the aux-loss-free bias.
- *The fix:* `decoder_router_evidence_from_files` scans the routing closure —
  the UNION of the dedicated router class (holds the bias buffer) and the MoE
  block (holds the selection algorithm), following the `sparsemixer` free fn one
  hop. It reports the enacted score transform, the aux-loss-free bias, and
  sparsemixer. Two adversarial cases solved: routing SPLIT across two classes
  (DSV3 router-class buffer + block algorithm — merge, don't either/or), and the
  gpt-oss EXPERT-GLU `torch.sigmoid(gate*alpha)` that must NOT be read as router
  scoring (score-transform detection keys on routing-logit-NAMED tensors —
  `logit`/`rout`/`score`, deliberately excluding bare `gate`). 8/8 families
  correct.
- *Resolution:* code-first — code decides scoring/bias/sparsemixer; a declared
  string only confirms (disagreement → code wins + `_scoring_declared` note).
  Resolved `bias_correction`/`sparsemixer` keys drive the render (three render
  sites switched off the `topk_method=="noaux_tc"` string). Geometry (n_group,
  topk_group, renorm, routed scale) stays config — genuine checkpoint values.
- *Witness:* real GLM-4.5 (92 layers, config.json only) rendered, pixel-reviewed
  (router draws sigmoid + learned-bias side-input + Gather-weights + ×2.5 scale;
  the architecture view shows GQA **(QK-Norm, Partial RoPE)** + MTP — the whole
  session's S1/S1b/S2 work in one gallery), 11/11 nets, blessed → corpus 25.
- *Corpus safety:* only DSV3 + gpt-oss are blessed MoEs; both byte-identical
  (code agrees with their declared facts). +7 router tests. Also: phimoe's
  pre-existing `lm_head_bias` audit fail classified as ignored (one YAML row).
- *Her Eyes lawful completion (same day):* her GLM-4.5 review (LOVE 7, no
  DISLIKE) flagged that the sigmoid was chip-only — the router flowed Gate→Top-k
  with the score transform invisible, so the drill's "expert scores" had no
  origin (the identical lawfulness line she raised for QK-norm). Fix: the score
  transform is now a drawn OP NODE between Gate and Top-k, but ONLY when the code
  scores BEFORE selection — a new code fact `scoring_before_topk` derived by
  source-order (score-call lineno vs first topk-call lineno). GLM/DSV3/Mixtral
  (score-then-topk) draw the node; gpt-oss/Granite (topk-then-softmax) correctly
  do NOT — a node before topk would misdraw them, so gpt-oss stayed byte-stable
  while DSV3+GLM re-blessed. En route, a general correctness win: framework
  container classes (`*ForCausalLM`/`*Model`) are excluded from router scanning
  (their `output_router_logits` aux-loss softmax was polluting the order
  verdict). +3 tests (incl. a `_parse_defs` lru_cache test-collision fix).
  485 tests green, corpus 25/25 locked.
- *Register status:* NO active wrong-drawing config dependency remains. The next
  items are non-lie hardening (norm multiplicity, A5 cross-check) and B1 (the
  general unclaimed-signal net, now with its test corpus).

**Verified hazards + their neutralizations (checked in real source):**
1. *Conditional construction* — StableLM builds q/k norms inside
   `if self.qk_layernorm:` (verified in installed modeling_stablelm.py).
   A naive init-scan would fabricate QK-norm on flag-off checkpoints. The
   `_code_qk_norm` reader MUST classify construction as unconditional vs
   config-conditional and DEFER to config when conditional (the diffusion
   reader's pattern-4 precedent). Negative-control test required.
2. *MLA latent norms* — DSV3's `q_a_layernorm`/`kv_a_layernorm` are LoRA
   latent norms, NOT QK-norm; the reader must not misread them (the DSV3
   fixture hash is the guard).
3. *Full-rope safety* — `_rope_dim` (parser.py:1233) returns None with no
   partial signal and head_dim on factor=1.0 → the strictly-less render
   rule cannot fire on full-rope models. NeoX's IR had rope_dim=null: its
   `rotary_pct` spelling needs an ALIAS ROW (data), not code.
4. *Swallowing wrappers* — every new `_code_*` reader gets narrow
   exceptions + a direct-call test (the lived AttributeError lesson).
5. *Twin tax* — extractor logic written ONCE in evidence/ against the
   existing registries, so A4's merge doesn't unpick it.
Precondition for all waves: suite GREEN first (R0 conftest fixture) — a
red suite means the corpus lock can't be trusted to catch drift.

---

## PART 8 — OPEN DECISIONS (Soumil's calls, blocking nothing until reached)

1. **Label unification / THE FOLD (A7):** fold the flagship LLM chain view
   onto the tower projector (deletes the last hand-authored SVG twin). Safe
   (hero-altitude lossless lock exists); needs Soumil's call on the label
   style differences. Until then every cell rule costs two fixes.
2. **Her Eyes review home:** in-gallery (original spec; destroyed by
   re-bless rmtree) vs `docs/dev/reviews/<slug>.md` pinned to
   hash_signature (survives, auditable history; RESTRUCTURE's taxonomy
   rule). My recommendation: docs/dev/reviews with signature pinning;
   CLAUDE.md amended accordingly.
3. **The second procedure** (sibling of Her Eyes) — announced, undesigned.
4. **Naming:** repo rename unfold-pkg → unfold vs PyPI package name
   `model-unfolder` (mismatch unaddressed in RESTRUCTURE_PLAN).
5. **Retro-tagging** the tagless releases 0.2.10–0.2.16 vs declaring tags
   restart at next release.

---

## PART 9 — THE NUMBERS THAT MATTER (memorize these)

| Metric | Value |
|---|---|
| Package LOC / files | 35,783 / 131 (+ 1,119 YAML) → target ≤ ~30k |
| Twin/duplication debt | ~700–1,000 LOC, ranked in MASTER_PLAN §3 |
| Tests | 442 pass / 3 fail / 5 skip (red = corpus-path rename; R0 fixes) |
| Blessed corpus | 21 fixtures + galleries (deliberately UNTRACKED — 41e1574; needs backup story) |
| 21-LLM sweep | 6 clean / 15 findings / 21 Dable-clean · 12 systemic fixes mapped (Part 7.5) |
| Catalogue | ~118 model ids / 18 families (toserve.md) |
| Serve audit | 94/105 clean · 0 schema · 0 coupling · 5 honest refusals (stale-low) |
| Cost of an unseen family | EXAONE = 2 findings (was ~30) · target 0 |
| Perf (deepseek-v3) | unfold ~1.3s · to_html ~8ms · HTML 337KB · JSON 65KB |
| Version triple | pyproject 0.2.16 ≠ __init__ 0.2.15 ≠ tag v0.2.9 |
| Roadmap | 27 MASTER_PLAN units + R0–R6 restructure, interleaved |

---

## PART 10 — THE TOWER CENSUS (2026-07-06): mapping every structural class-type
## so we can build the archetype library

**THE PURPOSE — read this first, it's the whole intent.** We are building a
CENSUS + CATEGORIZATION of every "TOWER" class across HuggingFace models. A
"tower" (a.k.a. semi-tower / structural sub-block) = a class whose `forward()`
COMPOSES multiple sublayers into a structure, but is NOT the standard decoder
cell we already draw — vision encoders, patch embeds, multimodal projectors,
audio conformers, mergers/poolers, VAE stages, cross-attention fusion blocks,
SSM mixers, etc. WHY we need this census: exotic tower classes currently LEAK
their raw camelCase class NAME as a block label (the `Llama4UnfoldConvolution` /
`Phi4MultimodalAudioConformerEncoderLayer` bug — identity leaking into the
drawing, forbidden by Law 1). The fix is TIER-2: each tower-type becomes a
DESIGNED FILL-IN DIAGRAM (like the MoE router view — an authored template the
parsed facts populate, NOT a raw op-chain dump). To build that ARCHETYPE
LIBRARY we must first KNOW THE LANDSCAPE: how many distinct tower class-types
exist, in what categories, so we can distribute the work into tackle-able
SECTIONS and cover ~99% of models with a bounded set of archetypes.

**WHERE WE ARE GOING (the architecture, Soumil's vision).** NOT a rigid
100%-confined archetype set — general STRUCTURING LAWS for any dynamic block +
a TYPE SYSTEM distributing classes into types, each a curated diagram Soumil
authors ("all classes go through me, we decide how their diagrams look"). The
census is the MAP that says which archetypes to build. Each archetype = a
triple: (1) a STRUCTURAL SIGNATURE that recognizes it name-independently, (2) a
DESIGNED diagram, (3) a FACT-BINDING that fills its slots. Recognition is
tri-state (archetype / generic-filtered-op-chain fallback / never-a-wrong-
archetype). The FILTRATION for the generic fallback = draw SUBMODULE CALLS
(self.<field>(...)), filter TENSOR METHODS (.permute/.reshape/.view/index/
dropout) — bounded by the class's constructed fields, so it can't explode.

**HOW TO NAVIGATE THE CODEBASE for towers (the map):**
- Tower CLASSES live in installed `transformers/models/<fam>/modeling_*.py`
  (vision via `vision_config`, audio via `audio_config`, projectors via
  `multi_modal_projector`/`*_projector`, etc.) and `diffusers` (VAE/UNet stages).
- Tower PARSING (the existing machinery to extend): `evidence/vision.py`
  (`layer_facts_from_block` = the ONE shared per-tower fact reader;
  `_patch_ops` = patch-embed op-chain; `_class_node`); `submodel.py`
  (`submodel_spec` = recursive sub-model tower; `:298` = MoE-drill; the
  standard-cell tower projector); `special_parts/modalities/` (vision/audio/
  fusion/projector builders; `schema.py:tower_submodel_spec` — HARDCODES
  supporting-tower ffn kind = "dense", a known limitation).
- Existing ARCHETYPES (the pattern to replicate): `tower_cell` (standard
  decoder cell), `moe_router.py` (MoE — the exemplar fill-in diagram),
  vision-encoder cell, VAE up/down stage.
- The LEAK site (raw class name → label): `evidence/vision.py:399` (conv label
  = raw `field_type`).

**WHAT WE ARE DOING RIGHT NOW:** a deep scan of as many installed models as
possible → extract every non-standard structural class → categorize into
SECTIONS (each a candidate archetype family) with counts + representatives →
deliver a class-wise distribution MD (`TOWER_CENSUS.md`) that becomes the
build-plan for the archetype library. Facts-first still holds: this census is
the MAP; the archetype diagrams are built on top of it afterward.

**CENSUS RESULT:** (DONE 2026-07-06 — full deliverable in `TOWER_CENSUS.md`.)

Scanned **466 modeling files**. Two passes: a name+role census (every class
with `forward()`) and a TRUE-tower pass (only classes composing ≥3 submodule
calls = a real internal structure worth an archetype). Distilled:

- **COVERED (no new work):** standard decoder cell 261/218, attention leaf
  286, FFN leaf 219, MoE 228/69, head+LM+model wrappers 796, embeddings 99.
  These map to `tower_cell` / attention-drill / FFN-drill / `moe_router` /
  containers-not-drawn.
- **TOWERS NEEDING ARCHETYPES (the work), grouped into 4 buildable SECTIONS:**
  - **§1 VISION** (~74 models — the largest): vision-encoder-cell (249 TRUE),
    patch-embed (26 TRUE — the `Llama4UnfoldConvolution` leak), patch-merger
    (`Glm4vVisionPatchMerger`), attention-pooler (`Aimv2AttentionPoolingHead`),
    vision-attn variants.
  - **§2 AUDIO/CONFORMER** (~22 models): conformer-cell (39 TRUE —
    `FastSpeech2ConformerEncoderLayer`, `Phi4MultimodalAudioConformerEncoderLayer`),
    conv-module (`Phi4MultimodalAudioConvModule`), audio-encoder (whisper-style).
  - **§3 MULTIMODAL BRIDGE** (~42 models): mm-projector (28 TRUE —
    `AyaVision`/`Cohere2Vision`/`Aria`), cross-attn-fusion (24 — mllama).
  - **§4 VQVAE/IMAGE-TOKENIZER** (~6 models — Chameleon/Emu3): resnet-stage
    (12) + down/up-sampler (8) → REUSE the diffusion VAE renderer.
- **DEFERRED:** SSM/Mamba mixers (18 models — C1); **UNTHEMED-779** = the big
  mixed bucket (heads/containers already covered + real un-named towers) that
  needs a STRUCTURAL triage pass before we can claim ~99% coverage.

**Coverage math:** standard-cell + MoE (done) **+ ~10 tower archetypes**
(vision-encoder-cell, patch-embed, patch-merger, attention-pooler,
conformer-cell, conv-module, audio-encoder, mm-projector, cross-attn-fusion,
vqvae-stage) ≈ 99%. **Build order:** §1 vision (patch-embed first — smallest
win, fixes the original leak) → §3 projector → §2 conformer → §4 vqvae →
triage UNTHEMED-779, defer SSM. Each archetype ships: structural signature
test + designed diagram (Soumil's look) + fact-binding + generic-fallback
(tail never a raw class name) + loud-fallback marker (un-archetyped = worklist,
not "covered"). Recognition stays STRUCTURAL — name themes were the census map
only, never the recognizer (gpt-oss MoE is `GptOssMLP`; Qwen3Moe names all Moe).

---

## PART 11 — THE HONESTY WAVES from MODEL_DISTRIBUTIONS.md (2026-07-06)

**SOURCE:** `MODEL_DISTRIBUTIONS.md` (workspace root, ~2500 lines) — a per-block
config-vs-code-vs-default provenance trace of 20 models (LLMs + DiTs + UNet +
video), each with a 🔴 generic-assertion section, plus a cross-model synthesis
(its §4 = 9 real honesty gaps ALL with sable 12/12 GREEN; §5 = the consolidated
generic-assertion hunt-list). Headline: nets green ≠ honest — only trace+pixels
caught these. Top claims RE-VERIFIED against live source this session:
`attention_score_scaling_from_files` (patterns.py:2390) is called ONLY on the
encoder-tower path (diffusor/parser.py:1830), never LLM/DiT main; unet.py
hardcodes GroupNorm+SiLU/GEGLU prose (:233-273,:326,:377,:551); `_dit_norm_kind`
(diffusor/parser.py:1382) is config-only while `diffusion_norm_from_classes`
(patterns.py:411) sits unwired for that slot.

**THE BUILD PLAN (3 waves, facts before presentation):**

**WAVE A — content bugs (fabrication / undercount / dropped op). Do FIRST.**
- A1 🐞 **Sana RoPE cross-component leak** (the one fabrication): `_source_files`
  unions DiT + Gemma-2 text-encoder source, so Gemma's rotary markers make the
  DiT claim RoPE it doesn't have. GENERAL FIX: scope every code reader to the
  ROOT component's files (same per-component-registry principle that already
  keeps Qwen2-VL text/vision FFN evidence apart).
- A2 **BLOOM param undercount (61.1B drawn vs 176B real)**:
  `decoder_intermediate_size_from_files` only evaluates the n_inner ternary in
  the DECODER LAYER init; BLOOM's `4*hidden` lives in `BloomMLP.__init__`.
  GENERAL FIX: follow into the FFN class's own init (the extractor already
  resolves field classes).
- A3 **Wan2.2 cross-attn QK-norm dropped**: reader gates `not cross_attention`
  globally. GENERAL FIX: per-attention-SITE QK-norm verdict (self and cross
  each read from their own class/branch), not a global gate.
- A4 **GPT-OSS attention sinks unclaimed**: name-vocabulary (`attention_sinks`)
  misses the bare `self.sinks` nn.Parameter. GENERAL FIX: structural detection —
  a learned Parameter concatenated/added into scores pre-softmax (evidence-not-
  identity poster child).
- A5 **Lumina over-drawn tanh gate**: class read takes the `modulation=True`
  branch but the refiner instance is built `modulation=False`. GENERAL FIX:
  construction-site-aware branch pruning — we ALREADY have `_ctor_kwarg_truthy`
  (HunyuanVideo lane norms); apply it to gate/modulation reads.

**WAVE B — kill the generic assertions (ranked by blast radius):**
- B1 ⭐ **highest-value single fix**: wire `attention_score_scaling_from_files`
  into the LLM + DiT MAIN attention paths (reader exists, proven on encoder
  towers; AuraFlow/SD3.5/CogVideoX/PixArt currently draw code-verified unscaled
  QKᵀ on their T5 tower while ASSERTING sqrt(dim) on the DiT in the same render).
- B2 **norm placement `or "pre"`** (parser.py:555): when `_code_topo` is None
  draw placement-UNKNOWN pale instead of asserting pre (OLMo-2 post / Gemma-2
  sandwich would mis-draw if source unreadable; SD3/FLUX/Lumina DiT main blocks
  effectively fire it today).
- B3 **UNet path onto the code-evidence rail** (biggest un-read surface):
  unet.py's hardcoded ResNet-SiLU/GEGLU/GroupNorm/sqrt idioms → route through
  the same `_code_*` readers the DiT path uses (`act_fn` is even already read,
  on a dead branch).
- B4 **unparsed router** → blocking `evidence_ambiguity` finding instead of
  `or "softmax"` (GLM-4.5 pre-S2 was the victim; S2 landed, make the fallback
  loud for the next family).
- B5 **defaults distinguishable-from-declared** in the spec (provenance enum on
  every fact: declared/code-proven/default-asserted) — the standing cure; makes
  §5's whole class of latent hazards visible in one shot.

**WAVE C — small drawn-fact gaps:** Gemma-2 softcapping chips (facts in bag,
draw them); Wan two-expert `transformer_2` spelling (vocabulary → everchanging
YAML); PixArt `attention_bias=true` honored in drawing; DiT block-norm from
`AdaLayerNorm*` class when config silent (wire `diffusion_norm_from_classes`
into the `_dit_norm_kind` fallback).

**SEQUENCING vs the Tower Census (Part 10):** two tracks. THIS part is the
FACTS track (Soumil's law: facts first); the census/archetype library is the
PRESENTATION track. Order: Wave A (contains a live fabrication) → B1+B4 (small,
sharp) → B3 (big, self-contained) → B2/B5 + Wave C interleaved with archetype
§1. Also: `DISTRIBUTION_OF_INVOCATION.md §5-7` and `INVOCATION_FLOW_DEEPSEEK_V3
.md §9` are STALE (list S1/S2/S3 as pending; they SHIPPED) — refresh when next
touched. Every fix stays GENERAL (no per-model patches), lands with a code-
evidence test + re-bless only where the drawing legitimately changes.

**STATUS 2026-07-07 — THE WHOLE ARC IS CLOSED: 529/529 tests, 25/25 corpus
locked, 16 galleries re-blessed with supersede + Her Eyes reviews.** All 14
wave units + follow-ups U1-U4 (anchored UNet FFN reader — SDXL GEGLU restored
WITH evidence, ⭐flag dissolved; MLP-bias code shift; MLA cross-check net;
cross-attn + hybrid-mixer schedule locks, whose cross lock immediately caught
and fixed the mllama bare-config suppression) + U5 (the pixel pass, which
caught 3 real defects in a mechanically-green batch: gemma-2 lm_head label
overflow; gpt-oss sink lane mis-wire → final form ONE spine box, weights are
never input nodes; LTX wrong norm kind → `_code_norm_kind` is now
BLOCK-scoped, never a file-wide vote). DISTRIBUTION §9 carries the moved
rows. Full build log + the ranked FOLLOW-UP LEDGER (top: pre-FFN AdaLN
conditioning tap missing on 7 DiT galleries) in
`SURGICAL_PLAN_HONESTY_WAVES.md`. Zero new config reads anywhere in the arc —
structure is HF-source-authoritative throughout; Soumil reviews + commits.**

---

## PART 12 — THE HER-EYES THEME MAP (2026-07-07): presentation debt, clustered general

**SOURCE:** the 21-LLM sweep's design reviews (`previews/llm_sable_sweep_
2026-07-05/*/images/her_eyes_review.md` — 146 images, LOVE 93 / FINE 50 /
DISLIKE 3) aggregated into GENERAL mechanisms in **`HER_EYES_THEME_MAP.md`**
(workspace root — the full map with quotes, counts, loci).  Reviews predate
the honesty waves; verified bucket-0 items already solved (persimmon QK-norm
drawn, gpt-j/stablelm partial-rotary rot/pass render, mixtral router softmax
node).  The map's shape:
- **Bucket 1 — polish wave (do first):** 8 small general fixes — port-caption
  format ("in (8,192)" tuple-misread), ReLU² typography, card-only newcomer
  glosses per mechanism (ALiBi/noPE/⊙/‖/fusion abbrevs), Expert-k ellipsis
  naming, ×1.0 no-op glyph suppression, SWA-tag attachment rule,
  window≥context ⇒ "full context" wording, strip retitled "Layer map".
- **Bucket 2 — layout-rhythm unit (pixel-reviewed):** MoE fan-out headroom
  (5 models), shared-norm centering under parallel children (falcon vs the
  praised gpt-j/neox), router page rhythm, KV-card density tier.
- **Bucket 3 — structural designs:** (1) THE VARIANT-FOLD ENGINE — her #1
  theme (8+ models): twin spines/panels/drills differing by one box → one
  toggled view; interacts with view identity + bless hashing + never-hide-
  an-op; design doc first.  (2) 1-type layer-strip demotion to a spine badge
  (9 models; removes a view class → Soumil's yes + consolidated re-bless).
  (3) Tower plumbing-op collapse = the Tower-Census archetype thesis (all 3
  DISLIKEs are vision projector/patch pages — census §1b/§3a requirements,
  never per-model).  (4) journey/footnote tier in the render spec (her Q4
  answers are uniform: drills are destinations, strips are footnotes).
**Meta:** 93/146 LOVE, 3 DISLIKEs — the format works; every complaint is one
of ~5 general mechanisms, and the ugliest pages are exactly what the census
already scheduled.  Nothing model-specific anywhere.

**THEME-L (the 3 DISLIKEs) — LANDED 2026-07-07.  Intent, mechanism, and the
navigation map for whoever touches tower op labels next:**
- **WHAT/WHY:** tower drill pages buried real ops under plumbing ladders and
  raw class-name boxes.  Two general mechanisms landed: (1) STRUCTURAL human
  labels — a class name is provenance (SourceOp.class_name + an "Implemented
  by X" description), never the drawn word; (2) `_collapse_plumbing_runs` —
  consecutive AXIS-plumbing reshapes fold into one step whose description
  enumerates every move; SEMANTIC reshapes (merge/split/join — the ops that
  change what tokens ARE) never fold.  The plumbing set is OUR OWN closed
  label vocabulary (`_PLUMBING_LABELS`, evidence/vision.py) — self-vocabulary,
  not identity.
- **THE THREE-BUILDER MAP (all tower op labels live here — none elsewhere):**
  (a) `evidence/vision.py` — patch-embed sites (`_owner_patch_flow_ops`,
  `_ordered_patch_callable`; humanizer `_patch_impl_label`: unfold token →
  "Patch unfold + project", conv → "Patch convolution"); (b)
  `evidence/projector.py::_callable_ops` — SHARED by projector AND audio
  towers (neutral noun "Convolution"; `_conv_flavor` reads `groups=` at the
  CONSTRUCTION SITE — a Conv subclass may have no __init__, so words-in-names
  are never the signal); (c) `evidence/audio.py::_label_for` + `_flow_ops`
  (the SSA path; depthwise from the developer's own field declaration
  `self.depthwise_conv1d`; the old `Gemma4Audio`-prefix strip — a hardcoded
  per-family string — is DELETED).  Plus one pre-existing engine gap fixed:
  `opgraph.py::ops_region` silently dropped declared op `description`s; now
  → `meta.desc` so every declared-ops card can carry prose.
- **RESULTS:** gemma-3 projector 5→4 boxes (merge stands out); patch pages
  read "Reshape patches → Patch convolution → Flatten tokens"; llama-4's
  `Llama4UnfoldConvolution` box now reads "Patch unfold + project" (derived
  from its own unfold token).  qwen2-vl (the one blessed witness) re-blessed
  + fresh Her Eyes (L4/F9/D0).  llama-4's remaining ladder + the lone-Linear
  projector page are the RECORDED boundary: census §1b/§3a archetype design
  work, not more filtration (reconciliation in HER_EYES_THEME_MAP.md).
- **OLD/NEW COMPARISON SET:** old pages untouched at
  `unfold-pkg/previews/llm_sable_sweep_2026-07-05/<model>/images/`; new
  renders at `unfold-pkg/previews/theme_l_fix_2026-07-07/<model>/` (gemma-3,
  llama-4-scout, qwen2-vl) — flip pairs by identical filename.
- **CLOSED at 530/530 tests, 25/25 corpus** (2026-07-07).
- **THE TIER DOCTRINE (Soumil-confirmed, guides all presentation work):**
  correctness never left — the collapse moved truth from BOX-COUNT to CARD
  PROSE (every folded move enumerated in order; provenance on the card;
  op-KIND presence preserved for the set-based nets), and anything that
  changes what the tokens ARE (merge/split/join, depthwise-vs-plain) keeps
  its own box.  VOICE tiers: cards now EXPLAIN ("letting a head attend to
  nothing"), matching the pages Her Eyes loves (the KV card that argues, the
  fusion strip that shows); the fallback tower PAGES remain honest
  fact-chains BY DESIGN — "not blatantly stating facts" (the fill-in way,
  the MoE-router exemplar) is the ARCHETYPE tier's deliverable (census
  §1b/§3a), not a fallback goal.  Fallback = never wrong, never a class
  name, never a ladder; archetype = the story.

---

## PART 13 — AUDIO-GEN + TTS SUPPORT (2026-07-07: scope BLESSED, first units LANDED)

**BUILD STATE:** U-D0 (latent-grid guard) + U-A (composite wrapper walk,
ADDITIVE cross-attention cell, T5 tower via the encoder round-trip — now
lifted to adapter-neutral `encoder_panel.py`) + U-B (K-codebook streams)
+ U-D core (1-D audio geometry, evidence-gated patchify, oobleck ladder)
are IN, suite-green with witness tests + coverage-corpus entries; unit
statuses live inline in `SURGICAL_PLAN_AUDIO.md` (the build log).
MusicGen went from hard refusal to a full honest gallery; Stable Audio
from would-be fabricated 1024×1024 grid to 1-D truth.  U-C/U-E next;
U-F deferred (MAGNeT repos are audiocraft-native, no config.json).
Soumil action: accept the stable-audio-open hub license so the class-
defaults witness can swap to the hub config.

**TTS check (2026-07-08):** Parler-TTS parses BY ID (raw-config loader
fallback + the U-A walk; cross/K-fan honestly unproven — its modeling
package isn't installed); VITS's parallel+double cell was a CLASSIFIER
FABRICATION on classic post-norm (``norm(residual+x)``) — fixed both
idiom forms, VITS/SpeechT5 now sequential-post; silent component drops
(flows/duration/HiFiGAN/pre-post-nets, flat-seq2seq decoder half) are
now STATED omissions from the undrawn_component_fields vocabulary.
Galleries: previews/tts_first_iterations_2026-07-08/.  U-E owes the
real structures; ledger in SURGICAL_PLAN_AUDIO.md U-E block.

Soumil asked what supporting MusicGen/AudioGen, MAGNeT, Stable Audio,
AudioLDM/2, Tango, ACE-Step, MusicLDM/Riffusion, YuE (+ TTS) needs.  Two
docs at workspace root: **`AUDIO_SUPPORT_MAP.md`** (the why/what map) and
**`SURGICAL_PLAN_AUDIO.md`** (the code-wise plan, recon-verified to
file:line against installed transformers/diffusers and our seams).  The
recon's load-bearing finds: `_component_configs` (sources.py:306) only
descends `*_config`-suffixed keys, so MusicGen's `text_encoder`/
`audio_encoder`/`decoder` slots are invisible today; `blocks.py:196-198`
would FABRICATE a square latent grid from Stable Audio's scalar 1-D
sample_size (U-D0 = kill that guard-first, zero new scope, safe pre-bless);
Patchify/Unpatchify are hardcoded in the DiT hero and must become
evidence-gated; Stable Audio's conditioning is a global PREPENDED token
(a new dialect beside AdaLN, read from source); the loader has no
`vocoder`/`projection_model`/`language_model` slots.  The shape:
every family is one of FOUR structures that map onto the EXISTING two
adapters — codec-token LM (MusicGen/Parler/Bark/YuE → transformer), masked
NAR LM (MAGNeT), audio-latent DiT (StableAudio/ACE-Step/F5-TTS → diffusor;
`StableAudioDiTModel` already matches the DiT marker, oracle installed),
spectrogram UNet+vocoder (AudioLDM/Tango/MusicLDM → UNet exists).  SIX
general units: U-A seq2seq/composite wrapper walk · U-B multi-codebook
streams (the one genuinely new LM fact: K embeddings summed, K heads fanned,
delay-pattern strip) · U-C codec/RVQ tower (census §4 audio twin; EnCodec
LSTM stays honest-opaque) · U-D audio-DiT 1D geometry + oobleck conv1d VAE ·
U-E mel+vocoder loop-hero tail + HiFiGAN vocoder archetype (gated on
DECLARED pipeline `vocoder` components) · U-F masked-iteration loop variant
(generalizes the block-diffusion hero).  PROBED today: Riffusion parses as
the SD it is; YuE as a llama decoder (true); vits/speecht5 parse
MISLEADINGLY-partial (silent component drops — U-A/U-E fix); encodec/bark
refuse honestly.  Recommended order: MusicGen unit (A+B+C, unlocks
Parler-TTS free) → Stable Audio (D) → AudioLDM/vocoder (E) → MAGNeT (F).
**Scope-law: this extends the recorded scope line — Soumil must bless.**

---

*If you are a future session reading this: start by checking Part 7 against
reality (suite state, token, corpus tracking) — those were live wounds when
this was written. The laws in Part 1 are not context, they are law.*


## PART 14 — THE EVIDENCE-HARDENING CAMPAIGN: WHY EVERY ERROR HAD ONE ROOT, AND THE GENERAL FIX (2026-07-12)

*Written after executing H0–H4 of docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md
and making — and catching — six real mistakes. Soumil asked that the issues be
solved GENERALLY, not patched one by one. This part is that general solution.
It is deliberately about the IMPLEMENTATION PROCESS, because the code laws
(Part 1) were never the thing that failed.*

### 14.1 The single diagnosis

Six errors happened this campaign. They look unrelated — a completeness flag, an
over-claimed DONE, three test regressions, a phantom-alias assumption, a masked
unread, a wrong marking design. They are not unrelated. **Every one of them
happened at a point where a rule existed as PROSE and had to be honored by the
implementer's memory and judgment. Not one happened where the same rule existed
as a MECHANICAL GATE that fails a build.**

That is not a coincidence — it is the plan's own thesis turned back on the
process that executes it. The plan's final definition of success (§12) is that
"the doctrine stops depending on reviewer intelligence and becomes a property of
the library." The campaign proved the contrapositive: **wherever the doctrine
still depended on the implementer's intelligence, the implementer (me) failed —
predictably, repeatedly, and in exactly the places the doctrine was not yet
mechanical.**

So the general fix is not six patches. It is one move applied everywhere:
**convert the discipline that governs the WORK from prose into gates, the same
way the plan converts the discipline that governs the CODE from prose into
gates.** The process must become self-enforcing, or a tired/optimistic/forgetful
implementer will breach it again.

### 14.2 The evidence — each error, its prose rule, its missing gate

| # | What I did wrong | The rule I breached | Was it prose or a gate? |
|---|---|---|---|
| 1 | Lift manufactured `completeness="complete"` | H1 amendment "never invent epistemic metadata" | PROSE (an amendment I hadn't re-read) |
| 2 | Marked H2 DONE with its 4th blocking gate unbuilt | §6 DONE = every exit + blocking gate verified | PROSE (DONE was my judgment, not a checklist) |
| 3 | Broke 3 tests editing before the blast-radius | §8.1 P-2 "enumerate every consumer incl. tests" | PROSE (I grepped code, skipped tests/) |
| 4 | Designed the net around a false residual (231 phantoms) | "measure, don't assume" (implicit everywhere) | PROSE (no probe-before-design step) |
| 5 | A global-scope fix exposed masked cross-scope unreads | I-10 / H3.6 "no flat global leak" | ARCHITECTURE (the leak the plan already names) |
| 6 | First designed `consumed` as a ledger derivation (misses geometry) | "acquire before interpreting" (G-3) | PROSE (designed before mapping the dataflow) |

Read the last column top to bottom. Five of six were prose. The sixth (5) was a
real architectural leak the plan already schedules a fix for (H3.6) — I hit it
because I made a global change while that leak was still open.

### 14.3 The four general solutions (each kills a whole class, not one bug)

**G1 — Every plan rule becomes a mechanical gate before it can be relied on.**
This is the master fix; it absorbs errors 1, 2, 6. Concretely:
- The twelve invariants (I-1…I-12), each unit's EXIT criteria, and every
  mid-course AMENDMENT get encoded as an executable assertion — a constructor
  error, a blocking test, or a census — the moment they are written. A rule that
  is only prose is treated as NOT YET IN FORCE; it may guide, but a unit cannot
  be called DONE on the strength of prose alone.
- A unit's DONE is not a sentence I write; it is a checklist FILE
  (`docs/H<n>_EXIT.md` or a `test_h<n>_exit.py`) that lists each exit criterion
  and the exact test that proves it. If a criterion has no test, the unit is not
  DONE — full stop. This is precisely what would have caught H2: the H2.4
  blocking gate ("new legacy structural write") had no test, so H2 was not DONE,
  and a checklist would have said so instead of my optimism.
- Amendments are re-read as a DIFF at the start of every unit. The plan has
  already changed under me three times (§8.1, the H1 and H0 amendments); the
  standing assumption is that it changes again. "Re-sync with the current plan
  text" is step zero of every unit, not a courtesy.

**G2 — Measure before you design; the probe is the first artifact of every
unit.** This absorbs errors 4 and 6. Before any reader/net/marking is designed,
run a probe against the ACTUAL runtime data — the real residual, the real
sources, the real present-fields, the real dataflow from acquisition to spec —
and write the numbers down. Every design assumption ("the residual is near the
multipliers", "consumed can be derived from the ledger") must be a MEASURED fact,
not an intuition. The phantom-alias disaster (231 fields I assumed were ~5) and
the ledger-derivation dead-end (which silently drops geometry) were both pure
assumption-without-measurement, and both dissolved the instant I probed.

**G3 — The blast-radius is a tool, not a grep, and it always includes the four
things I forget.** This absorbs error 3. §8.1's P-2 consumer graph must
mechanically enumerate, for any changed symbol: (a) production callers, (b)
TESTS that assert its behavior, (c) the mechanical NETS (advisory and blocking)
that read its output, and (d) CROSS-SCOPE effects when the symbol feeds a global
structure (the accessed set, the audit, the ledger). I broke three tests because
I searched `model_unfolder/` and not `tests/`; a real consumer-graph step would
have listed the `capture_accesses` test caller and the two census-pinning tests
before I touched a line.

**G4 — Scope everything by owner; the flat global leak is the root of every
cross-scope surprise.** This absorbs error 5 and is a genuine architecture fix
(H3.6). The config audit, the accessed set, and the ignore rules currently match
by KEY NAME across every adapter/component/scope. That single flat namespace is
why a transformer-side change (phantom-alias fix) silently unmasked diffusion-VAE
unreads: one scope's reads were papering over another scope's gaps. Until the
audit is owner-scoped, ANY global change will have unpredictable cross-scope blast
radius. The general fix is not to tiptoe around the leak — it is to close it:
make `accessed`/`consumed`/`ignored`/`unread` carry their owner path, so a
transformer field and a `_vae_config` field with the same name are DIFFERENT
entries, and no scope can mask or affect another.

### 14.4 The meta-principle (the thing to actually remember)

> The campaign's laws were never the risk. The risk was always the GAP between a
> law being written and a law being enforced. Close that gap the same way the
> product closes it for models: **detect the breach from evidence (a failing
> gate), never from vigilance (a careful implementer).** A process that relies on
> me being careful will fail exactly when I am tired, optimistic, or rushing —
> which is exactly when it matters. Make the process, like the library, correct
> by construction.

### 14.5 What this means for the next sessions (concrete order of operations)

1. **Before touching any unit:** re-read the plan text for that unit + a diff
   scan for new amendments (G1). Write/refresh its `H<n>_EXIT` checklist mapping
   each exit criterion → its enforcing test.
2. **First artifact of the unit:** a measurement probe of the real data (G2),
   its numbers recorded in the completion log.
3. **Before any production edit:** the §8.1 pre-change gate with a COMPLETE
   consumer graph — production + tests + nets + cross-scope (G3).
4. **When a change is global-scope:** stop and ask whether the flat global leak
   (G4) will make its blast radius cross scopes; if so, prefer scoping the target
   (H3.6) over a global edit.
5. **A unit is DONE only when its EXIT checklist is fully green** — never on a
   written judgment (G1).

None of this slows the real work; it front-loads the thinking that the six errors
proved I skip under pressure. The plan already made the CODE self-enforcing. Part
14 makes the PROCESS self-enforcing too.
