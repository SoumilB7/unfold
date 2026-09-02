# SURGICAL PLAN — EVIDENCE-COMPLETE ARCHITECTURE RECONSTRUCTION
*(2026-07-11 · from RUN77_PROBLEM_MAP.md + the external review Soumil endorsed.
Doctrine: code determines structure and meaning; config supplies checkpoint
values to expressions proven by that code. We eradicate config-only structural
guesses, defaults-as-facts, template selection by key presence, and
read-but-never-projected values — never config itself.)*

## FULL-CAMPAIGN EXECUTION STATE (2026-07-12, Soumil: "get it done perfectly,
## strategically, eradicated — nothing compromised")
★ U2 CLOSED 2026-07-12: 603/603 green · 25/25 zero-drift · SILU TIER RETIRED
STRICT with zero drift (MoE expert-hop made deepseek/glm/gpt-oss code_proven
— the arbitration dissolved) · witnesses: bert "bidirectional" in pixels,
t5-base dense-ReLU truth, flan gated-gelu declared, MiMo config-declared
RoPE + θ chip, gpt2/llama byte-stable · galleries previews/
u2_default_kill_2026-07-11/ + u2_readers_2026-07-11/ · P2/P3 builder's 6
disclosed deviations logged in its final report (task transcript).
Track A (transformer, serial on parser.py): U2 ✅ → U3 BUILDER RUNNING → U4
(spec = _problem_map/01 + PROJECT_CONTEXT Group-1 unit; the 6 broken
spellings onto the Group-1 engine, source-discovered gate + config-evaluated
values, honest-unknown fallback, heterogeneous per-variant views — this also
fixes Mllama's merged-view collapse per PM-4 R7) → U4 (spec = _problem_map/
03+05 + PM-2 §2: enc-dec topology views, router order/renorm from forward
AST, rope-dialect projection incl. context-stretch, shared-expert from
construction, score-scale VALUE, CLA periodic, MFA honest-unknown, composite
wrapper-local head facts).
Track B (diffusor, disjoint — RUNNING): U6 builder launched (spec =
_problem_map/06, sub-unit ladder in its contract). 2026-07-12 EXTERNAL
REVIEW of its mid-build state deep-verified (_problem_map/08): 8 BINDING
corrections sent to the builder — no identity tables (unet_blocks/
vae_classes/schedulers.yaml → class-resolved evidence; class name = address
only), no unknown→transformer2d/kl defaults (unknown stays pale), mid-block
absence = complete negative proof (ALREADY FIXED in-tree), per-stage
temporal, placement≠cell channels, schedulers from step(), test-import
(REFUTED — passes), IDENTITY-GUARD HARDENING (catch class-keyed structural
maps by BEHAVIOR — the scanner's fixed key-name list is a hole). Diffusion
layer stays UNCOMMITTED until corrected + fresh full-suite. ef9046c
(Soumil's transformer-sweep commit) stands. ADDED TO U4: project
embedding_multiplier/logits_scaling (parser.py:747/:750 still discarded —
U2 missed these two; confirmed). U5/U6 must extend
fact_projection.py DRAWABLE_FAMILY_SEGMENTS to their new surfaces.
Track C: U5 modality discovery (spec = PM-4 R5; after U4 — shares parser.py).
Track D: U7 closure (spec = _problem_map/07 minus already-landed
_MASK_SHORT; + frontier-fixture STAGING — galleries + reports prepared,
bless itself remains Soumil's act).
SOUMIL'S YAML DOCTRINE (2026-07-12, BINDING for all units + GROUND_RULES.md):
config = checkpoint truth; what dies is field-presence/spelling/identity/
convention determining architecture without source/schema-backed meaning.
(1) aliases.yaml keep, NARROW + TYPE every row (geometry-value/declared-flag/
schedule-values/address; presence never gates structure). (2) config_facts
.yaml: dismantle catch-all; migrate structural rows to fact families.
(3) text_encoders.yaml: display vocabulary only. (4) any identity→
architecture YAML: remove completely. The coherence audit outputs the
per-file migration plan; a cleanup unit executes it before U7 closes.
EVERY builder: FactLedger statuses on every touched fact · both P4 nets stay
blocking · 25/25 zero-drift absolute · witnesses w/ before/after galleries ·
full suite per gate · no git.

## UNIT LADDER (each unit gates on tests + corpus zero-drift + pixel review)
- **U1 SOURCE PARITY (this doc's active unit — spec below).**
- U2 provenance + projection accounting + the default-kill + missing code
  readers — FULL SPEC BELOW (⏳ ACTIVE).
- U3 general per-layer variant engine (source-discovered gate expression +
  config-evaluated values; config-normalizer fallback at oracle_missing;
  witnesses: gpt-neo, MiniMax, recurrentgemma, Phi-3-small, step3, Nemotron NAS).
- U4 transformer semantic completion (multiplier projection; score-scale VALUE;
  rope dialect/context-stretch; shared experts from construction; router
  order/renorm from forward; Llama4 dense width; MFA honest-unknown; CLA
  periodic extension; T5 gate + enc-dec topology; BERT bidirectional/post-norm/
  type-embeddings; wrapper-local head facts).
- U5 component/modality discovery from construction (config keys as address/
  declared fallback, honest placeholders).
- U6 diffusion evidence rail (UNet construction graph, temporal ops, scheduler
  step semantics per class, VAE stages/dimensionality, conditioning from
  declared components, prediction-symbol propagation, patch/merge geometry).
- U7 guard/corpus/presentation closure (frontier fixtures per root; unresolved-
  severity unification; shared-expert net un-gating; verified label one-liners).

---

## UNIT 1 — SOURCE PARITY  ✅ RIGOROUS-GATED (2026-07-11; Soumil review+commit pending)
**Thesis:** shipped parses and audited parses must share ONE source-resolution
context, and that context must find source wherever it lawfully exists
(repo id → hub snapshot (cache-first), unknown-model_type → declared class,
honest warnings never masked). Until this lands, every other fix is measured
against a parse the user never sees.

**Verified problem sites (PM-4, all reproduced live):**
R1 parser.py:411-436 raw-JSON rung returns unstamped dict → sources `_model_id`
None → `_hub_bundle` bails → remote-code repos evidence-blind (internlm ×3,
MobileLLM ×5, EXAONE, Qwen1 ×3, Orion, Ling, Baichuan ×3, chatglm-6b, Molmo,
MiMo, deepseek-moe, Phi-4-MM, Phi-3.5-V…).  R2 sources.py:183 `if not
model_type:` — unknown-but-present model_type blocks the declared-architecture
class lookup (Kimi-K2 → installed deepseek_v3).  R3a sources.py:607+ no
cache-first read.  R3b sources.py:121-123 fileless hub bundle's true warning
silently replaced by the local "no installed source" one.  R4(sable)
sable.py:215-217 harness-side stamp = audit-vs-ship gap.

**The four edits (surgical, general, no per-model rows):**
1. `_load_raw_config_json`: stamp `cfg["_repo_id"] = model_id` (the loader's
   provenance stamp `_model_id` already prioritizes; mirrors diffusor
   loader.py:116 and the Mistral rung's `_name_or_path`). Identity-as-ADDRESS
   at load time only; parse stays name-blind (`_repo_id` is not a fact channel;
   `name_blind_diff` pre-resolves context so the guard is unaffected).
2. `_installed_transformers_bundle`: when the model_type family walk yields NO
   files and the config declares an architecture, fall through to
   `_transformers_file_for_class` — the same address lookup an ABSENT
   model_type already gets. Unknown ≠ absent.
3. `_hub_bundle`: cache-first (`local_files_only=True`, fall to network only
   when the cached snapshot has no .py) — offline/gated users with the
   snapshot on disk get evidence.
4. `resolve_source_files` remote-code branch: when the hub bundle is fileless
   but carries a real warning, MERGE it into the returned fallback instead of
   dropping it. And DELETE the sable.py stamp — the loader owns the address;
   sable must never be better-informed than unfold().

**Witness matrix (four counterexample classes + controls):**
- raw remote-code: internlm/internlm-7b (expect: RoPE box appears; warning gone)
- unknown model_type + installed class: moonshotai/Kimi-K2-Base (expect:
  deepseek_v3 source resolves; nested_conformance un-vacates)
- cached/offline: facebook/MobileLLM-1B with HF_HUB_OFFLINE=1 (expect: source
  from cache)
- NEGATIVE CONTROL: huggyllama/llama-7b (installed, expect byte-identical) +
  full 25-fixture corpus zero-drift.
- warning honesty: unstamped dict w/ auto_map (unit test) → true cause surfaces.

**Gates:** new unit tests green · full targeted suite green · corpus
check_regression zero-drift · before/after galleries rendered + pixel-diffed
(before = run_77 folders; after = previews/unit1_source_parity_2026-07-11/) ·
Soumil reviews images before any bless.

**STATUS 2026-07-11: EDITS LANDED + WITNESSED.**
- Edits: parser.py `_load_raw_config_json` `_repo_id` stamp · sources.py
  class-address fallthrough for unknown model_type · `_hub_bundle` cache-first
  · fileless-hub warning merge (+`dataclasses` import) · sable.py stamp DELETED
  (loader owns the address; harness ≡ ship context).
- Tests: 4 new unit tests green (stamp / class-fallthrough / warning-unmasked /
  existing hub-rail regression); **FULL SUITE 550/550 GREEN (25m31s).**
- Corpus: **25/25 check_regression ZERO-DRIFT.**
- Witnesses (galleries in previews/unit1_source_parity_2026-07-11/):
  internlm-7b 0→3 source files, attention drill now draws "apply RoPE Q/K"
  (before: run_77/internlm__internlm-7b — positionless); MobileLLM-1B same fix
  + resolved OFFLINE from cache; Kimi-K2-Base resolves installed
  modeling_deepseek_v3.py via declared class; llama-7b negative control
  **byte-identical** on all 4 images vs run_77.
- Soumil: review the before/after pairs; bless decisions after his eye pass.

---

## UNIT 2 — KILL THE DEFAULT LAYER (tri-state everywhere + code readers)  ⏳ ACTIVE
*(Spec frozen 2026-07-11 from 4 recon maps in scratchpad/u2_recon/ —
attention_families.md · ffn_norm_families.md · provenance_machinery.md ·
blast_radius.md. Every site below is recon-verified file:line.)*

**Thesis (from CONFIG_ABLATION_CENSUS.md):** position is the only tri-state
fact family; ~10 families assert defaults at zero evidence. U2 gives every
family position's discipline AND a code channel wherever one is buildable.
**Acceptance metric: a no-evidence parse contains ZERO asserted structural
facts** (census harness = the permanent net).

**Reference implementations (the house patterns being generalized):**
render = position (typed unknown parser.py:1783 → pale chip labels.py:400 →
banner sections.py:42-49 → `unresolved` conformance tier); serializer =
scores_scale (emit-only-when-evidenced across all three projections);
honest-unknown primitive already exists on the tower path
(text_encoder.py:104) — PORT, don't invent.

**Recon's two hard discoveries (beyond the census):**
1. `tie_word_embeddings` is WRONG-VALUED today, not just untagged: gpt2 / t5 /
   bert / bloom / falcon omit the flag → we assert untied while the HF config
   CLASS default is tied. The class_default provenance tier is real evidence
   (hydration channel) — this fixes live wrong facts.
2. `param_count` silently miscounts on unknowns (gated=None → −33% FFN;
   tie=None → +38M phantom head) — the unknown tier needs an explicit
   param-count policy (consume class_default values with provenance; annotate
   estimates; never let unknown silently pick a branch).

**PHASES (each gated: new tests · 550-suite · 25/25 zero-drift · pixel review):**

**P0 — FactLedger foundation (pure bookkeeping, zero behavior change).**
`evidence/context.py` (ParseContext — call-local, ship≡sable since U1) gains
a FactLedger: `set_fact(owner, name, value, status, source)` with enum
{code_proven, config_declared, class_default, derived, ambiguous,
oracle_missing, asserted, unknown}. Unify the scattered provenance bools
(activation_from_class / activation_assumed / scores_scaled / projection_mode
/ the binary asserted tuple ir.py:90/125 born parser.py:1045-1123). debug.py
note_access gains intent (inspected/consumed); RenderEvent
(render_context.py:10-25) gains `facts_projected` (the nested-conformance
precedent — no new channel). Ledger serialized to ir.extras["fact_provenance"].

**P1 — default-kill, in the recon-verified cascade order
{activation, norm_kind} → gated → {ffn_storage, projection, labels}:**
replace the `or`-collapses at adapters/transformer/parser.py:655 (activation)
/ :679 (norm_placement `or "pre"`) / :1752 (`return norm_kind=="rmsnorm"` —
the gated heuristic terminal) / :791 (bias None→False) / :1174 (tie bare
bool) / ir.py:29 (mask="causal" dataclass default) with typed unknown +
ledger tags. Port the pale/banner primitive to decoder cells. Kill the
SECOND default layer: renderer `.get(..., "causal"/True)` fallbacks
(blast_radius.md §5 list). param_count policy per discovery #2. Blast radius
is CONTAINED on the corpus (all 25 evidence-rich; only mask tags +
attn_projection soft-move — pixel-review those pairs).

**P2 — the missing code readers ("code ways"), cheapest-first:**
- P2a bias: reader EXISTS (decoder_attention_bias_from_files,
  patterns.py:2337 + _linear_bias_value :2245) — wire it at parser.py:791.
- P2b tie: NEW reader — `lm_head.weight = embed.weight` / get_output_
  embeddings tying idiom via the ast.Assign precedent (transitive.py:298) +
  the class_default tier. Fixes the live wrong values.
- P2c activation: extend patterns.py:1343 (name-match-only) to the
  `ACT2FN[config.hidden_act]` dispatch idiom — code proves THAT an activation
  applies + names the config field; config supplies WHICH (doctrine-exact).
- P2d mask: the one genuinely new reader — `is_causal=` kwargs / causal-mask
  construction / `if not self.is_decoder` branches; config channel
  `is_decoder=False` + is_encoder_decoder as config_declared bidirectional
  (the BERT/T5-encoder fix, PM-5). Each reader ships the 4-counterexample
  test set (same source ≠ config value / same arch ≠ family / similar config
  ≠ source / source absent).

**P3 — config-declared channel completion:** position gains the config
fallback (PM-4 R4: rope_theta/rope_scaling as config_declared when code is
oracle_missing/ambiguous — θ chip + provenance, never silent); un-ignore
is_gated_act/feed_forward_proj as declared FFN evidence (scoped-ignore
dependency noted for U7); tie flag + class default.

**P4 — the two nets (advisory → blocking after corpus-clean, the
config_field_audit staging precedent):** net #13 projection-audit (every
ledger `consumed` structural fact needs a drawn witness in
RenderEvent.facts_projected — kills the granite-multiplier class forever);
net #14 census (the D-quadrant harness from CONFIG_ABLATION_CENSUS appendix:
zero `asserted` entries on the witness set). config_field_audit gains the
accessed-but-unprojected finding class.

**WITNESS MATRIX (8, from blast_radius.md §3):** bert-base (mask/bidir +
post-norm + tie), t5-base (enc-dec + gate + tie), gpt2 (tie + Conv1D bias),
falcon-7b (parallel + tie), phi-2 (dense control), code-blocked llama
(synthetic zero-evidence), llama-7b (byte-stable control), Qwen3 ± sliding
(window). Before/after galleries per phase under
previews/u2_default_kill_<date>/.

**STATUS: P0 ✅ · P1 ✅ GATED (baseline 554 → 564 green; 25/25 CLEAN ×3;
llama-7b byte-identical throughout; witness galleries
previews/u2_default_kill_2026-07-11/ — bert mask→unknown, t5 dense-relu
SOLID via config_declared, gpt2 tie=True→correct 124M, zero-evidence
synthetic = pale wiring + ZERO asserted beyond doctrine-kept) ·
P2-P4 confirmed by Soumil ("bleed it"), builders in flight.**

**P4 ✅ GATED (2026-07-12):** facts_projected channel live (record_graph/
render_graph kwargs + note_facts_projected + NEW fact_projection.py single
source of truth; attention/FFN/MoE-expert drills + architecture event
declare witnesses; keys derived from live parses). **Net #13
projection-audit: 0 findings on 25/25 → flipped BLOCKING** (100% witness
coverage incl. gpt-oss FFN via the expert drill). **Net #14 zero-asserted
census: BLOCKING day one — every parseable fixture asserts only the
doctrine-allowed {projection_mode, scores_scale, ffn_storage} at zero
evidence** (P1's default-kill confirmed by the net). config audit gained
accessed-but-unprojected (advisory; inert until the consumed rail
populates). 20 new tests; 25/25 zero-drift with both nets blocking;
end-to-end sable clean on llama/gpt-oss/flux/sd3.5. Design note: chips are
HTML not RenderEvents — chip-drawn facts attributed to the nearest reliable
graph/architecture event (full corpus coverage at key granularity).

**P1 disclosed deviations (both justified, both tagged in ledger):**
(a) norm_placement source-present-but-reader-abstained keeps conventional
pre (recorded ambiguous+asserted, banner disclosed) — the blocking
conformance oracle checks residual/norm ops against readable forward(), so
paling them broke MusicGen's code-proven accounting; source-ABSENT is fully
pale as specced. (b) ⚠ **ARBITRATION FOR SOUMIL — the silu tier:** strict
kill of the silu gate-family convention pales 3 blessed MoE expert drills
(deepseek-v3 / glm-4.5 / gpt-oss). P1 reinstated silu as the WEAKEST
`derived` ledger tier (code/config outrank; never marked fact). Strict
honesty instead = a deliberate 3-fixture re-bless with eyes on pale expert
drills. Flagged at `_is_gated` + both tests. Also noted: P1 and the P2/P3
stream collided mid-flight and were reconciled (P2/P3's _config_gated
channel kept — it landed P3b early).

### P2 — the four code readers (file-exact; build serially, one family gated at a time)
**P2a bias wire-up** — `adapters/transformer/parser.py` (the P1 bias tri-state
site): insert the CODE channel — `decoder_attention_bias_from_files`
(EXISTS, evidence/patterns.py:2337 + `_linear_bias_value` :2245) — order
config_declared → code_proven → unknown; ledger source = reader name.
`everchanging/transformer/aliases.yaml`: add the missing `bias`/`qkv_bias`
alias rows (internlm spelling). Tests: tmp-file ctor witnesses both values +
zero-evidence stays None. (Conv1D-declines nuance stays; Conv1D role fix is U7.)
**P2b tie reader** — NEW `lm_head_tying_from_files` in evidence/patterns.py:
signal = the UNCONDITIONAL manual assignment idiom
(`self.lm_head.weight = …embed…weight`, ast.Assign precedent
transitive.py:298). ⚠ SUBTLETY: transformers' `_tied_weights_keys` is
tie-CAPABILITY (tying actually happens in post_init gated on the config
flag) — the reader must treat it as capability only, NEVER proof, or we
fabricate tying on every llama. Wire at the P1 tie site: config_declared →
code_proven → class_default → unknown. Tests incl. the capability-≠-fact
negative.
**P2c ACT2FN dispatch** — extend evidence/patterns.py:1343 (name-match-only)
to the `ACT2FN[config.<field>]` / `get_activation(config.<field>)` idiom →
returns the deciding field name; parser records the activation as
**code_and_config** (NEW enum member in evidence/context.py FACT_STATUSES —
the endorsed envelope has it): code proves an activation applies + names the
field, config supplies which. Wire at the P1 activation site.
**P2d mask reader (the new build)** — NEW `attention_causality_from_files`
in evidence/patterns.py: signals (a) `self.is_causal = True` class attr /
`is_causal=` kwargs in sdpa-family calls, (b) causal-mask machinery
(`_update_causal_mask`/AttentionMaskConverter), (c) bidirectional evidence:
encoder-style mask path + `is_decoder` branch absence (the BERT fix at the
FACT level). Tri-state causal/bidirectional/None; wire ABOVE the P1
config-decoderness channel (code_proven → config_declared → unknown).
INCLUDES the labels.py:20-28 `_MASK_SHORT` `bidirectional` row (pulled
forward from U7 — without it the fixed fact still LABELS "causal").

### P3 — config-declared channel completion
**P3a position config fallback** — `adapters/transformer/parser.py:680-694`:
when code status ∈ {oracle_missing, ambiguous} AND config declares rope
(`rope_theta`/`rope_scaling`/rotary spellings → NEW alias rows in
everchanging/transformer/aliases.yaml), assert RoPE as config_declared with
a "θ=… (config-declared)" chip (labels.py) replacing the honest-unknown
banner. Witnesses: MiMo-7B (ambiguous residual — the cross-package
superclass case), OLMo/DCLM-shape (oracle missing), negative: no-rope
configs stay unknown.
**P3b un-ignore the FFN declarations** — remove `is_gated_act` /
`feed_forward_proj` from everchanging/transformer/ignored_fields.yaml
(:162-168) and read them as config_declared in the P1 gated/activation
decisions (order: code_proven → config_declared → class_default → unknown).
Fixes standalone-T5 gate truth (t5-base dense-relu, flan gated-gelu) at the
FACT level now; the enc-dec topology collapse itself remains U4 — stated
honestly. (U7's scoped-ignore design unaffected: we REMOVE rows, adding
scoping later stays open.)
**P3c** tie config flag — landed in P1; verification only.

### P4 — the two permanent nets (+ audit upgrade)
**Net #13 projection-audit** — renderers declare `RenderEvent.facts_projected`
at the card/chip emission sites (v1 scope: attention detail + FFN detail +
layer/norm cards + the position/scores/mask/tie/bias chips — the recon
serializer list; files: renderers/html/blocks/attention.py, ffn detail
builder, labels.py chip emitters). sable.py registers net #13: every ledger
fact with status ∈ {code_proven, config_declared, class_default,
code_and_config} on a DRAWABLE family must appear in ∪facts_projected.
Staged ADVISORY → flip blocking at 25/25 clean (config_field_audit
precedent).
**Net #14 census (zero-asserted)** — sable.py: strip the model's own config
to numbers+address (census appendix recipe), re-parse with an EMPTY
SourceBundle ParseContext (constructed directly — no monkeypatch), assert
`ledger.asserted()` ⊆ the doctrine-allowed set (v1: scores_scale/
ffn_storage). BLOCKING from day one (post-P1 the families are clean).
**config_field_audit upgrade** — sable.py consumes debug.consumed_fields():
structural fields touched-but-never-consumed become an
"accessed-but-unprojected" finding class (advisory v1) — the granite-
multiplier detector.

### FINAL SWEEP (the scorecard Soumil asked for)
8-witness galleries → previews/u2_final_<date>/ (bert, t5-base, gpt2,
falcon-7b, phi-2, zero-evidence synthetic, llama-7b byte-stable control,
Qwen3±sliding) diffed against run_77 + u2_default_kill; census harness
re-run → CONFIG_ABLATION_CENSUS.md updated with the post-U2 D-quadrant
table; full suite + 25/25; the how-well-did-we-did report with the
asserted-count metric (~10 families → allowed-set only).

**SEQUENCING/CONFLICT NOTES:** P2a-d and P3 all touch
adapters/transformer/parser.py + patterns.py → build SERIALLY (one builder,
family-gated); P4 touches sable.py + renderers (disjoint, may parallel P3);
P1's full-suite green is the precondition for any P2 edit. PR #14 rebase
risk unchanged (parser.py/sources.py).
