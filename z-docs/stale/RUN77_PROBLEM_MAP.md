# RUN77 PROBLEM MAP — the verified understanding (2026-07-11)
*(Deep-verification pass over the run_77 campaign's findings: 7 parallel agents,
one per mechanism cluster, each walking OUR code (file:line, quoted), the HF
source truth (mechanism + its purpose), expected-vs-actual output, why the nets
passed, and the generality gate. The campaign reports were treated as LEADS,
not truth — several proposed mechanisms were refuted and re-attributed below.
Full blocks live in `unfold-pkg/previews/run_77/_problem_map/01..07_*.md`.
This header is the synthesis; the sections are the evidence.)*

## THE VERDICT
**41 verified findings · 41 GENERAL · 0 MODEL-SPECIFIC · 0 problems refuted.**
Every symptom the campaign reported was real. What the deep pass corrected was
*root-cause attribution* — 8 proposed mechanisms were wrong or stale, and the
fixes would have gone to the wrong sites (ledger below). Under Soumil's gate
("act only if the difference is not model-specific"), **everything on this map
is actionable**, and every fix extends an existing rail or vocabulary — no
per-model rows anywhere.

## THE FIVE STRUCTURAL ROOTS (every finding reduces to one of these)

**ROOT A — accessed ≠ projected** *(PM-2; the ⭐B1 family)*
`_resolve` (parser.py:94-106) marks a config field "handled" via
`debug.note_access` even when its return value is DISCARDED — so
config_field_audit's silence is itself the bug. Reproduced: granite
embedding/logits multipliers, Cohere `logit_scale=0.0625` (accessed,
unread=[], absent from HTML); rope dialect parsed into `extras["rope"]` with
no renderer consumer (Llama-3.1 SVG **byte-identical** to Llama-2 while the
header claims 131k ctx); value-blind `self.scaling` presence check
(patterns.py:2678 — gemma-3n `scaling=1.0` drawn √dim; gpt-neo same via no-op
`softmax_scale=1.0`). gpt2 `scale_attn_by_inverse_layer_idx` ADJUDICATED
STRUCTURAL (denominator becomes √dim·(layer+1)) — must not be ignore-listed.
**Cure (one mechanism):** the B5 provenance enum + a PROJECTION-AUDIT net —
an accessed field/code verdict must have a *drawn-element witness*, not merely
an `ir.extras` key.

**ROOT B — declared config evidence refused** *(PM-1 + PM-4)*
Two shapes: (1) the per-layer schedule walk (parser.py:939) reads layer-type
schedules through exactly TWO channels (`layer_types` + MoE-schedule fields);
6 spellings have zero readers → silent uniform stacks. Smoking gun: the SAME
MiniMax arch draws heterogeneous via installed config class, uniform via raw
remote config. Broken spellings: `block_configs` (Nemotron NAS — **98**/162
attention-free, campaign said 92), `attn_type_list` (MiniMax), `_block_types`
(recurrentgemma RG-LRU), `dense_attention_every_n_layers` (Phi-3-small),
`moe_layers_enum` (step3), `attention_types` (gpt-neo). 12 spellings already
work — this is vocabulary widening on the Group-1 engine, not a new engine.
(2) config `rope_theta`/positional facts are UNIFORMLY refused when the code
oracle is missing (parser.py:680-694 — the config channel is never consulted
in any non-proven branch): a straight Law-1a violation; config IS declared
evidence. Honest-unknown markers are path-dependent (some models get the chip,
some get silence).

**ROOT C — identity lost on the way to the source** *(PM-4)*
`_load_raw_config_json` (parser.py:411-436) never stamps `_repo_id` →
`_model_id` None (sources.py:665) → `_hub_bundle` bails (:609-610) → every
raw-JSON remote-code model is evidence-blind. Live proof: internlm-7b flips
0→3 source files, unknown→proven/{rope}, the instant the id is stamped. The
stamp EXISTS on the diffusor loader and in sable.py:216-217 — which means
**the harness audits a source-resolved parse while the shipped gallery was
drawn blind** (the audit-vs-ship gap; explains batch-5's contradictory
oracle=present rows). Corollaries: cached snapshot never read before network;
fetch-failure warning masked (sources.py:113-123); arch-class fallback gated
behind `if not model_type` (sources.py:183/:189 — Kimi-K2 blocked from the
INSTALLED deepseek_v3 source); modality recognition keyed only on
`vision_config`/`audio_config` schemas (parser.py:918 + registry.py:72 —
Phi-4-MM/Phi-3.5-V/Molmo drawn text-only over config-DECLARED towers);
composite wrappers inherit sub-config facts they never enact (PaliGemma2
softcap over a bare nn.Linear head — wrapper facts must come from the
wrapper's own construction).

**ROOT D — branch facts from config fields where the forward AST is the truth**
*(PM-3 + PM-5)*
Router: renorm node gated on `norm_topk_prob` (moe_router.py:34) while
Mixtral's `/= sum()` is unconditional; softmax-after-topk has no branch
(:56-64) though parser.py:1612-1616 already computes the axis; granitemoe
card prose asserts Mixtral order. Shared experts: renderer fully built and
working for 3 spellings (Ling/dots/DeepSeek); Llama-4 (code-only declaration,
config says 0 — config CONTRADICTS code), Hunyuan (`num_shared_expert` +
`use_mixed_mlp_moe`), step3 (`share_expert_dim`), Qwen1.5/ERNIE (alias
misses) all drop the lane — and the ONE net that could catch it
(validate.py:59-60) is gated OFF exactly when a model has multiple layer
variants (:45-48), which is precisely when MoEs hit it. Llama-4 dense-FFN
width: `intermediate_size_mlp` 0 refs → dense layers drawn at expert width
(8192 vs 16384). step3 MFA: parser.py:558 `or num_heads` silently defaults
KV heads → standard-MHA card fabricated over factorized attention (honest
answer per tri-state: honest-unknown, never a guessed standard shape). CLA:
the cross-layer-KV axis EXISTS (CrossLayerEdge kv_share + detector + net) —
Hunyuan's alternating pattern extends it (campaign's "no axis exists"
refuted). T5 FFN gate: the code reader honestly ABSTAINS; the fabrication is
the abstain→`rmsnorm⇒gated` heuristic fall-through (parser.py:1752) + the
ignore-listed `is_gated_act` — while the encoder_panel path reads the same
field correctly (the Mochi natural experiment). Enc-dec collapse: the
seq2seq gate tests `encoder_layers`/`decoder_layers` spellings T5 lacks
(parser.py:629-631, reproduced `warnings == []` on codet5/mt5/switch/byt5);
BERT: `forced_bidirectional` never derived from `is_decoder=False`
(parser.py:815), post-norm invisible inside delegated BertSelfOutput
(:679 `or "pre"` default asserted), `type_vocab_size` concept absent.

**ROOT E — hand-authored templates on the diffusor rail** *(PM-6)*
The UNet/scheduler/VAE skeletons are presence-keyed templates, not
code-derived: anything that isn't UNet2DCondition+AutoencoderKL+first-order-
epsilon-Euler is DRAWN AS that trio. Kandinsky-2-2's image-embed conditioning
(config-declared `encoder_hid_dim_type="image_proj"`) gets a fabricated
"Text prompt → encoder" tower (blocks.py:945-961 fires when encoders==[]);
kandinsky-3 gets a fabricated mid block + SD ResNet internals over its AdaGN
[1,3,3,1] bottleneck + wrong skip count (and its honest WARNING promises
cross-attention the diagram never draws); SVD/Wan temporal structures vanish
(2D-only builder); the scheduler drill draws first-order epsilon-Euler for
DPM++/UniPC/LCM alike + ⊕ on a subtraction (T6); ε̂ hardcoded over
velocity/flow heroes whose own drills say v̂; patch-merger token counts skip
÷merge² (12,100 vs 3,025) and AutoConfig class-defaults masquerade as
geometry (576 vs 729); VAE conv_in+mid dropped on a config-silent gate
(blocks.py:593) though diffusers builds them unconditionally. Nets stay green
because hand-authored builders have NO forward() closure to diff against —
the whole rail sits outside the code-evidence machinery (Part 11 B3, now with
its full casualty list).

**Plus the cross-cutting net/corpus truth** *(PM-1's reframe — changes the plan)*:
config_field_audit ALREADY flags nearly every Root-B field — the signal
exists today. These shipped silently because (a) none of the frontier models
are among the 25 blessed fixtures, so the blocking gate never ran on them,
and (b) remote-code models are oracle-missing-exempt for the conformance
nets. **The fix campaign needs a corpus leg (bless the frontier witnesses),
not just parser legs.** Related net asymmetries: op_conformance filters
"unresolved" out of blocking (sable.py:262) while nested_conformance doesn't
(:282); Mllama's drill resolver can't bind one merged view to two divergent
closures (the scanner itself resolves both classes fine — campaign mechanism
refuted); the ignore vocabulary has NO scoping (flat global frozenset,
debug.py:41) so diffusion-scoped rows blind the LLM audit — scoped ignore
rows are the fix, not row deletion.

## THE REFUTATION LEDGER (campaign mechanisms corrected — fixes redirected)
1. T5 gate: NOT patterns.py:1291 branch-resolution — reader abstains; the
   heuristic fall-through + ignore rows are the target (PM-5, runtime-proven).
2. Conv1D: the "one-row YAML fix" is a NON-FIX — `_role_of` case-folds, so
   the row can't be deleted without breaking audio `nn.Conv1d`; needs
   evidence-based class resolution (import origin / addmm forward). And
   op_tokens.yaml:37 is innocent (PM-7).
3. Mllama: AST scanner follows the conditional ModuleList fine — the gap is
   the merged-view↔two-closures drill binding (PM-4).
4. CLA: the kv_share axis exists — extend, don't invent (PM-3).
5. `attention_score_scaling_from_files` "never on main paths" is STALE — B1's
   boolean half landed (parser.py:796, diffusor :868+); the VALUE-blind half
   remains (PM-2).
6. Nemotron-Ultra: 98/162 attention-free, not 92 (PM-1, measured).
7. llama3 rope sub-fields are LOUD (blocking audit finding) — the silent part
   is the dialect/context-stretch never drawn (PM-2).
8. Kimi conformance-red is THREE roots (resolver-map gap / model_type-unknown
   / genuine oracle-missing), not one (PM-7).
9. PROJECT_CONTEXT Group-1 recon ("schedules work, only hardening remains")
   is false for the out-of-corpus/remote frontier (PM-1).

## WHAT TO DO, AT THE HIGHEST LEVEL (the shape the evidence forces)
1. **R-C first (identity/stamp)** — one-line class of fix, ~15 models unblind,
   and it closes the audit-vs-ship gap so every later fix is measured honestly.
2. **R-A net (projection audit + provenance enum)** — makes the entire
   read-never-drawn class structurally impossible; catches future regressions
   of every other root.
3. **R-B vocabulary widening** — the 6 schedule spellings into the Group-1
   engine (+ honest-unknown fallback when config declares but code is absent);
   the config-positional channel admitted as declared evidence with tri-state
   marking.
4. **R-D forward-AST derivations** — router order/renorm; shared-expert from
   construction (detector exists, renderer exists — wire them); the abstain≠
   assert rule for every heuristic fall-through (T5 gate, step3 KV default,
   BERT defaults, enc-dec gate spellings).
5. **R-E the diffusor code-evidence rail** (B3, biggest single surface) —
   conditioning from declared pipeline components; block internals from the
   denoiser's own classes; scheduler drill per scheduler class; temporal axis.
6. **The corpus leg** — bless frontier witnesses (one per root) so the
   already-blocking nets finally gate these paths; fix the net asymmetries
   (unresolved-tier unification, scoped ignore rows, shared-experts gate).
7. **The label/guard batch** (PM-7's verified one-liners that survived
   scrutiny: window≥context, _MASK_SHORT, SW· prefix, ReLU², 404 typed-
   exception order, layout hints, port-caption format, bias chip).

*Sections with full evidence blocks follow the same numbering:
`_problem_map/01` schedules · `02` read-never-drawn · `03` router/MoE ·
`04` remote-code/modality · `05` enc-dec · `06` diffusor · `07` vocab/guards.*
