# U7 conditional ordinary/shared FFN proof

## Decision

DeepSeek-V3 and GLM-4.5 must retain their detailed split-gated FFN view. Their
temporary generic `Feed-forward` view was not an honest limitation of the
Hugging Face implementation; it was an ownership limitation in our reader.

This is a bounded U7 proof slice. It does **not** mark all of U7 complete.

## Exact source facts

The installed Hugging Face implementations independently expose all required
mechanism evidence:

- `DeepseekV3MLP` and `Glm4MoeMLP` construct distinct `gate_proj`, `up_proj`
  and `down_proj` affine projections;
- their forwards compute
  `down_proj(activation(gate_proj(x)) * up_proj(x))`;
- each decoder layer exhaustively assigns its one `mlp` field to either the
  ordinary MLP or the MoE wrapper;
- each MoE wrapper constructs an MLP child and unconditionally invokes that
  child in its forward; the child's output reaches the wrapper return;
- the ordinary and invoked shared paths therefore independently prove the
  same `gated=True`, `projection_mode=split`, config-dispatched activation;
- routed expert storage remains separately proven as `fused_gate_up`.

GPT-OSS is the permanent routed-only counterexample: it has no ordinary/shared
MLP lane and must remain unknown on the ordinary/shared FFN fact.

## Root cause

`ComponentOwner` correctly recorded the decoder's two guarded `self.mlp`
constructions as rival owner sites and refused to pick one. The old FFN reader
then treated that typed rivalry as no usable evidence. It could classify a
single graph child but could not evaluate every exact construction alternative.

The error was therefore:

> exact rival address evidence was preserved, but the mechanism reader had no
> unanimous-alternatives proof rule.

It was not missing source specificity, a renderer limitation, or a reason to
add a model-family branch.

## Implemented proof law

The FFN reader may resolve an exhaustive conditional field only when all of the
following hold:

1. one exact unguarded block call invokes the rival field;
2. all writes to that field are exact construction sites under one proven
   `if`/`elif`/`else` decision;
3. every site resolves to exactly one class candidate;
4. each candidate is inspected in an isolated owner graph tied to its exact
   construction site—no rival is fabricated as a child in the main graph;
5. a direct candidate independently proves its projection/activation dataflow,
   or one exact unguarded nested child both proves that dataflow and reaches the
   wrapper return;
6. every alternative proves the identical tuple:
   `(gated, projection_mode, activation, activation_config_path)`.

Missing, dynamic, guarded-use, non-exhaustive, multi-candidate, unused-child,
multi-FFN or disagreeing alternatives remain failed/ambiguous. No majority,
first-hit, class name, field name, model type or family identity is permitted.

## Downstream closure

Two consumers previously assumed an FFN mechanism always had one owner:

- projection bias now evaluates every exact variant and resolves only on a
  unanimous boolean;
- parallel-normalization evidence treats the one block call as a neutral
  observed node whose candidate implementations are mechanism-equivalent.

The parser continues to consume the same typed FFN result. No renderer branch
or model exception was added.

The proof also reduces the asserted-convention population from 599 to 593.
Those six rows were the initial dense layers of DeepSeek-V3 and GLM-4.5. With
their storage now source-proven, the obsolete `ffn_storage` fact definition,
structural-debt row, parser assertion and zero-evidence allowlist entry are
deleted together. The internal functions that analyze unresolved storage remain
because `projection_mode=None` is still a lawful unknown for other mechanisms.

## Permanent controls

The focused matrix pins:

- dense, split-gated and fused-gate-up ordinary FFNs;
- unanimous direct-versus-invoked-shared alternatives;
- disagreement between alternatives;
- a constructed but uninvoked shared-looking child;
- non-exhaustive construction guards;
- a guarded block invocation;
- a direct FFN beside a second rival invoked field;
- forged semantic consensus and forged unguarded entry evidence;
- cross-variant projection-bias disagreement;
- real DeepSeek-V3 and GLM-4.5 positives;
- real GPT-OSS routed-only negative;
- BLOOM, Llama, StableLM, Qwen2-VL and MusicGen preservation controls.

## Intended output delta

- DeepSeek-V3 and GLM-4.5 restore the already-blessed detailed
  gate/up/activation/multiply/down FFN view.
- Their Sable view hashes return to the existing fixture hashes; no gallery or
  Sable fixture re-bless is required.
- Their IR, expanded, parameter, HTML-metadata, evidence-ledger and Sable
  preservation surfaces move from the accidental temporary generic state back
  to code-proven split-gated semantics.
- GPT-OSS and ordinary dense/gated controls do not change.

## Remaining U7 boundary

This slice closes conditional ordinary/shared FFN mechanism equivalence only.
It does not complete the full U7 program: canonical shared-expert child
projection at every owner altitude, remaining norm/cell topology, Conv-GLU,
nested modality/diffusion consumers, default deletion, and all reverse
fabrication/parameter receipts remain governed by the master plan.

## Canonical embedded-encoder cutover (current U7 slice)

The next U7 slice removes the second FFN authority that survived in embedded
text-encoder panels.  The transformer parser's owner-qualified
`decoder_ffn_mechanism_for_path` result now authors the canonical `FFNSpec`, and
the encoder panel plus nested sub-model projections consume that exact spec.
The former `evidence/ffn.py` whole-file scan and
`decoder_ffn_activation_from_files` path are deleted rather than retained as a
fallback.

Two general substrate corrections were required; neither is model-specific:

1. A copied config is not all-or-nothing evidence.  Exact, unshadowed
   `copy.deepcopy(config)` preserves each unchanged path, while an assignment
   invalidates only that path and its descendants.  Escaped aliases, opaque
   mutation calls, local `copy` shadowing and ambiguous flows remain unproved.
   All config-path consumers use the same `ConfigBinding.resolved_path()` law.
2. FFN projection topology and activation identity are independent.  An exact
   two/three-projection graph may remain dense/split/fused when an intervening
   transform is proved on the data path but its activation kind is opaque.
   Unrelated activations, post-output activations and guarded activation bypasses
   still reject the mechanism.  Where source names an exact config dispatch
   path, the parser consumes that exact spelling; an equal alias cannot replace
   the occurrence the code actually reads.

The real-model review covers UMT5, T5, Gemma-2, Llama, Mistral, Qwen2/Qwen3-VL,
CLIP and CLIP-with-projection encoders across AuraFlow, HunyuanVideo, SD3.5,
SDXL, Sana, Lumina, PixArt, CogVideoX, LTX, Mochi, PRX, Qwen-Image, FLUX.2,
Wan and MusicGen.  The intended outcomes include:

- exact split-gated GELU/SiLU paths where source and config dispatch join;
- exact dense CLIP topology with an opaque activation when an inherited factory
  prevents exact constructor-config forwarding proof;
- an unresolved MusicGen conditioning FFN because `T5Model` exposes rival
  encoder/decoder stages and no lawful owner selection exists yet;
- no pixel change across the diffusion/embedded-encoder galleries except the
  Gemma-2 FFN drill: its source-selected `hidden_activation` value now renders
  `GELU` instead of the former generic `Activation` label.  The other five
  Gemma-2 views and every other reviewed PNG remain byte-identical.  This is an
  evidence-driven label correction, not a renderer redesign.

The pre-existing MusicGen top-level architecture label overflow is explicitly
not blessed as a new U7 behavior: it is byte-identical to the prior gallery and
remains separate visual debt.

## Exact intermediate-width cutover (current U7 slice)

The whole-file `decoder_intermediate_size_from_files` reader is deleted.  Its
replacement starts from the exact decoder-block occurrence and the canonical
FFN mechanism, then evaluates the input expression of that mechanism's proven
output projection through the exact constructor occurrence chain.  Every
config operand is retained as an exact premise and consumed under
`decoder.ffn.intermediate_size`; a missing condition operand is not treated as
an explicit `None`.  Operand origin is not flattened: checkpoint operands yield
`code_and_config`, while any installed-class operand keeps the fact at the
weaker `class_default` tier.  A directly supplied runtime `PretrainedConfig`
has no checkpoint-origin claim, but its exact current value may still certify
the runtime geometry as `code_and_config`; its access event remains explicitly
unestablished, never relabelled checkpoint-declared.  Loader-only metadata
cannot certify the geometry.

The same rule covers GPT-J/CodeGen/GPT-2 `n_inner=None` defaults, BLOOM's inline
`4 * hidden_size`, ordinary declared-width FFNs and Transformers' package-
relative `Conv1D` import.  `Conv1D` is recognized only through the exact
`...pytorch_utils.Conv1D` protocol binding; an unrelated class with the same
final name remains powerless.  The reader also requires every upstream
projection output to equal the down projection input, so inconsistent gated
lanes abstain instead of letting the down projection certify itself.

### Width-slice receipt

The final commit is `039f66c`.  Its detached-worktree coordinator receipt is
`/private/tmp/model-unfolder-verification/9c1a39dbaa`: 2,584 tests collected,
275 focused tests passed, 44 U2 authority tests passed, all 48 preservation
checks passed, and the exhaustive partition passed 2,477 tests with 13 skips
and two expected xfails.  Every lane's complete source-tree and ignored
blessed-artifact fingerprint was identical before and after.  The sanctioned
manifest generator changed exactly BLOOM's ledger hash; structural, expanded,
parameter, HTML, Sable, gallery and all per-view hashes stayed byte-identical.

## Exact cell topology and split-expert storage slice

The next bounded U7 slice replaces the whole-file cell-topology scan with
`decoder_cell_topology_for_path`.  It starts at the exact repeated decoder-block
occurrence and shares one attention/FFN invocation census with the parallel-norm
reader.  It resolves only from live positive residual equations:

- sequential means the attention merge reaches the FFN input and a second live
  merge reaches the final return;
- parallel means the exact attention and FFN calls consume one proven input and
  one live merge contains both contributions;
- pre/post/double derives independently from the exact norm boundary on both
  branches;
- a residual merge inside an exact addressed child is accepted only when its
  computed and residual terms both reach that child's return.

Guarded alternatives are never unioned.  An exact source-bound selector may
choose them only after its constructor assignment identifies one exact config
path.  StableLM therefore cites `use_parallel_residual`; Qwen3.5 cites
`layer_types` for both norm and residual facts.  Unknown selector values,
dynamic config subscripts, dead equations, opaque transforms, unrelated sibling
classes and incomplete path domains all abstain.

The same slice extends routed-expert storage without an identity branch.  A
split result requires three exact repeated Parameter fields, their exact
selection in the parent expert loop, exact binding to child formals, two input
matmuls, a live gate/up product, the exact down matmul consuming that product,
and a live return.  A dead side-product cannot certify an unrelated down path.
DBRX proves this split form; the existing DeepSeek-V3, GLM-4.5 and GPT-OSS
controls retain fused gate-up storage.

DBRX's old diagram was not merely generic: the whole-file scan had selected a
false parallel shell and the expert drill was opaque.  Its reviewed replacement
shows the real nested sequence—norm, attention, residual add, norm, routed FFN,
residual add—and the exact split expert gate/up/product/down path.  Only DBRX's
architecture and expert PNGs change; the other DBRX views and every unaffected
witness remain byte-identical.  Evidence-ledger hashes change corpus-wide
because the fact source moves from the deleted legacy reader to the exact
owner-qualified reader, not because those diagrams change.

This slice retires one quarantined semantic reader (23 to 22) and one broad
transformer-parser exception baseline (13 to 12).  U7 remains open: the
quarantine still contains `decoder_router_evidence_from_files` and
Conv-GLU/cross-altitude terminal closure still require explicit proof or a
documented non-applicability decision.
Falcon is an explicit topology carry-forward, not a hidden regression: its
parallel cell is distributed across nested `self.config` boolean guards and an
augmented assignment.  The removed whole-file scan happened to classify this
fixture correctly but did not prove that exact path.  Until an owner-qualified
control-path evaluator binds those conditions, Falcon retains its independently
proved MQA details and draws one `wiring unresolved` cell instead of accepting
`parallel_attn` as architecture by declaration alone.

The following bounded cleanup deletes `layer_class_count_from_files` rather
than replacing it.  That reader counted layer-looking classes across an entire
source bundle and used the count to decide whether owner-unqualified validation
findings should be emitted.  It could therefore let an arbitrary sibling class
silence every warning, while a single-class file could assign a feature found
in one owner to another parsed owner.  A class count cannot repair that missing
join.  The owner-unqualified MoE, KV-sharing, softcap, partial-RoPE, NoPE,
fine-grained-routing, shared-expert, PLE, AltUp and double-FFN-norm comparisons
are removed with it.  Independent MLA, ALiBi and MTP checks are intentionally
left for their own owner-qualification audit; this slice does not broaden its
scope merely because they share the validation file.  A permanent negative
control proves a whole-file PLE signal cannot be assigned to an unqualified
Llama owner, while the existing MLA positive remains green.  The quarantine
shrinks 22 to 21 and evidence-layer parse authorities 34 to 33 (legacy
model-source sites 23 to 22), with all 28 preservation witnesses unchanged.
Its committed-tree receipt is
`/private/tmp/model-unfolder-verification/34ef32ea0e`: 2,619 tests collected,
356 focused and 44 U2-authority tests passed, all 48 preservation tests passed,
and the exhaustive partition passed 2,512 tests with 13 skips and 2 expected
xfails.  Every detached lane retained identical source and blessed-artifact
fingerprints.  The gate caught and locked an additional real debt reduction:
deleting the warning wrapper removed `evidence/validate.py`'s last broad
`except Exception`, so that module is no longer present in the ratchet baseline.

## Exact routed-expert activation closure

The final transformer-local U7 gap was not expert storage or router selection:
those were already exact. The remaining lie was that a proved expert still
drew a generic activation because activation identity was borrowed only from
the ordinary/shared FFN lane. The correction extends the same exact
`RoutedExpertStorage` result; no second reader or family branch was added.

The reader now accepts an activation only when its exact call reaches the
already-proven gate/up product under the exact expert callable. It supports:

- a functional activation resolved through its import binding;
- an exact `ACT2FN[config.path]` dispatch;
- an exact local `dict.get(key, literal_default)` dispatch whose complete
  config prefix is propagated through the expert construction chain;
- a source-literal swish formula whose alpha, clamps and additive up operand
  each reach the same proved product.

Same-function proximity is not evidence. An unrelated activation, clamp or
numeric addition; an activation after the down projection; rival config
bindings; or competing formulae all abstain. Formula refinements are combined
only when one downstream product adds exact operands without contradicting any
weaker proof. The source span, expert owner and construction route remain closed
inside the typed evidence.

The four real controls are deliberately different:

- DeepSeek-V3 and GLM-4.5 cite `hidden_act` and draw fused gate/up, SiLU,
  multiply and down;
- DBRX cites `ffn_config.ffn_act_fn.name`, retains its source-literal
  `"silu"` fallback when the key is absent, and draws split gate/up/down;
- GPT-OSS proves the literal formula
  `clamp(gate,max=7) * sigmoid(1.702*gate)` with the up lane clamped to
  `[-7,7]` and incremented by `1`. Its same-named checkpoint
  `swiglu_limit` does not override source that never reads that field.

The parser records one owner-qualified
`decoder.ffn.expert.expert_activation_formula` fact and attaches it only to
MoE layers. Dense siblings in a heterogeneous schedule cannot inherit it.
Cards, the canonical op graph, HTML expert drills, expanded JSON and projection
audit all consume that same fact. The former `FFNSpec.activation_clip` lane is
deleted because it was a dead, config-shaped second authority.

The visual connector law found one final projection defect: the exact GPT-OSS
`+1` operand initially produced a one-input unlabeled plus glyph. Constant
operands now remain visible beside their connector, and an unlabeled one-input
connector still blocks. The whole 28-witness dangling-connector corpus is green.

## Exact final bookend and residual scale closure

The last two transformer-local U7 claims now have positive source proofs. They
are not declaration fallbacks.

`final_stage_norm_evidence` begins at the D0/B1-resolved model-stage
occurrence, consumes the exact repeated-child proof, classifies exact norm
construction/calls through the shared norm primitive, and follows value
lineage to the exact primary hidden-state slot of every observed return. It
does not borrow the repeated layer's norm kind. Entry norms, auxiliary output
fields, guarded final norms, an early unnormalized return, unsupported return
shapes and conditional-expression alternatives all abstain. Reaching-path
growth is bounded at 256 alternatives; overflow is typed incompleteness, never
truncation used as proof. BLOOM proves a final LayerNorm; Llama and Gemma-2
prove final RMSNorm through the same reader.

Residual scaling extends the canonical cell-equation proof rather than adding
a config reader. The exact canonical mixer/attention and FFN residual
contributions must both carry the same explicit multiplier. A `self.field`
operand is then bound through the
exact constructor assignment and owner config chain; an exact numeric source
literal remains code-proven. A bare `residual_multiplier`, a multiplier used on
only one branch, rival operands, dynamic arithmetic, or an identity value of
one draws nothing. An additive cross-attention branch cannot inherit that fact;
it remains unscaled until its own equation is proved. Sequential cells show one
scale connector per exact scaled branch. Parallel cells use the exact algebraic factorization
`residual + s*(attention + FFN)`, preserving the untouched residual input.
Granite is the real positive control; unused and asymmetric synthetic controls
remain powerless.

Tightening the constant-operand connector rule exposed one independent
projection defect: shared tower gates in a refiner were captions beside a
one-input multiplication, not wired operands. The shared tower projector now
draws variable conditioning/parameter gates as explicit side-source nodes and
edges. Only an actual numeric subtitle or typed numeric operand can discharge
the one-input constant exception. HunyuanVideo's token refiner and Lumina's
latent entry-stage are the two real preservation controls: both now show
separate, non-overlapping gate sources and the full connector corpus is green.

With these two facts implemented, the transformer-local U7 code scope is a
closure candidate. U7 is not marked complete until the exact preservation
delta is reviewed, the 29-witness manifest is consciously re-blessed, and the
unchanged-tree final gate passes. Universal root token/embedding/head
scaffolding remains U14; this reader closes only the independently evidenced
final-normalization bookend.

Per-layer MoE placement, MTP and codebooks are explicitly U8 schedule work;
diffusion Conv-GLU remains explicitly U10 owner-resolution work. The current
preservation delta includes the four exact expert formulae, final-bookend fact
promotion, and the intentional removal of the obsolete nullable
`activation_clip` schema lane. Its exact surface/view list still requires
Soumil's review and the final unchanged-tree receipt; implementation success
alone does not close U7.

## Exact pending acceptance delta

The first authoritative manifest measurement was run against the 28-witness
corpus before Granite entered. A later full Sable regression run found that
this measurement was incomplete: the manifest represented views as a mapping
keyed by their display label, so two distinct views both named `architecture`
silently overwrote one another. U7 now replaces that representation with an
occurrence-exact sequence, rejects extra as well as missing views, and pins the
same-label counterexample. The figures below are the corrected pre-acceptance
delta, relative to the last committed manifest:

- all 28 witnesses change `ir` and `ledgers` because this U7 slice changes the
  canonical FFN schema and its authority/debt projection: the obsolete nullable
  `activation_clip` lane is replaced by the owner-qualified
  `expert_activation_formula` lane, and final-bookend/residual-scale evidence is
  now explicit. Most non-applicable witness values remain null/unknown rather
  than acquiring a mechanism;
- 11 transformer witnesses also change `expanded` and `params`, 13 change
  `html_meta`, and 15 change the Sable evidence surface;
- 34 SVG view hashes change across 14 existing witnesses:

| Witness | Views requiring visual re-approval | Reason |
|---|---|---|
| `dbrx-base` | `expert_1`, `expert_k`, `expert_kp1`, `expert_n`, `router` | exact split-expert activation and exact router policy |
| `deepseek-v3` | `architecture_v1`, four expert views, `router` | exact routed-expert formula and router policy |
| `gemma-2-2b-it` | `architecture_v1` | exact final-bookend promotion |
| `glm-4-5` | `architecture_v1`, four expert views, `router` | exact routed-expert formula and router policy |
| `gpt-oss-20b` | `architecture_v1`, four expert views, `router` | exact Swish/clamp/up-offset formula and router policy |
| `hunyuanvideo` | `text_refiner` | variable conditioning gate is now an explicit wired operand |
| `lumina-image-2-0` | `entry_stage` | variable tanh-conditioning gates are explicit wired operands |
| `qwen3-5-27b-text` | `architecture_v1` | exact final-bookend/residual-cell projection |
| `bloom` | `architecture` | source-proven final LayerNorm replaces the unresolved pre-head placeholder |
| `qwen3-8b` | `architecture` | source-proven final RMSNorm replaces the unresolved pre-head placeholder |
| `llama-7b` | `architecture` | source-proven final RMSNorm replaces the unresolved pre-head placeholder |
| `olmo-2-1124-7b` | `architecture` | source-proven final RMSNorm replaces the unresolved pre-head placeholder |
| `stablelm-2-1-6b` | `architecture` | source-proven final LayerNorm replaces the unresolved pre-head placeholder |
| `sana-1600m-1024px-diffusers` | two embedded Gemma-2 self-attention views | exact `query_pre_attn_scalar=256` score scale survives the proven tanh-softcap path, replacing “scaling unresolved” with `QK^T/sqrt(dim)` |

The remaining 14 pre-Granite witnesses have no SVG-view change. The earlier
claim that BLOOM, Llama and StableLM were byte-equivalent was false because of
the duplicate-label collapse described above; Sable's occurrence-independent
hash multiset correctly exposed the changes. The current working tree has the
first eight existing galleries and Granite reviewed/blessed. Soumil approved
the six additional existing witnesses on 2026-08-06 after their seven changed
PNGs were individually inspected and tied to the exact final-bookend/score
evidence above. Their durable galleries and fixture signatures are re-blessed;
the full occurrence-exact 29-witness preservation manifest and final gate are
the remaining acceptance operations.

### Acceptance audit correction: Granite must become witness 29

The registry/conformance audit found one non-negotiable closure gap:
`decoder.layer.residual_scale` is a registered native fact, but none of the 28
frozen witnesses produces it. The real positive control is Granite, currently
covered only by a source-backed smoke test. Consequently
`test_every_registered_fact_is_observed_in_corpus` correctly fails with
`registered but never observed: ['residual_scale']`.

The gate must not be weakened and the fact must not be made synthetic merely to
turn it green. Granite must be added as witness 29 with its exact frozen config,
source-backed fact/receipt surfaces, complete rendered gallery and explicit
visual review. The final acceptance operation is therefore:

1. review/re-bless all 14 changed existing galleries listed above;
2. review/bless Granite as the new residual-scale witness;
3. rebuild the full occurrence-exact 29-witness preservation manifest from
   that exact tree;
4. rerun registry, authority, conformance, preservation and the exhaustive
   unchanged-tree bracket.

Until those four steps pass, U7 is code-complete but **not acceptance-complete**
and must remain a closure candidate rather than `DONE`.

Granite's acceptance inspection also exposed one real source-binding gap before
blessing: its dispatch helper receives `scaling=self.scaling`, while the helper
parameter performs the multiplication. The score reader now follows only that
exact selected call argument back through the exact attention owner's
constructor assignment to `config.attention_multiplier`; keyword-only and
positional parameters are both bound without any name search. This changes the
Granite drill from the conventional `QK^T/sqrt(dim)` to its real `QK^T/128`.
The same path retains Gemma-2's exact scale through its separately-proven tanh
softcap. A backwards-liveness poison prevents an unused scaled copy from
lending config provenance to the live raw-score path.

### Approved working-tree receipt (commit boundary remains)

Soumil approved Granite and every corrected existing-witness delta. The
guarded fixtures are re-blessed and the occurrence-exact manifest contains 29
witnesses. Verification on one unchanged working-tree fingerprint
`69ed17206d7cc5c4f921582bd1796451bd1ced098207d540ae2e420fab635ef6`
produced:

- 80 attention-mechanism tests passed;
- 232 focused U7 tests passed;
- 98 registry/debt/projection tests passed;
- 52 strict preservation tests passed, including duplicate-label,
  missing-view and extra-view poisons;
- 499 changed-file/affected tests passed;
- the complete working-tree suite, excluding only the committed-HEAD archive
  assertion, passed: **2706 passed, 14 skipped, 2 expected xfailed**;
- the Sable regression corpus passed for all 29 frozen fixtures;
- `git diff --check`, bytecode compilation and pyflakes are clean.

The fingerprint before and after the complete working-tree suite was identical.
U7 cannot be marked `DONE` yet because
`test_preservation_is_clean_checkout_reproducible` deliberately executes
`git archive HEAD`; the current HEAD still contains the preceding 28-witness
state. The next and only lawful step is a reviewed local U7 commit containing
the approved source, tests, fixtures, manifest and newly blessed galleries,
followed by that clean-checkout test and the committed-tree receipt. No parser,
renderer or evidence work remains pending inside U7.
