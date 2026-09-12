# U8 — Exact position, mask and layer-schedule execution plan

Status: **ACTIVE — semantic implementation and the full-corpus non-view audit
are complete through U8-G; exhaustive/preservation acceptance is still in
progress. No manifest or gallery has been re-blessed.**
Parent: `docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` §20.11
Precondition: U7 is `DONE` at `37f3b1b` with receipt `5c2eaa5`.

## 1. Outcome

U8 makes every transformer/text layer answer four independent questions from
the exact source occurrence that owns that layer:

1. which computation occupies the layer slot;
2. which attention mask is created and applied;
3. which positional operation is actually applied, and where;
4. which optional per-layer modules exist (MoE, cross-attention, MTP,
   per-layer embeddings and codebook streams).

Config supplies values and selector operands. It never supplies mechanism
meaning by field presence, token spelling, frequency convention or model
identity. A mixed stack remains mixed. An unproved layer remains unknown; it
does not borrow the majority layer's architecture.

## 2. Starting inventory

The starting tree contains:

- **3 quarantined semantic readers**:
  `attention_causality_from_files`, `decoder_moe_schedule_from_files`, and
  `decoder_rope_dim_from_files`;
- **5 U8 legacy parse-authority sites**: `_find_decoder_layer`, `_parse_defs`,
  `attention_causality_from_files`, `position._call_line`, and
  `position._class_forward`;
- **58 U8 StructuralDebt rows**: 23 raw-extras writers, 2 drawn leaves, and
  33 config occurrences;
- semantic token authority in `layer_types.yaml`, `layer_schedules.yaml`,
  `_normalize_layer_schedule`, `_mixer_kind_for`, `_layer_mask`, and
  `_is_sliding_label`;
- duplicated/raw structure in `extras.moe`, `extras.sliding_window`,
  `extras.dual_kv`, `extras.irope`, `extras.position_encoding`, `extras.rope`,
  `extras.mtp`, and `extras.codebooks`;
- manual submodel relay parameters (`norm_label`, `position_evidence`, and FFN
  provenance arguments) that can let embedded and standalone parses diverge.

The 29-witness preservation bracket is the frozen starting behavior. Its
important heterogeneous controls include:

- Gemma-2 and GPT-OSS: alternating sliding/global masks;
- Qwen3.5: three positionless gated-delta layers followed by one RoPE softmax
  layer, repeated;
- DeepSeek-V3 and GLM-4.5: dense prefix followed by routed layers;
- MusicGen: fixed absolute positions and additive cross-attention;
- BLOOM: ALiBi;
- Llama/Qwen/Granite/OLMo/StableLM: ordinary RoPE controls.

## 3. Binding laws

### 3.1 Occurrence before interpretation

Every result starts from a resolved component root and an exact owner
occurrence already present in the OwnerGraph. U8 may add a neutral owner
boundary only when a required occurrence is not representable; it may not rank
classes by name, field spelling or model likeness.

### 3.2 One selector proof, many mechanisms

A selector proof is the exact relation:

```text
layer index + config operand(s)
    -> source expression/dispatch
    -> exact selected construction occurrence(s)
```

Mask, position, mixer, MoE and cross-attention readers consume that proof. They
do not independently reimplement list, threshold, modulo or frequency logic.

### 3.3 Syntax is not semantics

YAML may normalize a spelling only after source proves what that spelling
selects in this exact implementation. A YAML row cannot map `local` to sliding,
`0` to lightning attention, or a class/token to MoE by itself.

### 3.4 Positive facts and complete negatives

Presence needs an exact applied operation. Absence (`none`, dense instead of
MoE, no cross-attention) needs a complete candidate census for that layer.
Partial source can prove a positive local relation but cannot manufacture a
negative schedule.

### 3.5 Geometry follows mechanism

Window length, RoPE theta/fraction, compression ratio, expert counts and module
counts remain checkpoint values. They are consumed only after the owning
operation and exact config path are proved. A declared value without its
operation stays visible, unprojected evidence.

### 3.6 Independent axes never vote for one another

Mask kind does not prove position kind. RoPE parameters do not prove RoPE
application. Expert count does not prove MoE placement. Modality presence does
not prove cross-attention placement. A mixer label does not prove its internal
mechanism.

### 3.7 Embedded equals standalone

The same text component parsed as a root or embedded tower must have identical
position, mask and schedule facts, modulo namespace and presentation-only
altitude. No caller-supplied structural override is allowed.

## 4. Scope decisions for T-03 through T-18

| Issue | U8 decision |
|---|---|
| T-03 | Fully close token/mixer schedule authority. Source-bound selector and candidate mechanisms replace semantic YAML maps. |
| T-04 | Fully close self-attention mask direction/window/compression selection. Delete substring and presence fallbacks. |
| T-05 | Fully close transformer/text position application and geometry binding. Diffusion-owner application remains U10. |
| T-06 | The unsafe epsilon-spelling behavior was already killed by U7. U8 removes stale doctrine/relay remnants and adds the source-missing regression. |
| T-13 | Close per-layer cross-attention placement. U9 still owns modality fusion and K/V source semantics. |
| T-14 | Close transformer/text MTP, per-layer embedding and codebook construction. Block-diffusion remains U10. |
| T-15 | U8 forbids slot/name vocabulary from selecting a schedule. Full composite-role ownership remains U9. |
| T-16 | Close standalone/embedded parity for U8 fact families and delete post-parse honesty overrides. |
| T-17 | Remove structural submodel override arguments reached by U8. Altitude remains presentation-only. |
| T-18 | Add exact local/chunk/compression and temperature facts only when the source operation/order is proved. |

Any debt row outside these owner boundaries is reassigned in the same commit
to its true unit with a concrete reason; it is never silently excused.

## 5. Execution phases

### U8-A — freeze and neutral selector substrate

Files:

- new `model_unfolder/evidence/layer_selector.py`;
- `program_index.py` only if a neutral observation is demonstrably missing;
- new `tests/test_layer_selector.py`;
- this plan and the master tracker.

Build a closed typed result for exact per-layer selection. It carries:

- resolved component root and owner occurrence;
- exact selector callable and source span;
- exact layer-index formal;
- exact config paths and provenance used as operands;
- every construction candidate and its occurrence;
- one result per requested layer index;
- unresolved rivals/unsupported expressions;
- completeness status.

Supported expression forms are admitted by evidence, not by a family list:
literal list indexing, membership, threshold comparisons, modulo/frequency,
boolean composition, literal dispatch dictionaries, ternary/if alternatives,
and exact aliases. Short/malformed lists, conflicting aliases, dynamic keys,
unresolved calls and incomplete candidates fail or remain ambiguous.

This phase is observation/address infrastructure only. It must cause zero IR,
HTML, JSON, parameter, Sable or gallery delta.

### U8-B — exact positional operation and geometry

Files:

- replace `evidence/position.py` internals with ProgramIndex/OwnerGraph readers;
- add `evidence/position_schedule.py` if per-layer application needs a distinct
  result contract;
- parser, fact registry, IR/opgraph/card projections and conformance;
- position-focused tests.

Prove independently:

- model-stage learned/fixed embedding addition;
- attention-score ALiBi/relative bias;
- Q/K RoPE application;
- partial rotary width from the applied slice/constructor expression;
- per-layer RoPE/NoPE selection;
- genuine no-position only from a complete position-operation census.

Implementation checkpoint: the learned-absolute reader is deliberately a
positive shadow proof before parser cutover.  It requires an exact coordinate
producer, exact embedding primitive, exact unconditional addition, and exact
reachability into the repeated-child invocation.  An embedding field, a
familiar class/model name, or a configured maximum position count is never
sufficient.  Fixed/sinusoidal positions, ALiBi and relative bias remain
separate proofs; failure here must not be interpreted as their absence.

Delete the two raw AST helpers in `position.py`, the whole-file RoPE-dimension
reader, `position_declared` as a mechanism channel, and any
`rope_theta`-presence mechanism inference. Register/rout position facts and
consume theta/fraction/scaling only through the exact applied RoPE proof.

Required controls: Llama, BLOOM, GPT/StarCoder learned absolute, T5 relative
bias, ChatGLM explicit rotary dimension, partial RoPE, Llama-4-style
interleaved NoPE, Qwen3.5 mixed mixer/RoPE, unused rotary config, and
source-missing.

### U8-C — exact mask mechanism and window schedule

Files:

- new `evidence/attention_mask.py` and/or `mask_schedule.py`;
- parser, registry, projection and conformance;
- mask-focused tests.

Prove causal, bidirectional, full/global, sliding/local, chunked and compressed
masks from the exact mask creation and application path. A per-layer selector
may choose between already-proven mask implementations; it cannot label a raw
token itself.

Delete `attention_causality_from_files`, `_is_sliding_label` substring logic,
semantic `layer_types.yaml` mask groups, `_layer_mask` convention branches, and
window/pattern presence fallbacks.

Required controls: causal decoder, bidirectional encoder, T5 encoder/decoder
from the same source, Gemma alternating masks, GPT-OSS alternating masks,
unused `sliding_window`, unknown token, short list, cross-attention mask that
must not classify self-attention, and an additive causal-mask implementation.

### U8-D — mixer and ordinary-attention placement

Use U8-A to bind each layer to its exact constructed/invoked candidate. The
candidate's own U6 mechanism evidence supplies `mha/gqa/mqa/mla/gated_delta/`
other mixer semantics. A schedule proves placement only; it cannot upgrade an
opaque candidate from its token.

Delete `_normalize_layer_schedule`, `_mixer_kind_for`, `mixer_kinds`, semantic
`value_aliases`, semantic token groups, and frequency/list convention paths.

Controls include Qwen3.5 mixed gated-delta/softmax, a recurrent/attention
hybrid, MiniMax-style integer dispatch, GPT-Neo nested patterns, two different
candidate classes sharing the same token, same class at two construction
occurrences, and an unrecognised candidate that remains opaque.

### U8-E — MoE/dense and per-layer QK/KV schedules

Replace `decoder_moe_schedule_from_files` with U8-A plus exact U7 ordinary/routed
FFN occurrence proofs. The selector chooses between proved candidates; expert
spelling/count never classifies the candidate.

Also route per-layer QK-normalization, global/sliding KV geometry and cross-layer
KV sharing only when their exact selectors/application paths are proved.

Delete `first_k_dense_replace`/frequency/list convention interpretation and
raw `extras.moe`, `extras.dual_kv`, `extras.num_kv_shared_layers` authority.

Controls: DeepSeek-V3/GLM dense prefix, all-MoE DBRX/GPT-OSS, dense Llama,
explicit membership, modulo, threshold, guarded dense/expert sibling, malformed
schedule and heterogeneous candidates.

### U8-F — cross-attention and auxiliary modules

Reuse the same selector for replacement and additive cross-attention. Module
construction and invocation prove placement; U9 later proves modality/fusion
meaning and the exact K/V source.

Add exact construction facts for MTP modules and per-layer embedding pathways.
Finish the already-exact codebook stream proof by making the fact/receipt—not
raw extras—the consumer authority. Counts remain config values bound to those
proofs.

Delete config-only `_cross_attention_layers`, raw MTP/per-layer/codebook
structure authoring and corresponding raw-extras debt.

Controls: MusicGen additive cross-attention, replacement cross-attention,
declared-but-unconstructed schedule, MTP count without module construction,
constructed MTP with shared/unshared embedding/head, per-layer embedding value
without its operation, MusicGen codebooks, and plain text negatives.

### U8-G — embedded parity and deletion sweep

Remove U8-reached structural relay parameters from `submodel_spec` and encoder
panel call sites. Group facts must derive only from the recursively parsed
ModelIR. Altitude may change labels/detail density but no architecture fact.

Then delete/update in one commit:

- all 3 U8 quarantined reader rows and implementations;
- all 5 U8 legacy parse-authority rows/sites;
- all satisfied U8 StructuralDebt rows;
- semantic YAML keys with no lawful syntax-only consumer;
- dead helpers, fallback fields and duplicated raw extras.

The generated reader inventory and COR5 census must be regenerated from their
generators. U8 is not complete with “new reader plus old fallback.”

## 6. Permanent adversarial matrix

Every reader family must include:

1. exact positive;
2. exact negative with complete source;
3. missing source;
4. partial/broken sibling source;
5. two rival owner occurrences;
6. same class constructed twice;
7. renamed classes, fields and local variables;
8. config path collision across root/text/vision siblings;
9. conflicting aliases;
10. present-but-unused config value;
11. malformed/short/long schedule;
12. conditional/dynamic construction;
13. embedded-versus-standalone parity;
14. mixed-layer schedule preserving every occurrence;
15. renderer/params unable to read raw config or source;
16. family/model identity scrub producing the same structural result.

## 7. Acceptance bracket

Each phase uses focused tests and affected authority gates. Full suite and
preservation run at mechanism phase boundaries, not after every small edit.
Independent lanes run in parallel worktrees/coordinator where safe.

Final U8 acceptance requires:

- static/lint/diff checks clean;
- U2 authority and U5 consumer firewall green;
- all U8 adversarial tests green and anti-vacuous;
- all 29 Sable witnesses mechanically clean;
- equivalent model examples inspected for every migrated mechanism;
- standalone/embedded parity green;
- exact U8 debt/readers/parse-authority census at its declared target;
- full suite green on one unchanged fingerprint;
- preservation manifest and occurrence-exact views green;
- every intentional PNG/card delta individually inspected and approved by
  Soumil before re-bless;
- committed `git archive` carries every fixture, gallery and new production
  module and passes import/parse plus gallery-hash verification.

## 8. Stop conditions

Stop the current phase and report rather than compensate when:

- the exact layer owner cannot be resolved;
- a negative fact needs callable completeness the ProgramIndex lacks;
- two candidates disagree or a selector cannot be evaluated exactly;
- a proposed fix needs a family/model/class-name semantic table;
- a config token is the only evidence for mechanism meaning;
- a renderer/parameter path needs raw config/source access;
- a change affects an owner assigned to U9/U10/U11 without a reusable neutral
  contract;
- a preservation delta is unexplained or an existing gallery disappears;
- verification changes the tree.

## 9. Definition of done

U8 is `DONE` only when a new transformer with a novel mixture of attention,
position, mask, dense/MoE and auxiliary layer types can be represented by
adding no model/family branch and no semantic YAML row: its exact source
occurrences and selector expressions are enough. Unknown source remains
unknown at each affected layer, and every downstream surface projects the same
owner-qualified typed facts.

## 10. Running implementation record

### U8-A — SHADOW GREEN (local, not pushed)

Implemented the neutral exact layer-selector boundary in
`model_unfolder/evidence/layer_selector.py` with its permanent poison matrix in
`tests/test_layer_selector.py`.

The boundary:

- derives the complete exact candidate census internally from one resolved
  owner/callable/target, so callers cannot omit a rival;
- selects an exact `ConstructionSiteId` plus its exact `ChildCandidate`, never
  merely a class spelling;
- evaluates membership, threshold, modulo, boolean, alias, list/subscript,
  ternary-guard and literal-registry forms without model/family vocabulary;
- retains every exact config path and whether the deciding value was
  checkpoint-declared or class-default;
- treats short lists, missing operands, symbol-less/dynamic candidates,
  multiple live occurrences and unsupported regions as typed uncertainty;
- refuses an `absent` result when unsupported code could hide a candidate;
- has no production consumer and therefore cannot change IR, renderer, params,
  Sable or gallery output.

Verification on the unchanged working tree:

- selector poisons: **20 passed**;
- U3 kernel plus selector: **175 passed**;
- affected U2 structural/debt blocking gates: **34 passed**;
- existing position/schedule smoke examples: **27 passed**, 140 deselected;
- `py_compile`, `pyflakes`, `git diff --check`: clean;
- repository reference scan: no production import/consumer of the new boundary.

One gate caught and corrected a real integration defect before closure: the
private interpreter originally stored `self.index`, which the writer census
correctly treated as a possible public structural-spec mutation. It now uses
the established private `_program_index` handle; no debt row or exemption was
added.

### U8-B1 — Q/K HALF-TURN APPLICATION SHADOW GREEN (local, not pushed)

Implemented the first positive-only position-operation boundary in
`model_unfolder/evidence/position_application.py`, with adversarial tests in
`tests/test_position_application.py`.

This is deliberately named `QKHalfTurnApplicationEvidence`, not RoPE evidence.
It proves that two exact projection-derived lanes enter a two-output rotation
helper, that the helper implements the exact split-half quarter-turn formula,
and that both outputs reach the first two projection-derived attention-compute
operands. It does **not** yet prove that the two factors are position-derived
cosine/sine values. Therefore it has no fact/IR/renderer consumer and cannot
draw or claim RoPE by itself.

Soundness closures include:

- exact decoder-block and attention-owner occurrences from the U3 graph;
- ordered, per-lane Q/K producer identity (one source per lane; fused storage
  must share one source and split storage must not);
- exact application/helper/half-turn symbols and source spans;
- exact factor expressions tied to application arguments 3 and 4;
- exact first-half and second-half slices, including their order in the
  concatenation;
- exact framework `unsqueeze` as the only admitted shape-only factor transform;
- projection-derived attention operands rather than raw positional argument
  numbers, so framework dispatch may prepend a module argument safely;
- multiple valid application paths become ambiguity; broken/partial source and
  every incomplete path remain failure/unknown, never NoPE.

Focused receipt on the final local tree:

- Q/K half-turn controls and poisons: **16 passed**;
- selector plus application tests: **36 passed**;
- affected U2 structural writer/blocking gates: **14 passed**;
- `py_compile`, `pyflakes`, `git diff --check`: clean.

Real controls: Llama, Gemma-2, Qwen3 and OLMo2 resolve; BLOOM remains a negative
control. Full RoPE evidence remains blocked on U8-B2 factor provenance, then
geometry and layer selection. No parser/render cutover is authorized yet.

### U8-B2 — TRIGONOMETRIC FACTOR PROVENANCE SHADOW GREEN (local, not pushed)

Added two narrowly separated boundaries:

- `evidence/call_arguments.py` binds explicit caller expressions to the exact
  callee `forward` formals only after an owner invocation is graph-resolved;
- `evidence/position_factors.py` follows the two B1 factor lanes backward over
  those exact calls to one constructed producer and proves its ordered output
  lanes are cosine and sine of one shared phase.

The call rail is neutral. Parameter names serve only as Python keyword
addresses. The implicit method receiver is bound by addressed-instance call
semantics, not by a `self` spelling. `*args`, `**kwargs`, invalid/duplicate
arguments and omitted explicit values stay partial or fail; exact bindings are
retained without claiming the unresolved expansion.

The factor proof additionally requires:

- direct-factor lineage ends at producer output lane 0 and rotated-factor
  lineage ends at lane 1, across tuple transport and owner boundaries;
- the producer is one exact addressed construction occurrence (an uninvoked
  trig-looking sibling is irrelevant);
- output lane 0 is an exact zero-argument tensor cosine call and lane 1 an
  exact zero-argument tensor sine call;
- both calls share one exact phase expression;
- that phase contains exact matrix multiplication between stored owner state
  and one producer formal explicitly bound at the producer call.

This closes the prior semantic gap where half-turn math with arbitrary factors
could have been mislabeled RoPE. Geometry and per-layer application are still
required before a RoPE fact may be authored.

Focused receipt on the final local tree:

- neutral call-binding controls: **8 passed**;
- factor-provenance controls: **10 passed**;
- B1/B2 plus affected structural writer/blocking gates: **48 passed**;
- real Llama, Gemma-2, Qwen3 and OLMo2 factor paths resolve; BLOOM does not;
- `py_compile`, `pyflakes`, `git diff --check`: clean.

No parser, fact, IR, renderer, params, Sable, manifest or gallery consumer was
added. The entire B1/B2 position rail remains shadow-only.

### U8-B3 — SECOND EXACT ROTATION PROTOCOL (local, not pushed)

The first application proof covered the common `x*cos + rotate_half(x)*sin`
form, including dtype-only return wrappers needed by OLMo2. A second exact
protocol now covers implementations such as GPT-OSS that:

1. split one lane with an exact framework `chunk(x, 2, dim=-1)`;
2. compute `(first*cos - second*sin, second*cos + first*sin)`;
3. concatenate the pair in exact source order; and
4. invoke that same proved one-lane kernel independently for Q and K.

The two protocols are closed typed values (`split_half_turn`, `chunk_pair`). A
helper/function spelling never selects one. Wrong signs, wrong chunk width or
axis, reversed concatenation, different Q/K kernels, factor laundering and an
uninvoked valid-looking kernel all fail.

Real positive controls now include Llama, Gemma-2, Qwen3, OLMo2, Granite,
StableLM and GPT-OSS. BLOOM remains negative. DeepSeek-style latent/conditional
projection paths and Qwen3.5's dynamically constructed embedded text stage
remain typed gaps; they are not patched with identity or config conventions.

Focused receipt: application plus factor controls **30 passed**; static checks
clean. The expansion remains shadow-only and causes no output delta.

### U8-B4 — SCORE-OPERAND DECOUPLING + GUARDED INTERLEAVED ROTATION

The B1 prototype coupled positional application to the projection-storage
classifier. That was unsound as a boundary: projection storage and use as the
query/key attention-score operands are separate facts. It rejected exact
latent, sliced or config-conditional projection programs even when their
rotation and eventual score participation were directly visible.

`evidence/attention_operands.py` now derives the two exact score operands from
the already-proven attention computation:

- a direct SDPA/framework protocol supplies its first two exact tensor
  operands;
- a dot/softmax helper must have one exact dot call whose result reaches the
  exact softmax call;
- exact versioned dataflow maps those dot inputs to compute formals and exact
  Python argument binding maps the formals back to the attention entry call;
- owner state can support reshaping/repetition but cannot masquerade as a
  score value;
- multiple score producers, uncertain flow, index/root
  mismatch or missing bindings remain unknown/failure.

`QKHalfTurnApplicationEvidence` now terminates at this operand proof rather
than requiring simple Linear projection occurrences. It no longer authors or
contains a projection-storage fact. Permanent controls show that conditional
or sliced projection production still resolves when the exact rotated outputs
reach query and key, while query/value substitution, overwritten outputs and
duplicate unpack targets fail.

The application boundary also gained exact source-guard selection through the
existing config guard interpreter. A config value is used only to execute the
source predicate; it never names the mechanism. The selected call retains the
exact deciding config path and provenance. Missing/non-boolean/unprovable
guards cannot choose a branch.

A third algebraic protocol, `interleaved_pair`, proves the even/odd-coordinate
formula independently for both lanes:

`(even*cos - odd*sin, odd*cos + even*sin)`.

The proof requires exact stride-2 slices, exact signs/products, exact ordered
concatenation and the same factor pair for both Q and K. A wrong sign or
one-lane match fails. DeepSeek-V3 is the real positive control: its
`rope_interleave` source guard selects the interleaved implementation, its
latent/conditional projection storage is irrelevant to this proof, and its
exact cosine/sine producer provenance resolves. No DeepSeek spelling exists in
production evidence code.

Focused receipt on the current shadow tree:

- application + factor counterexamples: **38 passed**;
- real Llama, Gemma-2, Qwen3, OLMo2, Granite, StableLM, GPT-OSS and
  DeepSeek-V3 positive paths checked;
- BLOOM remains a negative control;
- `py_compile`, `pyflakes`, `git diff --check`: clean.

This section supersedes B1's provisional projection-storage coupling. The
code remains shadow-only: no parser, fact, IR, renderer, params, Sable,
manifest or gallery consumer exists yet. Geometry, layer selection and
non-RoPE mechanism proofs remain mandatory before cutover.

### U8-B5 — APPLIED ROTATION GEOMETRY (local, not pushed)

`evidence/position_geometry.py` proves the extent of the already-proven Q/K
rotation from the executed tensor path. It has three closed layouts:

- `full`: the complete code-proven Q/K projection lanes enter rotation, its
  outputs reach the score operands, and no exact local split/recombination is
  present;
- `partial/prefix`: exact prefix slices are rotated and concatenated before
  their exact untouched suffixes;
- `partial/suffix`: an exact two-part split rotates the suffix and concatenates
  it after the exact untouched prefix.

A full result additionally requires exact Q/K/V storage and exact
all-elements Q/K producer paths. Producer origin alone is insufficient:
slicing or opaque narrowing can share the same producer and is therefore
explicitly rejected. Exact fused-QKV unpack is accepted only after the storage
boundary independently proves it. A partial result requires both Q and K to
prove the same layout, exact width evidence, exact pass-through lineage, and
correct recombination order. Merely declaring a partial factor, slicing only
one lane, swapping concatenation order or dropping the complement cannot
author partial geometry. Guarded
intermediate writes preserve lineage only when both guard alternatives retain
the same source; the selected interleaved/standard rotation rival is evaluated
by B4's exact config guard.

`evidence/expression_value.py` supplies the numeric half of the boundary. It is
a neutral exact-owner constructor interpreter, limited to:

- exact owner-qualified config paths;
- unique unguarded `self` fields and straight-line locals;
- numeric constants and `+ - * / // %`;
- unshadowed builtin `int`/`float` casts.

Every result retains all exact config/class-default premises and source spans.
Missing operands, guarded/rival writes, shadowed casts, unsupported syntax,
bad arithmetic, or constructor-normalized values without value-bearing input
premises remain unknown. No fallback or field-name vocabulary is used.

Real controls prove:

- Llama and GPT-OSS: full rotation;
- StableLM: prefix-partial, width `20`, evaluated from exact
  `hidden_size / num_attention_heads * partial_rotary_factor` premises;
- DeepSeek-V3: suffix-partial, width `64`, from the exact
  `qk_rope_head_dim` premise.

Focused receipt: expression + geometry + application + factor + operand
controls **60 passed**; affected U2 authority gates **49 passed**; static checks
clean. Counterexamples include sliced projection laundering, opaque narrowing,
wrong recombination, Q/K disagreement, and exact fused-QKV as a positive
control. This remains shadow-only. The next U8-B work is per-layer application
selection and the independent learned-absolute/ALiBi/relative-bias proofs.

### U8-B6 — UNIT-COMPLEX ROTATION + GUARDED LAYER VALUE (local, not pushed)

The application boundary now recognizes a fourth exact algebraic protocol,
`complex_pair`. Both Q and K must independently prove the complete chain:

1. dtype-only conversion followed by a reshape of the entire final dimension
   into exact pairs of width two;
2. `torch.view_as_complex`;
3. multiplication by the same factor through broadcast-only full slices and
   inserted singleton axes;
4. `torch.view_as_real` and exact pair-dimension flattening; and
5. both returned lanes reaching the exact Q/K score operands.

Names carry no authority. Missing real/complex conversion, pair width other
than two, addition instead of multiplication, a narrowed factor, a differing
factor between Q and K, wrong flattening, or a one-lane lookalike fails.

`PositionComplexFactorEvidence` independently proves that the single complex
factor is a unit phase made by exact `torch.polar(torch.ones_like(angle),
angle)`, with `angle` containing stored owner state multiplied by one explicitly
bound coordinate input. Optional scalar owner-state scaling may follow the
unit phase, but arbitrary complex construction cannot pass. This keeps
"complex multiplication" separate from "position-derived rotation" just as
B1 and B2 kept real half-turn application separate from cosine/sine provenance.

The exact guard interpreter gained a deliberately small constructor-parameter
channel. It can follow one unique unguarded `self` field assignment, exact
subscript lookup and Python truthiness for closed immutable scalar values.
Thus source such as `self.flag = config.schedule[layer_idx]` can be evaluated
for a supplied, exact constructor index. The value only executes the source
guard; it never names the mechanism. Missing/out-of-range parameters, guarded
or rival field assignments, non-scalar/user-defined truthiness and unknown
syntax remain unknown. Geometry threads the same parameter evidence to the
application boundary.

Real installed-source control: Llama 4 text layer 0 proves `complex_pair`, the
position producer proves `Llama4TextRotaryEmbedding.forward`'s unit complex
phase from `inv_freq @ position_ids`, and full applied Q/K geometry resolves.
Layer 3 does not select the guarded application. This is source-shape support,
not a Llama/model-family branch: production evidence contains no Llama 4
spelling.

Focused application/factor/geometry plus config-guard consumers pass; affected
U2 identity, taint, structural-write, blocking-net and reader-exception gates
pass; `py_compile`, `pyflakes` and `git diff --check` are clean. This remains
shadow-only. Supplying an index manually is not yet a schedule proof: the next
unit must prove the construction-time index transport from the repeated-layer
comprehension through the block constructor into the attention constructor,
then evaluate every layer without caller-authored parameter values.

### U8-B7 — EXACT POSITION-APPLICATION SCHEDULE (local, not pushed)

`evidence/construction_arguments.py` adds the missing neutral address boundary:
an authoritative `ConstructionSite` is bound to the exact constructed child's
`__init__` formals only after the site, parent occurrence, child occurrence,
unique candidate, child symbol and ProgramIndex record all round-trip through
the resolved OwnerGraph. Positional/positional-only/keyword arguments are
retained as exact expressions. Defaults are not evidence; variadics, duplicate
arguments, rival/dynamic candidates, forged site copies and missing callables
remain typed partial/failure. The boundary assigns no role to any name.

`evidence/position_schedule.py` composes that neutral binding into one exact
index-transport chain:

`builtin range(exact layer-count path)`
→ one filter-free construction comprehension target
→ repeated-child constructor formal
→ immediate attention constructor formal
→ the unique attention-field assignment
→ the exact guarded Q/K rotation application.

Every integer index in the proved `range(count)` is evaluated. Active indices
must resolve the same application identity and its factor + geometry proof;
inactive indices mean only that this exact application call is disabled. They
do **not** claim NoPE, because a complete census of other positional operations
has not landed yet. The schedule rejects transformed indices, an unproved owner
hop, shadowed `range`, short or long selector sequences, absent values, rival
field assignments, all-inactive schedules and rival per-index applications.

The real installed Llama 4 text source resolves all 48 indices and reproduces
its exact source-enacted sequence:

`1110 1110 1110 1110 1110 1110 1110 1110 1110 1110 1110 1110`

Each active entry cites the `complex_pair` application, unit-complex positional
factor and full Q/K geometry. Synthetic controls cover alternating schedules,
complete class/field/local renaming, positional and keyword constructor
transport, transformed-index attacks, missing/short/long values, shadowed
builtins, guarded rival fields and forged ProgramIndex-site copies. Focused
position application/factor/geometry/schedule tests and affected U2 authority
gates pass; static checks are clean. The rail remains shadow-only and has not
changed parser, facts, IR, renderer, params, manifest or pixels.

### U8-B8 — LEARNED-ABSOLUTE PRE-STACK ADDITION (local, not pushed)

`evidence/position_absolute.py` proves one exact learned-position mechanism:
an exact coordinate producer reaches one exact embedding primitive, that
lookup is added to the hidden stream, and the resulting stream reaches the
exact repeated layer call. The reader is model-stage and occurrence-scoped;
it never infers the mechanism from an embedding-shaped field, a class name or
a config key. Fixed/sinusoidal addition, score-side bias and rotary application
remain separate evidence families. Failure therefore stays unknown rather
than becoming a fabricated NoPE claim.

Real GPT-2/learned-position controls and adversarial source fixtures pin exact
coordinate origin, call ownership, addition order and pre-stack reachability.
The unit is shadow-only: no parser/fact/IR/renderer/params projection exists.

### U8-B9 — NEUTRAL SCORE-SIDE ADDITIVE APPLICATION (local, not pushed)

`evidence/attention_score_additives.py` begins the independent score-side rail.
It does **not** classify an operand as ALiBi, relative bias or a mask. The
result is an ordered inventory because several independently meaningful
operands can reach one score lane. Its first closed protocol composes the
existing exact attention score-product-to-softmax proof with an exact
`baddbmm` receiver, exact matrix operands and an explicit finite, non-zero
`beta` proven from the exact owner constructor and/or exact config occurrence.
An implicit framework default is refused because it is not present in
ProgramIndex. `beta == 0` proves the receiver is inactive;
dynamic/unsupported beta remains a typed failure. The second protocol proves
ordinary `score = score + operand` and `score += operand` only from the exact
live path plus its exact dataflow operation.

Real BLOOM exposes two distinct ordered applications: its enabled `baddbmm`
receiver, then its causal mask. Real Llama exposes only its mask addition. Real
T5 exposes the exact combined `position_bias_masked` augmented operand without
yet assigning that operand a semantic kind. A plain score-to-softmax fixture
proves absence. Synthetic controls pin zero and dynamic beta,
malformed/implicit argument shapes, config-value joining, complete local
renaming, inventory ordering and DTO-forgery rejection: **14 passed**. Static
checks are clean. This remains shadow-only and cannot author an ALiBi,
relative-bias or mask fact until separate producer proofs reach the exact
inventory entry.

### U8-B10 — END-TO-END LINEAR POSITION BIAS / ALIBI (local, not pushed)

`evidence/position_linear_bias.py` joins one exact enabled `baddbmm` receiver
to its exact producer across all ownership boundaries:

`attention formal -> exact block-to-attention argument`
`-> exact repeated block argument -> exact model-stage definition`
`-> exact direct model-stage wrapper -> exact local producer return`.

The producer is classified only when its returned value is an exact product
of a cumulative masked coordinate and head-dependent slopes. The coordinate
requires the same formal on both the exact `.cumsum(dim=-1) - 1` lane and its
masking multiplication. The slope lane requires an exact imported
`torch.pow`, an exact `torch.arange` controlling its count, and only closed
guarded extensions that concatenate the prior slopes with another exact pow
lane. The head-count formal is derived specifically from the arange count
arguments; device/dtype formals cannot impersonate it.

The wrapper is intentionally direct-only. An inherited method is not selected
through the older first-base shortcut because that is not a reusable exact
Python-MRO proof; such a source remains unknown until the B1 precedence rail
offers exact method resolution. A sibling producer, disconnected producer,
wrong coordinate, wrong slope, additive instead of multiplicative producer,
or disabled `baddbmm` receiver cannot author ALiBi.

Real BLOOM resolves end-to-end. Real Llama and Gemma-2 remain negative
controls despite their mask additions. A completely renamed synthetic model
resolves, while the producer/application attacks above fail: **10 passed**.
The combined U8/attention/U2-authority bracket is **277 passed**. This remains
shadow-only; no parser, fact, IR, renderer, params, manifest or gallery output
has changed.

### U8-B11 — FIXED SINUSOIDAL PRE-STACK ADDITION (local, not pushed)

`evidence/position_fixed.py` proves the fixed/sinusoidal mechanism separately
from learned lookup and score bias. It requires one exact constructed child to
return an exact generated-buffer lookup at a code-proven coordinate, that
exact call to be added into the hidden stream, and the result to reach the
exact repeated layer call. The buffer must be installed by an exact
`register_buffer(..., persistent=False)` call from a direct builder whose
returned table contains exact `torch.cos` and `torch.sin` of the same
position-derived angle. A guarded extension may only append exact zero
padding. Parameter/field/class/helper spellings are never role evidence.

The coordinate kernel gained one neutral missing observation: `ParamRecord`
now retains structural annotations, and an exact annotated tensor
`a, b, n = tensor.size()` unpack can supply an arange bound. Exact `int`/`float`
formals can supply scalar offsets. Untyped receivers, shadowed scalar types,
`.size(...)` calls with arguments, unrelated local objects and rival/guarded
unpacks remain unsupported. This fixes the kernel once instead of granting a
MusicGen-specific exception.

Real MusicGen resolves at its exact nested `decoder` occurrence. BLOOM and
Llama remain negative controls. A fully renamed synthetic source resolves;
persistent buffers, cos/cos lookalikes, non-multiplicative angles and a
disconnected hidden-stream application fail: fixed-position controls **9
passed**, coordinate + ProgramIndex + learned-absolute regression controls
**81 passed**. The rail remains shadow-only with no output projection.

### U8-B12 — LEARNED RELATIVE-BUCKET SCORE BIAS (local, not pushed)

`evidence/position_relative_bias.py` replaces the old transitive-marker guess
with two joined, independently closed source proofs. The producer proof starts
at one exact neutral score-additive operand and walks backward inside the exact
attention occurrence through its alias and guarded compute definition. The
compute callable must construct two exact imported `torch.arange` coordinate
lanes, subtract query coordinate from key coordinate, call one exact bucket
callable, use that bucket as the sole input to one exact learned
`torch.nn.Embedding`, and return the resulting table. The bucket callable must
contain one exact absolute-value lane, one exact logarithmic large-distance
lane, and one exact `torch.where(small, relative, large)` merge whose returned
state reaches the producer return. Class, method, field, formal, local and model
spellings have no semantic authority.

The ownership proof is code-only and symbolic. It transports the exact
constructor flag across every authoritative OwnerGraph hop from repeated block
to attention and proves the flag is selected by exact unshadowed
`bool(comprehension_index == 0)`. It therefore establishes only that repeated
index zero owns the learned table. It deliberately does **not** enumerate the
checkpoint's layer count and does **not** claim that later layers reuse the
returned table; loop-carried sharing is a separate execution-flow fact.

This work exposed and repaired a neutral construction-binding defect:
`ModuleList.append(...)` sites can occur identically on the general
construction and container-element index surfaces. `bind_construction_site`
now deduplicates identical authoritative records before checking uniqueness,
so one `ConstructionSiteId` can no longer be mistaken for two rival sites.

The real installed T5 source resolves end-to-end. Real Llama, BLOOM and the
non-T5 text-encoder control do not. Adversarial copies of the real HF source
cover total class/field/formal/local renaming, wrong embedding primitive, wrong
coordinate arithmetic, missing logarithmic bucket lane, reversed bucket
branches, disconnected table lookup, disconnected score application, wrong
first-index predicate, constant ownership, duplicate-index-surface binding and
DTO forgery: **16 passed**. This remains shadow-only; parser, facts, IR,
renderer, params, manifests, galleries and pixels are unchanged.

### U8-B13 — EXACT MASK BUILDER TO REPEATED-BLOCK FORMAL (local, not pushed)

`evidence/attention_mask.py` begins U8-C without reintroducing the old
whole-file causality scan. It resolves only closed framework mask APIs through
their exact unshadowed imports, retains the exact builder call and definition,
and uses local reaching definitions plus the exact repeated-child invocation
binding to prove which builder result reaches which formal of which exact block
occurrence. A local helper with a familiar `create_causal_mask` spelling has no
semantic authority. Model, class, field, formal and local names do not select a
mask kind.

This is deliberately a producer-to-block boundary, not yet the final mask
schedule. An exact single reaching builder resolves `causal`, `bidirectional`
or `sliding_causal`. Rival or conditional builders remain an incomplete typed
inventory; the reader cannot label the selector token. When the neutral
reaching-definition helper reports a conditional rewrite, the inventory
recovers every builder definition targeting the exact selected lane so an
earlier guarded producer cannot disappear merely because a later guarded
assignment was visited last. That recovery is a rival census only and never a
selection.

Real Llama and BLOOM resolve causal. Real Gemma-2 and GPT-OSS preserve the
exact causal and sliding-causal builders selected by
`causal_mask_mapping[self.config.layer_types[i]]` and remain incomplete until
the per-layer selector is proven. The real T5 encoder source preserves its
reachable bidirectional lane but remains incomplete because the current bundle
does not index the external `PreTrainedModel` assignment that would prove
`self.config` and the absent `PretrainedConfig.is_decoder` default. U8 must not
restore the old hand-written universal default to force this green; exact
external-framework config ownership is the prerequisite.

Synthetic same-source controls prove causal and bidirectional selection from
one exact constructor-bound config flag, unused builders do not qualify, local
lookalike helpers do not qualify, complete local/formal renaming is invariant,
and DTO cross-stage forgery is rejected.

The same module also closes the producer-to-score join. It consumes the exact
positive attention-child `invocation_path`, binds the mask formal across every
owner hop, crosses an indexed compute helper only through its exact entry call
and Python actual-to-formal binding, then requires that transported source to
reach the exact neutral score-additive operand. This is what prevents a T5
encoder-side cross-attention mask from certifying its self-attention lane: the
block exposes both mask formals, but only the self-attention formal reaches the
selected score application. Llama's free eager-attention helper, BLOOM's direct
attention forward and T5's two-hop self-attention wrapper are all covered. A
poison retaining the helper parameter spelling while replacing its actual with
`None` fails, pinning the cross-callable anti-laundering rule.

Mask controls now total **16 passed**. Gemma-2's exact score join succeeds but
the overall result stays incomplete at its still-unproven per-layer selector;
this is the intended separation between mechanism and schedule. The boundary
remains shadow-only: no parser, fact, IR, renderer, params, manifest, gallery or
pixel surface consumes it.

### U8-B14 — FRAMEWORK CONFIG ADDRESS + EXACT MASK SCHEDULE (local, not pushed)

The first U8-C schedule slice is now closed without turning `self.config` into
a magical config root. `evidence/framework_config.py` proves the missing address
edge through one exact owner constructor, one unshadowed
`super().__init__(config)` call, one owner-graph `ConfigBinding`, a conservative
single-base inheritance trace with no intervening constructor, and the closed
external `transformers.modeling_utils.PreTrainedModel` config-storage protocol.
This is framework address semantics only: it neither reads a value nor assigns
meaning to a config field. A local lookalike base, wrong external target,
shadowed `super`, transformed actual, intervening constructor, rival binding or
forged DTO fails. Transformed config objects remain path-exact: T5's exact
invalidated paths stay unavailable while unaffected paths can still resolve.

`decoder_attention_mask_layer_schedule_for_path` then accepts only an exact
enumerated selector of this form: the exact repeated-block call indexes one
literal source dictionary of already-proven mask builders with an exact config
sequence and the exact `enumerate` target. The repeated container count must be
an unshadowed builtin `range` over one exact config path, and it must equal the
selector sequence length. Each sequence value selects the dictionary entry the
source actually binds; no token spelling carries mask semantics. Unknown keys,
short lists, constant/wrong indices, shadowed `range`, rival maps and count
disagreement fail.

Real Gemma-2 now resolves its full per-layer causal/sliding-causal schedule from
the exact `layer_types` sequence and exact `num_hidden_layers` count. The result
retains the exact framework config alias, repeated invocation, literal map,
builder calls, score consumer and both config paths. The same code does not
force GPT-OSS green: GPT-OSS adds its mask before concatenating learned attention
sinks and subtracting a row maximum, so the current U6 exact score-to-softmax
reader stops at that unsupported transformation. This is recorded as a separate
score-lane prerequisite; U8 does not add a GPT-OSS branch or pretend the mask was
not built. T5 also remains honestly incomplete where its checkpoint omits the
decoder flag and the source bundle cannot yet prove the external config-class
default.

Focused controls: framework-config **11 passed** and complete mask boundary/
schedule **23 passed** (34 combined). The earlier mask/attention/identity
bracket remains **184 passed**. All work remains shadow-only; no parser, fact,
IR, renderer, params, manifest, gallery or pixel surface consumes it.

### U8-B15 — SCORE-TRANSFORM CLOSURE, SOURCE CONFIG DEFAULTS, UNIFORM
SCHEDULES AND EXACT MASK GEOMETRY (local, not pushed)

This slice supersedes the two upstream limitations recorded at the end of
U8-B14 without introducing a model/family exception.

The exact attention score reader now recognizes two closed algebraic
transformations between an already-proven score-plus-mask state and softmax:
concatenation with an independently-proven score fragment, and translation by
the row maximum.  The recognition is structural and dataflow-bound; neither a
function name nor a GPT-OSS identity participates.  That lets the same
per-layer schedule reader prove GPT-OSS's alternating causal/sliding-causal
schedule while renamed equivalents remain equivalent and disconnected or
wrong-axis transformations remain unresolved.

`SourceBundle.supporting_files` and `framework_config.py` add a separate,
strict source-default boundary.  Same-revision `configuration_*.py` files are
indexed by the one `ProgramIndex` but remain excluded from legacy whole-file
reader surfaces.  A default is usable only when the exact model-stage
constructor annotation resolves through an exact import to one indexed config
class and that class owns a literal default for the requested field.  A
checkpoint declaration always wins.  Computed defaults, unresolvable imports,
sibling components and annotation-name matching cannot supply a value.  This
proves T5's omitted `is_decoder=False` as `class_default`, so the exact same
installed `T5Stack` source resolves bidirectional for an encoder and causal for
a decoder.  The score join also keeps self- and cross-attention mask lanes
separate: an encoder-side formal cannot certify the selected self-attention
lane.

`decoder_uniform_attention_mask_layer_schedule_for_path` covers the other
lawful schedule shape: one exact score-applied builder repeated across one
exact config-bound container count.  Llama and BLOOM resolve uniform causal;
the real T5 encoder resolves uniform bidirectional.  This does not weaken the
alternating reader or turn an absent selector into guessed semantics.

`decoder_attention_mask_geometry_for_path` makes window/chunk values
architectural only after an exact enacted framework builder consumes the exact
stage config object.  For `**mask_kwargs`, the config actual must come from one
exact earlier dict definition on the same guard path; a nearby field, sibling
dict or rival rewrite cannot certify it.  The geometry path is selected by the
closed framework protocol, then resolved through the exact `ConfigBinding` and
typed config/default selector.  Real Gemma-2 resolves sliding window **4096**;
real GPT-OSS resolves **128**.  A source-proven chunked fixture resolves
`attention_chunk_size=256`.  Qwen3 remains absent even when a plausible
positive `sliding_window` value is injected, because its enacted source path
does not select the sliding builder.  Foreign config objects, rival kwargs
maps, unavailable values and non-positive geometry fail.

The remaining Qwen3 limitation is honest and separate: callers may supply an
already-built runtime mask mapping, so source alone cannot prove that the local
builder owns every invocation.  Qwen3.5's mixed gated-delta/softmax placement
belongs to U8-D rather than being folded into mask geometry.  This slice is
still shadow-only; parser, facts, IR, renderer, params, manifest, gallery and
pixel surfaces do not consume it yet.

### U8-C16 — SINGLE MASK EXECUTION AUTHORITY CUTOVER (local, not pushed)

The transformer parser now consumes one `AttentionMaskExecution`: the exact
framework builder, exact score-additive lane, exact per-layer schedule and
every required geometry operand. It writes one typed `mask_schedule` fact and
derives both the summary `mask` fact and every `AttentionSpec.mask/window_size`
from that same value. Config-only architecture suffixes, decoder flags, window
fields and schedule tokens no longer author masks; absent or incomplete source
remains `unknown` at every layer.

The old `attention_causality_from_files` whole-file interpreter, parser wrapper,
quarantine row and parse-authority row are deleted. Its implementation and
parse-caller fingerprints, generated U3 reader inventory and structural-write
baseline shrink in the same tree. The raw `extras.sliding_window` author and
its three debt/writer pins are also deleted. Exact config dependencies are
joined back to their owner-qualified U1 occurrences; legacy mask declarations
which the enacted code does not consume are recorded as scoped non-deciding
reads rather than global bare-key exemptions.

Real parser controls now project exactly: Llama uniform causal; BLOOM uniform
causal; Gemma-2 alternating sliding-4096/global; GPT-OSS alternating
sliding-128/global; Qwen3 uniform causal even when an unused positive window is
injected; Qwen2-VL uniform causal. MusicGen and Qwen3.5 remain honest unknowns
instead of borrowing a majority/family answer. All eight high-risk witnesses
pass every blocking Sable net, and parser/fact/IR equality is pinned directly.

The required BERT control is now closed through neutral boundaries rather than
an identity branch. `self_method_return.py` proves the exact same-class helper
call, Python argument binding, unique unguarded return lanes and caller-side
destructuring. The mask reader then binds the selected returned lane through
the exact `BertModel` -> `BertEncoder` call and into the repeated `BertLayer`
formal. The owner graph separately preserves a local conditional constructor
alias as rival candidates and selects one only from an exact literal
constructor actual/default. At a block with multiple attention children, the
score join admits only the child whose exact Q/K/V lineage proves self
attention; the replacement-cross child cannot certify or block that lane.

Installed BERT now resolves 12 bidirectional encoder layers when
`is_decoder=False` and 12 causal decoder layers when `is_decoder=True`, with
`num_hidden_layers` transported through the exact parent-to-stage config
address. The parser projects both through the same `mask_schedule` fact used by
all other models. Dynamic/reassigned constructor aliases, forged helper lanes,
disconnected formals and cross-attention replacement lanes remain unresolved.
Single-child BLOOM and T5 paths retain their exact formal-to-score proof even
when their fused projection layout is outside the stricter three-affine Q/K/V
classifier. No BERT/class/field spelling selects a mechanism.

### U8-D17 — OCCURRENCE-EXACT MIXER PLACEMENT (local, not pushed)

`evidence/mixer_schedule.py` replaces the parser's config-authored mixer
schedule with one exact proof chain:

`range(exact layer count)` -> repeated-block constructor index -> exact child
construction selector -> exact block-forward child invocation -> that exact
child's positive mechanism proof.

Ordinary attention is classified per occurrence by the positive
attention-computation census. Gated-delta is classified by a new
`gated_delta_geometry_at_occurrence` boundary. The schedule therefore does not
union sibling modules and does not collapse two construction sites merely
because they instantiate the same class. Every resolved layer round-trips
through the selected construction decision, invoked child occurrence and
mechanism evidence. An unknown selector value, short list, constructor/forward
disagreement, opaque child, multiple live candidates or incomplete index
transport fails the whole schedule; none becomes a conventional attention
layer.

The transformer parser consumes the schedule's exact config dependencies and
writes one typed `decoder.attention.mixer_schedule` fact. Every
`AttentionSpec.kind` is projected from that schedule plus the independently
qualified U6 geometry fact for the selected mechanism. The qualification net
compares the fact against every layer occurrence, so swapping two otherwise
valid mechanisms is a blocking mismatch. Config tokens, integer labels and
field/class spellings remain operands only.

The former authorities are deleted: `layer_schedules.yaml`,
`layer_types.yaml`, their loaders, `_normalize_layer_schedule`,
`_mixer_kind_for`, semantic mixer maps, frequency/list synthesis and the dead
compression alias. Config-only `attn_type_list`, `block_types`,
`attention_types`, dense intervals, `layer_types`, `compress_ratios` and MoE
membership lists now author no structure. Compressed-attention geometry stays
unknown until a separate exact operation reader exists.

The same cutover removes a remaining positional lie. A gated-delta mechanism
does not prove that no independent positional operation exists, and the old
positional reader used familiar `layer_types` tokens to invent NoPE. That
branch is deleted. Qwen3.5's mixer placement remains exact, while positional
application is honestly unknown until its independent per-occurrence
application schedule is complete. NoPE is never inferred from a mixer token or
from the absence of a detected RoPE call.

Permanent controls include installed Qwen3.5 and Qwen3-Next hybrid sources,
plain Llama with injected mixer-looking tokens, integer dispatch, complete
class/field renaming, the same attention class at two construction sites,
constructor/forward disagreement, short/unknown schedules, an opaque recurrent
candidate, and an exact fact-versus-layer placement mismatch. U8-D does not
claim MoE/dense selection, per-layer QK/KV geometry, KV sharing,
cross-attention or auxiliary module placement; those remain U8-E/F.

### U8-F18 — CROSS-ATTENTION + PER-LAYER SIDE INPUT (local, not pushed)

Replacement cross-attention is now an exact per-occurrence schedule.  The
reader joins the heterogeneous repeated-block selector to the selected block's
graph-authoritative attention child and proves Q uses one callable formal while
K/V projections use the other.  MusicGen additive cross-attention remains a
separate exact two-attention construction proof.  One typed
`decoder.attention.cross_attention_schedule` fact owns both shapes; a config
list without the two source mechanisms is powerless.  Real Mllama resolves its
40-layer schedule with replacement layers 3, 8, 13, 18, 23, 28, 33 and 38;
bare source-unaddressable text configs remain unknown.

Per-layer side-input rendering is no longer activated by
`hidden_size_per_layer_input`.  `evidence/per_layer_side_input.py` proves the
exact model-stage tensor, its exact loop-indexed repeated-call operand and the
block's gate -> activation -> multiply -> projection -> norm -> state-add
chain.  Only then may the exact width/vocabulary paths be consumed and one
typed `decoder.per_layer_embedding_pathway` fact project the pathway.  Real
Gemma3n resolves; Gemma4 and Llama do not.  Removing the multiply, removing the
layer index, or injecting plausible dimensions into Llama cannot create it;
complete operational-field renaming preserves the result.

The old config-only MTP renderer has also been switched off.  Installed
DeepSeek-V3 and GLM sources declare `num_nextn_predict_layers` but do not
construct or execute the detailed norm/embedding/concat/projection/block/shared-
head pipeline that was drawn.  Those declarations are now audited but
powerless, and the five raw MTP extras/debt/writer entries are deleted.  U8-F
was not accepted as evidence. A generic HF-shaped positive now proves the full
construction/execution/sharing path and authors one typed `mtp_modules` fact.
The architecture card consumes that fact and emits a strict real-consumer
receipt. Its inner repeated block remains opaque: equality of the block class
does not identify which occurrence in a heterogeneous main schedule may donate
internals. MusicGen's codebook-stream fact likewise carries its expected value
and status from consumption to the real architecture receipt; the card calls
them parallel embedding banks and does not invent codec semantics.

### U8-G19 — PARITY, DEBT RETIREMENT AND QUALIFICATION CLOSURE (local, not pushed)

Embedded text towers no longer receive manual norm or FFN-source relay
arguments. The recursively parsed `LayerSpec` is the sole norm authority, and
FFN provenance remains with the typed reader/fact rather than a caller-supplied
file/owner hint. Standalone and embedded schedules are compared occurrence by
occurrence; T5's one relative-bias layer followed by 23 unresolved layers is no
longer flattened into a false homogeneous summary.

RoPE initialization is now a separate exact boundary. It follows the selected
attention occurrence into the exact initializer return/buffer/local-helper
dependency closure and consumes theta only when that initializer actually uses
the exact config path. The boundary also follows an explicitly imported
framework registry only after the selected callable independently proves the
inverse-frequency formula and all of its required operands. Framework config
normalization is source-proven rather than alias-declared: the exact config
class, base/mixin closure, post-init conversion, legacy-key pops, runtime-map
writes and selector default must all be indexed. Llama earns direct theta;
GPT-OSS and DeepSeek retain exact external-YaRN dependencies; Granite's raw
legacy theta/scaling become runtime `rope_parameters` only through the proved
framework conversion. An altered conversion or familiar selector routed to a
different callable stays unknown. The real Q/K application drill emits the
corresponding receipt, including MLA's separate query/key RoPE consumers.

DBRX exposed a distinct and deliberately uncollapsed boundary. Its exact
applied rotation crosses a two-hop block-wrapper→attention constructor path and
a fused QKV lane containing a value-preserving clamp; both are now proved
structurally. The applied frequency initializer uses the framework's exact
source literal default `10000`, while the checkpoint's nested
`attn_config.rope_theta=500000` is copied into attention state but is not an
operand of that rotation. It remains one owner-exact U11 classification row:
the attention object is also passed to a selectable backend, so absence of a
direct field read is not yet a sound global deadness proof. U8 does not hide or
project the contradictory declaration.

All U8 `StructuralDebt` rows are retired. The live register contains zero U8
rows; the DBRX declaration above is an explicit U11 carry-forward, not U8
semantic debt. The generated config census contains zero standing occurrences
outside registered exact debt. This
does not let parser consumption certify rendering: parse-time obligations stay
unreceipted until the actual HTML consumer emits the matching receipt.

The closed registry now has two explicit evidence populations. The frozen
corpus remains the preservation population. A separate qualification
population runs exact HF and HF-shaped frontier sources through the production
parser for mechanisms absent from the gallery (heterogeneous head schedules,
per-layer side inputs and MTP). It supplies IR/fact rows, never a hand-written
fact-name allowlist; a registered fact that neither population really emits is
still blocking stale registry debt.

Focused receipts at this checkpoint: U8 semantic matrix **232 passed**;
registry contract/closed-world matrix **30 passed**; reverse-fabrication and
reader-exception ratchets **11 passed**. The broad authority, full-suite,
preservation and representative-unfold brackets remain required before U8 may
be proposed for acceptance. No re-bless, commit or push is authorized by this
checkpoint.

Post-checkpoint hardening receipts: position initialization/schedule/geometry
plus StructuralDebt **81 passed**; blocking Sable controls are clean for real
Granite, DBRX, GPT-OSS, DeepSeek-V3 and Llama. MusicGen remains covered by the
dedicated audio/mixer suite because its Transformers validation logger disrupts
the ad-hoc console probe. No manifest/gallery re-bless, commit or push has been
performed.

Final pre-bless hardening exposed two more boundaries and closed them without
weakening a gate. First, Qwen2-VL's exact `text_config` component can now be
addressed even when the unavailable outer wrapper cannot be resolved: the
component key and selected config path must agree exactly, and no wrapper
construction claim is invented. Its ordinary decoder position reader still
does **not** claim multimodal RoPE. The `rope_parameters` mapping is classified
only as an address namespace, while its four exact child operands
(`mrope_section`, `rope_theta`, `rope_type`, legacy `type`) remain visible,
owner-exact U9 projection debt for the wrapper/modality coordinate mechanism.
They cannot author a U8 decoder position fact or be hidden by the projector
audit.

Second, the old heterogeneous-encoder fixture was found to be a config-authored
counterexample: it pointed at ordinary Llama source and injected an alternating
`layer_types` list which that source never reads. U8 correctly returns one
uniform causal group. The two former preservation assertions are now permanent
anti-fabrication controls: familiar config tokens cannot split a tower or alter
its detected period without an exact enacted mask-builder schedule.

Current unchanged-tree semantic receipts: position **133 passed**; attention,
mask and score mechanics **195 passed**; FFN/mixer/layer selection **91 passed**;
authority, receipts, debt and structural writers **152 passed**; representative
cross-model parser/render controls **50 passed**; coverage invariants **6
passed**; the two corrected heterogeneous anti-fabrication controls **2
passed**. `git diff --check` is clean. The full non-preservation suite is the
last engineering bracket in progress; preservation remains deliberately held
for Soumil's explicit visual/manifest decision.

### U8-G20 — FULL-CORPUS OUTLIER CLOSURE (local, not pushed)

The first corpus-wide non-view regression sweep exposed four independent
boundary losses. They were fixed at shared evidence/address boundaries rather
than by model or family exceptions:

- embedded text-tower attention groups had lost the exact occurrence/file
  coordinate already proved by the mixer schedule. `submodel_spec` now projects
  an attention source only when every layer in that group has one identical
  exact mixer-decision owner; mixed or incomplete groups remain unqualified.
  This closes Flux2's nested attention conformance without searching by class
  name or borrowing a sibling source;
- some RoPE helpers rotate only a prefix of Q/K internally and concatenate the
  untouched suffix. The position-application reader now accepts that algebra
  only when both lanes use complementary slices at one identical boundary,
  recombine prefix then suffix with `cat(..., dim=-1)`, and the rotated result
  actually depends on the prefix. Gap, overlap, reversal, wrong axis and
  one-lane-only poisons all fail;
- a framework base may install `self.config`, after which a parent constructs a
  rotary child from that stored value. `FrameworkConfigChildRelay` proves this
  one exact address edge from the closed framework-storage protocol, the exact
  parent occurrence and construction actual, then delegates to the canonical
  child-binding resolver. It neither changes `OwnerGraph` nor treats a field
  spelling as evidence. Conflicting bindings, unrelated/rival actuals and
  cross-site DTO forgeries fail. StableLM therefore retains exact initializer
  operands without adding a StableLM path;
- local as well as imported inverse-frequency initializers now enumerate their
  exact optional config operands. Once the enacted nested
  `rope_parameters.partial_rotary_factor` input is proved, a duplicate legacy
  top-level spelling is classified as an unselected alias occurrence rather
  than globally ignored. GLM and StableLM are clean without converting config
  presence into a mechanism claim.

Qwen3.5 is deliberately not over-claimed. Its ordinary text rotation algebra
is proved, but the multimodal wrapper owns the phase-coordinate construction,
so the exact `mrope`/theta/type/partial declarations remain owner-qualified U9
`StructuralDebt`. Those rows have explicit deletion conditions and cannot
project a U8 decoder position fact.

Permanent boundary receipts after this hardening: owner/framework/application
**116 passed**; position semantics **74 passed**; embedded diffusion/submodel
projection **114 passed**; structural debt/writer/registry authority **75
passed**. The one-time parallel scan of every blessed witness passed with **zero
non-view findings** in **458.17s** and was removed afterward so it does not add
eight minutes to every routine test run. `git diff --check` and the affected
module compile gate are clean. Exhaustive suite and preservation/artifact
approval remain the final gates; this section does not mark U8 `DONE`.

The diagnostic exhaustive run completed after the outlier fixes: **3,151
passed, 14 skipped, 2 expected xfailed**. Its only **30 failures** were the two
deliberately blocking acceptance mechanisms: all 29 parametrized canonical
preservation witnesses still cite the pre-U8 manifest, and the Sable corpus
stops at AuraFlow's first stale SVG lock. There were zero engineering-test
failures outside those acceptance locks. Because this execution record was
edited while that diagnostic run was active, it is evidence about the failure
set but is not claimed as an unchanged-tree final receipt.

A candidate manifest was generated only under `/private/tmp`; no blessed file
was touched. Against the reviewed pre-U8 manifest, all 29 witnesses change
`ir`, `ledgers`, `html_meta` and Sable evidence; 20 change expanded output and
18 change parameter estimates. AuraFlow's only visual change is its final
layer-map architecture occurrence (15 views remain 15): both layer groups now
state `position unresolved`; topology, group counts, storage uncertainty and
ordering are unchanged. The remaining model-specific view deltas are the
already-recorded U8 position/mask/mixer/cross-attention/auxiliary honesty
changes. A manifest/gallery rebuild still requires Soumil's explicit approval,
after which the final gate must run on one frozen tree.

### U8-G21 — ARTIFACT RULING AND GUARDED RE-BLESS (approved)

Soumil approved the reviewed U8 honesty delta and the 29-witness artifact
transition.  This approval covers the already-audited position, mask, mixer,
cross-attention and auxiliary-view changes; it does not authorize unrelated
visual changes or a weaker preservation gate.

Every corpus entry was regenerated through the production Sable path with
`render_images=True`.  A fixture was blessed only after its mechanical report
passed, its oracle was present and its review state was explicitly set to
`CLEAN`.  All 29 fixtures and their occurrence-exact galleries completed that
guarded path.  The canonical expected manifest and the legacy shadow baseline
were then rebuilt independently from the reviewed corpus; both contain exactly
29 witnesses.  The final committed-tree verification receipt is still required
before U8 may be called complete or pushed.

### U8-G22 — FINAL COMMITTED-TREE RECEIPT

U8 closed on commit `fd20ac4e66260a1c7fcfd8a7d6a8885fb8b6c7c2`.
The detached-worktree coordinator reported: static clean across 151 changed
Python files; 3,197 tests collected; focused 245 passed; U2 authority 44
passed; preservation 52 passed with zero drift; full suite 3,085 passed, 14
skipped and 2 expected xfailed with zero failures.  Every lane's complete-tree
fingerprint was identical before and after.  The clean-checkout gallery poison
also proved every newly occurrence-named reviewed PNG is committed rather than
available only through a local ignored artifact.

This receipt completes U8's source-evidence migration and reviewed artifact
transition.  It does not complete later U9 coordinate/modality ownership debt
or authorize family/config-driven architectural fallbacks.
