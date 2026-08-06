# U8 — Exact position, mask and layer-schedule execution plan

Status: **ACTIVE — U8-A and U8-B application/factor substrate shadow-green; U8-B geometry and non-RoPE mechanisms next**
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
