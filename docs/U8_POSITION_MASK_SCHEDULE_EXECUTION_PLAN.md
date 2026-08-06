# U8 — Exact position, mask and layer-schedule execution plan

Status: **ACTIVE — U8-A selector substrate shadow-green; U8-B next**
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
