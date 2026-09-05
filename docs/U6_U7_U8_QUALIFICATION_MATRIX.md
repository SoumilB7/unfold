# U6/U7/U8 qualification matrix and independent audit

Status: **ACTIVE AUDIT RECEIPT — U6/U7 are closed; U8 exact position, mask,
mixer, cross-attention and auxiliary-module authorities are cut over locally,
and the 29-witness non-view audit is clean. Final exhaustive/preservation
acceptance and Soumil's artifact decision remain.**

This document is the binding review matrix for the current U6/U7/U8 pass.  It
does not promote U8 to `DONE`, and it does not let a green corpus stand in for a
negative proof.

## 1. Qualification law

A structural value may reach the canonical spec only when all of these are
true:

1. an exact source occurrence owns the mechanism;
2. the source expression that selects the value is identified exactly;
3. every value-bearing config operand is resolved at its exact component path;
4. the instance carries an owner-qualified typed fact with the same value;
5. every downstream surface derives from that canonical value; and
6. missing, conflicting, heterogeneous or unsupported evidence remains
   unknown at the affected occurrence.

Registration of a fact *kind* is not evidence for a fact *instance*.  A
plausible checkpoint number is an operand candidate, not architecture.

## 2. Runtime qualification matrix

The blocking `qualified_projection_values` Sable check is implemented by
`model_unfolder/evidence/qualification.py`.  Its current cutover is deliberately
owner-scoped to the transformer adapter's decoder.  Diffusion transformer and
UNet layers carry `extras.diffusion` and remain U10/U11 owners; borrowing the
transformer decoder's fact owner would be cross-altitude laundering.

| Owner | Typed fact | Canonical structural fields | Required consumers | Unknown behavior |
|---|---|---|---|---|
| `decoder.attention` | `head_geometry` | `kind`, `num_heads`, `num_kv_heads`, `head_dim`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` | spec, opgraph, card, expanded JSON, params | withhold every unproved member; an ordinary mechanism may retain exact counts with `head_dim=None`; an MLA lane may not reconstruct one missing auxiliary width from its siblings |
| `decoder.attention` | `gated_delta_geometry` | `num_kv_heads`, `num_heads`, `head_dim`, `v_head_dim`, `conv_kernel_size` | spec, opgraph, card, expanded JSON, params | omit the recurrent lane geometry independently of ordinary attention; params must omit rather than apply the ordinary Q/K/V/O formula until U14 supplies the recurrent formula |
| `decoder.ffn` | `intermediate_size` | `intermediate_size` | spec, opgraph, card, expanded JSON, params | omit ordinary/shared width |
| `decoder.ffn.expert` | `expert_intermediate_size` | `expert_intermediate_size` | spec, expert graph/card, expanded JSON, params | omit expert width independently of ordinary width |

The runtime gate compares the typed instance fact to the canonical per-layer
spec values and blocks:

- a projected value with no instance fact;
- a fact/spec value mismatch;
- a fact that claims a value every spec withholds; and
- heterogeneous layer values laundered through one global fact.

The five consumers are also covered by the existing U4 cross-surface,
projection-audit and parameter tests.  `qualification_findings()` does **not**
render those surfaces at runtime: it validates the single canonical spec they
must consume.  `surfaces` is the mandatory consumer-test inventory.  A future
renderer is not qualified merely because its name appears there; it needs a
consumer poison or a real projection receipt.

## 2.1 Evidence qualification matrix

| Family | Exact owner / mechanism proof | Value proof | Ambiguity and negative law | Positive controls | Rival / negative controls | Runtime cutover |
|---|---|---|---|---|---|---|
| ordinary split/fused attention | exact repeated block → invoked attention-compute child → exact affine storage and Q/K/V flow | exact count paths plus exact shared projection/reshape factor | rival active/possible attention children are ambiguity; one uniquely positive conditional lane remains a positive mechanism, not a universal schedule claim | Llama, BLOOM, Qwen3, GPT2, Qwen2-VL | source missing, conflicting aliases, two conditional attention children, unsupported factor | blocking |
| MLA | exact compressed/expanded K/V projection pair, split and live attention flow | independently bound heads, latent ranks, noPE/RoPE/V widths | no auxiliary dimension borrows from `head_dim`, `rope_dim` or a sibling dimension | DeepSeek-V3 | incomplete latent pair or missing auxiliary operand | blocking |
| gated-delta | exact Q/K/V split/reshape, repeat ratio, Conv1d kernel and recurrent terminals | five exact config operands | never borrows the ordinary attention fact; hybrid occurrences remain separate | Qwen3.5 | ordinary-only lane and cross-mechanism laundering poison | blocking |
| ordinary FFN width | exact FFN owner and live input/output projection mechanism | output-projection input expression equals every proved input-projection output | guarded/rival/unequal widths fail; a literal width is code-proven | GPT-J, CodeGen, GPT2, BLOOM, Llama | source missing, unequal gate/up widths, unrelated Conv1D | blocking |
| routed-expert width | exact routed storage and live fused gate/up/down parameter path | one literal two-lane fused factor equals the down-parameter dimension | flattened/split storage is not enough to assign expert-count versus width axes | DeepSeek-V3, GLM-4.5, GPT-OSS | DBRX flattened storage; dense Llama | blocking |
| ordered coordinate origin | exact callable expression reaches exact `torch.arange`; learned/rotary consumers separately prove use | numeric literals, exact tensor shape coordinate, or exact optional-cache length protocol | arbitrary tensors/method names/cumsum are powerless; this fact alone does not classify a positional scheme | synthetic learned absolute, GPT2, exact optional cache | arbitrary tensor, unguarded cache method, wrong receiver/default/arguments | shadow only — U8 not complete |
| mask schedule and geometry | exact framework builder → repeated block formal → exact score application; alternating selectors use the exact enumerate index, uniform schedules cite the one enacted builder, helper-return lanes and nested stages retain exact call/formal ownership, and geometry cites the enacted builder's exact stage-config actual | exact config sequence/count plus protocol-selected direction/window/chunk path with typed checkpoint/class-default provenance | config tokens only select already-proven builders; same-name fields, foreign/rival kwargs, unknown token, short list, wrong index, dynamic constructor alias and runtime-supplied mask maps cannot fabricate a result | Gemma-2 and GPT-OSS alternating causal/sliding with windows 4096/128; Llama/BLOOM uniform causal; T5 encoder/decoder; BERT encoder bidirectional and decoder causal through the same helper/nested-owner source; synthetic chunked causal | Qwen3 unused injected window, constant index, disconnected/forged helper lane, replacement-cross child, foreign config actual, rival kwargs and zero geometry | blocking for exact resolved executions; incomplete owners remain unknown — U8-C source authority is closed, final U8 bracket pending |

## 3. U6 — attention geometry qualification

### Implementation

- `evidence/attention.py::_fused_equal_head_binding` now emits the exact
  per-head reshape dimension as `common_factor`.  It formerly emitted the
  packed projection's hidden-width factor, while the grouped-fused protocol
  emitted per-head width.  That inconsistent DTO made BLOOM report `14336` as
  one head dimension.
- `evidence/attention_geometry.py` evaluates that exact factor through the
  exact attention occurrence and records only its real operands.
- `adapters/transformer/parser.py` no longer reads head counts/dimensions as
  free-standing structural truth.  The mechanism supplies exact count paths;
  the geometry reader supplies the exact per-head factor.
- The parser consumes the source-selected spelling.  Thus BLOOM's `n_head` is
  accepted even when the config class also exposes an equal canonical alias;
  no model or family branch was added.
- Modeling code may read a config-class runtime property while the checkpoint
  serializes an audited input spelling (`hidden_size` <- `n_embd`).  The shared
  evaluation view bridges it only when every present spelling agrees.  It does
  not mutate the checkpoint or create a canonical occurrence; the U1 ledger
  still records the actual spelling.  Unequal rivals remain blocking ambiguity.
- Conditional child construction is qualified at the exact occurrence.
  GPT2's absent `add_cross_attention` is retained as a `class_default=False`
  premise and excludes the cross-attention child.  A per-layer selector that
  cannot be evaluated at a symbolic repeated-block occurrence may retain one
  uniquely positive mechanism, but it can never choose among two possible
  attention children.
- A resolved MLA protocol explicitly retires only the exact generic
  `head_dim` / `num_key_value_heads` occurrences that its source does not
  consume.  DeepSeek's latent Q/K/V operands remain the sole geometry
  authority; this is a mechanism-scoped classification, not a family ignore.
- A symbolic repeated-block selector may leave the attention constructor's
  non-config formals unresolved without erasing direct config-bound geometry
  inside the exact selected class.  The evaluator therefore permits only
  direct config/literal expressions in that case; a missing formal remains
  unknown.  This restores Qwen3.5's ordinary GQA `head_dim=256` without using
  its gated-delta sibling or its layer schedule as evidence.
- Owner occurrence means component occurrence, not “same physical file as the
  wrapper.”  A permanent two-file wrapper/block control proves exact geometry
  across files inside one component; cross-component provenance still fails.
- Nested components are evaluated against their exact component document, and
  premise paths are lifted back to the root document.  Missing checkpoint
  operands supplied by the installed class retain `class_default` status.

### Qualification cases

| Case | Permanent control | Expected result |
|---|---|---|
| split Q/K/V, grouped counts | synthetic 8/2 fixture | `gqa`, 8/2, head dim 8 |
| unused plausible `head_dim=99` | same fixture | ignored; source computes 8 |
| fused equal QKV | real BLOOM | 112/112, head dim 128 |
| transposed affine (`transformers.Conv1D`) plus scalar 3-way split | real GPT2 raw aliases | MHA 4/4, head dim 16; `add_cross_attention=False` stays a class-default premise |
| fused grouped QKV | real DBRX | 48/8, head dim 128 |
| framework-normalized RoPE initializer | real Granite | legacy root theta/scaling → runtime mapping only through exact config-class/mixin conversion source |
| code-default RoPE initializer across wrapper/fused QKV | real DBRX | two constructor hops + reshape/transpose/clamp/fused split; applied theta 10000, disconnected nested 500000 retained as U11 classification debt |
| imported registry RoPE initializer | real GPT-OSS + DeepSeek-V3 | exact selector → exact YaRN callable → complete required dependency set; wrong callable/missing operand blocks |
| split grouped QKV | real Qwen3/GLM/GPT-OSS | exact GQA counts and dimension |
| hybrid ordinary/recurrent stack | real Qwen3.5 | ordinary GQA 24/4/256 and gated-delta 48/16/128 stay independently qualified |
| class-default operands in nested text component | Qwen2-Audio smoke control | MHA 32/32, dim 128, fact tier `class_default` |
| mechanism counts proven, factor unsupported | GPT-BigCode-style control | exact counts, `head_dim=None` |
| source absent | synthetic missing-source control | kind/counts/dimension all `None`; no fact |
| custom config with plausible counts only | Gemma4/ChatGLM smoke controls | geometry withheld |
| heterogeneous per-layer values | qualification poison | global fact rejected; per-occurrence facts required |
| wrapper and attention implementation in separate files | synthetic two-file component | resolves through the same exact occurrence; no same-file assumption |
| conflicting runtime-property spellings | GPT2 `hidden_size=96`, `n_embd=64` poison | head dimension withheld and U1 ambiguity remains visible |

## 4. U7 — ordinary and expert width qualification

### Ordinary FFN width

`evidence/ffn_width.py` evaluates the input dimension of the exact, already
proved output projection and confirms every proved input projection emits the
same width.  It now runs even when `intermediate_size` is present; the former
`config first, code only if missing` order let arbitrary values bypass source
qualification.

Controls cover GPT-J's conditional default, explicit GPT-J width, CodeGen,
GPT-2/Conv1D, BLOOM, Llama, nested Qwen2-Audio, missing operands, missing
source, guarded/rival sites and DTO provenance forgeries.

Symbolic repeated blocks are evaluated from their exact owner-level config
bindings even when the one-shot constructor argument environment is not
available.  This restored MusicGen's exact `ffn_dim` without inventing a value
for any unbound formal; an unbound expression still remains unknown.

### Routed-expert width

`evidence/expert_width.py` composes the existing exact routed-expert storage
proof.  It accepts only a fused parameter dimension containing one literal
two-lane factor (`2 * width` or `width * 2`) whose remaining expression is the
same structural dimension in the proved down parameter.

| Storage shape | Controls | Result |
|---|---|---|
| fused gate/up plus matching down | DeepSeek-V3, GLM-4.5, GPT-OSS | exact 2048 / 1536 / 2880 expert width and matching typed fact |
| flattened split storage | DBRX | unknown; the shape alone cannot identify expert-count versus per-expert-width factor |
| no routed expert | Llama | unknown/no expert fact |

Ordinary and expert widths are independent.  A missing expert width cannot
borrow the ordinary width, and a routed-only model cannot put its flattened
expert declaration into the ordinary lane.

## 5. U8 — coordinate-origin hardening

`evidence/position_coordinate.py` is the shared neutral prerequisite used by
learned-absolute and rotary-factor readers.  It accepts exact `torch.arange`
origins, approved shape-only wrappers, exact `None`-defaulted `arange`, and
scalar/cache-length offsets.  It rejects:

- a caller tensor merely named `position`, `coordinate` or similar;
- arbitrary `cumsum` (OPT's mask-to-coordinate relationship needs its own
  exact U8-C protocol);
- `arange + tensor`;
- hidden states passed as a trigonometric phase coordinate; and
- hidden states passed into a complex-phase producer.

The cache offset is accepted only when the receiver is the exact callable
parameter, that parameter defaults to `None`, the same conditional tests it as
`is not None`, and `get_seq_length()` has no arguments.  A shape coordinate is
accepted only from a structurally well-formed one-receiver `.shape[index]`.

The exact schedule fixture now originates coordinates with `torch.arange`;
the former fixture used an arbitrary caller tensor and therefore tested the
very false positive the new boundary forbids.

The coordinate boundary now feeds the cut-over U8 position readers. Positional
evidence, exact mask execution, occurrence-exact mixer placement,
cross-attention placement and auxiliary-module construction have replaced the
corresponding parser/config convention paths. Incomplete owners remain unknown.
The final artifact inspection and U8 acceptance bracket remain governed by
`docs/U8_POSITION_MASK_SCHEDULE_EXECUTION_PLAN.md`.

The U8-C shadow boundary now proves both alternating and uniform mask schedules.
A typed framework-config address proof closes `super().__init__(config)` →
`self.config` without treating the field spelling as evidence; exact imported
config-class ownership separately supplies only literal omitted defaults. The
layer sequence selects only source-proven mask builders and agrees with the
exact repeated-container count. A closed score-transform proof carries
GPT-OSS's mask state through concatenation and row-max translation, so its
alternating schedule resolves without an identity branch. The same installed
T5 source resolves bidirectional for the class-default encoder and causal for a
declared decoder. Exact enacted builder/config joins additionally prove
Gemma-2/GPT-OSS sliding geometry and a synthetic chunked geometry protocol.
Qwen3 remains causal under an injected plausible window because its enacted
path disables the sliding builder; runtime-supplied mask mappings remain an
explicit completeness boundary. BERT now crosses its exact same-class helper
return, destructured lane, parent-to-encoder call and repeated-layer formal;
the `is_decoder` value selects the source-proven builder and never acts as a
mask by itself. Multiple BERT attention children are separated by exact Q/K/V
lineage, while single-child fused BLOOM/T5 paths remain governed by their exact
formal-to-score transport. No family exception, raw-token semantic table or
config-presence inference was added.

## 6. Breakage and compatibility matrix

| Potential breakage | Correct handling | Evidence |
|---|---|---|
| source-present ordinary models lose real geometry | exact source path retains it | Llama, BLOOM, Qwen3, DBRX examples |
| nested text components lose defaults | evaluate exact component; classify class defaults honestly | Qwen2-Audio |
| name-blind replay loses a nested class default | reuse the original address result only through the exact self-verifying `DocumentBinding`; loose/sibling paths fail | FLUX T5 `d_kv` metamorphic control + binding forgery poisons |
| custom/source-missing fixtures retain fabricated numbers | intentional honesty delta: values become unknown | Gemma4, ChatGLM |
| fused-QKV protocols disagree on DTO meaning | producer corrected to one per-head meaning | BLOOM + DBRX pair |
| mixed dense/MoE layers borrow sibling widths | independent ordinary/expert facts; global heterogeneity blocks | DeepSeek/GLM/GPT-OSS/DBRX + poison |
| diffusion layers are judged by transformer owner | matrix scoped by typed adapter altitude, not identity | FLUX Sable control |
| arbitrary tensors become positional coordinates | shared coordinate-origin prerequisite rejects them | position absolute/factor poisons |
| optional-cache method spelling becomes position evidence | require exact parameter/default/guard/zero-argument protocol | positive cache control plus wrong-default/guard/argument poisons |
| wrapper and mechanism classes live in different files | qualify by exact symbol and component, never root-file equality | two-file geometry control |
| source runtime property and checkpoint spelling differ | agreement-only syntax bridge, then exact U1 occurrence join | GPT2 positive + unequal-rival poison |
| unknown values crash params/renderers | canonical specs use `None`; downstream unknown-safe tests remain required | U4 cross-surface suites |
| exact gated-delta geometry is counted as ordinary attention | omit recurrent attention terms with an explicit U14 carry-forward; never use the Q/K/V/O formula | parameter qualification poison |

## 7. Current verification receipt

Green on the current working tree after this independent pass:

- attention mechanism/geometry/child/storage/cache/operands/clip/softcap/sinks,
  bias, QK norm and dispatch family: **296 passed**;
- FFN mechanism/ordinary width, expert storage/width, router, norm, bookend,
  cell topology and fact-ledger family: **296 passed**;
- layer selector/schedules plus position application/absolute/factor/geometry/
  schedule family: **151 passed**;
- qualification, U4 consumer closure, opgraph, receipts and U2
  authority/register family: **209 passed**;
- every frozen corpus fixture passed **every blocking Sable mechanical net**:
  **29/29 green** (parallel, four minutes);
- parameter/fact closure spot bracket: the stale pre-U6/U7 expectation was
  corrected to the exact current omission (`ordinary/shared FFN inner width
  unknown`) and real Llama now has no fabricated attention omission;
- real parses/unfolds checked: Llama, BLOOM, GPT2, Qwen3.5 hybrid,
  DeepSeek-V3, GLM-4.5, GPT-OSS, DBRX, MusicGen and nested Qwen2-VL;
- changed-file `py_compile`, `pyflakes` and `git diff --check`: clean at the
  time recorded.

The first broad non-manifest run found two real integration gaps after 2,836
passes: MusicGen's test asserted the wrong diagnostic surface, and FLUX's
name-blind replay re-resolved a scrubbed T5 class instead of consuming the
root's already-resolved class-default address.  The test now pins the
authoritative occurrence-exact debt row; the slot-context API now accepts the
original overlay only through the exact `DocumentBinding`, rejects a sibling
path or different object, and the FLUX metamorphic test is green.

The corrected broad non-manifest rerun is green on one frozen tree:

- **2,839 passed, 14 skipped, 2 expected xfailed, 0 failed**;
- before/after tree fingerprint
  `2630ac8b2982c48cb614b0dcc0ffc0f2f437737b2a7926e2eea85bf434f823ab`
  was identical;
- the focused ownership/context bracket is **193 passed**;
- final `compileall`, changed/untracked-file `pyflakes` and `git diff --check`
  are clean.

The current production rendering path was also rerun into temporary, unblessed
directories for DeepSeek-V3, GLM-4.5, GPT-OSS, DBRX, Qwen3.5,
FLUXTransformer2DModel, SDXL and MusicGen.  Every PNG for the first seven is
byte-identical to its durable gallery.  MusicGen is the sole visual delta: its
main decoder self-attention changes from an unresolved stub to source-proven
MHA and gains one exact self-attention drill.  Its cross-attention drill still
withholds unresolved storage, scale and output-path details, and the embedded
conditioning encoder's own unresolved attention/FFN drills remain unresolved.
The architecture therefore gains only the detail that the source proof earns.

The preservation gate is intentionally red against the pre-change manifest:
all 29 hashes report the unblessed evidence/structure deltas.  No old view is
missing.  MusicGen is the only view-sequence delta (9 -> 10) because its newly
source-proven self-attention drill is now reachable.  Its conditioning card no
longer copies raw `num_heads`/`d_ff`: those stay unknown, and the exact
`root.conditioning:num_heads` occurrence is visible U9 debt until the rival
T5 encoder/decoder stage owner is resolved.  DeepSeek's head dimension changes
64 -> 192 because exact MLA Q width is `qk_nope_head_dim + qk_rope_head_dim`;
ordinary/shared FFN widths on DeepSeek/GLM remain unknown while their exact
expert widths remain independently proven.  These deltas still require
Soumil's visual/manifest decision; no blessing has been changed.

The broad suite and visual delta review are complete.  The preservation
manifest decision remains Soumil's explicit gate before any re-bless, commit,
push or completion claim.  No fixture or gallery may be re-blessed silently.

The final U8 pre-bless audit also pins two non-laundering controls. An exact
nested component address may make Qwen2-VL's text source reachable, but it does
not promote the wrapper's multimodal coordinate operands into ordinary decoder
RoPE; those four exact declarations stay registered U9 debt. Conversely, a
Llama-backed text tower with injected alternating `layer_types` stays one
uniform causal group because the enacted source never consumes that schedule.
These controls distinguish lawful component addressing from mechanism proof,
and source-proven schedules from familiar config vocabulary.

## 8. U8-D mixer-placement qualification (local, not pushed)

Mixer placement is now a separate owner-qualified fact rather than a config
token interpretation. `decoder.attention.mixer_schedule` must agree with every
projected layer kind after ordinary MHA/GQA/MQA/MLA lanes are normalized to
`ordinary_attention`; gated-delta remains occurrence-specific. Candidate
mechanism proofs, selector decisions and invocations all retain exact owner
occurrences and spans. A valid head-geometry fact cannot launder an incorrect
mixer schedule, and a valid mixer schedule cannot supply missing head geometry.

Real controls cover Qwen3.5 and Qwen3-Next mixed schedules plus uniform Llama,
BLOOM, Gemma-2, GPT-OSS, Qwen3, Qwen2-VL and MusicGen parses. Config-only
schedule and compression dialects stay unknown. The NoPE claim formerly
derived from a `layer_types` token is deliberately removed: mixer placement is
proved, positional absence is not. This is an intentional honesty delta and
must be visually reviewed before any preservation re-bless.

## 9. U8 full-corpus outlier qualification

The all-witness non-view audit found and closed four shared-boundary defects,
not four model cases: exact embedded attention provenance (observed on Flux2),
internal prefix-rotation/suffix recombination (GLM), framework-stored config
relay plus local initializer operands (StableLM), and a multimodal
phase-coordinate ownership boundary (Qwen3.5). The first three now have
positive and adversarial mechanism tests. The fourth remains exact U9 debt
because the text decoder proves rotation algebra but the wrapper—not the text
attention owner—constructs the multimodal coordinate phase.

This distinction is binding: addressing a config input does not prove its
mechanism; proving an internal Q/K formula does not transfer ownership of a
wrapper-produced coordinate; and a group receives a source coordinate only
when every represented layer agrees on that exact owner. The final parallel
scan of all 29 blessed witnesses produced zero non-view findings. It was an
acceptance probe and was deleted rather than becoming an eight-minute permanent
test. Focused permanent receipts are 116 owner/framework/application, 74
position, 114 embedded diffusion/submodel and 75 authority tests.

The post-hardening exhaustive diagnostic produced **3,151 passes** with only
the expected 29 canonical-manifest witness failures plus the first stale Sable
SVG lock. A `/private/tmp` candidate manifest measured the acceptance delta
without modifying any blessing: 29/29 `ir`/ledger/HTML-metadata/Sable surfaces,
20/29 expanded surfaces and 18/29 parameter surfaces changed. These are not
auto-approved by a green semantic audit. The final manifest/gallery rebuild and
unchanged-tree bracket remain contingent on Soumil's explicit artifact ruling.

Soumil subsequently approved that artifact ruling.  All 29 witnesses passed
the guarded Sable clean-review path and were re-blessed; the canonical and
shadow preservation artifacts were regenerated at the same 29-witness
boundary.  This is an approved qualification transition, not permission to
normalize future drift.  U8 remains under final committed-tree verification
until the closing receipt is recorded.

Closing receipt: `fd20ac4` passed static (151 changed Python files), focused
245, U2 authority 44, preservation 52/zero drift and the full 3,197-test
collection (`3,085 passed, 14 skipped, 2 expected xfailed, 0 failed`).  All
detached-worktree fingerprints were identical.  U8 is therefore `DONE`; the
explicit Qwen3.5 multimodal-coordinate rows remain later-unit debt rather than
being laundered into U8.
