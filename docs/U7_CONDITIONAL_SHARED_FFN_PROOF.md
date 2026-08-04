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
