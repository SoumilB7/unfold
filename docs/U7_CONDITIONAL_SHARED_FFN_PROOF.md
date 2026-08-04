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
