# U3-F0 — Exact Repeated-Child Address Reconnaissance

> **Status:** binding reconnaissance for U3-F1/U3-F2.  This document authorizes
> only neutral address evidence.  It does not authorize a “decoder block,”
> “layer stack,” attention, FFN, positional, or rendering classification.

## 1. Question

Starting from an already resolved component root (D0) and declared model-stage
occurrence (B1), can the existing neutral evidence prove the exact child
occurrence repeatedly invoked by the model-stage forward?

The required proof chain is:

```text
resolved model-stage occurrence
  -> authoritative container construction record
  -> exact forward loop and loop-target binding
  -> exact call through that target inside the loop body
  -> exact element construction template
  -> exact OwnerGraph child occurrence
```

Every arrow must be backed by an authoritative ProgramIndex/OwnerGraph record.
No class name, field name, model type, family, role vocabulary, checkpoint
value, or “most model-like child” may complete the chain.

## 2. Existing boundary coverage

The current substrate proves only this narrow loop form:

```python
for child in self.container:
    hidden = child(hidden)
```

It deliberately leaves sliced iterables and Python iteration adapters unresolved.
That makes only one of the nine actual transformer model-stage stacks directly
reachable today.  Qwen2-VL is a separate recursive-modality case: B1 resolves
`Qwen2VLModel`, while the repeated text stack lives inside an exact nested text
child.  It belongs to U3-G and must not be laundered into U3-F by scanning the
whole file for its loop.

| witness | observed loop form | current result | missing neutral proof |
|---|---|---|---|
| stablelm-2-1-6b | direct `self.<container>` | template | none |
| deepseek-v3 | sliced `self.<container>[:limit]` | unresolved | slice preserves the same base container |
| glm-4-5 | sliced `self.<container>[:limit]` | unresolved | same |
| llama-7b | sliced `self.<container>[:limit]` | unresolved | same |
| olmo-2-1124-7b | sliced `self.<container>[:limit]` | unresolved | same |
| bloom | `enumerate(self.<container>)` | unresolved | tuple target ↔ enumerated element binding |
| gpt-oss-20b | `enumerate(self.<container>)` | unresolved | same |
| qwen2-vl-7b-instruct | no repeated loop in the B1 occurrence | incomplete / U3-G | nested text owner must be resolved first |
| gemma-2-2b-it | `enumerate(self.<container>[:limit])` | unresolved | enumerate binding + slice base |
| qwen3-8b | `enumerate(self.<container>[:limit])` | unresolved | enumerate binding + slice base |

This is a syntax/protocol gap, not ten model gaps:

- direct container iteration: 1;
- sliced direct iteration: 4;
- `enumerate` over a direct container: 2;
- `enumerate` over a sliced container: 2.

After the neutral U3-F1 change, the unpatched corpus proves one exact
template-to-OwnerGraph-child join for all nine model-stage stacks:

| witness | iteration proof | exact graph child |
|---|---|---|
| bloom | enumerated | `BloomBlock` |
| deepseek-v3 | sliced | `DeepseekV3DecoderLayer` |
| gemma-2-2b-it | enumerated+sliced | `Gemma2DecoderLayer` |
| glm-4-5 | sliced | `Glm4MoeDecoderLayer` |
| gpt-oss-20b | enumerated | `GptOssDecoderLayer` |
| llama-7b | sliced | `LlamaDecoderLayer` |
| olmo-2-1124-7b | sliced | `Olmo2DecoderLayer` |
| qwen3-8b | enumerated+sliced | `Qwen3DecoderLayer` |
| stablelm-2-1-6b | direct | `StableLmDecoderLayer` |

The names in the last column are audit output only.  No resolver queries them.

## 3. Smallest lawful prerequisite (U3-F1)

Extend the neutral invocation resolver with an exact **iteration binding** for:

1. direct `self.<field>`;
2. a slice whose exact base is `self.<field>`;
3. the Python built-in `enumerate(...)` applied to either form.

The binding must preserve:

- the authoritative `LoopObservation`;
- the complete iterable `ExprNode`, including the slice expression;
- the exact container `ContainerAddress`;
- the exact target variable that receives the container element;
- whether the relationship is direct, sliced, enumerated, or
  enumerated+sliced;
- the exact call through that target inside the loop body.

It must not:

- evaluate slice bounds or a config value;
- infer count, role, block type, or execution completeness;
- expand symbolic repetition into N runtime occurrences;
- treat `zip`, `ModuleDict`, `Sequential`, a dynamic wrapper, or an unknown
  callable as equivalent to `enumerate`;
- select the first tuple target, container, element, candidate, or graph child.

Unsupported adapters remain typed unresolved.  This unit adds no whole-callable
completeness claim.

## 4. Repeated-child boundary (U3-F2)

After U3-F1, a neutral resolver may join each positively proven repeated
invocation template to the authoritative OwnerGraph.

A resolved result requires:

1. a resolved D0 component root;
2. a resolved B1 model-stage occurrence;
3. a B2 inventory for that exact occurrence;
4. a positively proven invocation template;
5. one uniquely resolved element construction candidate;
6. one graph child whose parent occurrence, container field, and final
   construction-site identity exactly match the template.

The resolver returns:

```text
resolved | ambiguous | incomplete | failed
```

- `resolved`: exactly one child occurrence is proven; all contributing
  templates and spans are retained.
- `ambiguous`: at least two exact graph children remain viable; every rival is
  retained.
- `incomplete`: no positive template proves the address, or an iteration form
  remains unsupported.  This is not `absent`, because the execution substrate
  is open/non-exhaustive.
- `failed`: D0/B1/index/graph consistency is broken or a carried record cannot
  round-trip to its authority.

The resolver is neutral.  The later transformer reader may consume its result
as the exact repeated model-stage child, but the resolver itself must not call
the child a decoder layer.

## 5. Mandatory counterexamples

U3-F1/U3-F2 are not acceptable without:

- direct iteration;
- sliced iteration with literal and dynamic bounds;
- `enumerate` direct and sliced;
- reversed tuple targets;
- nested/shadowed loops;
- call before/after the loop with the same spelling;
- two containers in one model stage;
- same child class constructed at two different sites;
- one container with heterogeneous/rival/dynamic element candidates;
- sibling owner with the same field and target names;
- `zip`, `ModuleDict.items`, `Sequential`, helper/factory, and indexed access
  remaining unresolved unless independently proven;
- broken/partial source;
- file-order reversal;
- complete class, field, and local-variable renaming;
- forged cross-index, cross-root, cross-owner, cross-field, cross-loop, and
  cross-call records rejected by DTO closure.

## 6. Acceptance boundary

The achieved output is an exact graph-authoritative repeated-child occurrence
or typed ambiguity/incompleteness/failure.  It is not a semantic mechanism
classification and changes no parser, fact, IR, renderer, parameter estimate,
manifest, or gallery.

`embedding_stage_norm_from_files` remains blocked until this boundary and a
separate code-proven norm-kind classifier both exist.
