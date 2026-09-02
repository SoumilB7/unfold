# The next product architecture

## Product destination

The next version of model-unfolder is not a larger collection of model-family
templates. It is a general evidence compiler for neural architecture.

A new model should become accurate when its shipped declarations express
mechanisms the compiler already understands. A genuinely new mechanism should
require one new mechanism interpreter and projection contract—not a family
renderer, identity table, or copied parser branch.

This destination is bounded. It targets the official Transformers and Diffusers
mechanism surface in explicitly supported versions, plus compatible custom code
that the source index can resolve. It does not promise to understand every
arbitrary repository. Unsupported code remains visible as unresolved.

## Target compiler stages

```text
RawProgramIndex
  syntax, classes, calls, assignments, branches, construction, spans

BoundProgramGraph
  exact component instances, owners, call/construction relationships

EvidenceGraph
  typed mechanism facts, config bindings, premises, completeness, failures

SemanticGraph
  canonical architecture without presentation-specific deletion

ProjectionPlan
  selected truthful detail for architecture, drills, cards, JSON, and params
```

Each stage has a narrower responsibility and may not import authority from a
later stage.

## Product capabilities this unlocks

### Unseen-model generality

Class and repository names locate source only. Shared construction/dataflow
mechanisms determine architecture. Equivalent implementations normalize to the
same semantic representation even when their names and config spellings differ.

### One drill system

An FFN in the main decoder, a diffusion encoder, a vision tower, or a secondary
stack is the same semantic mechanism with a different owner and surrounding
context. They use one evidence interpretation and one drill projection, not
separate hand-authored stories.

### Honest partial understanding

The product can show established structure while marking one unresolved
mechanism, owner, or projection as unknown. It does not need to choose between a
fully guessed diagram and total refusal.

### Extensible detail policy

Raw evidence remains lossless. Product views select conceptual operations and
group incidental tensor plumbing at the projection boundary. A transpose may be
available to a deep trace without becoming a dominant architecture block.

### Symmetric trust

Every evidenced fact owes a projection, and every projected claim owes evidence.
This applies equally to HTML, JSON, cards, block drills, parameters, and future
output formats.

## What must disappear to reach it

- family/model identity selecting architecture;
- parser branches that also hand-author prose and layout;
- raw config/YAML directly producing structural cards;
- renderers and parameter estimators independently inferring facts;
- parallel AST scanners that disagree about the same source;
- default-to-familiar behavior after ambiguity or missing source;
- permanent “new rail plus old fallback” compatibility paths.

## Delivery strategy

The product is reached mechanism by mechanism while preserving existing correct
renderers:

1. finish exact evidence ownership and unknown-safe substrate;
2. build the shared program/binding/evidence graph boundaries;
3. convert one mechanism end to end through all projections;
4. prove equivalent, collision, missing-source, and unseen-shape cases;
5. delete or quarantine the old authority path;
6. repeat until domain adapters contain acquisition glue rather than parallel
   architectural intelligence.

Every cutover must shrink an old authority path. Adding a second reader,
projection ledger, or compatibility route without a deletion boundary is not
progress toward this architecture.
