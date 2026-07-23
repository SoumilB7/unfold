# U3-F3/F4 — Exact Primitive Calls and Embedding-Stage Bookend Cutover

> **Status:** binding next implementation sequence after U3-F2.  This document
> records a prerequisite discovered from the real BLOOM occurrence graph; it
> supersedes any direction that assumes every invoked module has an indexed
> `OwnerOccurrenceId`.

## 1. The discovered boundary

U3-F2 proves the repeated internal block occurrence.  BLOOM's embedding and
embedding-stage normalization are different:

```python
self.word_embeddings = nn.Embedding(...)
self.word_embeddings_layernorm = LayerNorm(...)
```

Both constructors are external PyTorch primitives.  The OwnerGraph correctly
does **not** fabricate internal nodes for them.  It retains exact
`UnresolvedChild(kind="external")` records and construction sites, while the
execution resolver currently keeps their calls unresolved.

Therefore `embedding_stage_norm` cannot be migrated by:

- guessing from `word_embeddings` / `word_embeddings_layernorm`;
- manufacturing OwnerOccurrenceIds for external classes;
- treating every unresolved field call as a primitive;
- reading `Embedding` / `LayerNorm` substrings;
- writing a local def-use interpreter beside `execution_flow.py`.

## 2. U3-F3a — external construction-call address boundary

Add an exact, neutral construction occurrence identity:

```text
ConstructionOccurrenceId
  parent OwnerOccurrenceId
  exact ConstructionSiteId
```

It is distinct from `OwnerOccurrenceId`: the latter is legal only when
`OwnerGraph.node_for()` succeeds.

Resolve `self.<field>(...)` by joining:

1. the exact caller occurrence and call observation;
2. the caller's exact class symbol;
3. exact `ConstructionSite` records for that field;
4. the graph's exact internal child or exact typed unresolved child at that site;
5. exact import binding for an external constructor reference.

Output:

```text
resolved_internal | resolved_external | ambiguous | incomplete | failed
```

An external resolution carries the exact import target (for example
`torch.nn.LayerNorm`) plus the `ImportRecord`, reference expression,
construction site, call site and spans.  Multiple imports/sites/candidates
remain rivals.  Dynamic or unsupported construction remains incomplete.

Add external addressed invocations to the existing execution-flow census and
def-use graph as a distinct node kind.  Do not create a second dataflow
interpreter.

## 3. U3-F3b — code/library-proven primitive semantics

Classify an exact construction occurrence, never a name in isolation.

Initial closed protocol:

| exact external target | mechanism |
|---|---|
| `torch.nn.Embedding` | embedding |
| `torch.nn.modules.sparse.Embedding` | embedding |
| `torch.nn.LayerNorm` | layernorm |
| `torch.nn.modules.normalization.LayerNorm` | layernorm |
| `torch.nn.RMSNorm` | rmsnorm |
| `torch.nn.modules.normalization.RMSNorm` | rmsnorm |

This table is a framework API protocol in code, not a model/family table.
Consumers must cite the exact import binding and construction occurrence.

For indexed custom norm classes, classify exact implementation math:

- exact `torch.nn.functional.layer_norm` protocol or mean subtraction:
  `layernorm`;
- mean-of-squares / reciprocal-square-root rescaling with no mean subtraction:
  `rmsnorm`;
- conflicting signals: ambiguous;
- insufficient or unsupported implementation: incomplete.

For a custom embedding, require an exact embedding operation/protocol; do not
infer from class or field spelling.

## 4. U3-F4 — embedding-stage norm reader

The reader consumes only positive proof from the shared execution graph:

```text
exact embedding construction call
  -> exact norm construction call
  -> exact U3-F2 repeated-child invocation template
```

Required:

- all calls belong to the exact B1 model-stage occurrence;
- F3b classifies the first as `embedding`;
- F3b classifies the second as `layernorm` or `rmsnorm`;
- shared versioned def-use proves embedding output enters the norm;
- shared versioned def-use proves norm output reaches the repeated-child call;
- no unsupported/ambiguous relation is promoted.

Return `ReaderResult[str]` with the norm kind and exact construction/call/dataflow
spans.  Without the complete positive chain, return typed
incomplete/ambiguous/failed—not a negative.

Rewire both:

- transformer parsing; and
- conformance

to the same reader, then delete `embedding_stage_norm_from_files` and its
whole-file AST helpers in the same commit.

## 5. Controls

- BLOOM positive: external `Embedding -> LayerNorm -> repeated block`.
- Llama/Gemma/OLMo negative controls remain no-drawing-change; absence is not
  claimed without reader-specific completeness.
- external constructor imported directly and through a module alias;
- locally/module-shadowed protocol names;
- two imports with the same alias;
- two construction sites for one field;
- internal custom RMS math and misleading class name;
- T5-style RMS math behind a “LayerNorm” spelling;
- mean-subtracting custom LayerNorm;
- conflicting/unsupported math;
- same primitive used by sibling owners;
- call before/after the repeated loop with the same local spelling;
- class/field/local renaming;
- partial and broken source;
- cross-index/root/owner/site/call forgery.

## 6. Acceptance

F4 is complete only when the legacy reader is deleted, parser and conformance
consume the one exact reader, all existing outputs are byte-identical except an
explicitly approved truth correction, and no renderer or parameter path imports
ProgramIndex/owner/reader machinery.
