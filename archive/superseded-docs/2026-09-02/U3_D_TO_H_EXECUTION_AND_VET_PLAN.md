# U3-D to U3-H — Definitive Reader-Migration and Vet Plan

> **Status:** Binding execution plan after the independent U3-B/U3-C audit.
>
> **Authority:** This document narrows `U3_RUNBOOK.md`; it does not replace the
> U3 boundaries or `U_PLAN_RISK_OWNERSHIP_AND_STOP_GATES.md`.
>
> **Objective:** Move every production model-source reader onto the one
> observation-only `ProgramIndex`, exact construction-occurrence ownership and
> `ReaderResult[T]`, then delete the parallel parsers and unsafe selection
> paths. U3 changes the evidence substrate, not the meaning or appearance of a
> model.

## 1. Current verdict

U3-A is a suitable observation substrate. The independently corrected U3-B and
U3-C now express the intended laws:

- owners are root-plus-construction-site occurrence chains, never class names;
- two constructions of the same class remain different owners;
- same-field guarded constructions are rivals, not simultaneous children;
- exact import bindings may locate symbols, while suffixes and class-name
  resemblance may not;
- ambiguous config prefixes remain ambiguous and never inherit a convenient
  parent prefix;
- factory inputs are not falsely relabelled as constructor parameters;
- reader success has exact occurrence ownership and structured provenance;
- absence, failure, incompleteness and ambiguity cannot be consumed through a
  defaulting helper.

The focused U3 tests, the affected U2 gates and the complete suite have passed
on an unchanged tree. This means the **kernel is now fit to build on**. It does
not mean U3 is complete: no production reader uses this path yet, the legacy
parsers remain authoritative, and one address boundary described below is
still missing.

## 2. Non-negotiable scope

The only lawful migrated path is:

```text
SourceBundle component address
  -> ComponentRootResolution
  -> OwnerOccurrenceId / OwnerGraph
  -> ProgramIndex observations for that exact occurrence
  -> specialized ReaderResult[T]
  -> existing fact/IR consumer
```

During U3:

- do not change a fact schema, canonical IR, renderer, projection DTO or
  parameter formula;
- do not add a model/family/class-name table, suffix chooser, best-candidate
  rule, allowlist or new debt row;
- do not make an architecture more detailed as part of a parser migration;
- do not turn ambiguity/unsupported/missing evidence into a conventional
  architecture;
- do not retain the old reader as a fallback after a cutover;
- do not parse model source outside `evidence/program_index.py` once the owning
  reader has migrated;
- do not stage unrelated user-owned files.

An intended architecture or pixel delta is **not a U3 cleanup**. Stop and assign
it to the appropriate U6-U13 mechanism unit.

## 3. Prerequisite — close the corrected B/C baseline

Land one correction commit containing only:

- `model_unfolder/evidence/component_owner.py`
- `model_unfolder/evidence/reader_result.py`
- `tests/test_component_owner.py`
- `tests/test_reader_result.py`
- the U3 status correction in `docs/U3_RUNBOOK.md`
- `docs/U_PLAN_RISK_OWNERSHIP_AND_STOP_GATES.md`

Do not stage `MODULARIZATION.md`, `TRUE_CONFIG_PLAN.md`, the U0/U1/U2 plan
documents, `architecture_atlas.html`, or any other concurrent file.

The commit must reproduce from its own committed tree. Minimum receipt:

```text
41 focused B/C tests pass
69 U3-A/B/C tests pass
identity, structural-write, registry and projection gates pass
affected evidence/config/conformance tests pass
full suite passes
tree fingerprint before == after
git diff --check and pyflakes are clean
```

This correction is already independently designed and vetted. It may be
executed mechanically, but its exact file scope may not grow.

## 4. U3-D0 — finish the component-root address boundary

### Why this is required

`resolve_owner_graph()` correctly refuses to select a root; it requires a
`SymbolId`. Production callers currently possess a `SourceBundle` component
key and a declared architecture address. Without a typed bridge, the first
migration would invent a local “find this class” helper and would likely revive
shortest-name, suffix, first-hit or whole-file selection.

### Codex-owned contract

Add a typed, observation-only result in
`model_unfolder/evidence/component_owner.py`:

```text
ComponentRootCandidate
- exact component key
- exact declared architecture spelling
- SymbolId
- class SourceSpan

ComponentRootResolution
- status: resolved | absent | ambiguous | failed
- exact component key
- declared architecture, if supplied
- resolved OwnerOccurrenceId and OwnerGraph only when resolved
- all exact rival candidates when ambiguous
- typed ProgramIndex parse failures when failed
```

Add:

```text
resolve_component_root(index, bundle, component_key, *, root_param_prefixes=None)
```

Resolution law:

1. Read the address only from
   `bundle.component_architectures[component_key]`; `bundle.architecture` is a
   lawful compatibility address for `root` only.
2. Search only `ProgramIndex` classes whose `SourceId.component_key` equals the
   requested component.
3. Match the qualified class name exactly. Exact identity is an address, not a
   mechanism claim.
4. One candidate resolves. Zero is `absent` unless an indexed parse/read failure
   prevented the lookup, in which case it is `failed`. More than one is
   `ambiguous` with every rival preserved.
5. Never use `model_type`, suffixes, substrings, shortest name, most fields,
   first file, import order or role markers.

Required poisons:

- same class spelling in two component files -> ambiguous;
- same class spelling under root and vision -> root selects root only;
- exact class missing but similar suffix present -> absent;
- one component file fails parsing while another looks similar -> failed, not
  selected;
- `bundle.architecture` cannot address a non-root component;
- same source indexed for two components remains two candidates/addresses;
- resolved root preserves supplied constructor-prefix bindings;
- empty architecture remains absent, never “pick the only class.”

**Stop gate V1:** Codex implements or directly reviews this contract. The
implementation agent must not design an alternative root-selection API.

## 5. U3-D1 — first production-reader pilot

### U3-A1 prerequisite discovered by the pilot

The first implementation correctly stopped because the original ProgramIndex
embedded names only inside selected call/dataflow/control records. That could
not prove the negative “this exact forward contains no temporal identifier.”
The Codex-owned correction adds neutral `IdentifierObservation` records:

- one exact `ast.Name`/`ast.arg` occurrence;
- exact owning and enclosing callable addresses;
- exact `SourceSpan`;
- syntactic context (`parameter`, load/store/delete, annotation, default or
  decorator);
- strict lexical-scope separation—nested callables/classes/lambdas never
  contaminate their parent;
- complete unsupported-expression/nested-scope records for the exact callable,
  so an opaque region blocks a negative instead of disappearing.

This remains observation-only. ProgramIndex does not interpret an identifier as
a temporal axis; the denoiser reader is the semantic consumer. A future reader
needing bindings represented by Python's AST as plain strings (for example an
exception alias or match capture) must stop for a separately proven token-aware
observation—it may not invent a span.

### Selected reader

Migrate `denoiser_temporal_axis_from_files` from
`model_unfolder/evidence/patterns.py` first.

It is the bounded pilot because it has one production caller, addresses the
denoiser root rather than traversing sibling owners, has a small deletion
surface, has positive video and negative image witnesses, and can be answered
after the U3-A1 identifier-observation correction. It requires no fact, IR shape
or renderer rule.

### Exact implementation

1. Add an owner-bound reader in a domain evidence module (prefer
   `model_unfolder/evidence/denoiser.py`; do not add it to the generic index):

   ```text
   denoiser_temporal_axis(index, owner: OwnerOccurrenceId)
       -> ReaderResult[bool]
   ```

2. Query the exact owner class and its exact `forward` callable from
   `ProgramIndex`. Read its `IdentifierObservation` census directly; never
   reconstruct names by unioning selected calls/dataflow, reopen a file, call
   `ast.parse`, inspect `source_segment`, or search another class.
3. The existing temporal marker vocabulary may remain temporarily as a
   specialized-reader code-shape vocabulary. It may nominate a temporal-axis
   source symbol; it may not select the owner. Its later semantic replacement
   belongs to U10.
4. Return:
   - `resolved(True)` with the exact matching observation span;
   - `resolved(False)` only when the exact forward identifier census is
     complete, the callable has no unsupported/nested lexical region, and no
     temporal observation exists;
   - `failed(missing_source/parse_failure/unsupported_syntax)` when absence
     cannot be proven;
   - never `False` for an unreadable, unsupported or unresolved body.
5. In `adapters/diffusor/parser.py::_temporal_axis`, obtain the call-local index,
   resolve component `root`, invoke the reader only for a resolved root, and
   consume only a resolved value. Existing checkpoint-declared temporal fields
   remain the lawful fallback for a non-resolved source result; no name fallback
   may be introduced.
6. Delete `denoiser_temporal_axis_from_files` and its direct AST loop in the same
   commit. Update imports and tests; no compatibility shim remains.

### Required tests

- positive video owner and negative image owner;
- class/field/local-variable renaming that preserves the same temporal
  observation gives the same result;
- same class at a sibling component cannot influence root;
- two same-spelled roots produce typed ambiguity before the reader;
- syntax failure and unsupported expression never produce `False`;
- temporal names in assignment/loop/comprehension binding positions and bare
  name statements are observed with exact spans;
- a nested callable/lambda is incomplete, not a false negative;
- a child occurrence cannot be stamped with the component root's forward;
- missing forward never produces `False`;
- an unrelated `num_frames` in a sibling/helper does not affect the exact
  owner's forward;
- current image controls remain image and current video controls remain video;
- all 26 preservation witnesses have byte-identical pixels and structural
  surfaces.

**Stop gate V2:** Hold the pilot commit for Codex vet before wider migration.
Codex checks owner exactness, negative-proof completeness, old-path deletion and
the no-pixel-delta receipt.

## 6. U3-E — leaf-reader migration cluster

After V2 approval, migrate only single-owner leaf readers whose answers use
record families already present in ProgramIndex. One reader per commit.

Candidate ordering must be generated from `U3_READER_INVENTORY.md` using:

1. one production caller before multiple callers;
2. exact owner already available before a new traversal;
3. no raw checkpoint walk inside the reader;
4. smallest old AST/deletion surface;
5. at least one positive and one negative corpus witness;
6. no new mechanism/fact/IR/projection contract.

For every reader:

- change the API from `(files, architecture, ...)` to `(index, occurrence, ...)`;
- return `ReaderResult[T]`;
- preserve exact source spans and config paths;
- delete its raw parse/cache/helper path in the same commit if no other live
  reader uses it;
- add the six frontier controls: rename, same-name collision, partial source,
  missing source, unsupported syntax and equivalent implementation;
- add a sibling-owner contamination poison;
- prove output parity on all corpus witnesses that exercise the reader.

Do not migrate `decoder_qk_norm_from_files` or router/FFN/position readers in
this phase merely because they have one caller: they traverse nested mechanism
owners and belong to U3-F.

## 7. U3-F — nested transformer-mechanism readers

Migrate in this order, one independently receipted reader/cluster at a time:

1. exact decoder block occurrence discovery;
2. exact attention-child readers, including QK norm and storage;
3. exact FFN-child readers, including activation dispatch and FFN structure;
4. exact router/expert occurrence readers;
5. positional/application-site reader.

Rules:

- the parser supplies an exact occurrence; a reader may not search the whole
  file for “the attention/FFN”;
- same-role sibling unions, majority votes, sorted-first, shortest-name and
  `matched or fallback` are deleted as their readers migrate;
- code/config arbitration stays in the owner-bound specialized reader and uses
  exact ConfigAccess events; ProgramIndex never reads checkpoint values;
- disagreement becomes `ReaderResult.ambiguous` with rival chains/spans;
- an old known-wrong behavior discovered here is reported and assigned to
  U6/U7/U8 rather than silently folded into U3.

**Stop gate V3:** Codex reviews the first attention, first FFN and positional
cutover. Repetitions under an unchanged contract are delegable.

## 8. U3-G — recursive modality readers

Migrate, in order:

1. vision owner and per-block occurrence readers;
2. audio owner and per-block occurrence readers;
3. projector construction and config-prefix binding;
4. fusion routes.

The existing projector config-chain walker is a behavior reference, not a
second resolver. Delete it when the shared owner graph reaches parity.

Mandatory controls:

- identical class used by text and vision with different mechanisms;
- gated text FFN beside dense vision FFN;
- two projectors using the same class at different fields/config prefixes;
- embedded versus standalone component equivalence;
- pipeline slot versus root modality separation;
- dynamic/external component remains opaque;
- class and field rename metamorphism;
- Qwen2-VL positive projector, FLUX/Qwen-Image negative projector and MusicGen
  composite ownership controls remain green.

**Stop gate V4:** Codex reviews the first recursive vision cutover and the
projector resolver deletion. These are high-blast-radius ownership boundaries.

## 9. U3-H — conformance cutover and legacy-parser eradication

### Conformance

Conformance must consume the same ProgramIndex, OwnerGraph and ReaderResult
used by parsing, while remaining an independent comparison of evidence to the
drawn artifact. It must not rerun a parallel AST interpreter.

Delete or replace:

- conformance-local `ast.parse` sites over model source;
- broad all-block fallbacks;
- name/suffix drill resolution;
- sibling role unions;
- duplicate config-scope walks;
- imported-file closure parsing already represented by ProgramIndex.

### Final static gates

Add blocking scans proving:

- model-source `ast.parse` exists only in `evidence/program_index.py`;
- no reader accepts a raw `files` union after migration;
- no migrated reader returns bare `None`, `False`, `{}`, `[]` or `0` for
  uncertainty;
- no production use remains of the four legacy parse caches;
- no shortest-name, sorted-first, role-union or class-suffix path selects an
  owner;
- renderers and params do not import ProgramIndex, owner resolution or readers;
- the count of old parser/helper symbols is zero, not allowlisted.

U3 is complete only when the inventory is regenerated with every model-source
reader marked migrated or explicitly assigned to a later semantic unit, and
every superseded parser/cache/union is deleted.

## 10. Per-commit receipt (U3-D through U3-H)

Every commit must reproduce from its own committed tree and include:

1. focused reader tests and all new poisons;
2. `test_program_index.py`, `test_component_owner.py` and
   `test_reader_result.py`;
3. all blocking U2 audit/net files;
4. affected evidence, conformance and corpus controls;
5. `tests/test_preservation.py` with zero drift;
6. full suite, zero unexpected failures;
7. fingerprint before equals fingerprint after;
8. 26-witness structural surfaces and pixels byte-identical;
9. clean-checkout isolated-tree import/parse and focused test;
10. `git diff --check` and pyflakes clean;
11. exact old symbols deleted, demonstrated by `rg`;
12. no manifest/gallery bless without Soumil's explicit decision.

If a commit passes only from the dirty working tree, it is not a receipt.

### 10.1 Executable receipt and runtime discipline

For every reader migration, execute the checklist through:

```bash
python3 scripts/verify_commit.py --focus tests/test_<reader>.py \
  --forbid <deleted_legacy_identifier>
```

This is an exhaustive staged partition, not a reduced suite. A fail-fast
preflight runs focused, U2-authority, collect-only and static/eradication lanes
in separate detached worktrees. Once those lanes release their workers, a
dedicated lane owns all preservation tests (including one independently
schedulable case per blessed witness) while the full-core lane owns every other
test file. Host-aware scheduling gives those two exhaustive lanes the complete
bounded CPU budget instead of permanently reserving idle workers for already
finished preflight lanes. Each lane must return zero and preserve
both the project-tree fingerprint and the content fingerprint of the ignored
blessed galleries/baselines staged into that worktree. The emitted
`receipt.json` is the machine-readable completion record.

Run the same command with `--serial-full` at each U3 phase boundary and final
U3 closure. That slower run is the cross-file/order-dependence backstop; it is
not repeated after every one-reader migration unless the parallel receipt or a
stop condition exposes an ordering concern.

## 11. Mandatory stop-and-report conditions

The implementation agent stops without adding a workaround when:

- a reader needs a new ProgramIndex record or query semantic;
- a component root or construction occurrence is ambiguous;
- parity requires a name, family, model type, table, vote or first-hit;
- exact ownership changes an existing diagram;
- source and checkpoint evidence disagree;
- unsupported syntax would need to be interpreted;
- a new fact, status, debt row, exception or projection surface is required;
- a renderer/params change seems necessary;
- a preservation witness changes for an unexplained reason.

The report to Soumil must contain:

```text
exact model/witness
exact component occurrence chain
old reader and old result
new observations and typed result
rival/missing source spans
which invariant prevents continuing
smallest lawful options
likely visual/consumer blast radius
```

Do not report merely “tests fail” and do not make the most familiar model pass.

## 12. Work ownership and review cadence

### Implementation agent may execute

- the exact B/C correction commit scope;
- U3-D pilot implementation after D0 exists;
- one bounded reader migration at a time;
- tests, poisons, parity fixtures and mechanical old-code deletion;
- subsequent repetitions under a contract already reviewed by Codex.

### Codex must enter

- U3-D0 component-root resolution;
- U3-D pilot vet;
- first attention, first FFN and position cutovers;
- first recursive modality and projector resolver cutovers;
- any stop condition;
- U3-H final authority-deletion audit.

### Soumil alone decides

- any intentional diagram/structural delta;
- manifest or gallery re-bless;
- accepting a new canonical mechanism or carrying explicit debt;
- final U3 DONE status.

## 13. Achieved output

At U3 completion, every model-source file is parsed once into neutral
observations; every architectural reader is tied to one exact constructed
occurrence; uncertainty is typed; parser and conformance share evidence without
sharing conclusions; parallel AST parsers and best-candidate fallbacks are
gone; and renderers remain consumers rather than architecture detectors.

This does not “support every model.” It makes a new model either resolve through
the same generic program/evidence laws or remain explicitly opaque/ambiguous.
That is the boundary that turns future support from recurring family patches
into bounded mechanism work.
