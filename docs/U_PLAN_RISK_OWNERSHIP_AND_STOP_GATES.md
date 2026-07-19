# U-Plan Risk Ownership and Stop Gates

> **Status:** Binding supplement to
> `EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` and `U3_RUNBOOK.md`.
>
> **Purpose:** Prevent the code-authority migration from recreating the old
> config/name/default architecture under new abstractions. This document does
> not replace the U3-U15 sequence. It defines who may make the dangerous design
> decisions, what must be proven before implementation continues, and when work
> must stop for review.

## 1. The invariant we are protecting

The only lawful architectural path is:

```text
model ID
  -> artifact/source address resolution
  -> PreparedDocument + exact SourceBundle
  -> ProgramIndex observations
  -> ComponentOwner construction occurrence
  -> specialized ReaderResult[T]
  -> owner-qualified typed architectural fact
  -> canonical IR / operation region / component DAG
  -> projection DTO
  -> HTML / cards / expanded JSON / params / conformance
```

The model ID locates artifacts. Configuration supplies checkpoint-specific
values. Source observations describe literal program structure. The owner
resolver identifies an exact constructed occurrence. A specialized reader—and
only that reader—may interpret a mechanism. Renderers format already-proven
structure.

No layer may silently perform the job of the next layer.

## 2. Why this plan is stronger than the former approach

The former approach mixed identity, config fields, class names, source tokens,
defaults and renderer conventions into one architectural decision. The current
plan separates those authorities and makes every transition inspectable.

The new plan is sound only if the separation is enforced. A single giant
`ProgramIndex`, a clever owner resolver, a permissive projection helper, or a
renderer fallback could otherwise recreate the same defect with different
names.

## 3. Binding ownership rule

Work is divided into **kernel decisions** and **bounded migrations**.

### 3.1 Kernel decisions reserved for Soumil + Codex

The implementation agent must stop before changing any of these contracts:

1. `ProgramIndex` observation-versus-interpretation boundary.
2. `ComponentOwner` identity, occurrence-chain, conflict and ambiguity laws.
3. `ReaderResult[T]` success/failure/completeness vocabulary.
4. Fact status, provenance, arbitration or negative-proof laws.
5. Canonical IR/Region/Component-DAG schemas.
6. Projection DTO and receipt contracts.
7. Unknown/not-applicable/false semantics.
8. Renderer/parser dependency-firewall rules.
9. Parameter or expanded-JSON authority rules.
10. A new debt, exception, allowlist, abstraction or identity vocabulary.

These are systemic boundaries. A locally plausible decision here can make every
later model confidently wrong. They must be designed or directly implemented by
Codex, then ratified by Soumil before repetitive migrations build on them.

### 3.2 Work the implementation agent may perform autonomously

After the relevant kernel contract is frozen, the implementation agent may:

- populate already-defined observation records;
- migrate one reader to an existing `ReaderResult[T]` contract;
- bind an additional exact owner using the frozen owner API;
- add a fact already present in the closed registry;
- project an existing canonical region through an approved DTO;
- add fixtures, counterexamples and preservation witnesses;
- delete the replaced legacy reader/fallback in the same commit;
- make mechanical repetitions that do not alter evidence strength or meaning.

If a migration needs a new fact kind, owner rule, evidence status, projection
surface or unsupported-syntax policy, it is no longer mechanical and must stop.

## 4. Unit ownership map

| Unit | Primary purpose | Reserved work for Codex | Delegable work after the contract freezes |
|---|---|---|---|
| U3-A | Observation-only ProgramIndex | Final schema/boundary audit | Walker population and frozen AST fixtures |
| U3-B | Exact ComponentOwner resolver | **Direct design and implementation** of occurrence identity, rival chains and ambiguity | Additional fixture coverage only |
| U3-C | ReaderResult substrate | **Direct design and implementation** of failure, completeness and provenance laws | Mechanical wrapping after acceptance |
| U3-D-H | Reader migration and old-parser deletion | Select and personally vet the first pilot; approve cluster boundaries | One reader/cluster per independently receipted commit |
| U4 | Unknown-safe semantic substrate | **Direct implementation** of tri-state fields, opaque regions and default-removal rules | Repetitive call-site adaptation after each primitive is proven |
| U5 | Consumer firewalls | **Direct implementation** of DTO/import rules and blocking static gates | Mechanical import/read cleanup |
| U6 | Attention | Define and vet the first owner-bound attention fact/region contract | Migrate other altitudes using the same contract |
| U7 | FFN/norm/cell | Define and vet canonical FFN and cell-region contracts | Migrate dense/gated/MoE and nested consumers without changing schema |
| U8 | Position/mask/schedules | Vet selector and application-site semantics | Migrate mechanisms one at a time |
| U9 | Recursive modalities | Vet recursive-owner and fusion ownership boundaries | Migrate tower/projector/fusion mechanisms under existing readers |
| U10 | Diffusion root | Vet root/stream/conditioning DAG contract | Migrate bounded denoiser families by mechanism, never identity |
| U11 | UNet | Vet canonical stage/cell DAG and skip ownership | Populate exact stage/cell readers and remove templates |
| U12 | VAE/codec | Vet component DAG and output-domain evidence law | Populate exact mechanisms and remove templates |
| U13 | Scheduler | Vet state/update-graph abstraction | Migrate scheduler mechanisms without class-name semantics |
| U14 | JSON/params/conformance | **Direct implementation** of consumer contracts and first cutovers; final audit of every surface | Mechanical serialization/formula migrations after contracts freeze |
| U15 | Authority deletion/release | **Direct final audit and deletion decisions**; Soumil alone blesses artifacts | Test/document cleanup explicitly requested by Codex |

The useful division is not “Codex writes difficult code and another agent writes
easy code.” It is: Codex owns decisions whose mistakes multiply across the
system; the implementation agent performs bounded repetitions after those
decisions are made executable and testable.

## 5. Risk register and mandatory countermeasures

### R1 — ProgramIndex becomes a new architecture classifier

**Failure:** The index records `is_attention`, `is_gated`, `is_vision`, or other
semantic conclusions, turning the shared substrate into a new family table.

**Prevention:** ProgramIndex may record syntax, spans, assignments, calls,
control flow, dataflow observations and construction sites only. Architectural
terms belong in specialized reader results.

**Blocking poison:** Insert a class named `Attention` whose forward is a plain
linear projection. The index may record its name and operations; it must not
emit an attention mechanism.

### R2 — Class identity replaces construction-occurrence identity

**Failure:** Two instances of the same class, or two same-role siblings, share
evidence.

**Prevention:** Every resolved owner includes the full construction-site chain.
No fact join may use class name alone.

**Blocking poisons:** Same class at two fields with different config prefixes;
gated text FFN beside dense vision FFN; identical child class constructed twice
under different static guards.

### R3 — Names complete resolution

**Failure:** `MLP`, `Attention`, `Processor`, model type or field spelling turns
a candidate into a proven role.

**Prevention:** Names may nominate candidates only. Construction binding plus
reachable behavior must prove the role.

**Blocking poison:** Rename every relevant class and field while preserving the
program graph. Architectural facts and diagrams must remain equivalent.

### R4 — Dynamic or unsupported Python becomes a guess

**Failure:** Factories, monkey patching, dynamic dispatch or external code are
not understood, so a conventional mechanism is supplied.

**Prevention:** Use typed `unsupported`, `external`, `ambiguous` and
`incomplete` results. An unsupported child is an opaque component.

**Blocking poison:** A dynamically selected callable must not become MHA, gated
FFN, causal mask, RoPE or a standard UNet/VAE cell.

### R5 — Configuration is “eradicated” too aggressively

**Failure:** The project attempts to derive checkpoint values such as width,
layer count or head count from source code, even though source legitimately
reads `config.hidden_size` and the checkpoint supplies the value.

**Prevention:** Eradicate configuration as **mechanism authority**, not as the
source of instance values. A value is lawful only when exact code consumption
binds it to a fact.

**Blocking poison:** Two checkpoints use the same class with different widths.
They must share a mechanism graph but retain their own dimensions.

### R6 — Raw operations are mistaken for architectural importance

**Failure:** Every transpose, reshape, cast or temporary tensor becomes a box,
making diagrams literal but unusable.

**Prevention:** ProgramIndex records operations without ranking them. A
mechanism reader selects architecturally meaningful state changes and dataflow.
Projection altitude decides which proven facts are visible; it cannot invent or
erase them silently.

**Blocking controls:** Equivalent implementations with extra transpose/view
noise must yield the same high-level region; a transpose that changes head or
token axes in a mechanism-sensitive way must remain available in the drill.

### R7 — Canonical schemas cannot represent a new mechanism

**Failure:** A novel architecture is squeezed into MHA, gated FFN, sequential
residual or another familiar enum because the IR has no representation.

**Prevention:** Stop and extend the canonical schema with an opaque/partial or
new generic mechanism shape. Never encode novelty in a renderer branch.

**Stop condition:** Any implementation request of the form “map this unknown
thing to the closest existing kind so it renders.”

### R8 — Renderer resumes interpretation

**Failure:** HTML/cards infer owner, mechanism, topology or applicability from
IDs, titles, raw extras, missing fields or familiar defaults.

**Prevention:** U5 dependency firewall plus projection DTOs. Renderers accept
geometry, display values and canonical regions only.

**Required static rules:**

- no renderer imports config readers, source resolvers or evidence readers;
- no renderer searches `id`, `title`, `view` or `kind` text to infer owner;
- no renderer defaults missing semantic fields to a real mechanism;
- no renderer reconstructs child operations when a region is absent.

### R9 — Parser authors renderer-ready architecture

**Failure:** Adapters continue emitting architectural `title`, `description`,
`view`, `children` and hand-built op cards.

**Prevention:** Adapters emit facts, topology and canonical regions. A dedicated
projection layer creates presentation DTOs.

**Required U15 gate:** Outside the approved projection package, writes to
architectural `title`, `description`, `view`, `children`, node kind or SVG
metadata are blocking. Explicit non-architectural diagnostics are typed and
separately allowed.

### R10 — JSON, params or conformance becomes a second interpreter

**Failure:** HTML is correct while expanded JSON invents cache/QKV, params infer
gating, or conformance unions sibling candidates.

**Prevention:** U14 makes all three exact consumers of the same facts, regions
and owner graph.

**Blocking poison:** Change one fact to unknown and verify that HTML, cards,
JSON, params and conformance all weaken consistently without selecting their own
fallback.

### R11 — Preservation protects an existing lie

**Failure:** Pixel equality is treated as proof of semantic correctness.

**Prevention:** Every migration has both preservation witnesses and semantic
counterexamples. A known correction may intentionally change an artifact, but
only with a fact-level explanation and Soumil's approval.

**Gate:** Preservation answers “did an unrelated model move?” Semantic controls
answer “is the retained result true?” Neither substitutes for the other.

### R12 — Corpus coverage creates false confidence

**Failure:** The 26 witnesses exercise only common paths; unused code and rare
composites remain broken.

**Prevention:** Combine static poisons, synthetic AST fixtures, equivalent-model
controls, same-class/different-owner controls and real corpus witnesses. Add a
new witness when a shared boundary defect escapes the corpus.

### R13 — Dual truth paths survive migration

**Failure:** A new reader lands while the old scanner/default remains as a
fallback indefinitely.

**Prevention:** Delete the old reader or quarantine it with an exact owner,
reason, count and deletion unit in the same commit. A migration is not complete
when both paths remain available.

**Blocking gate:** Search and call-graph checks prove the retired symbol has no
production consumer.

### R14 — Evidence strength is laundered

**Failure:** Class defaults, config declarations, partial source or inferred
premises become code-proven facts.

**Prevention:** The winning evidence occurrence supplies the fact's provenance,
completeness and premises. Renderers cannot choose or upgrade status.

**Blocking poisons:** checkpoint declaration conflicting with class default;
presence-only evidence attempting to prove absence; partial source attempting a
complete negative.

### R15 — Source version or external dependency mismatch

**Failure:** The checkpoint config and inspected source describe different
versions, or the decisive mechanism lives outside the fetched bundle.

**Prevention:** Content-fingerprint exact source files, record external-node
provenance, and expose mismatch/unavailable-source status. Do not silently use a
nearby installed implementation as checkpoint truth.

### R16 — Performance pressure weakens correctness

**Failure:** Repeated parsing makes tests unusably slow, encouraging global
caches, skipped gates or shortcut inference.

**Prevention:** One immutable ProgramIndex per SourceBundle and call-local
ParseContext attachment. Cache identity is content-based. Performance work may
reuse observations but cannot reuse facts across owners/documents.

### R17 — U3 becomes an endless refactor

**Failure:** The team tries to migrate every reader before proving one complete
path, or continually expands the index for speculative future syntax.

**Prevention:** U3-A freezes observation completeness against the inventory;
U3-B/C freeze owner/result contracts; U3-D proves one pilot; later clusters are
bounded and delete old code. Unsupported syntax is a valid typed result, not a
requirement to build a Python interpreter.

## 6. Mandatory commit protocol

Every U3-U15 commit must include:

1. Exact mechanism/owner scope and files changed.
2. The old authority path being deleted or quarantined.
3. Positive witness proving the intended mechanism.
4. Equivalent implementation or renamed-class control.
5. Sibling/collision counterexample.
6. Missing, partial, ambiguous or unsupported-source counterexample.
7. Focused tests and all blocking U2 nets.
8. Full suite on an unchanged tree fingerprint.
9. Isolated committed-tree reproduction—not merely working-tree green.
10. Pixel/IR/JSON/params comparison for affected and equivalent models.

No unit is `DONE` from a focused test alone.

## 7. Immediate stop-and-report conditions

The implementation agent must stop and report the exact evidence boundary when:

- more than one owner or config-prefix chain remains viable;
- a class/name/family token is needed to select a mechanism;
- a renderer or parameter path needs raw config/source access;
- an unknown would have to be displayed as a known mechanism;
- an existing canonical schema cannot express the observed program;
- a new allowlist, debt category or exception appears necessary;
- the same fact is interpreted separately by two consumers;
- a preservation witness changes without a fact-level causal explanation;
- verification mutates the tree;
- a commit passes only from the working tree and not from its committed tree.

The report to Soumil must say:

```text
Observed program/evidence:
What can be proven:
What remains ambiguous or unsupported:
Which boundary lacks a representation:
Models/surfaces potentially affected:
Smallest general contract change proposed:
Why a local/model/renderer/config shortcut is forbidden:
```

## 8. Final acceptance after U15

The migration is complete only when all of the following are mechanically true:

- identity strings are address/display data only;
- configuration fields supply values only through exact proven consumption;
- ProgramIndex contains no architectural role conclusions;
- every architectural fact belongs to an exact construction occurrence;
- no renderer infers owner, mechanism, applicability or topology;
- adapters/evidence readers do not author architectural presentation trees;
- HTML, cards, JSON, params and conformance consume the same facts/regions;
- missing/ambiguous/unsupported evidence remains unknown at every depth;
- no model-family table or renderer branch is required for an equivalent
  mechanism;
- old readers, defaults, raw-extras paths and temporary authority tables have
  been deleted or remain as explicit shrinking debt owned by a later approved
  unit;
- every acceptance gate reproduces from the committed tree with an unchanged
  fingerprint;
- Soumil has reviewed every intentional artifact change.

Until those conditions hold, “the diagram looks right” is evidence of a useful
output, not proof that the architecture pipeline is finished.
