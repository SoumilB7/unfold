# EVIDENCE ARCHITECTURE HARDENING PLAN

> **SUPERSEDED AS TRACKER (2026-07-13).** The authoritative tracker and
> execution plan is now `docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md`
> (independent audit of commits `daf056f`..`d475d87` + the binding §20 runbook
> U0→U15 and the §21 live tracker). Where any status, sequence, or completion
> claim below conflicts with that document, **that document wins**. This file
> remains the historical record and rationale for §16 and the nine
> `procedure N` commits.

*A root-level plan for making architectural fabrication difficult by construction,
not merely forbidden by doctrine. Written 2026-07-12 after the Run 77 campaign,
the U1/U2 evidence work, and the review of the active U6 diffusion changes.*

---

## 0. Executive decision

The recurring problem is not that the library has too many legitimate model
branches. Model code is inherently conditional. The problem is that architectural
decisions are distributed across source readers, config readers, parsers, raw
dictionaries, YAML marker tables, labels, and renderers, with no compulsory typed
boundary between **what was observed** and **what is drawn**.

The project already has the beginnings of the right architecture:

- `ParseContext` resolves source once and carries call-local state.
- `FactLedger` records provenance for a subset of transformer facts.
- positional evidence distinguishes proven / ambiguous / oracle-missing.
- render events can carry `facts_projected` receipts.
- the identity guard name-blinds parses and scans production sources.
- nested conformance can compare a drill to a transitive callable closure.

These rails are valuable but optional. Diffusion code can still create structural
facts without using them, renderers can still expand one weak boolean into several
strong claims, and a newly named YAML table can evade an enumerated identity guard.

**Decision:** preserve the completed evidence work, but put a closed evidence
contract underneath every architecture domain. U6 diffusion is the first full
vertical migration because it currently exposes every failure mode at once. Once
the contract is proven there, migrate the remaining transformer/modalities tail.

This plan does **not** replace `SURGICAL_PLAN_EVIDENCE.md`. It hardens its
foundation:

- U1 source parity remains complete and foundational.
- U2's ledger/default-kill/projection work becomes the seed of the closed contract.
- U3/U4/U5 continue, but new facts must enter through the contract.
- U6 is revised: no class-name-to-structure maps; exact resolved code supplies
  stage, VAE, and scheduler evidence.
- U7 closes the generalized guards, corpus, and presentation gates described here.

---

## 1. Goal and non-goals

### 1.1 Goal

For every structural statement shown in SVG, HTML cards, JSON, parameter counts,
or warnings, the library must be able to answer mechanically:

1. **What exact fact is being claimed?**
2. **Who owns it?** Root, component, stack, variant, layer, or stage.
3. **What evidence decided it?** Code, code-evaluated config, direct declaration,
   class default, derivation, or no evidence.
4. **Where is that evidence?** File/class/method/line and/or config path.
5. **Was the inspection complete enough to prove absence?**
6. **Where was the fact projected?** Diagram node, card, chip, JSON path, parameter
   formula, or explicit honest-unknown surface.
7. **Which mechanical check compares it to the oracle?**

An unregistered structural statement is a build/test failure. An unknown fact is a
valid result. A guessed fact is not.

### 1.2 What “code-based, not config-based” means

The project must not try to eliminate configuration. Hugging Face modeling code is
parameterized by configuration. Removing config would make branch resolution less
accurate, not more.

The correct law is:

> **Code proves that a field controls a structural branch or expression; the
> checkpoint config supplies the value selected for this checkpoint.**

Examples:

- Lawful: source proves `config.add_cross_attention[i]` gates construction;
  config supplies the booleans.
- Lawful: source proves `hidden_size` is a Linear width; config supplies 4096.
- Lawful fallback: source is unavailable, but config directly declares
  `encoder_hid_dim_type="image_proj"`; render it as `config_declared`, not
  `code_proven`.
- Forbidden: `"SimpleCrossAttn" in class_name` directly selects a cell topology.
- Forbidden: an absent field defaults to Transformer2D, KL, causal, gated, or
  first-order Euler.

### 1.3 Non-goals

- Do not execute remote model code.
- Do not require downloading weights.
- Do not build a universal Python interpreter.
- Do not make all old views change at once.
- Do not replace honest unknowns with broader heuristics.
- Do not introduce user-facing “model family options.”
- Do not use per-repository or per-model exception tables.
- Do not let this refactor block a safe, isolated correctness fix; use shadow-mode
  evidence and unit-sized cutovers.

---

## 2. The invariant set

These invariants are the source of truth for implementation and review.

### I-1 — Identity is an address, never a fact

`model_type`, `_class_name`, `architectures`, repo id, and component class names may
locate exact source. They may be displayed. They may not select architecture without
passing through inspection of the resolved source.

### I-2 — Config values require an ownership binding

A structural config value is consumable only when one of these is true:

1. resolved code names/reads the field for the relevant owner; or
2. the external config schema directly declares a public semantic contract and the
   fact is explicitly marked `config_declared`; or
3. source is missing and an approved compatibility fallback is used with a visible
   provenance/unknown policy.

### I-3 — Absence requires complete negative proof

`False`, `none`, “no mid block,” “no gate,” “no temporal path,” and similar negative
facts may be `code_proven` only when the relevant inspection is complete for the exact
owner and path. Partial extraction yields `unknown` or `ambiguous`, never absence.

### I-4 — Unknown never silently becomes a conventional structure

No `None → bool`, `value or default`, or terminal default may create an architectural
fact. Unknown UNet cells do not become Transformer2D. Unknown VAEs do not become KL.
Unknown schedulers do not become Euler. Unknown masks do not become causal.

### I-5 — Evidence strength cannot increase downstream

A renderer cannot turn `has_temporal_axis=True` into
`has_alpha_blender=True`. A placement tuple cannot become a proof of a cell's FFN.
Derived facts must list their exact premises, and their status cannot outrank the
weakest required premise.

### I-6 — One fact, one structural author

Each fact has one registered author. SVG, cards, JSON, parameter counts, labels, and
conformance consume that fact; they do not independently recompute it.

### I-7 — Every evidenced fact owes a projection

Every `code_proven`, `code_and_config`, `config_declared`, or `class_default`
structural fact in a drawable domain must be projected or explicitly classified as
non-drawable with a reviewed reason. New domains are blocking by default; they are not
silently exempt because their family name is absent from a whitelist.

### I-8 — Every drawn structural claim owes evidence

The projection audit must be symmetric:

- evidence with no projection = omission;
- structural projection with no evidence key = fabrication.

Presentation-only glyphs and prose must be explicitly typed as presentation, not
silently exempted strings.

### I-9 — Source failures are data

Unsupported AST, unresolved imports, ambiguous ownership, missing source, and reader
bugs are distinct typed results. Broad `except Exception: return None` is forbidden at
evidence boundaries.

### I-10 — Component and variant ownership is exact

Facts are qualified by component and variant. Wrapper source cannot prove a nested
vision fact unless construction binds it. A union of same-role sibling modules cannot
prove a particular drill. Root-level temporal evidence cannot prove every stage.

### I-11 — Tests prove rules with counterexamples

Every reader requires witnesses that would defeat identity matching, config-only
guessing, incomplete negative proof, and conventional defaults. A known-model golden
alone is insufficient.

### I-12 — Pixels remain the release oracle

Mechanical gates establish evidence and wiring correctness. Every intentional visual
change still requires exhaustive Dable inspection and an explicit Soumil bless/re-bless
decision.

---

## 3. Target architecture

### 3.1 The pipeline

```text
Input config / model id
        │
        ▼
ParseContext + SourceBundle
  (address resolution only)
        │
        ▼
Code registry / construction graph / callable graph
        │
        ▼
Config-aware branch evaluator
  (evaluate only source-bound fields)
        │
        ▼
Typed EvidenceGraph
  (facts + owner + status + completeness + source spans + premises)
        │
        ▼
Architectural IR
  (no identity/config inference)
        │
        ├────────► parameter accounting
        ├────────► conformance
        └────────► renderer projections
                         │
                         ▼
             SVG + HTML + cards + JSON
```

No later layer may reach backward to raw config identity to decide structure.

### 3.2 Core evidence types

The exact implementation can evolve, but it must carry these semantics:

```python
FactStatus = Literal[
    "code_proven",
    "code_and_config",
    "config_declared",
    "class_default",
    "derived",
    "ambiguous",
    "oracle_missing",
    "unknown",
    "legacy_asserted",   # migration-only; must trend to zero
]

Completeness = Literal[
    "complete",          # sufficient to prove presence and absence in scope
    "presence_only",     # may prove a found signal, never its absence
    "partial",           # known incomplete
    "uninspected",
]

@dataclass(frozen=True)
class SourceSpan:
    component: str
    class_name: str | None
    method: str | None
    file: str | None
    line: int | None

@dataclass(frozen=True)
class EvidenceFact(Generic[T]):
    key: FactKey
    owner: OwnerPath
    value: T | None
    status: FactStatus
    completeness: Completeness
    source_spans: tuple[SourceSpan, ...]
    config_paths: tuple[str, ...]
    premises: tuple[FactKey, ...]
    reason: str
```

`EvidenceFact(False, code_proven)` is invalid unless completeness is `complete`
for the relevant scope.

### 3.3 Owner paths

Use explicit owner paths rather than loosely encoded strings:

```text
root
root.decoder
root.decoder.variant[full_attention]
root.decoder.layer[7].attention
root.vision.encoder.variant[0].ffn
root.denoiser.down[2].resnet[1]
root.denoiser.mid
root.vae.decoder.up[3]
root.scheduler.step
```

This owner is what prevents a sibling gated FFN from proving a dense drill and a
root temporal axis from proving every child stage.

### 3.4 Fact registry: closed world, not whitelist exemption

Create a central registry where every structural fact declares:

```python
FactDefinition(
    key="unet.stage.attention_cell",
    value_type=AttentionCellKind,
    allowed_statuses={...},
    negative_requires_complete=True,
    projections={"svg", "card", "json"},
    unknown_policy="opaque_cell",
    conformance="nested_callable",
    parameter_consumer=None,
)
```

Rules:

- Creating an unknown fact key fails tests.
- A structural renderer receipt for an unknown fact key fails.
- A registered drawable fact without a projection definition fails.
- A projection may not claim a stronger value than the ledger record.
- Temporary legacy facts must be explicitly registered as `legacy_asserted` and counted.
- The registry is domain-neutral; transformer and diffusion use the same contract.

### 3.5 Evidence graph, not only a flat ledger

Keep the current serialized ledger for compatibility, but add premise edges. This is
needed to audit inference strength:

```text
denoiser.has_temporal_axis ──┐
stage.class_constructs_temporal_attention ──► stage.temporal_attention
```

The first premise alone cannot decide the output. A derived fact without all required
premises fails validation.

### 3.6 Renderer firewall

After a domain migrates, its renderers must not import config accessors, source
resolvers, identity vocabularies, or evidence readers. They receive architectural IR
plus fact keys and emit projection receipts.

Add a static dependency test:

- migrated renderer modules may import IR, labels, graph primitives, and projection
  helpers;
- they may not import `everchanging` structural maps, `_g`, source resolution, or
  identity fields;
- renderer prose that states an op/fact must be generated from the same typed fact as
  the graph node.

---

## 4. Why the current nets missed the problem, and how each changes

| Existing mechanism | Current value | Current hole | Required hardening |
|---|---|---|---|
| FactLedger | good per-fact provenance seed | optional and transformer-heavy | all structural facts registered; legacy count trends to zero |
| projection audit | catches dropped keys in four families | open whitelist exempts new domains | closed fact registry + symmetric drawn-without-evidence audit |
| identity AST/YAML scan | catches known identity idioms/tables | enumerates suspicious names | semantic taint + arbitrary-name negative controls |
| name-blind differential | strong end-to-end invariant | pre-resolved source and display dropping can mask some channels; not enough alone | retain as backstop; add fact-level identity provenance ban |
| nested conformance | compares real leaf closures | hand-authored diffusion views have no bound callable | every Tier-1 compute drill requires owner/callable or typed unresolved |
| config field audit | finds unread fields | accessed is treated as handled; consumed census incomplete | inspected/consumed/projected intents mandatory and blocking |
| golden/corpus tests | protects existing pixels | known witnesses reward local family fixes | add metamorphic counterexample matrix and frontier corpus |
| broad `_code_*` wrappers | keep parser alive | swallow extractor bugs into fallback | typed failures; narrow expected exceptions; reader-direct tests |

---

## 4A. Responsibility forensics — where the three jobs are mixed

This section distinguishes three jobs that must not be allowed to silently perform
one another's work:

1. **Observation / acquisition** — record what the source and config literally say.
2. **Architectural interpretation** — decide which mechanism those observations prove.
3. **Editorial projection** — decide how much of that proven mechanism belongs in the
   diagram at a particular altitude.

The problem is not that one file contains several helper functions. The problem is a
**strength leak**: a later job can manufacture a stronger statement than the previous
job supplied, and no type or dependency rule stops it.

### 4A.1 The required boundary objects

Every stage needs a distinct product so reviewers and gates can identify where a claim
was introduced.

```text
RawProgramIndex       observation only
    │
    ▼
BoundProgramGraph     exact owners + construction + config-use bindings
    │
    ▼
EvidenceGraph         architectural mechanisms with status/completeness/premises
    │
    ▼
SemanticGraph         normalized operations/regions, still presentation-neutral
    │
    ▼
ProjectionPlan        Tier-1/Tier-2/Tier-3/hidden choices for one altitude
    │
    ▼
Rendered artifacts    SVG/cards/JSON plus exact fact receipts
```

#### `RawProgramIndex` — no architectural conclusion

Contains literal syntax and source locations:

- classes, methods, bases, imports;
- assignments and constructed field call expressions;
- calls, operators, kwargs, branches, loops, return/dataflow references;
- literal config attribute reads;
- raw shape transforms such as reshape/transpose/permute;
- source completeness and unsupported-syntax diagnostics.

It may say “a call to field `foo` occurs and its constructor expression references
class `Bar`.” It may not say “`foo` is the FFN” or “this is Transformer2D.”

#### `BoundProgramGraph` — ownership, not visual meaning

Resolves:

- exact root/component/variant/stage instance;
- which class/callable a constructed field can invoke;
- which config field controls a constructor/forward branch;
- which branch is active for the current checkpoint when statically evaluable;
- complete versus presence-only inspection scope.

It may say “down stage 2 constructs `X`, whose forward calls fields A then B, and branch
B is enabled by `config.add_cross_attention[2]`.” It must not decide that B deserves a
box or infer unobserved internals.

#### `EvidenceGraph` — typed architectural meaning

Mechanism readers consume the bound graph and emit facts such as:

- `ffn.gated=True`;
- `ffn.storage=fused_gate_up`;
- `stage.has_cross_attention=True`;
- `stage.attention_cell=plain_cross`;
- `stage.temporal_blender=unknown`;
- `scheduler.uses_output_history=True`.

This is the only layer allowed to create architectural facts. It records premises and
cannot prove a negative from incomplete acquisition.

#### `SemanticGraph` — normalization without editorial deletion

Turns evidence into canonical mechanisms:

```text
Linear + split + activation + multiply + Linear → gated FFN region
Q/K/V projections + QKᵀ + scale + mask + softmax + AV → attention region
spatial path + temporal path + learned blend → spatial-temporal mixing region
```

Raw operations are folded, not forgotten. The semantic graph retains provenance from
each canonical operation back to raw observations.

#### `ProjectionPlan` — the only editorial layer

For a requested altitude, marks each semantic item as:

- Tier 1 — named computation block;
- Tier 2 — structural connector/dataflow operation;
- Tier 3 — annotation/chip/caption;
- hidden — implementation detail already represented by its parent mechanism.

This is where transpose/reshape/dropout-like details are filtered. The evidence reader
must not omit them merely because they are usually visually minor.

### 4A.2 Concrete conflation points in the current tree

The following are responsibility findings, not accusations that every current output
is wrong. Several functions are good transitional implementations. The problem is that
their contracts permit the wrong job.

#### Point C-1 — `forward_ops` / `transitive`: syntax collection and role meaning

The AST registries collect useful literal calls/fields and simultaneously classify
roles/op kinds. Low-level normalization is necessary, but today the raw observation is
not a distinct public product. A later reader cannot always distinguish:

- “the AST contained a multiply” from
- “the multiply was classified as a gate” from
- “this owner has a gated FFN.”

**Bifurcation:** one shared syntax registry emits literal events; role classifiers
consume it into a separate bound/semantic view. Preserve `ForwardOps` versus
`CallableInfo` semantic differences, but stop duplicating the raw scan.

#### Point C-2 — `evidence/ffn.py`: discovery, interpretation, and arbitration

`ffn_structure_evidence` currently:

1. discovers reachable classes;
2. decides whether each looks like an FFN;
3. infers gated/dense/fused from field counts, field spellings, split/chunk, and ops;
4. accepts `expected_gated` from an earlier config/parser decision;
5. filters candidates using that expectation;
6. selects a representative and returns `proven`.

This is close to the desired mechanism reader, but steps 1–5 are not separately
auditable. In particular, using `expected_gated` to choose among reachable candidates
can become confirmation rather than exact owner resolution: config expectation helps
select the code witness that then “proves” the expectation.

**Bifurcation:** construction binding must first identify the exact FFN owner instance.
The FFN interpreter then classifies that callable without a desired answer. Config is
used only to evaluate a source-proven branch inside that exact callable. Candidate
equivalence is a fallback for genuinely unresolved alternatives, never a substitute for
owner attribution.

#### Point C-3 — `evidence/vision.py::layer_facts_from_block`: observations,
architectural facts, and layout eligibility

The shared tower reader resolves roles, computes closures, infers norm placement/gating,
and returns `standard_cell`. `standard_cell` is not a literal code fact; it is an
architectural/layout eligibility decision: whether this block may be projected through
the canonical norm→attention→residual→FFN cell.

**Bifurcation:** emit exact mechanism facts first (number/order of sublayers, conv path,
norm dataflow, gate operations). A separate semantic-cell classifier determines whether
those facts match `StandardTransformerCell`. The altitude policy then decides whether to
use the standard tower projection or a literal op-chain.

#### Point C-4 — transformer parser: acquisition, channel arbitration, defaults,
fact accounting, and IR construction in one function

The transformer parser resolves config aliases, invokes code readers, arbitrates config
versus code versus class defaults, chooses unknown/default behavior, records FactLedger
entries, constructs specs, creates warnings, and prepares extras. This is the largest
responsibility concentration.

The ledger is currently written **after/in parallel with** decisions made in local
variables. It documents decisions but is not always the object from which the spec must
be constructed. That allows spec and ledger to diverge.

**Bifurcation:** the adapter orchestrates independent fact resolvers. Each resolver
returns `EvidenceFact`; specs can only be constructed through evidence-aware builders
that consume those facts. Warnings derive from statuses. The parser itself does not
contain per-fact evidence precedence ladders.

#### Point C-5 — `encoder_panel.py`: reuse followed by contextual re-interpretation

Embedded encoders correctly reuse the universal transformer parser, but then compensate
for decoder-oriented defaults by re-resolving norm, gating, storage, and activation and
passing override arguments into `submodel_spec`.

This is evidence that the shared parser output is not yet context-neutral. Reuse occurs,
then a second interpretation corrects it.

**Bifurcation:** parse a component into context-neutral evidence first. Apply only
declared altitude transforms afterward (for example, `cached=False` for a one-shot
encoder tower). Altitude transforms may suppress/present facts; they may not repair
architectural truth.

#### Point C-6 — diffusion parser: config interpretation and presentation topology

The diffusion parser does more than adapt input into IR. It selects denoiser variants,
computes conditioning stories, scheduler display/family facts, and render-spec flags
such as text rails/cross-attention sublayers. Some of those are architecture; some are
presentation decisions.

**Bifurcation:** diffusion adapter orchestration requests evidence for conditioning,
denoiser construction, scheduler, and VAE. Semantic graphs decide topology. A separate
pipeline altitude policy chooses hero layout and rails.

#### Point C-7 — adapter `blocks.py` / `unet.py`: architecture and prose authored
together

Adapter block builders produce block ids, structural children, graph/view selection,
titles, detailed explanatory prose, fact chips, and sometimes architecture-specific
operation lists in the same dictionary. A sentence such as “TemporalResnetBlock +
AlphaBlender” can therefore be introduced at the presentation layer even when upstream
evidence proved only a frames axis.

**Bifurcation:** adapter-side semantic builders emit canonical regions and fact keys.
Editorial policy emits block hierarchy/altitude. Card prose is generated from the same
fact/region templates and may not introduce an unreferenced mechanism noun.

#### Point C-8 — `opgraph.py`: canonicalization still contains compatibility defaults

The op-graph is the correct home for canonical mechanisms, but some regions still use
fallback reads such as truthy defaults (`gated`, RoPE, expert gating) or contain
model-specific explanatory text alongside structure. That means the canonicalizer can
still strengthen incomplete input.

**Bifurcation:** op-graph constructors accept typed, already-resolved mechanism facts.
Unknown selects an opaque region. Canonicalization never supplies a conventional
architecture. Descriptive copy lives in projection templates keyed to region/fact ids.

#### Point C-9 — labels/metadata/cards: presentation helpers re-derive semantics

Summary helpers often inspect loosely typed dicts and branch on combinations of fields
to generate titles, descriptions, and chips. If the combination is incomplete, fallback
wording can imply a specific mechanism. These helpers are effectively a second semantic
interpreter.

**Bifurcation:** labels consume typed enums and evidence status. Structural nouns and
formulae are mapped from canonical semantic operations; unknown has explicit wording.
No `.get(..., architectural_default)` is allowed in presentation code.

#### Point C-10 — fact projection: editorial coverage inferred globally

`fact_projection.py` manually lists drawable family segments and leaf names, and a
rendered FFN event can collect all matching FFN ledger keys rather than only facts owned
by the clicked block. This mixes registration, ownership, and projection policy and can
over-credit one surface for another owner's fact.

**Bifurcation:** the closed fact registry declares projection obligations; a render
event receives exact owner-qualified fact keys from its input semantic region. No global
family scan can manufacture receipts.

#### Point C-11 — YAML: editability confused with evidentiary authority

Aliases, spellings, labels, and low-level vocabulary are appropriate data. Class-name
marker tables that select structural kinds are architectural interpreters, regardless
of being stored outside Python.

**Bifurcation:** YAML schemas are typed by purpose:

- `lexical_aliases` — spelling normalization only;
- `display_vocabulary` — presentation only;
- `semantic_enum_contract` — direct public config declarations, status limited to
  `config_declared` until source-bound;
- structural identity maps — forbidden.

The loader must require a declared YAML purpose instead of returning arbitrary dicts.

#### Point C-12 — tests: expected pictures can encode the same interpretation

Known-model tests often assert the presence of a phrase or block that the new code itself
invented. This proves consistency between implementation and expectation, not that the
interpretation follows source.

**Bifurcation:** mechanism tests target observation/binding/evidence separately;
semantic graph tests target canonicalization; projection tests target altitude and
receipts; end-to-end model tests remain integration witnesses, never the sole proof.

### 4A.3 What caused the conflation in the first place

The repository history supports a concrete causal explanation.

#### Cause R-1 — the product was built as vertical visual slices

The early goal was to get a model config to a useful diagram quickly. A vertical slice
naturally bundled:

```text
recognize model shape → construct architecture dict → write labels/cards → render
```

That was effective for product discovery, but it made each adapter a miniature compiler
and presentation system.

The UNet history is the clearest example: commit `919f0a1` (“introduce unet”,
2026-06-07) added config detection, a UNet parser/template, block text, a renderer,
examples, and tests together. Its initial contract explicitly targeted
`UNet2DConditionModel`, used class-name/`Attn` string recognition, and supplied default
mid/resnet/transformer counts. The feature predated the general evidence rail.

#### Cause R-2 — canonical semantic graphs arrived after family templates

The canonical FFN op-graph arrived later (`a6573fc`), and canonical attention followed
(`f9c75c7`). Those changes successfully unified rendering, but they consumed facts
already decided by existing parsers/templates. They centralized **projection of a
decision** before centralizing **acquisition and interpretation of that decision**.

This is why the library can have one excellent FFN renderer while still having several
paths that decide what the FFN is.

#### Cause R-3 — evidence was introduced as reinforcement around an existing IR

The large identity/evidence/context work arrived later (`d5cc37d`, 2026-06-29), after
the adapters and visual structures existed. FactLedger and projection receipts arrived
later again (`ef9046c`, 2026-07-12).

Consequently, evidence was integrated as readers, validators, overrides, and accounting
around pre-existing decision sites instead of being the only constructor of IR facts.
The old path remained for compatibility, and every new reader had to coexist with it.

#### Cause R-4 — the original IR represented values, not epistemic state

Fields such as `gated`, `mask`, `rope`, and `norm_placement` began as ordinary booleans
or strings with useful defaults. They did not carry:

- who decided the value;
- whether source was complete;
- whether false meant absent or unobserved;
- which component/variant owned it;
- which projections consumed it.

Once defaults were serialized into ordinary values, downstream code could not know
whether it was rendering evidence or convention.

#### Cause R-5 — availability was mistaken for truth authority

Config is cheap, stable, offline, and easy to test. Source AST is harder: packages vary,
remote files may be unavailable, factories/inheritance hide construction, and code must
not be executed. The system therefore used config and names to keep diagrams complete
while source support matured.

The product's desire for a complete, attractive diagram created pressure to prefer a
plausible template over an opaque unknown. That tradeoff was understandable early and
became incompatible with the later honesty doctrine.

#### Cause R-6 — “vocabulary belongs in YAML” lacked a semantic boundary

Externalizing changing strings was a good modularization rule. It reduced duplicated
hardcoding. But the rule did not distinguish:

- lexical vocabulary from
- architectural truth tables.

That allowed identity inference to be moved rather than removed. Reviewers saw a general
YAML list and reasonably interpreted it as doctrinal compliance.

#### Cause R-7 — compatibility locks rewarded preservation of old assumptions

Blessed SVG hashes and byte-stability controls are essential, but during migration they
also make an honest unknown look like a regression. Developers are pushed toward keeping
the old confident structure when a new reader abstains, especially when conformance
expects the old nodes. This produced transitional rules such as “source present but
reader abstained → retain conventional pre-norm and banner it.”

The solution is not weaker regression locking. It is shadow/cutover/deletion with
explicitly reviewed honesty deltas, so compatibility cannot silently outrank truth.

#### Cause R-8 — adapter boundaries were organized by input ecosystem, not compiler stage

`transformer` and `diffusor` are necessary ownership domains, but each grew acquisition,
interpretation, structural assembly, and presentation helpers. Shared mechanisms were
then extracted horizontally after duplication appeared.

The new boundary must be two-dimensional:

- vertical domains: transformer, diffusion, audio, vision;
- horizontal compiler stages: acquisition, binding, evidence, semantics, policy,
  projection.

#### Cause R-9 — tests emphasized shipped witnesses over indistinguishability laws

Golden models and pixels protected the product users saw. They did not ask whether the
same result survived class renaming, identity collision, partial source, or equivalent
config with different code. Local fixes therefore had strong positive evidence and weak
generality evidence.

#### Cause R-10 — no import/type rule made the doctrine executable

The doctrine existed in documents and review practice, but modules were free to import
raw config, evidence helpers, opgraph constructors, labels, and renderer helpers across
stages. Raw dictionaries made those crossings easy. A sufficiently persuasive local
implementation could violate the intended layer boundary without any failing build.

### 4A.4 The deepest root cause, stated precisely

The start-on-config decision is only the historical trigger. The durable root cause is:

> **The architecture has no compulsory epistemic intermediate representation and no
> enforced direction of dependency.**

Without an epistemic IR, “the value is false” cannot be distinguished from “we did not
find it.” Without dependency direction, a renderer can inspect raw config or a parser can
author presentation. Without a separate projection policy, acquisition readers are
tempted to decide what is important enough to draw.

This is why more branches or better reviewers alone cannot close the problem.

### 4A.5 The concrete bifurcation to implement

Use existing modules where possible; the goal is responsibility extraction, not a
directory-renaming rewrite.

```text
model_unfolder/evidence/syntax/
    registry.py          # one literal AST/source index
    events.py            # RawCall, RawAssign, RawBranch, RawTransform

model_unfolder/evidence/binding/
    construction.py      # exact owner → constructed class/callable
    config_use.py        # code expression → config path binding
    completeness.py      # complete/presence-only/partial scopes

model_unfolder/evidence/mechanisms/
    ffn.py
    attention.py
    norm.py
    unet.py
    temporal.py
    vae.py
    scheduler.py         # emits EvidenceFacts only

model_unfolder/semantic/
    regions.py           # EvidenceGraph → canonical SemanticGraph
    rewrites.py          # reviewed folding/equivalence rules
    salience.py          # semantic importance metadata, not layout

model_unfolder/projection/
    policy.py            # altitude → ProjectionPlan/Tiers
    receipts.py          # exact owner-qualified fact receipts

model_unfolder/renderers/
    ...                  # ProjectionPlan → pixels; no raw config/source inference
```

Temporary facade functions preserve current imports while implementations move behind
the boundaries.

### 4A.6 Enforced dependency direction

Add an import-boundary test with this allowed graph:

```text
syntax → nothing architectural
binding → syntax + source context
mechanisms → binding + fact registry
semantic → evidence facts + canonical op types
projection policy → semantic graph + editorial constants
renderer → projection plan + graph primitives
adapter → orchestrates stages; does not implement their internals
conformance → raw/bound/semantic graphs; never renderer prose
```

Forbidden examples:

- syntax importing labels/opgraph/renderers;
- mechanism readers importing cards or SVG helpers;
- renderers importing `_g`, `everchanging` structural tables, source resolution, or AST;
- labels reading raw config identity;
- adapters directly constructing unregistered structural nodes;
- projection policy changing evidence values.

### 4A.7 Filtering minor operations without losing truth

The separation also answers the transpose/reshape problem.

1. Acquisition records the operation.
2. Semantic rewrites determine its role in a mechanism.
3. Projection policy decides whether it is visible.

Examples:

| Raw operation | Semantic treatment | Typical projection |
|---|---|---|
| Q/K transpose used by QKᵀ | part of attention-score operation | folded into attention formula/card |
| reshape heads + transpose | attention lane organization | Tier-3 head-shape annotation or hidden |
| BCHW → token sequence | modality boundary | Tier-2/3 “flatten to tokens” when load-bearing |
| spatial↔temporal rearrange before temporal attention | changes attended axis | visible boundary/axis annotation |
| pixel shuffle/unshuffle | changes resolution and channel semantics | Tier-1/Tier-2 named operation |
| `.contiguous()`, dtype/device cast | runtime plumbing | hidden, retained only in raw graph |
| tuple unpack after fused projection | proves storage/split topology | Tier-2 split in FFN/attention region |

No global blacklist decides importance. Reviewed semantic rewrite + altitude policy does.

### 4A.8 Migration rule for existing mixed functions

For each mixed function:

1. Freeze its current input/output with focused tests.
2. Mark every statement as O (observation), B (binding), E (evidence), S (semantic),
   P (projection), or X (cross-cutting diagnostic).
3. Extract O/B first without behavior change.
4. Return typed E facts and compare them in shadow mode to the old decision.
5. Make S consume only E.
6. Make P consume only S plus altitude.
7. Cut over one fact family.
8. Delete old cross-stage reads and add a forbidden-import/raw-write test.

This classification exercise is required in the implementation PR description; it
prevents “modularization” that merely moves the same conflated function to a new file.

---

## 5. Migration strategy: surgical and behavior-controlled

The migration must not be a rewrite. Each unit has a shadow phase, a cutover phase,
and a deletion phase.

### The three-phase rule for every fact family

1. **Shadow:** new reader produces typed facts, but old rendering remains active.
   Compare old decision vs new evidence across corpus and frontier witnesses.
2. **Cutover:** renderer/parser consumes the typed fact. Intentional differences get
   focused galleries and pixel review. No old fallback deletion yet.
3. **Deletion:** once equivalent controls and full corpus are green, delete the old
   branch/table/default and add a static dead-symbol/forbidden-path test.

Never combine all three phases for several high-blast-radius facts in one patch.

### Compatibility bridge

Existing `AttentionSpec`, `FFNSpec`, `LayerSpec`, and diffusion dictionaries do not
need immediate replacement. Add constructors/projectors that require evidence facts
and record how each field entered the old IR. Direct raw assignments remain temporarily
allowed only in an explicit legacy allowlist that must monotonically shrink.

### No regression by silent fallback

During dual running:

- new `unknown` may intentionally replace an old guess, but it requires a reviewed
  visual delta;
- new evidence may not silently fall back to old structure when the reader errors;
- parameter estimates must declare their unknown policy rather than treating `None`
  as dense/gated/tied/untied;
- no blessing is refreshed merely to make regression tests green.

---

## 6. Unit plan and tracker

Status meanings:

- `DONE`: code, counterexamples, isolated tests, full suite, corpus, and required
  pixels are all verified.
- `ACTIVE`: currently being implemented; at most one high-conflict unit per file area.
- `READY`: specification is complete and prerequisites are satisfied.
- `PENDING`: prerequisites remain.

| Unit | Status | Outcome |
|---|---|---|
| NAS quarantine (H0 sub-unit) | DONE | Unsafe `block_configs` config-only NAS projection removed (Soumil's 6-file edit) + pinning test; 0 corpus fixtures use it |
| H0 Baseline and unsafe-path quarantine | REPAIRED (§16.2 exit green; awaiting Soumil's commit) | Lawful-resource manifest (`_LAWFUL_TABLES`, 21 tables, each path·table·category·permitted-consumers·content-fingerprint); blanket `conformance/` exemption REMOVED (registered tables only); single-entry + single-capital + dict-comprehension detection; typed `@identity_address`/`@identity_display` wrappers replace the function-name set (3 sites); 12 E-criteria poison controls. Fixed a would-be regression: `role=Class` marker tables (`component_class_markers`, `drill_class_markers`) surfaced + manifested. Verified: 25 alone + 145 blast-radius (`test_h4_taint`/`test_smoke`/`test_conformance`), tree quiescent. The 12 exit criteria are the poison controls in `tests/test_identity_guard.py`. |
| H1 Evidence primitives and negative-proof law | REPAIRED (§16.3 exit green; awaiting Soumil's commit) | `migrated_legacy` is now `init=False` INTERNAL provenance set only by the private `_lift` path (reached via `from_record`), so a native caller can no longer opt out of the negative-proof law; a derived NEGATIVE requires effective completeness `complete`; `reason` (human) separated from a stable `legacy_source` label — a reason is never serialized as `FactRecord.source`; structured `SourceSpan`/config paths survive the typed channel. Verified: 330 passed alone (incl. FactLedger/`context.py` blast-radius + corpus round-trip), tree quiescent. Exit criteria are the tests in `tests/test_evidence_facts.py`. |
| H2 Closed fact registry and legacy census | DONE (§16.4 exit green; awaiting Soumil's commit) | **Part A (write-side):** `FactLedger.record_typed` validates every typed write against the registry (key/owner/status/value-type/negative-completeness); typed `legacy_asserted` is REPRESENTED (projection_mode/ffn_storage/attention_kind), not laundered into `asserted` (still serializes `asserted` for baseline stability). **Part B (StructuralWrite census):** `evidence/structural_writes.py` — a line-insensitive census keyed by sink·target (module·symbol provenance) over a **202-entry surface**: `ledger`, `spec` (5 classes), `spec_field` (78 dataclass fields), `extras` (27 incl. nested leaves), `opgraph` (88 Op/Region kinds), `card`, `params`. Static no-growth + no-stale gate (`new_structural_writes`/`stale_surface_entries`), a structured legacy register (`LegacyExtrasWrite`, 18 rows each carrying owner·reason·unit·deletion — replaces the bare `RAW_EXTRAS_BASELINE`), a runtime top-level gate over the corpus, and **5 poisons** (new nested-extras/spec-field/opgraph-kind/card-claim/param-formula). A new structural author cannot bypass the registry via a different representation. Renderer free-text label fabrication-defaults are scoped to H8 (the opgraph/label fallbacks). Verified: 359 + 41 passed, corpus `asserted` baseline unchanged (641), tree quiescent. Exit criteria = `tests/test_structural_writes.py` + `tests/test_fact_registry.py`. |
| H3 Config ownership and consumption | RESTART IN PROGRESS (§16.5): steps 1-2 landed, step 3 (wiring) remains | **Step 1 (remove the 3 audit-clearing diffusion reads) DONE** — `max_sequence_length`, VAE `act_fn`, VAE `temporal_compression_ratio` removed with pending-H7 markers; each confirmed to have NO structural consumer (the denoiser-level temporal read is a distinct, consumed one that stays); `config_field_audit` is non-blocking advisory, so the honest unread state IS the declared pending debt. **Step 2 (the owner-scoped ledger substrate) DONE** — `evidence/config_access.py`: `ConfigAccessEvent` (10 §16.5 fields), `ConfigAccessLedger` with owner-qualified `bound`/`consumed`/`accessed`/`ignored` joins (NOT global set subtraction), the 2 nets (accessed-but-unconsumed; consumed-but-unprojected clearing on projection/pending), `resolve_aliases` (records the actual spelling; unequal aliases → ambiguous, not first-wins; absent → default premise, never fictional consumed), a `current_owner` ContextVar, and derived compat name-views. **16 counterexamples PASS** (the 7 §16.5 cases — aliases/missing/conflicting/sibling-same-key/nested/concurrency/source-missing — + constructor laws + owner-qualified nets + compat). **Step 3 (REMAINING, high-conflict per §8.1):** wire the accessor to emit owner-scoped events, set `current_owner` in every parsing scope, DELETE the global `_touched/_bound/_consumed` (derive from the ledger), rewrite the 2 live nets owner-qualified, full-corpus verify. |
| H4 Semantic identity/taint guard | ACTIVE, early slice only (§16.6) | Fact-provenance rule + negative controls exist; full semantic taint (identity/config-name sources → structural sinks, interprocedural, YAML keys+values) and the renderer/parser dependency firewall remain |
| H5 Typed source failures and reader consolidation | PENDING | broad exception removal; unified registry/construction extraction |
| H6 Symmetric projection and renderer firewall | PENDING | evidence→projection and projection→evidence across all domains |
| H7 Diffusion vertical migration | PENDING | conditioning, UNet stages, temporal, VAE, scheduler all evidence-backed |
| H8 Transformer/modalities tail | PENDING | schedules, wrappers, modality discovery, remaining raw facts migrate |
| H9 Metamorphic/equivalence harness | PENDING | rule-level tests become mandatory for every reader |
| H10 Corpus, Dable, performance, release closure | PENDING | frontier corpus, all images, performance budget, deleted legacy paths |

---

## 7. Unit specifications

### H0 — Baseline and unsafe-path quarantine

**Purpose:** stop the hole from growing before the full substrate exists.

1. Record current:
   - full-suite count/result;
   - blessed corpus result;
   - identity debt and declared-vocabulary snapshots;
   - number of broad evidence catches;
   - number of `legacy_asserted`/unregistered structural writes;
   - projection coverage by domain;
   - representative render time and source-resolution time.
2. Produce a responsibility census for the high-risk entry points. Mark each
   statement/branch O (observation), B (binding), E (evidence), S (semantic), P
   (projection), or X (diagnostic), beginning with transformer `parse`, diffusion
   `_parse_unet_model`, `layer_facts_from_block`, `ffn_structure_evidence`,
   `submodel_spec`, UNet block builders, and `ffn_region`.
3. Fix test isolation (`tests/test_projection_audit.py` must pass alone).
4. Extend the current guard immediately to catch the active structural marker tables,
   while H4 builds the general semantic guard.
5. Classify the active/recent U6 diffusion work (some unsafe tables may already have
   been removed by the concurrent correction; verify they remain absent):
   - **keep/rework:** conditioning enum evidence, attention placement reader,
     temporal-axis reader, UI/card improvements, focused witness configs;
   - **do not ship or reintroduce as structural truth:** `unet_blocks.yaml`,
     `vae_classes.yaml`, `schedulers.yaml`, terminal Transformer2D/KL defaults;
   - **strengthen before cutover:** mid absence, cell internals, stage temporal facts.
   Quarantine the contemporaneous config-only `block_configs` NAS projection as
   pre-H0 cleanup: it hardcodes one width formula, ignores `replace_with_linear`,
   forces pre-norm topology, and creates phantom removed-sublayer blocks. Preserve
   the raw field for H8, but do not project NAS structure until source binds the
   exact per-layer construction and replacement semantics.
6. Add a temporary no-growth gate for class-name marker tables that produce structural
   values.

**Exit:** isolated audit tests green; unsafe tables cannot be added unnoticed; no
behavioral deletion yet.

### H1 — Evidence primitives and negative-proof law

1. Extend/refactor `FactLedger` into typed `EvidenceFact` records while keeping the
   existing serialized dictionary stable for consumers.
   A compatibility lift of a legacy `code_proven=False/None` row must **not**
   manufacture `completeness="complete"` from the status. Mark its completeness
   as legacy-unknown/uninspected and mark the fact as migrated debt; enforce the
   negative-proof constructor law on native facts. Representability is not
   permission to invent epistemic metadata the old row never recorded.
2. Add distinct `RawObservation`/`BoundObservation` types so a literal AST signal cannot
   be passed where an architectural fact is required.
3. Add completeness, source spans, config paths, premises, and typed failure reason.
4. Validate negative facts: a code-proven negative requires complete inspection.
5. Replace boolean coercion at evidence boundaries with explicit value/status checks.
6. Add unit tests for:
   - proven true;
   - proven false with complete inspection;
   - illegal false from presence-only inspection;
   - ambiguous;
   - oracle missing;
   - derived fact strength no greater than premises.

**Exit:** new evidence types are behavior-neutral and can represent every current
FactLedger record.

### H2 — Closed fact registry and legacy census

1. Inventory structural facts currently authored by:
   - specs/dataclasses;
   - `extras` dictionaries;
   - opgraph/tower builders;
   - diffusion render specs;
   - labels/cards;
   - parameter estimators.
2. Register existing transformer facts first, then diffusion fact definitions.
3. Introduce `legacy_asserted` for unconverted paths; record an exact baseline count.
4. Blocking gates:
   - unknown fact key;
   - new legacy structural write;
   - drawable fact without projection policy;
   - parameter consumer without unknown policy.
   The registry and census may not be a vacuous scaffold: `REGISTRY` must be
   non-empty, every ledger fact emitted by the blessed corpus must be checked
   against it, and a poisoned unregistered fact/owner/status must make the
   corpus census fail. A test module that validates only `FactDefinition`
   constructor errors does not satisfy H2 even if it is green.
5. Never key the registry on model family. Keys describe mechanisms.

**Exit:** debt may remain, but cannot grow or hide.

### H3 — Config ownership and consumption

1. Replace the current binary accessed tracking with mandatory intents:
   - `inspected`: read while exploring;
   - `bound`: source/schema proves ownership;
   - `consumed`: decided a fact or geometry;
   - `projected`: reached a visible/machine consumer;
   - `ignored`: scoped reason and owner.
2. Source readers report which config fields gate branches/expressions.
3. The branch evaluator reads only those bound fields for that owner.
4. Keep direct semantic enums as `config_declared` only when registered with a schema
   contract and honest fallback policy.
5. Make accessed-but-unconsumed and consumed-but-unprojected blocking after corpus
   cleanup.
6. Scope ignore rules by adapter/component/owner; eliminate the flat global leak.

**Exit:** discarded multipliers and future read-never-drawn values cannot remain silent.

### H4 — Semantic identity and taint guard

Implement multiple complementary nets; no single AST heuristic is sufficient.

1. **Fact provenance rule:** a structural fact cannot cite identity fields or a
   class-name vocabulary as its deciding source.
2. **Dataflow/static rule:** flag class/model/family/name comparisons or mapping lookups
   whose result reaches structural constructors, spec fields, evidence facts, render
   kinds, or parameter formulas.
3. **YAML semantic rule:** inspect both keys and values. Arbitrary table names must be
   flagged when class-like markers map to structural vocabulary.
4. **Dependency rule:** migrated parsers/renderers cannot import declared structural
   class maps.
5. **Name-blind differential:** retain and expand existing whole-parse comparison.
6. Negative controls:
   - rename the forbidden table key;
   - move it between Python and YAML;
   - hide it behind a helper;
   - use a generator/substring/exact match;
   - map class identity to an intermediate enum before structure.
7. Lawful controls:
   - class name locating source;
   - class label displayed in a card;
   - low-level resolved callable role derived from its own code;
   - config enum whose branch is proven by source.

**Exit:** changing vocabulary/table/helper names cannot evade the guard.

### H5 — Typed source failures and extraction consolidation

1. Replace broad evidence `except Exception` wrappers with narrow expected exceptions.
2. Unexpected reader exceptions fail reader tests/Sable with file/class context; they do
   not silently enter a fallback.
3. Introduce the explicit `RawProgramIndex` and `BoundProgramGraph` products from
   Section 4A; mechanism readers cannot query AST ad hoc after migration.
4. Consolidate shared AST scanning/init construction work so `ForwardOps` and callable
   closure views consume one raw class registry where semantics genuinely overlap.
5. Preserve their distinct products: literal top-level forward facts versus folded
   followable callable closures.
6. Add inspection completeness to each reader. Presence-only readers cannot prove
   absence.
7. Cache source registry/construction results in `ParseContext`; no repeated source
   resolution or reparse per fact.
8. Add the stage import-boundary test from Section 4A.6.

**Exit:** unsupported syntax is visible and typed; reader bugs cannot masquerade as
oracle missing.

### H6 — Symmetric projection and renderer firewall

1. Replace `DRAWABLE_FAMILY_SEGMENTS` with registry-driven projection obligations.
2. Add projection receipts for modalities, fusion, projectors, diffusion conditioning,
   UNet stages, VAE, scheduler, positional embedding stage, and parameter facts.
3. Add the reverse audit: structural graph node/card assertion with no evidence fact.
4. Bind prose templates to fact keys. Phrases such as “Transformer2D,” “AlphaBlender,”
   “KL,” “GEGLU,” and “Euler” count as structural claims and owe evidence.
5. Introduce `ProjectionPlan`: editorial Tier-1/Tier-2/Tier-3/hidden decisions consume
   the semantic graph and never raw source/config.
6. Keep the full raw/bound graph for audit while semantic rewrites fold plumbing such as
   transpose/reshape into mechanisms; do not discard it during extraction.
7. Enforce the renderer import firewall.
8. Keep presentation-only facts explicit and reviewed.

**Exit:** adding a new architectural domain automatically creates proof obligations.

### H7 — Diffusion vertical migration

H7 is serial by sub-unit. Each sub-unit uses shadow → cutover → deletion.

#### H7.1 Conditioning

- Keep `encoder_hid_dim_type` / `addition_embed_type` as direct declared evidence.
- Inspect the resolved constructor/forward to bind enum values to projector modules and
  consumed kwargs when source exists.
- Discover actual pipeline components; never fabricate a text tower when none exists.
- Unknown declared enum renders external/unknown conditioning.
- Project the same fact into hero, K/V label, side input, cards, and JSON.

#### H7.2 UNet construction graph

- Resolve the exact root denoiser and exact down/mid/up stage classes.
- Extract stage construction, order, skip production/consumption, sampling placement,
  repeated counts, and mid presence.
- Config block-type strings are addresses/constructor records only.
- Negative mid proof requires complete root construction/forward inspection.
- If internals are unresolved, draw the proven U skeleton with opaque stage cells.

#### H7.3 UNet cell evidence

For each exact stage owner, separately evidence:

- ResNet cell class and ops;
- self-attention placement;
- cross-attention placement;
- attention wrapper/cell ops;
- FFN presence, storage, and activation;
- normalization and timestep conditioning;
- sample op;
- call order.

Placement evidence cannot prove implementation evidence. Delete `unet_blocks.yaml` and
all terminal Transformer2D defaults after cutover.

#### H7.4 Temporal evidence

- `has_temporal_axis` remains a model-level shape fact.
- Per-stage code separately proves TemporalResnet, temporal attention, temporal conv,
  AlphaBlender, and spatial/temporal order.
- Only proven stages render those ops.
- Video with unresolved internals shows frames in geometry and opaque temporal stages,
  never a fabricated flat-2D or universal-AlphaBlender cell.

#### H7.5 VAE evidence

Resolve from exact VAE code:

- latent distribution/quantization kind;
- conv dimensionality and causality;
- quant/post-quant conv;
- decoder entry/mid/up construction;
- norm/activation;
- temporal sampling;
- stage counts and geometry.

Unknown does not mean KL. Delete `vae_classes.yaml` as structural vocabulary after
cutover. Class names remain addresses/display only.

#### H7.6 Scheduler evidence

Inspect the exact scheduler `step()` construction and state usage. Evidence mechanisms,
not marketing families:

- previous model-output/history use and order;
- predictor/corrector or substep structure;
- consistency coefficients;
- stochastic noise injection;
- velocity/epsilon/sample interpretation;
- additive/subtractive update operands;
- state carried across steps.

Config supplies coefficients and prediction type where source proves their use. Unknown
scheduler gets a generic opaque step, not Euler. Delete structural `schedulers.yaml`.

#### H7.7 Diffusion conformance binding

- Every Tier-1 UNet/VAE/scheduler drill binds to an exact callable or emits typed
  unresolved.
- Extend op, wiring, fact, and nested conformance to these owners.
- Projection audit covers all H7 facts.

**H7 witness matrix:**

| Mechanism | Positive | Equivalent/control |
|---|---|---|
| conventional Transformer2D UNet | SDXL | unknown Attn-named class must stay unknown |
| plain cross attention | DeepFloyd / Kandinsky 2.2 | same config geometry with different resolved stage code |
| custom no-mid conv-U | Kandinsky 3 | incomplete source must not prove no-mid |
| image conditioning | Kandinsky 2.2 decoder | text-conditioned SDXL remains text |
| temporal UNet | SVD | video root with a non-temporal stage; image UNet negative |
| VQ/MoVQ | Kandinsky VQ | AutoencoderKL control; renamed equivalent class |
| 3D/causal VAE | Wan/CogVideoX witness | unknown video VAE stays unknown |
| first-order current-output step | Euler witness | DPM++/UniPC/LCM distinct mechanisms |
| consistency step | LCM | renamed class with same step code |
| multistep/history | DPM++/UniPC | class-name twin with no history must not match |

### H8 — Transformer and modality tail

1. Migrate remaining raw/default facts into the registry.
2. Complete source-bound schedule expressions for all six Run 77 spellings.
   This includes `block_configs`: source must bind `no_op`,
   `replace_with_linear`, `ffn_mult`, width rounding, surviving norm/residual
   topology, and parameter ownership before any of them may change a layer drawing.
3. Project embedding/logit multipliers and RoPE/context-stretch dialect facts.
4. Bind router order/renormalization and shared experts to exact construction/forward.
5. Complete encoder-decoder topology ownership and wrapper-local head facts.
6. Discover vision/audio/projector/fusion components from wrapper construction; config
   keys remain addresses/declared fallback.
7. Replace any remaining renderer/config inference with evidence consumption.

**Exit:** transformer/modalities and diffusion obey the same architecture contract.

### H9 — Metamorphic and equivalence harness

Create reusable tests every new reader must instantiate:

1. **Rename identity:** same code, renamed class/model id → same structural facts.
2. **Identity collision:** same class string, different code → facts follow code.
3. **Config branch flip:** same code, config value flips only the source-bound fact.
4. **Irrelevant config flip:** unrelated field cannot change structure.
5. **Source missing:** direct declarations remain qualified; code-only facts unknown.
6. **Partial extraction:** may prove found presence, never absence.
7. **Ambiguous candidates:** no union-vote fabrication; owner remains unresolved.
8. **Factory/helper/inheritance:** exact constructed callable still resolves.
9. **Component collision:** sibling component cannot prove the target owner.
10. **Projection symmetry:** remove a receipt → omission fails; add an unevidenced node →
    fabrication fails.
11. **Test isolation:** every test file must collect/pass independently; no suite-order
    dependencies.

The harness should generate synthetic minimal Python sources so it tests mechanisms,
not installed model names.

### H10 — Corpus, Dable, performance, and release closure

1. Expand the regression corpus with at least one frontier witness per fact mechanism,
   not merely per family.
2. Run every isolated unit file, targeted domain suite, then full suite.
3. Run name-blind and config-ablation matrices.
4. Render all distinct views for every affected witness; inspect every PNG.
5. Compare equivalent-model controls and document intentional visual differences.
6. Measure source parse/cache performance. One source registry per `ParseContext`; set a
   regression budget before enabling the rail globally.
7. Delete compatibility tables, dead loaders, old fallbacks, and legacy allowlist rows.
8. Update `PROJECT_CONTEXT.md`, `sable_dable_playbook.md`, and the public extension guide
   with the final contract.
9. Soumil alone decides and performs blessings/commits.

**Campaign completion:** zero unregistered structural facts; zero new legacy assertions;
zero class-name-to-structure paths; all drawable facts symmetrically audited; full suite
and corpus green; all changed pixels reviewed.

---

## 8. Required review checklist for every architectural change

This checklist is deliberately explicit so a lower-intelligence reviewer has enough
guardrails to catch local-but-ungeneral fixes.

### 8.1 Mandatory pre-change renderer-preservation gate

This gate runs **before production code is edited**. Evidence hardening is not
permission to damage a correct renderer. A principled change can still regress an
existing model through a shared fallback, grouping fingerprint, block/card id,
registry dispatch, parameter estimator, source-resolution mode, or layout contract.

#### Pre-check P-1 — Freeze the exact baseline without re-blessing

Record, for the affected witness set:

- IR JSON and structural signatures;
- selected evidence values, status, owner, source, and completeness;
- top-level and nested block ids, view keys, child ids, and click targets;
- SVG/HTML hashes and existing blessed regression state;
- parameter totals and assumption ledger;
- warnings, unresolved findings, conformance results, and config-consumption audit;
- source provenance and whether the witness is code-resolved or fallback-only.

Never regenerate blessings during this gate. A moving baseline cannot detect damage.

#### Pre-check P-2 — Build a consumer and blast-radius graph

For every field/function/type being changed, use call-site search to enumerate:

```text
producer → binder → fact/IR field → semantic builder → renderer/card/JSON
         → grouping/signature → params → conformance → corpus/blessing
```

Explicitly inspect these high-risk boundaries:

1. a default/fallback shared by several mechanisms;
2. unknown-kind dispatch at both summary and drill depth;
3. owner qualification across root/component/layer/stage;
4. `signature()` changes that merge or split block types;
5. block ids/view keys that control click routing and card deduplication;
6. parameter code that assumes a detailed mechanism;
7. source-present versus source-missing behavior;
8. serialization keys consumed outside the edited adapter;
9. global/context render state and concurrent renders;
10. visual layout assumptions such as required mid/stage/card nodes.

The pre-check is incomplete until every consumer is either covered by a control or
listed as an unresolved risk to Soumil.

#### Pre-check P-3 — Select witnesses by mechanism and evidence state

The minimum matrix is not “the model being fixed plus one popular model.” Include:

- the failing witness;
- an equivalent model with the same mechanism but different identity;
- a same-name/same-config-shaped counterexample with different internals;
- an unaffected model using the shared renderer/builder;
- dense and gated, attention and mixer, 2-D and temporal, or other applicable
  sibling mechanisms;
- source-present, source-partial, source-missing, and ambiguous cases;
- one blessed corpus witness for each touched view archetype;
- one nested drill witness, not merely the closed top-level diagram.

#### Pre-check P-4 — Predict and classify every expected delta

Before implementation, write an expected-delta ledger:

| Delta class | Meaning | Release treatment |
|---|---|---|
| preservation | already-correct output must remain byte/structure/pixel stable | any drift is a blocker |
| correctness repair | a proved wrong claim becomes correct or honestly opaque | requires source evidence and Soumil visual review |
| provenance-only | pixels stable; evidence status/owner/source improves | mechanical + ledger review |
| intentional editorial | truth unchanged but presentation changes | Gate C and explicit blessing decision |
| unexplained | not predicted by the ledger | always a blocker |

“The tests changed because the new law is stricter” is not a classification. Every
changed model/card/op must have a fact-level reason.

#### Pre-check P-5 — Run the no-break sequence in increasing radius

1. reader/binder unit tests and counterexamples;
2. affected adapter tests in isolation;
3. shared builder, block-schema, view-registry, and click-card tests;
4. `test_projection_audit.py`, conformance, config ownership, and identity nets;
5. parameter and serialization tests;
6. `test_coverage.py` to exercise every registered view;
7. `test_sable.py` regression corpus without blessing;
8. full suite;
9. before/after galleries for only the predicted visual deltas;
10. Dable review of top-level and every affected nested drill.

If any preservation witness moves, stop and diagnose before continuing. Do not widen
the accepted diff, alter a golden, weaken the assertion, or hide the delta behind an
opaque fallback merely to make the suite green.

#### Pre-check P-6 — Two-key release rule

A change may ship only when both are true:

1. **fidelity key:** every changed claim is supported by correctly scoped evidence;
2. **preservation key:** every already-correct control is unchanged across all
   affected consumers, or its deliberate change was separately approved by Soumil.

The fidelity key cannot excuse renderer breakage. The preservation key cannot preserve
a known lie. When they conflict, keep the code unblessed, show Soumil the exact source,
IR, card/pixel delta, and risk, and request the product decision.

1. State the architectural fact in one sentence without a model name.
2. Name its exact owner path.
3. Show the source line/class/method that proves the mechanism.
4. If config participates, show where source reads that field.
5. State whether the reader is complete enough to prove absence.
6. List every premise used for a derived fact.
7. Show what happens when source is absent, partial, or ambiguous.
8. Search for every downstream projection and parameter consumer.
9. Confirm the renderer does not infer additional facts.
10. Rename the witness class mentally/testably: would the fix still work?
11. Give the same class name different code: would the result follow code?
12. Give similar config different code: would the result remain honest?
13. Give unknown code: does the result stay opaque/unknown?
14. Add a negative control that the old heuristic would misclassify.
15. Run the test file independently.
16. Run targeted suites, full suite, regression corpus, and Dable images.
17. List possible breakages and intentional visual deltas before blessing.

A change is not “general” merely because its marker moved into YAML, its helper has a
generic name, or it fixes several related models.

---

## 9. Risk register

| Risk | Failure mode | Mitigation |
|---|---|---|
| wide visual churn | honest unknowns replace guesses across many blessings | shadow mode, one fact family at a time, explicit galleries |
| false blocking | source unavailable interpreted as extractor failure | typed oracle-missing vs ambiguous vs unsupported |
| false negative proof | incomplete registry lacks a field | completeness law; presence-only readers cannot return false |
| performance regression | repeated AST scans per fact/stage | one cached registry/construction graph per ParseContext |
| new bypass APIs | developer writes directly to dict/renderer | closed registry, raw-write census, renderer firewall |
| guard gaming by renaming | new YAML/helper name avoids enumeration | semantic taint + arbitrary-name tests |
| parameter-count drift | unknown gate/tie/storage silently chooses formula | registered unknown policy and estimate provenance |
| test false green | known witness encodes the same assumption | metamorphic counterexamples and isolated collection |
| source-version dialect | installed and remote source differ | source spans/version provenance; exact resolved bundle |
| migration collision | several units edit monolithic parsers simultaneously | serialize high-conflict units; narrow domain cutovers |
| overly ambitious interpreter | AST evaluator grows without bounds | support explicit expression subset; unresolved is valid |
| doctrine drift | future documentation revives identity vocab | extension guide and blocking gates reference invariants |

---

## 10. Measures of progress

Track mechanisms, not only passing tests:

- registered structural facts / total discovered structural writes;
- `legacy_asserted` count;
- facts with complete projection coverage;
- projections with evidence receipts;
- config reads by intent: inspected / bound / consumed / projected / ignored;
- broad evidence exception count;
- code-proven negative facts with complete inspection;
- identity-taint findings and declared address-only uses;
- exact callable binding rate for Tier-1 drills;
- metamorphic reader matrices completed;
- frontier corpus mechanisms covered;
- full-suite and isolated-test results;
- changed view count and Dable-reviewed count;
- source-resolution/AST-cache performance.

No unit is DONE because its focused tests pass. The tracker row moves to DONE only after
its stated gates, equivalent controls, corpus, and required images are complete.

---

## 11. Immediate execution order

When Soumil authorizes implementation, start in this order:

1. **H0:** fix isolated test collection; baseline current tree; quarantine active class
   maps/defaults without deleting the useful U6 witness work.
2. **H1 + H2:** land behavior-neutral evidence primitives and closed registry/census.
3. **H4 early slice:** generalized identity-table negative controls, so U6 cannot add
   another bypass while the full taint system is built.
4. **H3:** make config ownership/consumption real, activating the currently inert
   accessed-but-unprojected net.
5. **H7.1–H7.3:** conditioning + exact UNet construction/cells; cut over the active
   Kandinsky/DeepFloyd fixes without class maps.
6. **H7.4–H7.6:** temporal, VAE, scheduler.
7. **H5 + H6:** finish typed failures/consolidation and projection symmetry as the
   diffusion domains migrate.
8. **H8:** remaining transformer/modality facts.
9. **H9 + H10:** permanent metamorphic harness, frontier corpus, exhaustive Dable,
   deletion and release closure.

This order fixes the enforcement substrate before asking it to carry the highest-risk
diffusion facts, but it still preserves shippable, per-mechanism units rather than a
single deep refactor bundle.

---

## 12. Final definition of success

The project is hardened when an engineer trying to add:

```python
if "SomeFamily" in class_name:
    stage_kind = "transformer2d"
```

or:

```python
stage_kind = detected_kind or "transformer2d"
```

cannot obtain a green build, regardless of whether the logic is placed in Python,
YAML, a helper, a renderer, or an intermediate enum.

The only green path is:

```text
exact owner resolved
→ exact source mechanism inspected
→ source-bound config branch evaluated
→ typed fact with provenance/completeness
→ registered IR projection
→ symmetric conformance + counterexamples + pixels
```

That is the point where the doctrine stops depending on reviewer intelligence and
becomes a property of the library.

---

## 13. Project working constitution — ground rules for every future change

This section is the short operational constitution for Soumil, implementers, reviewers,
and coding agents. It translates the deeper architecture above into rules that must be
answered before a change is accepted.

### 13.1 The project intent

The library is not a catalogue of model-family diagrams. It is a static architecture
reconstructor and explainer.

Its contract is:

> Given a checkpoint declaration and the source code that implements it, reconstruct the
> architecture this checkpoint actually uses, reduce it into a truthful semantic model,
> and present the important computation without inventing what could not be proven.

The product serves three truths simultaneously:

1. **Source truth** — what code is constructed and can execute.
2. **Checkpoint truth** — which code-controlled branches and values this config selects.
3. **Editorial truth** — the smallest diagram that preserves the correct mental model.

Editorial simplicity may fold details. It may never change source/checkpoint truth.

### 13.2 Ground rules for building future functionality

#### Rule G-1 — Start with a mechanism statement, never a model name

Before coding, describe the problem without identity:

```text
BAD:  Add support for Kandinsky attention.
GOOD: A conv-U can declare per-level self/cross-attention in source rather than
      down_block_types; acquire that construction schedule and bind it per stage.
```

If the statement cannot be made without a model/family name, the mechanism is not yet
understood.

#### Rule G-2 — Identify the exact owner

Every fact belongs to an exact root/component/variant/layer/stage/callable. “The model
has a gated FFN” is insufficient when one sibling is dense. “The model is temporal” is
insufficient to claim that every stage has AlphaBlender.

#### Rule G-3 — Acquire before interpreting; interpret before presenting

Implementation order is mandatory:

```text
literal observation → exact binding → typed fact → semantic region → projection plan
```

Do not start in a renderer and work backward. Do not begin by adding a label, card, YAML
row, or special block.

#### Rule G-4 — Code proves field meaning; config selects checkpoint value

Config remains authoritative for declared values and geometry. Code establishes what
those values control. A direct public semantic enum may be used as `config_declared`,
with that weaker status visible in provenance.

#### Rule G-5 — Identity may locate source and provide a display name only

Class names, model types, repo ids, architecture strings, and family labels cannot
select topology, operations, formulas, or block kinds.

#### Rule G-6 — Unknown is a successful result

When evidence is missing, partial, or ambiguous, return a typed unknown. Do not use the
most common architecture, the previous diagram, or a visually satisfying template.

#### Rule G-7 — Negative facts require stronger proof than positive sightings

Finding a call can prove presence. Not finding a call proves absence only when inspection
is complete for the exact owner and relevant path.

#### Rule G-8 — One fact has one author and all projections consume it

SVG, cards, JSON, labels, prose, parameter counts, and conformance must not independently
recalculate the same fact. They consume the owner-qualified registered fact.

#### Rule G-9 — Preserve raw truth; filter only at the editorial boundary

Record transpose, reshape, split, concat, casts, and calls during acquisition. Semantic
rewrites fold them into mechanisms. Projection policy decides their visibility. Never
discard a signal early merely because it usually looks unimportant.

#### Rule G-10 — Shared mechanisms, qualified instances

Use one FFN/attention/norm/scheduler mechanism schema and canonical semantic builder,
but create a separate evidence instance for every owner. Sharing must not become a
same-role union that erases context.

#### Rule G-11 — Every new fact creates two proof obligations

The implementation must prove:

1. evidence → projection: the fact is not silently dropped;
2. projection → evidence: the diagram does not claim more than the fact.

#### Rule G-12 — Generality is demonstrated by counterexamples

Every mechanism reader needs tests for renamed identity, identity collision, config
branch flips, partial source, missing source, ambiguous candidates, and an equivalent
control. A known-model golden is integration evidence only.

#### Rule G-13 — A green mechanical suite is necessary, not sufficient

Changed views require pixel inspection. Changed architecture requires comparison with
the exact source. A re-bless is a reviewed product decision, never a test repair.

#### Rule G-14 — Build in shadow, cut over narrowly, delete afterward

For high-risk facts, compare the new evidence path to the old path first. Cut over one
mechanism/owner domain, inspect the delta, then delete the obsolete fallback. Do not mix
several migrations into one opaque refactor.

#### Rule G-15 — The implementation must make the rule easier to follow next time

A valid fix does not merely correct output. It adds or strengthens the fact type, source
binding, registry, conformance net, counterexample harness, or extension documentation
so the same failure class is harder to recreate.

#### Rule G-16 — Familiar names never complete a missing mechanism

A token such as `recurrent`, `lightning_attention`, `cross_attention`, `temporal`,
`VQ`, or a familiar class name proves only what its declaring contract proves. It
does **not** license the implementer to fill in a remembered implementation.

The following inference is forbidden:

```text
config/source says "recurrent"
    → this is probably RG-LRU
    → draw causal conv + real gate + recurrent state
```

The lawful result is:

```text
placement/type-name evidence
    → opaque declared mixer at this owner
    → acquire exact source construction + forward/dataflow evidence
    → only then expose proven conv/gate/state operations
```

This applies even when the guessed mechanism is correct for the model currently
being tested. Correctness by familiarity is not reusable evidence.

#### Rule G-17 — Evidence scope may stay equal or shrink downstream; never grow

Every evidence producer must state its proved scope explicitly, for example:

- `placement_only`;
- `declared_type_name`;
- `construction_presence`;
- `forward_order`;
- `complete_callable_graph`;
- `complete_negative_scope`.

A consumer must declare the minimum scope it requires. If the producer's scope is
weaker, the consumer abstains. A mapper, renderer, card builder, label helper, or
parameter estimator may not supply the missing proof from a default or convention.

Mandatory review trace for every new architectural claim:

```text
claim → required scope → producing evidence → exact owner → completeness
      → all consumers → counterexample where the familiar implementation differs
```

If any arrow cannot be filled, the output must be opaque/unresolved and the agent
must report the missing proof to Soumil rather than implementing a plausible shape.

#### Rule G-18 — Unknown-kind fallthroughs are prohibited at every projection

Dispatch code must distinguish these cases explicitly:

1. recognised mechanism with sufficient evidence → canonical detailed builder;
2. recognised name/placement with insufficient internals → opaque builder;
3. truly unknown → opaque unresolved builder.

`mapping.get(kind, standard_attention_builder)` and equivalent defaults are banned
for architecture-bearing consumers. Every drill level must obey the same rule: an
opaque summary may not open into a conventional detailed card.

Required regression tests include:

- a familiar token whose real implementation differs from the presumed one;
- an unknown kind that must remain opaque in SVG, card children, JSON, and params;
- partial source that proves presence but not absence or order;
- model-level evidence that must not be stamped onto every stage/layer;
- a happy-path known model, used only after the counterexamples pass.

### 13.3 How a discovered problem must not be solved

The following are non-fixes unless explicitly proven to be presentation-only.

#### Never solve by identity

Do not add:

- model/repository branches;
- class-name substring/exact-match structure selection;
- family profiles;
- class-keyed structural YAML tables;
- “known models of this kind” fallbacks.

Moving the branch into YAML or a generically named helper does not change its nature.

#### Never solve by conventional defaults

Do not turn missing evidence into:

- gated FFN;
- causal attention;
- pre-norm;
- Transformer2D;
- KL VAE;
- Euler scheduler;
- text conditioning;
- two ResNets or one transformer per block;
- global temporal operations.

Defaults are permitted only for presentation conventions that cannot alter architectural
meaning, and they must be registered as such.

#### Never solve only in prose or labels

Changing “Feed-forward” to “SwiGLU” does not fix missing FFN evidence. Adding
“AlphaBlender” to a card does not establish a temporal branch. Repair the fact first;
the wording must follow automatically.

#### Never solve only in a renderer

Renderers may choose layout, tier, spacing, grouping, and phrasing. They may not inspect
raw config/source/identity to decide structure.

#### Never treat an extractor failure as absence

Do not catch a broad exception, return `None`, and enter a structural fallback. Report
the unsupported syntax/ambiguous ownership/reader defect distinctly.

#### Never use a union to prove an individual owner

Facts from sibling classes, layer variants, modalities, or stages cannot be unioned and
assigned to one drill unless equivalence is proven for that exact fact.

#### Never make the test repeat the implementation assumption

An assertion that a known class name produces a known label is not a generality test.
Test source mechanisms and adversarial equivalents.

#### Never refresh regression locks before understanding the delta

First classify the visual change as correction, honest degradation to unknown,
presentation-only improvement, or accidental regression. Only Soumil authorizes the
new lock after code and pixels are reviewed.

#### Never broaden scope silently

If a safe fix requires a new interpreter capability, source acquisition path, new
architecture domain, or a materially larger refactor, stop and report the expansion.
Do not hide it inside the local model fix.

### 13.4 What must be conveyed to Soumil when a problem is found

Every report must separate **symptom**, **claim**, **cause**, **evidence gap**, and
**proposed mechanism**. Use this template.

#### Mandatory problem report

```text
1. USER-VISIBLE SYMPTOM
   What the diagram currently shows and why it is misleading.

2. ARCHITECTURAL CLAIM AT STAKE
   One identity-free sentence describing the fact.

3. EXACT OWNER
   Component / variant / layer / stage / callable affected.

4. SOURCE + CHECKPOINT TRUTH
   Exact source class/method/line and relevant config path/value.

5. FAILURE ALTITUDE
   [ ] Acquisition: source/config signal was not recorded
   [ ] Binding: signal was attached to the wrong/ambiguous owner
   [ ] Interpretation: observations were mapped to the wrong mechanism
   [ ] Semantic normalization: mechanism graph is wrong/incomplete
   [ ] Projection: correct fact was omitted/overstated visually
   [ ] Verification: guard/corpus/test failed to expose it

6. EVIDENCE STATUS
   code_proven / code_and_config / config_declared / class_default /
   ambiguous / oracle_missing / unknown; state inspection completeness.

7. WHY EXISTING NETS MISSED IT
   Name the exact scope, exemption, false-green, or missing counterexample.

8. BLAST RADIUS
   Which mechanism shapes are affected; do not list only model names.

9. COUNTEREXAMPLES
   Same code renamed; same identity different code; similar config different
   mechanism; missing/partial source; equivalent unaffected control.

10. PROPOSED FIX SHAPE
    Which acquisition/binding/fact/semantic/projection layer changes.

11. FORBIDDEN SHORTCUTS CONSIDERED
    Identity row, default, prose-only fix, renderer branch, union vote, re-bless.
    State explicitly that they are rejected and why.

12. POSSIBLE BREAKAGES
    Existing facts/views/params/conformance/performance likely to move.

13. VERIFICATION PLAN
    Reader tests, metamorphic matrix, isolated test collection, full suite,
    corpus, exact changed galleries, Dable inspection.

14. DECISION REQUIRED FROM SOUMIL
    Pixel/editorial Gate C, scope expansion, honest-unknown visual change,
    or blessing decision. "None" if purely mechanical and already authorized.
```

### 13.5 How the nature of a problem should be described

Use one primary category and optional secondary categories:

| Category | Meaning | Typical response |
|---|---|---|
| Acquisition gap | source/config signal was never captured | extend raw reader without choosing presentation |
| Binding gap | correct signal, wrong component/class/stage | fix construction/owner resolution |
| Interpretation gap | observations mapped to wrong mechanism | fix mechanism reader and counterexamples |
| Completeness gap | partial reader claimed absence | downgrade to unknown; add completeness proof |
| Identity leak | name/family selected structure | remove path; source-resolve; strengthen taint guard |
| Default leak | missing evidence became a conventional value | tri-state fact; delete fallback |
| Ownership leak | sibling/union fact assigned to target | owner-qualified evidence and equivalence |
| Projection omission | fact exists but is not shown | add owner-qualified projection receipt |
| Projection fabrication | renderer/card claims an unevidenced fact | remove claim; bind to fact or show unknown |
| Abstraction error | too much/too little low-level detail shown | adjust semantic rewrite or projection policy |
| Verification gap | current nets/corpus could not catch class | add permanent negative control/frontier witness |
| Presentation defect | truth correct; pixels/wording/layout misleading | renderer-only fix, still inspect all affected views |

This vocabulary prevents every problem from being described vaguely as “parser support”
or “model family handling.”

### 13.6 What an implementer must tell Soumil before changing code

Provide a short preflight containing:

1. mechanism statement;
2. owner;
3. failure category;
4. evidence currently available;
5. proposed layer(s) to edit;
6. why no identity/default/renderer shortcut is needed;
7. expected changed views and controls;
8. risks and any decision gate.

If these cannot be stated clearly, investigation is not complete and implementation
must not start.

### 13.7 What an implementer must tell Soumil after changing code

Report:

1. exact facts and owners changed;
2. exact files and responsibility layers changed;
3. old shortcut/default/table deleted or remaining transitional debt;
4. evidence provenance and completeness behavior;
5. counterexamples and equivalent controls run;
6. isolated, targeted, full-suite, and corpus results separately;
7. every changed gallery/image inspected and what moved;
8. parameter/performance/conformance changes;
9. possible remaining breakages;
10. explicit items requiring Soumil's visual/blessing decision.

“Tests pass” is never a complete handoff.

### 13.8 Soumil's fast rejection questions

Soumil should be able to reject or pause a proposal by asking:

1. Would this still work if the class and repo were renamed?
2. What exact code proves this fact?
3. What exact owner does that proof belong to?
4. Does config select a branch that code actually binds?
5. What happens with partial or missing source?
6. Is any renderer or YAML table deciding architecture?
7. Did a weak fact become several stronger visual claims?
8. What adversarial counterexample was tested?
9. Which old path is deleted when this lands?
10. Which pixels will change, and who reviewed them?

If the answers are vague, the proposed fix is not ready.

### 13.9 Final working principle

The project should optimize for this outcome:

> A future implementer does not need exceptional architectural judgment to avoid
> fabrication. The types, dependency graph, evidence statuses, counterexample harness,
> and reporting contract make the honest implementation the easiest implementation.

# ------------------------------ STUFFF STARTSSS NOW ---------------------------------

# 16. Independent audit correction and binding recovery plan (AUTHORITATIVE)

**This Section 16 is AUTHORITATIVE. It supersedes every conflicting status and
direction claim in Sections 6 (tracker), 14 (completion log), and 15 (judgment
handoff) without deleting their historical record.** Where §6/§14/§15 say a unit
is DONE and §16 says ACTIVE/RESTART, §16 wins. The implementer follows §16's
execution order and acceptance conditions exactly; nothing in §14/§15 authorizes
skipping a §16 requirement.

## Summary

No current H0–H4 unit may remain marked `DONE` until the corrected exit criteria below pass. Hold the entire uncommitted implementation; split it into reviewed commits only after recovery.

Verified audit facts to record:

- Current collection: **1004 tests**.
- Hardening subset excluding Sable: **423 passed**.
- The prescribed grouped gate is currently invalid: `test_sable.py` fails collection because it imports `tests.test_diffusion`.
- H0 display-map pinning does not detect added entries.
- H1 permits a public `migrated_legacy=True` negative-proof bypass.
- H1 permits a derived negative from presence-only premises.
- H1 drops structured source spans when serializing a native fact.
- H2 accepts a code-proven negative without completeness.
- H2 sees top-level `extras` keys only; nested structural additions remain invisible.
- H3 records absent canonical fields as accessed and consumed.
- H3 loses the actual alias that supplied a value.
- H3 unions bare field names across components, allowing sibling components to clear each other’s debt.
- Three diffusion reads were added to clear audit findings but have no structural consumer.
- Existing parser/render/default seams can still turn unknown evidence into conventional architecture.

## Corrected Status and Decisions

Append this authoritative status table:

| Unit | Corrected status | Reason |
|---|---|---|
| NAS quarantine | DONE sub-unit | Unsafe `block_configs` projection removed and pinned |
| H0 | ACTIVE | Baseline exists, but display pins, broad exemptions, single-entry maps and helper/dataflow evasions remain |
| H1 | ACTIVE | Useful types exist, but legacy bypass, derived-negative completeness and provenance serialization are unsound |
| H2 | DONE (repaired) | record_typed registry gate + typed legacy_asserted (Part A); StructuralWrite census over all author surfaces, static+runtime, structured legacy register, 5 poisons (Part B) — 359 + 41 passed |
| H3 | DONE (procedure 2, 22ddaf0) | RESTART cutover complete: owner-scoped `ConfigAccessEvent` ledger is the single source; global `_touched/_bound/_consumed` + `capture_accesses` DELETED; `config_audit` DERIVED (compat==old). Re-vet (proc 9) fixed one un-ported `capture_accesses` test the fast smoke had skipped |
| H4 | DONE (procedure 3, ea15aff) | Taint net: class-name→structural-sink caught (production=0, boundary flipped); code-shape exempt; renderer/parser dependency-firewall gate. Interprocedural/mapping-lookup preventive-future (0 real flows) |
| H5 | DONE-core (procedure 4, 706818b) | No-broad-reader-exception RATCHET (shrink-only) + 2 evidence readers typed (8→6). Full program-index is the ratchet's shrink-work |
| H6 | DONE (procedure 5, a969b0f) | Registry-driven reverse-fabrication audit (drawn ⊆ registered ∪ DRAWN_UNLEDGERED_DEBT) + projection obligation; firewall in H4; prose-receipts via label-lint |
| H9-core | DONE (procedure 6, 6447e42) | Reusable metamorphic harness: 5 relations, proven on decoder+multimodal+diffusion; missing-source FIRES |
| H7 | RAIL + slice (procedure 7, 6ed88a3) | 3 removed reads → registered typed facts w/ pending-projection debt; metamorphic on FLUX. Per-family migration ongoing |
| H8 | RAIL + slice (procedure 8, 10f6fa5) | `sinks`: drawn-but-unledgered → registered code-proven fact (gpt-oss witness), census intact. Per-mechanism migration ongoing |
| H9/H10 | DONE (procedure 9) | H9 frontier metamorphic matrix (5 archetypes, 6 passed). FORESEER RE-VET found + fixed TWO real defects, both in `test_sable.py` (the file every fast smoke skipped): (1) proc-2's un-ported `capture_accesses` call — ported to the `capture_events()` ledger, suite-wide sweep found no siblings; (2) the 3 diffusion reads proc-2 removed on a FALSE "audit is advisory" premise (it is BLOCKING) — 10 corpus regressions across 8 models; fixed by teaching `config_field_audit` to EXCUSE registry `PENDING_PROJECTION_DEBT` canonicals (a declared classification — reads stay removed, proc-7's guard holds, a NEW unread field still blocks) + `test_config_field_audit_excuses_the_pending_projection_fields`; verified test_h7 5-passed / units 50-passed / corpus blast 0-drift. H10 = full-suite release run (first pass 986/1 CAUGHT defect 2; final re-run with both fixes = the green gate) |

Lock the four decisions from Section 15:

- **D1:** Preserve present-only access semantics, but replace the current bare-name implementation with a scoped event ledger.
- **D2:** Do not move to H7. Complete corrected H3, full H4, H5 and H6 first.
- **D3:** Top-level `extras` census is insufficient. Cover nested extras, specs, opgraph, blocks/cards, renderer claims and parameter consumers.
- **D4:** Hold the combined commit. Land small unit commits after their own stable gates.

## Recovery Implementation

### 1. Stabilize verification and fixture boundaries

- Move shared model fixtures into an importable top-level `test_support` package.
- Remove every `from tests.*` import from Sable, conformance and vision tests.
- Add a static gate forbidding production or tests from importing another test module.
- Run every hardening/audit test file alone and in the official grouped gate.
- Record a tree fingerprint before and after every full run. A test result is invalid if tracked or untracked source files changed during execution.
- A completion-log test claim must include tree fingerprint, collection count, command, result and duration.

### 2. Repair H0’s provisional guard

- Replace table-name exemptions with an exact lawful-resource manifest containing path, table, category, permitted consumers and canonical content fingerprint.
- A display-table entry change must alter the fingerprint and fail until reviewed.
- Remove the blanket `conformance/` exemption; exempt only exact registered code-role vocabularies and consumers.
- Detect single-entry class maps and single-capital class names where they reach structural sinks.
- Remove function-name-wide address/display exemptions. Use typed address/display wrappers instead.
- Add poison controls for:
  - one-entry maps;
  - display-map population growth;
  - dict comprehensions;
  - renamed tables;
  - helper-returned enums;
  - structural data hidden under a lawful display table;
  - mappings moved into conformance or aliases files.

H0 is complete only when all observed identity-table growth paths are blocking and no exemption is based solely on a filename or table name.

### 3. Make H1’s evidence types sound

- Make legacy lifting an internal constructor path. Remove `migrated_legacy` from the public initializer.
- Native callers must be unable to opt out of the negative-proof law.
- Reject derived negative facts unless the derived fact’s effective completeness is `complete`.
- Separate `reason` from legacy `source`; never serialize a reason as source provenance.
- Define a stable legacy source label and preserve structured `SourceSpan`/config paths in the typed channel.
- Require:
  - code facts to carry source provenance;
  - config facts to carry exact config paths;
  - derived facts to carry premises;
  - failure facts to carry typed failure context.
- Add tests proving:
  - public legacy bypass is impossible;
  - derived false from presence-only evidence raises;
  - native source spans survive typed ledger storage;
  - reason and source do not alias;
  - legacy lifts remain countable debt without claiming completeness.

### 4. Complete H2 across every structural author

Introduce a line-insensitive `StructuralWrite` census keyed by module, enclosing symbol, sink kind and normalized target.

Cover:

- typed and legacy ledger writes;
- IR/spec/dataclass construction and mutation;
- every nested `ir.extras` leaf;
- opgraph `Region`/`Op` construction;
- block/card structural keys;
- renderer structural phrases and kinds;
- parameter-estimator structural reads and assumptions.

Use both static and runtime gates:

- Static scanning catches unused/new code paths.
- Corpus runtime scanning proves exercised values, owners and types.
- Every legacy entry must include owner, reason, migration unit and intended deletion—not merely a string in an allowlist.
- `FactLedger.record_typed` must validate key, owner, status, value type, completeness and parameter/projection policy against the registry.
- The registry must represent typed `legacy_asserted` debt rather than silently translating it into ordinary `asserted`.
- Add poisons for a nested extras field, new spec field, new opgraph default, card claim and parameter formula.

H2 is complete only when a new structural author cannot bypass the registry by choosing a different representation.

### 5. Replace H3 with an owner-scoped config event ledger

Delete the global `_touched/_bound/_consumed` truth model. Store audit state call-locally on `ParseContext`.

Add:

```text
ConfigAccessEvent
- component path
- config path
- canonical field
- actual alias/path used
- present/absent
- intent
- exact fact/spec owner
- fact key or geometry target
- source-binding reader
- reason
```

Required semantics:

- An absent field produces a default/class-default premise, not a fictional accessed/consumed config field.
- Alias resolution records the actual spelling that supplied the value.
- Multiple aliases with unequal values are ambiguous and cannot silently choose the first.
- Equal redundant aliases are recorded explicitly; only the selected source path is consumed.
- Root, text, vision, audio, VAE and denoiser fields remain separate even when their leaf keys match.
- `bound`, `consumed`, `projected` and `ignored` are owner-qualified joins—not global set subtraction.
- Ignore rules require adapter/component/owner and a reason.
- Preserve old diagnostic lists only as derived compatibility views during migration.

Remove the three audit-clearing diffusion reads for `max_sequence_length`, VAE `act_fn` and VAE temporal compression. Reintroduce them only through H7 typed facts with actual projections or declared pending debt.

Add counterexamples for aliases, missing fields, conflicting aliases, same key in sibling components, nested parses, concurrency and source-missing cases.

The two blocking nets become:

- accessed/bound but neither consumed nor scoped-ignored;
- consumed but neither projected nor registered as shrinking pending-projection debt.

### 6. Finish H4, then build H5/H6 before domain migration

Complete H4’s semantic taint system:

- identity/config-name sources;
- local/interprocedural propagation;
- mapping lookups and intermediate enums;
- spec/opgraph/block/renderer/params sinks;
- YAML keys and values;
- renderer/parser dependency firewall;
- lawful typed address and display sinks.

Then complete:

- **H5:** one raw program index, owner-bound program graph, typed reader failures, completeness and no broad reader exceptions.
- **H6:** registry-driven projection obligations, reverse fabrication audit, structural prose receipts and renderer firewall.
- **H9-core:** reusable metamorphic harness before H7/H8, so every migrated reader must provide rename, collision, partial-source, missing-source and equivalent-control tests.

Only after these foundations pass should H7 diffusion and H8 transformer/modalities resume.

## Existing Nonlinear Seams That Must Not Be Extended

Record these as forbidden foundations:

- Transformer facts use specs plus a partial ledger; diffusion uses raw extras; modalities use separate evidence objects; renderers and params still infer independently.
- The transformer parser interleaves observation, binding, interpretation, normalization, projection and diagnostics.
- `config_facts.yaml` reads fields and emits chips without registered architecture facts; dismantle it into typed facts or scoped non-architectural ignores.
- Class markers, scheduler markers, topology fallbacks and detailed mixer mappings remain transitional structural config debt.
- Aliases may remain only as syntax vocabulary. They cannot prove mechanism semantics.
- Conditioning enums may provide an opaque declared fallback; detailed projector/operation graphs require source binding.
- Opgraph and labels still contain default-to-MHA, causal, RoPE, gated, split-storage, SiLU and transformer fallbacks.
- Parameter estimates consume raw specs and conventions independently from evidence.
- There are still 69 broad exception catches; evidence-reader failures can collapse into fallbacks.
- Raw extraction is duplicated across forward, transitive and specialized readers.
- Corpus-only gates miss unused code and unseen mechanisms.
- Test fixtures and collection order currently hide isolation failures.

Every migrated unit must delete or quarantine the corresponding old path. “New reader plus old fallback forever” is not completion.

## Execution Order and Commit Gates

1. Append Section 16 and mark it authoritative.
2. Fix fixture isolation and verification quiescence.
3. Commit the already-reviewed NAS quarantine separately.
4. Repair and commit H0.
5. Repair and commit H1.
6. Complete and commit H2.
7. Replace H3 with the scoped event ledger in shadow mode.
8. Clean corpus accounting and flip H3 nets blocking.
9. Complete H4.
10. Complete H5, H6 and H9-core.
11. Migrate H7 one fact family at a time.
12. Migrate H8 one mechanism at a time.
13. Finish H9 frontier matrices and H10 release/Dable closure.

Only one high-conflict unit may modify parser/evidence infrastructure at a time.

## Acceptance and Stop Conditions

A unit is complete only when:

- its anti-vacuous poison tests fire;
- each audit file passes alone;
- targeted and full suites pass on the same unchanged tree fingerprint;
- corpus and conformance are green;
- preservation witnesses remain unchanged;
- every intentional image/card delta is inspected;
- no blessing is modified without Soumil’s decision;
- old paths and temporary allowlists shrink as specified.

Stop immediately when:

- the tree changes during verification;
- a new allowlist entry lacks an owner and deletion unit;
- a renderer or params path needs raw config/source access;
- a weak fact would be projected as a stronger mechanism;
- a fix requires a new interpreter capability outside the current unit;
- a test is changed merely to accept an unexplained delta.

Final achieved output:

> Every architectural claim is an owner-qualified typed fact backed by exact source/config evidence, all structural consumers use that fact, unknown remains unknown at every drill depth, identity/config vocabulary cannot select architecture, and no new model can require a renderer branch or family table to become accurate.

---

# 17. COR-5 superseding completion receipt (2026-07-14)

This receipt supersedes every earlier U0/U1 status line in this document. The
binding authority for the corrections is
`docs/U0_U1_FINAL_RECOVERY_CORRECTION_PLAN.md` (COR-0 through COR-5); the
historical REC log above is preserved unedited as history.

## Correction commits

| Unit | Commit | Content |
|---|---|---|
| COR-0 | `602dcb0` | clean-checkout anti-vacuous preservation (committed corpus inputs, authoritative 25-witness manifest, zero-skip verifier, poison matrix, git-archive isolation) |
| COR-1 | `b8cea8c` | config access primitive repair (value_state law, semantic equality incl. the 2^53 trap, null-beside-value ambiguity, consume/ignore/bind laws) |
| COR-2 | `7cd9066` | exact occurrence-to-projection accounting (ConfigOccurrenceKey primary join, structured obligations, receipts-unavailable honesty) |
| COR-3 | `ff92b4a` | unknown-safe transformer and diffusion projection (hidden_size unknowable stays None at every depth; params incomplete-not-zero; activation rivals author nothing and block) |
| COR-4 | `08d2bfe` | exact modality scopes and source-authoritative projector evidence (dotted container paths + path_exact; construction-site width binding code_bound/config_bound/derived/unavailable; generic language-width arms DELETED; wrapper rivalry law) |
| COR-5 | this commit | migration-claim gate cutover, Net-2 conditional blocking, debt census, this receipt |

## Unit states

| Unit | State |
|---|---|
| U0 | REVIEWED/DONE only after COR-0 clean-checkout zero-skip preservation — machine-green as of this receipt; awaiting Soumil's review mark |
| U1 | REVIEWED/DONE only after COR-1 through COR-4 and all exact/unknown-safe gates — machine-green as of this receipt; awaiting Soumil's review mark |
| U2 | UNLOCKED only by this COR-5 receipt upon Soumil's approval |

Claude never blesses and never marks REVIEWED/DONE; both marks are Soumil's.

## Blocking policy landed (COR-5 §10, as refined by Soumil 2026-07-14)

- Migration is claimed at exact **(owner, mechanism)** scope
  (`MigrationClaim` in `model_unfolder/evidence/registry.py`), never per
  adapter or source file. A claim names its exact config paths.
- Net 1 (`config_migration_claims`) BLOCKS every claimed scope immediately:
  within it, a present read must carry an exact path and be consumed,
  scoped-ignored, or precisely classified — violations are structured rows.
- First claimants: `root.vision/projector_out_width` and
  `root.video/projector_out_width` (COR-4's source-bound width).
- Unclaimed reads remain VISIBLE migration debt:
  `docs/COR5_NET1_MIGRATION_DEBT.md` (252 exact rows, 2 zero-consumption
  owners, each assigned to H7/H8). The census must shrink; mass registration
  was explicitly rejected as debt-laundering.
- Net 2 (`config_consumed_unprojected`) independently verifies projection and
  blocks exactly where a parse declares `projection_receipts_available=True`;
  declaring receipts with unreceipted obligations fails. No parser/renderer
  path claims completion while receipts are unavailable — U2 lands receipts.
- Ambiguities (`config_ambiguity`) remain blocking regardless of claims.
- Poison suite: `tests/test_projection_audit.py::test_cor5_*` proves an empty
  or violated declaration cannot pass vacuously (unconsumed-read poison,
  bare-funnel poison, fabricated-receipts poison, constructor guards).

## Official grouped hardening gate

Every audit file must pass ALONE and inside the grouped run; a grouped pass
does not excuse a lone failure or a collection error.

```bash
# layer 1 — each audit file alone
pytest -q tests/test_config_access.py
pytest -q tests/test_config_intents.py
pytest -q tests/test_authority_probes.py
pytest -q tests/test_projection_audit.py
pytest -q tests/test_preservation.py
pytest -q tests/test_isolation.py

# layer 2 — official grouped hardening gate
pytest -q tests/test_config_access.py tests/test_config_intents.py \
  tests/test_authority_probes.py tests/test_projection_audit.py \
  tests/test_vision_evidence.py tests/test_projector_evidence.py \
  tests/test_fusion_evidence.py tests/test_audio_evidence.py \
  tests/test_h7_diffusion.py tests/test_sable.py tests/test_preservation.py \
  tests/test_isolation.py

# layer 3 — full suite on the same unchanged tree
pytest -q

# layer 4 — clean-checkout preservation/isolation (post-commit)
pytest -q tests/test_isolation.py
```

## §12 verification capture

Recorded below after the runs on this exact tree (see the run block appended
by the COR-5 close; fingerprints from `test_support.tree_state.fingerprint`).

### §12 capture — final verification run (2026-07-15, tree of COR-5)

```text
HEAD: 08d2bfe (COR-5 working tree; this commit closes it)
git status: COR-5 named files only (registry/parser/sable/vision consumer,
            projection-audit + expanded-json tests, rebuilt manifest,
            debt census, this tracker)
tree fingerprint (before == after every layer):
  f3631924a7334224ce969ecb77ba0b11ebbbb521549decff555c5de9183ad838
collection: 1197 tests

layer 1 — each audit file alone:
  tests/test_config_access.py      46 passed                (12s)
  tests/test_config_intents.py      8 passed                (<1s)
  tests/test_authority_probes.py   26 passed, 5 xfailed     (1:04)
  tests/test_projection_audit.py   26 passed                (17s)
  tests/test_preservation.py       20 passed                (8:29)
  tests/test_isolation.py           5 passed                (2s)
layer 2 — official grouped hardening gate:
  215 passed, 5 xfailed                                     (16:29)
layer 3 — full suite (same tree):
  1192 passed, 5 xfailed, 0 failed                          (32:18)
layer 4 — clean-checkout preservation/isolation:
  run post-commit on the pushed COR-5 HEAD; result recorded in the
  current-state chapter (z-docs/07-current-state).

corpus gates on this tree:
  all-25 blessed-signature regression (blocking nets incl.
  config_migration_claims): 0 findings
  manifest rebuild: STRUCTURAL deltas [] · EVIDENCE deltas 50
  (ledgers+sable ×25 — the migration_claims register key), documented.

final receipt:
  defects closed: C0 C1 C2 C3 C4
  preservation witnesses: 25/25
  preservation skips: 0
  working-tree fingerprint: unchanged through layers 1-3
  full-suite collection/result: 1197 collected; 1192 passed, 5 xfailed
  visual matrix: previews/cor5_visual_matrix_2026-07-14 — 185 images,
    13 models across transformer/multimodal/diffusion-pipeline/DiT/UNet;
    spot-inspected (llama architecture; qwen2-vl vision path + projector
    drill, in=5120 five-op PatchMerger chain; FLUX dual-stream denoiser);
    FULL approval pass is Soumil's.
  one full-suite delta during closing, explained and lawful:
    test_expanded_json gemma4 projector assert encoded the pre-COR-4
    text-width fabrication; the embedder projects through a raw Parameter
    einsum no construction site proves, so the assert now pins
    out_features ABSENT with out_width_source="unavailable".

U0/U1 REVIEWED/DONE marks and the U2 unlock remain Soumil's alone.
```
