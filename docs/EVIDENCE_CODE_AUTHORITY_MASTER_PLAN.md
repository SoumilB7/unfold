# Evidence and Code Authority — Binding Master Execution Plan

**Status:** authoritative recovery and completion plan  
**Written:** 2026-07-13  
**Scope:** `unfold-pkg` evidence, parser, IR, renderers, parameters, conformance,
configuration vocabulary, tests, and release procedure

This is the single execution document for finishing the evidence-hardening and
config-to-code-authority transition. It incorporates:

- the governing intent in `PROJECT_CONTEXT.md`;
- the conversion analysis in `CONFIG_VS_CODE_CONVERTIBILITY.md`;
- the requirements and historical record in
  `docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md`;
- the independent audit of commits `daf056f` through `d475d87`;
- adversarial probes run against the resulting tree.

Where this document conflicts with a status, sequence, or completion claim in an
older plan or completion log, **this document wins**. The older documents remain
valuable history and rationale; they are not the current tracker.

---

## 1. The outcome this plan must produce

The project is a truth machine about architecture. It must reconstruct a model
from the model's own declarations without allowing identity, convention, field
presence, presentation vocabulary, or renderer defaults to manufacture structure.

The final invariant is:

> Every architectural claim is an owner-qualified typed fact backed by exact
> source/config evidence. Every structural consumer uses that fact. Unknown stays
> unknown at every drill depth. Identity and vocabulary cannot select
> architecture. A new model never requires a family table or renderer branch to
> become accurate.

This does **not** mean “delete config.” It means “delete structural authority from
config where the mechanism lives in code.”

### 1.1 The authority law

> **Code proves the mechanism and proves what a field controls. The checkpoint
> config supplies the value selected by that checkpoint.**

Examples:

- `hidden_size=4096` remains config truth. Source often contains only
  `Linear(config.hidden_size, ...)`; it cannot recover the checkpoint's number.
- A forward pass containing gate projection, up projection, activation, multiply,
  and down projection proves a gated FFN. A config activation string alone does
  not.
- Source may prove `if layer_idx in config.moe_layers` controls expert
  construction. The list value remains config truth; its architectural meaning is
  code-bound.
- `_class_name`, `model_type`, `architectures`, and repository identity may locate
  source or provide a human label. They may never become mechanism evidence.

### 1.2 Four lawful evidence roles

| Role | Lawful input | Lawful output | May decide structure? |
|---|---|---|---|
| Address | identity/config locator | source/component address | No |
| Display | identity or declared name | typed human label | No |
| Checkpoint value | exact config path | number, enum, bool, list selected by checkpoint | Only after code binds what it controls, except pure geometry |
| Mechanism evidence | owner-bound program graph | typed architectural fact | Yes, subject to completeness |

### 1.3 What “code-based” must not become

Code authority does not mean substring-matching arbitrary class names, scanning a
union of unrelated classes, treating token occurrence as execution, or replacing a
config default with an implementation convention. Code-based evidence must be:

1. resolved to the exact component;
2. bound to the exact constructed class and callable;
3. restricted to reachable construction/forward paths;
4. explicit about completeness and failures;
5. converted from raw observations to facts by a mechanism reader;
6. projected through the same typed fact consumed by renderers and parameters.

---

## 2. Verified repository checkpoint

Branch: `audio-composite-support`, nine commits ahead of its remote checkpoint:

```text
daf056f procedure 1
22ddaf0 procedure 2
ea15aff procedure 3
706818b procedure 4
a969b0f procedure 5
6447e42 procedure 6
6ed88a3 procedure 7
10f6fa5 procedure 8
d475d87 procedure 9
```

Independent verification on the unchanged tree:

- `1093` tests collected;
- `459` targeted hardening tests passed in `266.04s`;
- `tests/test_sable.py` passed alone: `33 passed in 317.87s`;
- the tree was clean before and after verification;
- Sable's former cross-test import/collection failure is repaired.

These results prove the current implementation matches its current tests. They do
not prove that the tests express the full contract. The adversarial probes in this
document demonstrate several missing contract edges.

---

## 3. Authoritative status

| Unit | Current status | Binding interpretation |
|---|---|---|
| Fixture isolation | **DONE** | Shared fixtures are importable and Sable passes alone |
| NAS `block_configs` quarantine | **DONE sub-unit** | Unsafe projection is removed and pinned |
| H0 identity guard | **ACTIVE** | Exact-table manifest is useful; decorator/code-shape escape hatches remain |
| H1 evidence primitives | **DONE substrate** | Constructor laws are sound; production adoption belongs to H7/H8 |
| H2 structural-author census | **ACTIVE** | Current key and runtime depth do not cover every author |
| H3 config ownership | **RESTART/CUTOVER INCOMPLETE** | Ledger substrate is useful; live resolver and blocking nets violate the contract |
| H4 semantic taint | **ACTIVE, direct-pattern slice** | Direct branch-to-dict detection landed; propagation and typed exemptions did not |
| H5 program index/readers | **ACTIVE, ratchet only** | Broad-exception no-growth exists; unified owner-bound graph does not |
| H6 projection firewall | **ACTIVE, scaffold** | Key declarations exist; actual renderer authorship and owner-qualified debt remain open |
| H7 diffusion migration | **ACTIVE, rail only** | Three pending debts are declared; no typed facts/readers/projections landed for them |
| H8 transformer/modalities | **ACTIVE, rail only** | `sinks` is an interim legacy-ledger migration, not a full typed migration |
| H9 metamorphic frontier | **ACTIVE, harness skeleton** | Relations are incomplete or too weak at the actual frontier |
| H10 release verification | **CHECKPOINT DONE; FINAL PENDING** | Current suite is green; final release waits for preceding units and visual closure |

“H10 ran” must never be translated into “H0–H9 are complete.” H10 is evidence
about a particular tree, not authority to weaken incomplete unit exits.

---

## 4. Commit ratification

| Commit | Accepted | Not accepted as complete |
|---|---|---|
| `daf056f` | fixture isolation, lawful-resource fingerprints, H1 primitives, NAS quarantine | typed decorator claim; full H0/H2 closure |
| `22ddaf0` | call-local event capture, removal of global access sets, initial owner scopes | H3 cutover and equivalence claims |
| `ea15aff` | direct class-comparison-to-direct-sink poison | semantic H4 taint completion; role/bool blanket exemption |
| `706818b` | shrink-only broad-exception ratchet | H5 program index, typed failures, complete readers |
| `a969b0f` | initial reverse-fabrication/debt vocabulary | H6 owner/provenance/value-level closure |
| `6447e42` | reusable metamorphic API skeleton | complete H9-core contract |
| `6ed88a3` | explicit acknowledgement of three pending diffusion facts | registered typed facts or H7 migration |
| `10f6fa5` | positive-only `sinks` semantics and a real witness | owner-bound typed `sinks` migration |
| `d475d87` | Sable fixture repair and checkpoint verification | key-only pending-debt excusal; final completion |

Do not rewrite this history merely to make commit boundaries prettier. Repair it
with narrow forward commits whose messages state the defect, the old false claim,
the new invariant, and the counterexample that now fails.

---

## 5. Blocking findings from the independent audit

### 5.1 H3's live path does not use its sound alias resolver

`evidence/config_access.py::resolve_aliases` correctly models:

- the canonical field;
- the actual spelling that supplied the value;
- equal redundant aliases;
- unequal ambiguous aliases;
- absent-default premises.

It has no production caller. The live transformer `_resolve` still selects the
first hit, marks present spellings as inspected, and separately records the
canonical as consumed.

A GPT-2-shaped probe using `n_embd`, `n_layer`, `n_head`, and `n_inner` produced:

- correct geometry by accident of the old resolver;
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and
  `intermediate_size` recorded as consumed;
- the real aliases recorded as accessed-but-unconsumed;
- absent canonicals inserted into compatibility `accessed`/`config_consumed`;
- conflicting `hidden_size=96` and `n_embd=64` silently choosing one value.

**Decision:** `accessed` and `touched` are present-only. Absence lives only in
`absent_default`. A temporary union may be named `considered_fields`, but it may
not feed blocking ownership checks.

### 5.2 Blocking unread coverage is still bare-key based

`config_field_audit` recursively scans dotted config paths but subtracts a flat set
of leaf names. A text `hidden_size` can therefore clear an unread vision
`hidden_size`. The owner-scoped ledger exists beside the old semantic collision;
it has not replaced it.

### 5.3 H3's two promised nets are not live

- accessed-but-unconsumed is advisory;
- it is disabled for an owner that has no consumed census, which makes diffusion
  largely invisible;
- consumed-but-unprojected exists as a unit-tested method but is not published or
  consumed by production Sable.

### 5.4 Pending-projection debt is excused by leaf key

The current code removes every unread path ending in `act_fn`,
`temporal_compression_ratio`, or `max_sequence_length`, regardless of component.
The debt entries carry owners, but the join ignores them.

The three entries are also not facts in `REGISTRY`, have no reader, emit no
`EvidenceFact`, and have no config-access event. They are honest debt declarations,
not “registered typed facts.”

**Decision:** pending debt is a lawful fourth resolution only when matched by exact
owner and config path and shown explicitly in diagnostics. It must not silently
disappear from unread accounting.

### 5.5 H4 catches only the direct syntax shape

The following identity-to-structure paths currently evade the guard:

```text
class comparison -> intermediate enum -> structural dict
class comparison -> helper role -> structural dict
class comparison -> subscript structural write
class comparison -> bool -> structural fact
```

A string or bool is not automatically lawful code-shape. It is often architecture
delayed by one function call.

### 5.6 Identity decorators are markers, not types

`@identity_address` and `@identity_display` return the original function and place
no restriction on its result. The scanner exempts the whole decorated function.
An `@identity_display` function returning `{"kind": "flux"}` is therefore
currently blessed, even though `kind` is a structural sink.

**Decision:** a decorator alone can never suppress a structural sink. Address and
display paths require typed outputs and sink-restricted consumers.

### 5.7 H2's structural-write identity is too weak

The plan requires `(module, enclosing symbol, sink, normalized target)`. The
implementation keys only `(sink, target)` and discards later authors of the same
target. A new module can therefore write a familiar structural field without
growing the census.

Runtime coverage deliberately stops at top-level `extras`, so nested values,
owners, types, and dynamic leaves remain invisible.

### 5.8 H6's debt and drawn-surface claims overstate their proof

`DrawnUnledgeredFact` claims to carry an owner but has no owner field. The reverse
audit trusts hand-authored `ATTENTION_DRAWN`/`FFN_DRAWN`/other sets. A renderer can
draw a new phrase or branch and omit it from those sets, making the audit blind.

### 5.9 `sinks` is not yet a full vertical migration

The positive-only policy is correct: record/draw a sink only when its presence is
proven. But the writer still calls legacy `FactLedger.record`, cites only the reader
function name, and carries no structured span or completeness. The detector scans
all classes in the supplied files instead of binding to the exact decoder-attention
class, so a sibling attention component can vote for the decoder.

### 5.10 H5 and H9 are explicit partials

- There are currently 71 broad `except Exception`/bare catches across production.
- H5 converted two and added a no-growth ratchet; it did not create one raw program
  index or owner-bound graph.
- H9's equivalent-control signature covers only layer count, hidden size, attention
  kind, FFN kind, and norm fields.
- H9's frontier matrix does not run missing-source or equivalent-control.
- Its collision relation proves only that multiple owner prefixes exist, not that
  the same field was kept separate or could not clear sibling debt.

---

## 6. Correcting `CONFIG_VS_CODE_CONVERTIBILITY.md`

The document's governing law is correct. Two conclusions must be superseded:

1. Transformer structure is **not yet essentially fully code-authoritative**.
   Config-normalized schedules, declared decoderness, legacy facts, renderer
   defaults, and incomplete owner binding still decide structure.
2. The diffusion debt is no longer primarily the deleted
   `unet_blocks.yaml`/`vae_classes.yaml`/`schedulers.yaml` trio. The active debt now
   lives in `config_facts.yaml`, `conditioning.yaml`, structural sections of
   `typing.yaml`, parser fallbacks, raw `extras`, and renderer/opgraph defaults.

### 6.1 Permanent config authority — keep

These are checkpoint values that code generally cannot recover as literals:

- dimensions: hidden/intermediate/head/latent/channel widths;
- counts: layers, heads, experts, selected experts, codebooks;
- tensor and patch geometry;
- vocabulary and maximum lengths;
- numeric constants: epsilon, theta, scaling factors, thresholds, ratios;
- the selected value/list for a branch whose controlling relation code proves;
- exact component/repository identifiers used only to locate source;
- public semantic declarations when shown as declarations rather than upgraded to
  implementation detail.

These must still be owner-qualified, alias-safe, and recorded with exact config
paths.

### 6.2 Convertible structural authority — move to code

The following must be code-proven or honest-unknown:

- component ownership and construction;
- attention/FFN/mixer mechanism;
- topology and execution order;
- norm kind and placement;
- projection storage/fusion/splitting;
- positional application and location;
- routing order and expert construction;
- conditional/dormant operations;
- UNet/VAE stage and cell internals;
- temporal/spatial operation placement;
- scheduler algorithm/history/state transitions;
- conditioning projectors and fusion path;
- parameter ownership and formula selection;
- whether a config value is live, dead, conditional, or merely metadata.

### 6.3 Lawful vocabulary is not evidence

Aliases, syntax tokens, role words, and display labels may live in YAML when they
are data that helps parse syntax or present a fact. Their presence does not prove a
mechanism. Every vocabulary consumer must terminate in a `RawObservation` or typed
address/display object unless an owner-bound reader independently establishes the
fact.

---

## 7. Current YAML/config inventory and required fate

### 7.1 Transformer vocabulary

| File | Current role | Fate |
|---|---|---|
| `transformer/aliases.yaml` | equivalent config spellings | **KEEP as syntax only**; consume through the exact alias resolver; unequal aliases block |
| `transformer/ignored_fields.yaml` | global unread suppression | **MIGRATE** to owner/component/reason-qualified ignore records; global leaf suppression is forbidden |
| `transformer/typing.yaml` | UI/schema stage vocabulary | **KEEP**, presentation/schema only |
| scope-qualified `input_format.*` entries in `transformer/aliases.yaml` | raw foreign-format field conversion selected by file layout and required keys | **KEEP as syntax adapter**; it may not manufacture model/class identity or mechanism semantics |
| `transformer/composite_slots.yaml` | wrapper/component slot roles and undrawn labels | **CONSTRAIN**: slot keys may locate candidates; construction/source proves ownership; undrawn labels remain presentation debt |
| `transformer/decoderness.yaml` | architecture suffix to decoder-role declaration | **QUARANTINE** as weak declared role; it cannot independently prove causal attention |
| `transformer/layer_schedules.yaml` | field dialects and token-to-mixer mappings | **SPLIT**: spellings may remain syntax; mixer/attention semantics require code-bound schedule expressions |
| `transformer/layer_types.yaml` | config token to mask/compression kind | **SPLIT/REMOVE AUTHORITY**: token normalization may remain, but source must prove what each token builds/applies |
| `transformer/layer_topology.yaml` | historical topology maps | **DELETE if dead**; never restore family/config topology authority |

### 7.2 Diffusion vocabulary

| File/section | Current role | Fate |
|---|---|---|
| `diffusor/aliases.yaml` | equivalent config spellings | **KEEP as syntax only**, through owner-scoped resolution |
| `diffusor/config_facts.yaml` | reads arbitrary present fields and emits chips/silent ownership | **DELETE through migration**; every architectural row becomes a registered typed fact, every non-architectural row a scoped ignore/display value |
| `diffusor/conditioning.yaml` | enum directly selects modality/projector/prose | **REMOVE structural authority**; enum may yield an opaque declaration, while source binds the constructed projector and executed fusion |
| `diffusor/text_encoders.yaml` | class-to-friendly title | **KEEP display-only** under fingerprint; source/component graph decides the actual tower |
| `typing.yaml:stages/block_ids/part_kinds` | schema/display taxonomy | **KEEP**, with no parsing authority |
| `typing.yaml:dit_class_markers` | class substring locates DiT source | **QUARANTINE as address** and replace where construction/AutoModel mapping can resolve directly |
| `typing.yaml:norm_type_kind` | config enum directly selects norm implementation | **DELETE as structural authority**; resolve constructed norm class/math |
| `typing.yaml:scheduler_display` | human labels | **KEEP display-only** |
| `typing.yaml:scheduler_flow_matching_markers` | scheduler class substring selects algorithm | **DELETE as structural authority**; inspect `step()` and state/history operations |
| `typing.yaml:temporal_forward_markers` | source token vocabulary | **KEEP as raw observation vocabulary**, not standalone proof |
| `typing.yaml:temporal_config_fields` | field-presence video fallback | **CONSTRAIN** to weak declaration; source-bound temporal operations decide detailed structure |
| `typing.yaml:audio_vae_fields` | field-presence audio-domain inference | **CONSTRAIN** to owner-qualified declaration; code/component type proves operations and axis |
| `typing.yaml:stack_lane_params` | forward parameter to lane mapping | **KEEP only after exact callable binding**; raw parameter names cannot vote globally |
| `typing.yaml:companion_denoiser_fields` | pipeline slot declarations | **KEEP as address/declaration**, with exact component ownership and stated omission |

### 7.3 Conformance vocabulary

`conformance/op_tokens.yaml`, `type_roles.yaml`, `fact_markers.yaml`,
`transitive.yaml`, and `wiring_roles.yaml` may remain as interpreter vocabulary only
when they classify syntax on an already owner-bound program graph. They may produce
raw/bound observations, never architecture merely because a token or class substring
matched.

Identity-specific presentation/conformance exceptions such as `cogvideox` and
`flux2` rows in `conformance/abstractions.yaml` are transitional debt. Replace them
with structural predicates or exact evidence records; do not expand that table.

### 7.4 Hardcoded fallback authority to eliminate

The current tree still contains conventional fallbacks including:

- unknown attention kind becoming SDPA/MHA in `opgraph.py`;
- absent cache evidence becoming cached autoregressive K/V;
- diffusion self-attention defaulting to MHA;
- UNet activation defaulting to SiLU;
- denoiser/block style defaulting to transformer;
- conformance storage defaulting to split;
- raw specs and parameter conventions being interpreted independently of facts.

Each must become one of:

1. a typed fact from exact evidence;
2. a typed declared value whose use code binds;
3. a pale opaque unknown;
4. a scoped presentation-only default that makes no architectural claim.

---

## 8. Target architecture

```text
checkpoint config ------------------------------+
  exact owner/path/alias/value                   |
                                                 v
resolved source -> RawProgramIndex -> OwnerGraph -> BoundObservation
                                                 |
                                mechanism reader + config gate evaluation
                                                 v
                                      EvidenceFact registry
                                                 |
                         +-----------------------+------------------+
                         v                       v                  v
                   semantic IR/opgraph      renderer receipts   parameter facts
                         |                       |                  |
                         +-----------------------+------------------+
                                                 v
                             symmetric conformance and release gates
```

### 8.1 One parse context

`ParseContext` owns, call-locally:

- one resolved source bundle;
- one raw program index per source file set;
- one component/constructor/callable ownership graph;
- one config-access ledger;
- one typed fact ledger;
- typed reader failures and completeness;
- projection receipts and pending debt joins.

ContextVars may route nested calls to the current context, but module-level
ContextVars must not become a second truth store.

### 8.2 One fact, all consumers

For a given concept, the renderer, JSON serializer, params estimator, labels, and
conformance checks consume the same `EvidenceFact`/semantic IR projection. None may
re-read config, source, identity, or a family table.

### 8.3 Unknown is structural data

Unknown is not an exception and not permission to choose a familiar drawing.

- unknown mechanism -> one opaque node/region;
- unknown topology -> no invented residual/norm arrangement;
- unknown parameter formula -> range/omission with an assumption receipt;
- ambiguous source-present evidence -> blocking finding;
- oracle missing -> visible honest degradation, never a family fallback.

---

## 9. Execution plan

Only one high-conflict unit may edit parser/evidence infrastructure at a time.
Every unit lands as a separate reviewed commit after its own gate.

### R0 — Freeze the checkpoint and correct status

**Goal:** make over-claiming mechanically difficult before further implementation.

1. Adopt this document as the authoritative tracker.
2. Record the current clean-tree fingerprint, 1093 collection, targeted gate, and
   Sable-alone result.
3. Add the adversarial probes from Section 5 as failing regression tests before
   changing production code.
4. Do not change blessings.

**Exit:** current false completion claims cannot be read as authorization to skip a
recovery unit.

### R1 — Complete H3 with one exact config resolver

1. Put `ConfigAccessLedger` on `ParseContext`.
2. Replace transformer and diffusion `_resolve`/`consume` funnels with one resolver
   returning both resolved value and exact access result.
3. Record:
   - component owner;
   - full config path;
   - canonical field;
   - actual spelling/path;
   - presence;
   - intent;
   - exact fact/geometry target;
   - source-binding reader where applicable.
4. Unequal simultaneous aliases become typed ambiguity and block structural use.
5. Equal aliases record every occurrence but consume exactly one selected path.
6. Add explicit `root.text`, `root.vision`, `root.audio`, `root.vae`, and
   `root.denoiser` scopes.
7. Remove absent fields from `accessed`, `touched`, and `config_consumed`.
8. Rewrite recursive unread coverage as exact-path/owner joins, not leaf subtraction.
9. Publish both owner-qualified nets.
10. Migrate corpus debt, then make both nets blocking.

**Required counterexamples:** GPT aliases, unequal aliases, sibling `hidden_size`,
nested wrappers, direct adapter parse, nested capture, threads/tasks, absent
default, source missing, diffusion owner without an existing consumed census.

### R2 — Make pending projection explicit and owner-tight

1. Replace leaf-key filtering with an exact join on owner, config path, canonical,
   fact/debt name, and intended projection.
2. Add a `pending_projection` diagnostic list. Do not erase pending fields silently.
3. Require owner, reason, migration unit, projection, creation fingerprint, and
   deletion condition.
4. Add a shrink-only baseline and stale-entry failure.
5. Prove `root.vae.act_fn` does not excuse `root`, vision, audio, or denoiser
   `act_fn`.

### R3 — Complete H0/H4 semantic identity confinement

1. Introduce typed `IdentityAddress` and `IdentityDisplay` values.
2. Decorators may document a function but may not exempt structural sinks.
3. Track identity/config-name taint through:
   - assignments and returns;
   - helper calls;
   - booleans and intermediate enums;
   - mappings/subscripts/comprehensions;
   - constructor kwargs and dataclass fields;
   - opgraph/spec/extras/card/label/params sinks;
   - YAML keys and values.
4. Distinguish a class-name role observation from a structural fact by type.
5. A role observation can guide owner binding; it cannot be recorded or drawn as
   architecture without a mechanism reader.
6. Enforce typed display/address consumer restrictions.

**Poison matrix:** direct branch, helper, intermediate enum, bool, subscript write,
decorated structural return, YAML laundering, role table used by renderer, and a
lawful address/display control.

### R4 — Complete H2 structural-author coverage

1. Key `StructuralWrite` by `(module, symbol, sink, normalized target)`.
2. Detect duplicate authors of a familiar target in a new module or symbol.
3. Recursively enumerate every runtime structural leaf with owner and value type.
4. Cover dynamic extras mutation, dataclass replacement, attribute writes,
   constructor wrappers, renderer prose/kinds, and parameter formulas.
5. Give every legacy author owner, reason, unit, and exact deletion target.
6. Validate typed writes at their production decision sites.

**Exit:** changing representation or reusing an existing target cannot bypass the
registry.

### R5 — Complete H5 raw program and ownership substrate

1. Parse every source set once into a reusable `RawProgramIndex`.
2. Build component -> constructed class -> callable -> field/call ownership edges.
3. Resolve factories, registries, classmethods, delegates, and wrapper assignments
   without family names.
4. Make all reader results typed with completeness and failure context.
5. Remove broad exception catches from evidence readers to zero; parser tolerance
   catches must become typed and narrow separately.
6. Consolidate duplicated forward/transitive/specialized extraction primitives.

**Exit:** every mechanism reader starts from the same owner-bound graph, and failure
cannot collapse into a conventional fallback.

### R6 — Complete H6 projection and consumer firewall

1. Add owner to `DrawnUnledgeredFact` and every projection debt row.
2. Derive projection receipts from actual render events/semantic builders, not a
   hand-authored declaration alone.
3. Require receipts for structural prose, chips, op kinds, topology, and omissions.
4. Make renderer and params imports/read paths accept semantic IR/facts only.
5. Add reverse checks:
   - fact owed but not projected;
   - projection with no fact/debt;
   - projected value stronger/different from the fact;
   - params formula with no fact/assumption receipt.

### R7 — Strengthen H9 before domain migrations

Build a non-vacuous harness that every migrated mechanism must invoke:

1. rename identity while pre-resolving the same source;
2. rename fields through equivalent aliases;
3. inject unequal conflicting aliases;
4. collide the same field across sibling owners;
5. force source missing;
6. force partial source/one missing component;
7. add a same-name/different-mechanism counterexample;
8. add a different-name/same-mechanism control;
9. move a signal to an unreachable helper;
10. add multiple rival candidate classes;
11. compare full semantic IR, facts, opgraphs, params assumptions, block/card paths,
    and projected receipts—not a four-field signature.

### R8 — Remove structural config authority, one vertical fact at a time

No YAML file is deleted wholesale before its rows are classified. Each migration
follows the recipe in Section 10 and deletes/quarantines its old path in the same
commit.

#### R8A — Transformer schedules and remaining defaults

- bind every layer schedule field and value token to the source expression that
  selects/constructs the layer mechanism;
- remove `mixer_kinds` and `layer_types` as independent architectural authority;
- make decoderness a weak declaration corroborated by code for mask behavior;
- migrate Llama-4 chunked attention and temperature schedules;
- remove unknown->MHA, cache, split-storage, and other conventional defaults;
- migrate remaining asserted/drawn-unledgered transformer facts.

#### R8B — Diffusion conditioning

- bind `encoder_hid_dim_type`/`addition_embed_type` values to the constructed
  projector and executed add/concat/cross-KV path;
- source missing -> opaque external conditioning with declared enum label only;
- dismantle `conditioning.yaml` structural templates.

#### R8C — DiT/MMDiT blocks

- bind root denoiser class, heterogeneous block variants, norm, attention, FFN,
  fusion, positional, temporal, and conditioning paths;
- remove class markers and MHA/transformer fallbacks as structural authority.

#### R8D — UNet

- derive down/mid/up construction, optional mid presence, layers per stage,
  attention/ResNet order, temporal operations, and activation from exact code;
- negative mid/attention claims require complete inspection;
- unknown cell -> opaque, never Transformer2D/SiLU.

#### R8E — VAE

- derive KL/VQ/other latent mechanism, spatial/temporal/audio axes, causal convs,
  stage schedule, norms, activation, attention, and quantization ops from the VAE
  owner graph;
- consume the three owner-tight pending facts through typed writers/projections;
- delete `config_facts.yaml` VAE architecture rows as they migrate.

#### R8F — Scheduler

- inspect `step()` and bound helper state to derive history, predictor/corrector,
  stochastic noise, flow/denoise target, and update form;
- config supplies coefficients and prediction selection only where code binds them;
- unknown scheduler -> opaque state update, never Euler;
- retain scheduler name mapping only as display.

#### R8G — Modalities and recursive submodels

- use one recursive submodel contract for text, vision, audio, video, codec, VAE,
  denoiser, and future components;
- bind tower/projector/fusion facts from each component's own source;
- class/title tables remain display-only;
- direct and embedded rendering of the same component must share facts and opgraphs.

### R9 — Make every H7/H8 fact natively typed

For each existing legacy `_note_fact` writer:

1. produce `EvidenceFact` with exact owner;
2. include `SourceSpan` and/or exact config paths;
3. state completeness;
4. use `FactLedger.record_typed`;
5. remove the legacy writer and asserted fallback;
6. prove registry, projection, params, and conformance consume the same fact.

The `sinks` migration is the first correction: bind it to the exact decoder
attention owner and retain positive-only semantics.

### R10 — Parameter and conformance unification

- params consume semantic facts/opgraph, never raw config conventions;
- unknown formula emits an assumption/range receipt;
- conformance consumes the owner-bound program graph and the same projection
  receipts;
- remove identity-specific abstraction exceptions and family-scoped omissions;
- vocabulary may classify syntax but may not create expected architecture.

### R11 — Delete dead compatibility paths and shrink data

- remove compatibility accessed/consumed lists after all consumers migrate;
- delete `config_facts.yaml` when its last row has a typed owner;
- remove obsolete structural sections from schedule/typing/conditioning YAML;
- remove dead loaders, render defaults, legacy ledgers, and allowlist rows;
- require a dead-symbol and stale-debt gate.

### R12 — Final H10 and visual closure

1. Run every hardening file alone.
2. Run the grouped hardening gate.
3. Run the full suite on the same unchanged fingerprint.
4. Run corpus/conformance over source-present and source-missing controls.
5. Render every distinct changed view.
6. Inspect summary and deepest drill views, click routing, cards, parameter totals,
   warnings, and unknown states.
7. Do not re-bless without Soumil's decision and an explained evidence-backed delta.

---

## 10. Mandatory per-mechanism migration recipe

Every structural fact migration is a narrow vertical slice:

1. **State the claim.** One owner and one fact; do not bundle unrelated mechanisms.
2. **Freeze witnesses.** Failing model, equivalent different-identity model,
   same-shaped counterexample, unaffected renderer control, and missing/partial
   source cases.
3. **Map consumers.** Config/source -> binder -> fact/spec -> opgraph -> renderer ->
   cards/JSON -> params -> conformance -> blessings.
4. **Bind source.** Identify the exact constructed class and reachable callable.
5. **Observe, then interpret.** Raw syntax becomes `RawObservation`, owner binding
   becomes `BoundObservation`, and only a mechanism reader emits `EvidenceFact`.
6. **Evaluate config gates.** Use the exact owner/path/alias only when code names
   that gate.
7. **Record epistemics.** Status, completeness, spans, paths, premises, failures.
8. **Project once.** All structural consumers derive from that fact/semantic region.
9. **Delete the old authority.** Remove the family/config table, fallback, raw
   extras writer, or renderer inference in the same unit.
10. **Run counterexamples and visuals.** A happy-path image is insufficient.

If step 9 cannot happen, the unit is not complete; record why as explicit shrinking
debt with an owner and deletion condition.

---

## 11. Guardrails for future fixes

### 11.1 Never solve a model problem this way

- add a `model_type`, architecture, repository, or family branch;
- add a class-name-to-structure table;
- add a config field to `config_facts.yaml` merely to silence unread audit;
- translate a config enum directly into detailed ops without source binding;
- choose the most common attention/FFN/norm/topology when evidence is missing;
- let a renderer or params estimator re-read raw config/source;
- scan all classes and union their votes for one owner;
- catch a reader exception and return a familiar default;
- broaden an allowlist to make a poison/corpus pass;
- re-bless an unexplained visual change;
- call a rail or ratchet the completed end state.

### 11.2 What to report to Soumil instead

When the evidence system cannot model a case, report:

1. the exact model/component/owner;
2. the exact fact that is unresolved;
3. what source/config evidence exists;
4. why the current interpreter cannot bind or understand it;
5. whether this is missing source, ambiguity, unsupported syntax, ownership, or a
   missing projection;
6. which existing models share the mechanism;
7. the smallest general interpreter capability required;
8. the affected renderer/params/conformance surfaces;
9. preservation and counterexample witnesses;
10. the expected pixel/card/parameter delta requiring review.

The answer must be “we need capability X,” not “we need a row for model Y.”

---

## 12. Unit acceptance gates

A unit is DONE only when all are true:

- its production path uses the new abstraction, not merely its unit test;
- its negative controls fail before the fix and pass after it;
- equivalent and counterexample models pass;
- source-present, partial, missing, and ambiguous cases behave honestly;
- owner collisions and alias conflicts are tested;
- every audit file passes alone;
- targeted and full suites pass on one unchanged tree fingerprint;
- corpus, conformance, params, and projection obligations are green;
- the old path/debt/allowlist shrinks in the same commit;
- intentional visual changes are inspected at summary and deepest drill;
- no blessing changes without Soumil's explicit decision;
- the completion record includes command, count, duration, fingerprint, expected
  deltas, and residual debt.

### 12.1 Immediate stop conditions

Stop the current unit and report instead of improvising when:

- a renderer or params path needs raw config/source;
- a weak fact would be projected as a stronger mechanism;
- ownership has multiple rival candidates;
- a negative claim lacks complete inspection;
- a new interpreter capability is required outside the current unit;
- a field can be cleared only by a global key ignore;
- pending debt cannot be joined to an exact owner/path;
- the tree changes during verification;
- a test is being changed only to accept an unexplained delta;
- the proposed solution requires an identity/family-specific exception.

---

## 13. Commit and tracker discipline

Each commit message must contain:

```text
Unit / fact:
Old unsound path:
New evidence path:
Owner and completeness:
Old path deleted/quarantined:
Counterexamples:
Preservation witnesses:
Tests and fingerprint:
Visual deltas requiring Soumil:
Remaining debt:
```

Tracker statuses are only:

- `PENDING`
- `ACTIVE`
- `BLOCKED` with exact external prerequisite
- `DONE` with a stable gate receipt

“Done-core,” “rail complete,” or “procedure ran” may describe progress in prose but
must not appear in the authoritative status column.

---

## 14. Final definition of done

The campaign is complete when:

1. config remains only checkpoint values, exact declarations, syntax/address, and
   display—not unbound structural authority;
2. all mechanisms are code-proven, code-and-config, or visibly unknown;
3. all facts are natively typed and owner-qualified;
4. all structural consumers use the same semantic fact/opgraph;
5. identity taint cannot reach structural sinks through helpers, enums, mappings,
   decorators, YAML, renderers, or parameters;
6. every structural author and projection is mechanically discoverable;
7. reader failure and incomplete source cannot become a conventional default;
8. all compatibility paths, family tables, terminal defaults, and temporary debt
   registers have been deleted or reduced to explicitly lawful display/syntax data;
9. a new model with a known mechanism works without a model-specific code or YAML
   addition;
10. a genuinely new mechanism becomes one honest opaque node plus an actionable
    unsupported-capability report—not a plausible but false familiar diagram.

That is the direction. Anything less is an intermediate checkpoint and must be
named as such.

---

## 15. Whole-tree code-authority audit — the actual remaining frontier

This section records the second, producer-to-consumer audit performed after the
YAML inventory. It is binding and supersedes any earlier statement that the
remaining conversion is “mostly diffusion.” That statement was too optimistic.

The audit followed all 270 direct config-access sites, every `everchanging/`
loader and consumer, the structural defaults in the dataclasses, the canonical
op-graph, raw `extras`, render/card reconstruction, parameter formulas, and the
conformance expectations. The important result is:

> The remaining problem is not merely configuration files. Structural authority
> also survives as Python defaults, config-presence predicates, semantic enum
> maps, class-default hydration, hand-authored drill graphs, renderer fallbacks,
> parameter conventions, and conformance exceptions.

The current guards being green does not disprove this finding:

- `scan_identity_debt() == []` checks identity-taint shapes, not every path by
  which a config field or a default can author architecture;
- the structural-write census currently collapses sites by `(sink, target)`, so a
  new producer of an already-known `kind`/spec/extras target is invisible;
- its runtime extras census sees top-level keys, not the dynamically populated
  structural leaves under `unet`, `diffusion`, `modalities`, or `render`;
- the fact-projection net uses manually maintained drawn-leaf sets, so it proves
  that a declared key has a nominal surface, not that the rendered value and
  mechanism came from the same evidence;
- corpus success exercises known paths; it cannot prove an unused fallback is
  safe for the next model.

This is why this ledger is necessary before implementation.

### 15.1 What “move it to code” means

It does **not** mean replacing config names with class-name conditionals or
searching source text for a familiar word. A structural conversion is complete
only when:

1. the exact component and owning class/callable are resolved;
2. the owner graph proves that the mechanism exists and where it executes;
3. when code gates the mechanism on config, the AST names the exact config path
   and the checkpoint supplies only the selected value;
4. the result is a typed fact with source span, config path/premises, and honest
   completeness;
5. IR, op-graph, cards, SVG/JSON, params, and conformance consume that one fact;
6. source-missing or reader-ambiguous cases become opaque/unknown;
7. the config-presence/default/table path is deleted in the same unit.

The following remain lawfully config-supplied after the conversion:

- widths, depths, counts, tensor/patch sizes, capacities, constants, ratios, and
  thresholds;
- the selected value/list for a branch that the exact owner code proves it reads;
- component addresses used only to resolve source;
- explicit public declarations when displayed as declarations and not upgraded
  into an implementation graph.

Everything below concerns a mechanism, topology, operation, placement, ownership,
or formula selection and therefore needs a code-bound fact or an honest unknown.

---

## 16. Exact migration ledger by producer and consumer

Priority meanings:

- **P0** — can fabricate a familiar architecture today; fix before broad domain
  migration.
- **P1** — structural authority is real but usually constrained by current
  fixtures; migrate in the named vertical.
- **P2** — address/display/schema or hardening debt; constrain it so it cannot
  become structural authority.

### 16.1 Core evidence, IR, and op-graph substrate

| ID | Priority | Current site and behavior | Required end state |
|---|---:|---|---|
| C-01 | P0 | `ir.AttentionSpec` defaults `rope=True`, `bias=False`, `qk_norm=False`, `shared=False`, and several mechanism booleans to a confident negative/positive. A caller omitting a field can therefore author structure accidentally. | Structural booleans become tri-state or typed fact references. Every constructor must pass a proven value; omission means unknown/absent-by-policy, never the common architecture. |
| C-02 | P0 | `ir.FFNSpec.gated=True`, `LayerSpec.norm_kind="rmsnorm"`, and `LayerSpec.norm_placement="pre"` make the conventional modern decoder the object-level default. | Remove architectural dataclass defaults. Factories accept typed resolved facts or emit an opaque/unknown cell. |
| C-03 | P0 | `assembly.decoder_layer`, `parallel_decoder_layer`, and `single_stream_decoder_layer` repeat the RMS/pre/fused topology defaults. | Assembly becomes a pure projection of a resolved cell/topology fact; it may not choose a topology. |
| C-04 | P0 | `opgraph.attention_region/_sdpa_region` treats `kind is None` as SDPA/MHA, missing storage as split Q/K/V, `cached is None` as autoregressive cache, and missing score evidence as `sqrt(dim)`. | Unknown kind/storage/cache/scaling produces typed opaque or partial regions. SDPA, split storage, cache, and scaling appear only from facts. |
| C-05 | P0 | `opgraph.ffn_region` still has `conv_glu -> silu`, `gated` default-true, expert gated default-true, and `ops_region(... source="config")`. | Region source is an evidence receipt, not a string default. Activation/gating/storage are projected only when proven; declared op lists need typed provenance. |
| C-06 | P0 | `labels.mask_*` defaults missing/unknown values back to causal; `kind_*` defaults an unknown kind to MHA. | Unknown stays “unresolved” in every label/title/chip. Presentation vocabulary can format a fact but never supply it. |
| C-07 | P0 | `renderers/html/metadata._make_info` creates a synthetic MHA + gated-SiLU fallback spec when a model has no layers. | Empty/unresolved models receive an explicit empty/opaque presentation state; no synthetic dominant layer. |
| C-08 | P1 | `FactLedger.record` remains a public legacy writer, and most production decisions still use `_note_fact` rather than native `EvidenceFact`. | Native typed writer only for migrated facts; legacy lift remains counted quarantine and cannot feed a stronger projection. |
| C-09 | P0 | `StructuralWrite.key` and its scanner documentation say site-qualified, but the actual key is `(sink,target)` and `seen` uses that pair. New authors of an existing target evade growth detection. | Key and pin are `(module,enclosing symbol,sink,normalized target)->count`; poison a second writer for an existing target. |
| C-10 | P0 | `runtime_structural_targets` explicitly checks only top-level extras despite H2’s stated nested-leaf requirement. | Recursively census all structural leaves with owner, type, source writer, and exercised value. |
| C-11 | P0 | `fact_projection.py` hand-lists `*_DRAWN` leaf sets and matches only owner-family segments. | Projection receipts are emitted by the actual projector from fact ID + owner + value hash + surface; no nominal hand-authored witness list. |
| C-12 | P1 | `renderers/html/cards._fallback_sub_inspect_children` reconstructs FFN structure separately, including default-true gating. | Delete the fallback reconstruction; cards derive only from the same namespaced canonical region used by the SVG/JSON. |
| C-13 | P1 | Raw `extras` remain parallel structural IRs (`unet`, `diffusion`, `modalities`, schedules, position, render specs). | Migrate mechanisms to typed facts/recursive component specs. Retain `render` only as non-architectural layout policy. |
| C-14 | P1 | `_hydrate_config_class_defaults`, `hydrate_encoder_config_facts`, and `_installed_config_defaults` hydrate a whole config-class dictionary; downstream code can treat any hydrated default as structure without proving modeling code reads it. | Hydration may supply a value only after an owner-bound source expression proves the field is read. Unbound class defaults are metadata, not facts. |

### 16.2 Transformer root stack

| ID | Priority | Current site and behavior | Code-bound replacement and deletion |
|---|---:|---|---|
| T-01 | P0 | `adapters/transformer/parser._resolve` is still first-hit and does not call `evidence.config_access.resolve_aliases`; it marks all present spellings but loses the selected alias and conflict. | One resolver records exact owner/path/alias. Unequal aliases block; equal redundant aliases are recorded but only the selected path is consumed. Delete both parser-local resolvers. |
| T-02 | P0 | `_attention_kind` chooses MLA from `kv_lora_rank` presence and MHA/GQA/MQA from head counts plus `multi_query*` flags. Counts describe geometry but do not prove the projection/storage mechanism. | Resolve the exact attention class; prove Q/K/V or latent paths and bind their count fields. Config supplies counts only. Unknown mechanism is opaque attention with known dimensions. |
| T-03 | P0 | `layer_types.yaml`, `layer_schedules.yaml`, `_normalize_layer_schedule`, `_mixer_kind_for`, and `_layer_mask` map config tokens directly to linear/recurrent/softmax mixer and mask structure. | Code proves the per-layer selector expression, candidate classes/callables, and token equivalence. YAML may normalize syntax only. Delete `mixer_kinds` and semantic token groups after migration. |
| T-04 | P1 | sliding/full/compressed masks can still be selected from field/value presence and label substrings (`_is_sliding_label` also accepts any token containing `sliding`). | Bind mask creation and per-layer gate in the attention owner. Window/compression values stay config; substring fallback is deleted. |
| T-05 | P0 | source-missing positional fallback upgrades `rope_theta`/`rope_scaling` presence to a RoPE mechanism and projects it onto every non-mixer layer. | A config value can say “declared rotary parameters,” but Q/K rotation is drawn only when the owner graph applies it. Delete `position_declared` as a mechanism override; retain a declaration-only chip if desired. |
| T-06 | P1 | `_norm_kind_evidence_src` derives norm kind from `rms_norm_eps`/`layer_norm_eps` spelling when source math is absent. | Epsilon remains a numeric value. Norm kind comes from the constructed norm class/math; source missing means generic Normalization. Delete spelling-to-kind derivation. |
| T-07 | P0 | FFN activation/gating may come from checkpoint fields or hydrated class defaults without first proving the exact FFN uses that dispatch; plain activation-name derivation can then select a shape. | Code proves the FFN callable, dispatch field, and multiply/storage graph. Config/class default supplies the selected activation only through that proven read. |
| T-08 | P1 | attention/MLP bias and tying accept config/class-default fallbacks even when the precise constructor/tying operation was not bound. | Bind each value to the exact `Linear(...bias=...)`/tie operation. A class default becomes lawful only as a premise of that source read. |
| T-09 | P1 | `decoderness.yaml` architecture suffixes and config roles can cause a causal mask when code causality abstains. | Keep role declaration for address/display; mask fact requires owner-bound mask construction/use. Source missing renders mask unknown. |
| T-10 | P1 | QK-norm, parallel residual, cross-layer KV sharing, residual scaling, output gates, and several topology flags are still applied from config values even when the reader has not proved how the owner consumes them. | Give each a code-gate fact (`mechanism`, exact config path, selector semantics, completeness). Values without a binding remain declaration-only. |
| T-11 | P0 | config schedules (`first_k_dense_replace`, frequency fields, `moe_layers`, `mlp_only_layers`) remain a complete fallback for MoE placement. | Code proves which per-layer branch constructs dense vs expert FFN and names its gate fields. Config supplies indices/frequency; unresolved gate yields an unknown layer type, not a computed schedule. |
| T-12 | P1 | router fields such as `n_group`, `topk_group`, renormalization, and routed scaling flow directly into the router view; code evidence currently covers only part of the ordering/score behavior. | Bind every drawn router operation and order to the exact route callable. Numeric values remain config. Delete any present-field-to-node fallback. |
| T-13 | P0 | `_cross_attention_layers` derives a per-layer cross-attention schedule from config lists/frequency; modality presence then supplies the side-state story. | Code proves cross-attention module construction and per-layer selector, while config supplies selected indices. Fusion source and K/V owner are separate facts. |
| T-14 | P1 | per-layer embeddings, MTP, block-diffusion, codebooks, speech/codec omissions, and several `extras` are created from field presence. | Resolve each component/module from the root construction graph. Field values parameterize it; absent source produces a declared-but-unresolved component, not a detailed block. |
| T-15 | P2 | `composite_slots.yaml` and source-resolution component keys can decide candidate roles from names. | Slot names remain candidate/address vocabulary only. Construction/forward edges prove role and ownership; warnings use display records, not architecture. |
| T-16 | P1 | `encoder_panel` reparses hydrated sub-configs and then carries manual honesty overrides because the root parser itself can assert decoder conventions. | Remove the need for overrides: the universal parser is role/altitude neutral, and decoder/encoder behavior is an explicit owner-bound fact. Embedded equals standalone modulo declared altitude policy. |
| T-17 | P1 | `submodel.ALTITUDE_TRANSFORMS` forces `cached=False` for every tower, while `submodel_spec` still accepts manual structural override parameters. | Altitude may suppress editorial detail but not alter mechanism. Cache behavior and all group facts come from evidence; remove structural override arguments as migrations land. |
| T-18 | P1 | Llama-4 chunk/local masks and temperature tuning remain unread; current config values can be surfaced without full operation/order proof. | Add exact mask/temperature readers with counterexamples; constants remain config. This stays a separate vertical fact family. |

### 16.3 Vision, audio, video, projector, and fusion paths

The modality code contains the largest transformer-side body of config-presence
architecture. The comments often say “config-driven, no family names”; removing
family names was useful, but it did **not** make the inference code-proven.

| ID | Priority | Current site and behavior | Code-bound replacement and deletion |
|---|---:|---|---|
| M-01 | P0 | `modalities.detect.is_unified_grid_stream` turns grid fields or an `mrope` enum into a unified multimodal-stream mechanism. | Wrapper forward proves grid inputs, placeholder replacement, and multimodal position construction. Config supplies grid/section values only. |
| M-02 | P0 | `has_cross_attention_adapter`/`cross_attention_layers` turns config fields/frequency into cross-attention modules and schedules. | Bind to decoder-layer construction and exact selector; reuse T-13’s one fact. |
| M-03 | P0 | `vision_path` chooses path kind, token kind, final operation, projection operation, and K/V target from the two predicates above. | Project these from typed tower/projector/fusion facts; the initial path before source overlay is opaque, not a detailed provisional story. |
| M-04 | P0 | `vision_position_encoding` maps field presence to learned 2D/RoPE/multimodal RoPE. | Read the vision embedding/attention owners and actual add/rotation site. Config values parameterize the proven op. |
| M-05 | P0 | `image_tiling` maps `max_num_tiles`/`image_grid_pinpoints` to fixed tiling/anyres operations. | Prove preprocessing/wrapper tiling call and bind its config fields. If preprocessing source is outside the modeling package, emit a declared preprocessing fact, not a model-code op. |
| M-06 | P0 | `token_reduction` maps pooling/downsample/scale fields directly to average pooling or pixel shuffle and computes reduction factors. | Resolve the projector/merger forward and ordered op. Config supplies kernel/scale only. |
| M-07 | P1 | `multilayer_features` and `feature_selection` turn field type/value into concatenation, layer selection, and CLS dropping. | Wrapper forward must prove indexing/cat/slice behavior and name the fields it reads. |
| M-08 | P0 | `projector_kind` chooses perceiver, patch merger, declared type, MLP, or linear from config presence; missing values default to linear. | `ProjectorEvidence` is authoritative from the exact constructed projector class and forward graph. Before evidence, use `code_defined_projector`; delete config mechanism classification. |
| M-09 | P0 | projector input/output widths are derived differently depending on config-classified `cross_attn`/`unified_grid`, including merged-patch multiplication. | Code proves the merger/flatten/concat expression and exact dimension fields; config supplies values. |
| M-10 | P0 | `audio_projector_kind` accepts a config enum as mechanism; otherwise path assembly still creates a projector stage. | Resolve the audio projector field/class and op chain. Unresolved remains one opaque connector with known I/O. |
| M-11 | P1 | `audio_path` and `video_path` author detailed stages and operations from component presence even before source evidence is applied. | Component source evidence builds the stage graph. Config-only paths are declaration/geometry shells and may not state encode/project/merge details. |
| M-12 | P0 | `conditioning_path` infers a linear width projector when encoder and decoder widths differ, and `is_encoder_decoder` selects emitted cross-attention states. | Code proves the bridge module and route. Width mismatch alone cannot prove a projection. |
| M-13 | P0 | `fusion_path` uses a config-declared conditioning token kind as a cross-attention fallback when wrapper evidence is unresolved. | Wrapper/decoder source proves fusion; a declared seq2seq role may label an opaque conditioning relation but cannot build the mechanism. |
| M-14 | P0 | `schema.tower_submodel_spec` maps any non-`linear` attention verdict to MHA, hardcodes FFN kind dense, marks structure proven, and forces uncached attention. | Carry exact typed attention/FFN facts from each variant. Unknown remains unknown; no binary “else MHA” or unconditional `structure_status="proven"`. |
| M-15 | P1 | `metadata_modalities._encoder_attention_child` independently infers MHA/GQA from config counts, and `_MODALITY_BLOCK_SPECS` provides structural fallback kinds. | Delete renderer inference. Metadata consumes the canonical component spec/region and uses presentation defaults only for words. |
| M-16 | P2 | modality existence and component class/config keys remain config-address driven. | This is lawful only for candidate discovery. The root construction graph confirms the component is owned and used; unused declared components remain stated inventory, not active paths. |

### 16.4 Diffusion root, DiT/MMDiT, conditioning, and scheduler

| ID | Priority | Current site and behavior | Code-bound replacement and deletion |
|---|---:|---|---|
| D-01 | P1 | diffusion adapter selection uses `_class_name` substrings in `dit_class_markers`; `is_unet` uses a config signature. | Treat `_class_name` only as an address. Resolve the exact root class and classify its constructed/call graph capability (transformer stack vs U-shaped stages). Unknown adapter remains unsupported/opaque. |
| D-02 | P0 | diffusion `_resolve` is another first-hit alias resolver with no conflict/selected-path semantics. | Reuse the one owner-scoped resolver from T-01. |
| D-03 | P0 | missing DiT intermediate size falls to `mlp_ratio=4`; head-dim/head-count interpretations and several dimensions are derived using assumed Diffusers conventions. | Parse the exact constructor expressions to learn which config field and default expression are used. Values remain config; unresolved geometry stays partial. |
| D-04 | P0 | `_conditioning` chooses dual-stream, concat-joint, cross-attention, KV-join, or plain self-attention solely from which dimension fields are present. | Root/block source proves streams, joins, self/cross attention sites, and upstream fusion. Dimension fields only size already-proven edges. |
| D-05 | P0 | `_code_attn_kind` returning `None` becomes MHA and an asserted tag. | Unknown becomes opaque attention; only an owner-bound softmax/linear/recurrent fact selects the region. |
| D-06 | P0 | a single-stream layer with unresolved `_code_single_fusion` falls into the fused `single_stream_decoder_layer` path. | `parallel`, `concat_fused`, `sequential`, and unknown are explicit typed outcomes; unknown renders an opaque single-stream block. |
| D-07 | P0 | `code_block_conditioning is None` falls through to AdaLN gates, so reader abstention creates modulation/gate operations. | Only positive source evidence creates AdaLN/gates; false creates plain cells; unknown creates unresolved conditioning wiring. |
| D-08 | P1 | RoPE, learned positions, QK-norm, cross-attention pre-norm, and attention bias still accept config-presence/enums as sufficient mechanism evidence. | Bind each field to exact construction/forward operations. Declaration-only facts may be displayed without adding the op. |
| D-09 | P1 | `_dit_norm_kind` uses `norm_type_kind` and epsilon spelling to choose norm implementation. | Constructed norm class/math decides. Delete the structural portion of `norm_type_kind`. |
| D-10 | P0 | `typing.yaml:temporal_config_fields` and `_temporal_axis` allow field presence/3-element patch size to classify video when source is missing. | Separate weak “declared temporal geometry” from proven temporal computation. Only source proof stamps temporal ops or a video tower; unknown remains geometry-only. |
| D-11 | P1 | `_audio_latent_domain` turns two VAE fields into an audio architecture and output wording. | Component code proves 1-D waveform/latent operations; config fields remain declared audio geometry. |
| D-12 | P0 | `_resolve_conditioning` and `conditioning.yaml` map enums directly to modality, projector class, labels, and text/no-text topology; absence plus a text encoder defaults to text conditioning. | Bind enum branches to root constructor/forward and exact projector class. Delete structural templates and the text-presence fallback; keep enum/display wording only. |
| D-13 | P0 | `config_facts.yaml` is a general present-field-to-card mechanism, including structural lists/booleans and “silent” fields whose consumption is never proved. | Classify every row: typed fact with code binding; pure numeric display attached to a proven owner; or scoped ignore. Delete the generic chip author. |
| D-14 | P1 | secondary stack lane classification maps raw forward parameter names through `stack_lane_params`. | Lawful only after the stack owner and its exact forward parameter are bound. Replace lane strings with typed dataflow edges when H5 can express them. |
| D-15 | P2 | `companion_denoiser_fields` detects components by pipeline keys and assumes architecture sharing. | Keys locate candidates; resolve each component source and compare structural equivalence before saying it shares architecture. Otherwise state an unresolved companion. |
| D-16 | P0 | `_scheduler_geom` uses scheduler class substrings to mark flow matching; `_scheduler_step_view` selects Euler-like/denoise prose from class/prediction config. | Inspect the exact scheduler `step()` plus helpers/state. Derive update graph, history, noise, predictor/corrector, target, and state use. Config supplies coefficients/selected prediction only through bound reads. |
| D-17 | P1 | text encoder friendly map is display-only, but source-missing encoders can still receive generic text-encoder pipeline semantics from component presence. | Keep titles only. The component spec is opaque until its own source/config parse resolves; no inferred concatenation/projection from family or width alone. |
| D-18 | P1 | `blocks.diffusion_loop_blocks` assumes patch defaults, derives temporal latent formulas, defaults denoiser style to transformer, and authors pipeline operations from raw geom. | Loop blocks project typed root/component facts. Geometry formulas carry provenance; unknown denoiser/style is opaque. |

### 16.5 UNet — a whole hand-authored architecture that must become owner-bound

`adapters/diffusor/unet.py` is not merely a config reader. It is currently a
second architecture interpreter containing a substantial encoded model of
Diffusers’ common UNet implementation. This is the highest-risk conversion after
the substrate defaults.

| ID | Priority | Current site and behavior | Code-bound replacement and deletion |
|---|---:|---|---|
| U-01 | P0 | `is_unet` classifies a U-shaped architecture from config field presence. | Resolve the exact denoiser class and prove down/mid/up ownership/construction. Config signature may be a candidate only. |
| U-02 | P0 | `parse_unet.has_attn` tests whether a block-type string contains `Attn`; block strings directly author attention placement. | Use each string only to address the exact class, then inspect its construction/forward. Class name never proves its cells. |
| U-03 | P0 | ResNet counts default to 2, transformer depth to 1, up ResNet count adds one, and sample placement is “all but final.” | Read constructor/list comprehensions/default expressions and bind their config fields. Missing proof yields unknown stage detail. |
| U-04 | P0 | `attention_head_dim` is reinterpreted as head count under a Diffusers convention. | Bind the exact attention constructor arguments for each stage and record the expression. |
| U-05 | P0 | width mismatch implies `text_proj`; block lists and cross width imply cross-attention story. | Prove the encoder projection class and per-stage K/V flow from source. |
| U-06 | P0 | absent `act_fn` becomes SiLU and render/card prose treats GroupNorm, two Conv3x3 operations, timestep add, residual, and optional 1x1 shortcut as the universal ResNet cell. | Build each resolved ResNet/temporal cell from its source op/dataflow graph. Class defaults are values only after binding. Unknown cell is opaque. |
| U-07 | P0 | `_unet_transformer_subblocks` creates MHA self-attention, MHA cross-attention, no RoPE/cache, and a four-times-width FFN from convention. | Resolve the exact nested transformer class/callables; reuse canonical typed facts/regions. Delete the hand-authored Transformer2D template. |
| U-08 | P0 | temporal root detection can stamp detailed temporal conv/attention/AlphaBlender prose; some per-stage evidence exists but the hardcoded op chain remains. | Require positive per-stage owner evidence for every temporal op and order. Root frames-axis evidence cannot stamp child mechanisms. |
| U-09 | P1 | downscale/upsample factors and stride/nearest-neighbor descriptions are conventional formulas. | Read exact sampler class/forward and its scale expression; config supplies factors. |
| U-10 | P0 | UNet cards and `block_views/unet.py` can independently restate structural order and modality. | Project the same stage/cell regions; renderer accepts geometry/layout only. |

### 16.6 VAE and codec components

| ID | Priority | Current site and behavior | Code-bound replacement and deletion |
|---|---:|---|---|
| V-01 | P0 | `_vae_class_kind` infers VQ from config field presence and otherwise labels the component “VAE,” with KL/2-D-neutral prose. | Resolve exact encode/decode/quantize mechanisms. Unknown is “latent decoder,” not KL/VAE/VQ. |
| V-02 | P0 | `_vae_geom` converts config stage lists/ratios into decoder structure; `base_dim*dim_mult` and audio ladders are interpreted without binding their source expressions. | Bind VAE constructor/stage graph and expressions; config supplies values. |
| V-03 | P0 | `_vae_decoder_children` defaults output channels to 3, ResNet count to `layers_per_block+1`, upsample to all but last, scale to powers of two, and constructs conv/mid/output stages. | Exact VAE owner graph supplies stages and operations. Unknown leaves dimensions without fabricated cells. |
| V-04 | P0 | `_vae_resnet_ops` and `block_views/vae.py` hardcode GroupNorm+SiLU, Conv3x3 twice, residual, and nearest-upsample+conv; renderer repeats the labels independently. | One owner-bound op region drives cards/SVG/JSON. Delete renderer-local graph authorship. |
| V-05 | P1 | mid attention, quant/post-quant convs, causal/temporal/audio behavior, and activation are config-presence facts or pending debt. | Migrate one exact VAE fact at a time with source completeness and owner-tight config premises. |
| V-06 | P1 | output “Image/RGB/Waveform/Frames” comes from inferred domain and default channel count. | Output domain/type is a component fact; presentation formats it without inferring. |

### 16.7 Renderers, parameters, conformance, and source interpretation

| ID | Priority | Current site and behavior | Required end state |
|---|---:|---|---|
| X-01 | P0 | renderer modules still branch on raw `extras`/spec leaf combinations and supply fallback kinds, paths, titles, norms, and cell graphs. | Enforce a dependency firewall: renderers import typed projection DTOs/regions only; no raw config, source, family, or mechanism inference. |
| X-02 | P0 | `params._attn_params` defaults missing KV heads to Q heads and missing head dim to `hidden/heads`; `_ffn_params` treats unknown gating as dense; norms/final norm use fixed formulas. | Parameter formulas are selected by typed mechanism/storage/ownership facts. Unknown yields a range/partial total with explicit missing terms, not one nominal architecture. |
| X-03 | P1 | parameter estimates ignore or conventionally collapse biases, embedding/position tables, fused/shared matrices, routed storage, diffusion embeddings, VAE/UNet/codec components, and component sharing. | Build an owner-bound parameter graph from the same constructed modules and config dimensions. Deduplicate shared parameters by owner identity. |
| X-04 | P0 | `conformance.py` defaults unproven projection mode to split and contains fallback candidate unions. | Conformance compares each rendered region to its exact owner graph. Unknown is unresolved, not split; rival owners block. |
| X-05 | P0 | `conformance/abstractions.yaml` carries `cogvideox` and `flux2` family/view exceptions and citations. | Replace with typed structural abstraction receipts tied to exact owners/ops, then delete identity-scoped rows. |
| X-06 | P1 | type-role, fact-marker, component-class, processor, and drill-marker vocabularies can turn substrings/tokens into semantic roles after broad closure. | Vocabulary emits candidate observations only. Exact construction/call/dataflow binding promotes them; owner-qualified counterexamples prove no sibling contamination. |
| X-07 | P1 | global omissions and presentation-kind drops can hide a real missing/fabricated operation across unrelated owners. | Abstraction policy is fact/owner/altitude scoped with an explicit receipt and reverse witness. |
| X-08 | P0 | broad reader exceptions return `None`/False, and many callers interpret that silence as the familiar default. | Typed reader failures distinguish no source, unsupported syntax, rival owners, incomplete graph, and proven absence. Callers cannot branch on bare `None`. |
| X-09 | P1 | `evidence/sources.py` uses class/model mappings and DiT markers to locate files; this is lawful address work but currently shares untyped strings with structural consumers. | Return opaque typed addresses resolved once. Structural modules never receive raw identity strings; exact component owner is mandatory. |
| X-10 | P1 | class-name role vocabulary (`type_roles.yaml`, `_role_of`) is treated as code-shape truth. Names like `MLP`, `Attention`, or `Processor` can misclassify a novel callable. | Role match is a candidate; constructor field, call reachability, signature, and forward ops confirm the role. Unknown roles remain opaque. |
| X-11 | P1 | raw extraction and callable resolution are duplicated across forward/transitive/specialized readers, so the same owner can produce different closures. | H5 program index is the single raw graph. Specialized readers query it and return typed facts; no independent reparse/union. |
| X-12 | P1 | display maps (`text_encoders`, scheduler labels) are fingerprinted but their consumers sometimes combine display identity with structural fallback behavior. | Pass display labels in a separate typed channel that cannot reach structural keys or fact values. |

### 16.8 Expanded JSON — a public structural consumer with independent assertions

`Diagram.to_json()` calls `expanded.build_expanded`; therefore this package is a
public architecture surface, not diagnostic serialization. Most modules project
IR or canonical regions cleanly, but the sites below still strengthen missing
facts or reconstruct topology. They must be governed by the same projection
receipts and unknown laws as HTML, cards, parameters, and conformance.

| ID | Priority | Current site and behavior | Required end state |
|---|---:|---|---|
| J-01 | P0 | `expanded.stack.build_stack` always publishes `kind="decoder_only"`, including diffusion, encoder-only, and unresolved stacks. | Stack kind is a typed root-topology fact. Unknown remains unknown; denoisers, encoders, encoder-decoders, and decoder-only stacks project their own proven kind. |
| J-02 | P0 | `expanded.attention._cache` enables KV cache for every MHA/GQA/MQA and MLA, while `_operation_graph` splices a cache for MHA/GQA/MQA/**unknown** without consulting `AttentionSpec.cached` or cross-attention. | JSON cache nodes and descriptors consume the same cache fact as the canonical attention region. `False` suppresses cache, `True` emits it, and unknown stays explicit. Remove the JSON-local cache policy. |
| J-03 | P0 | `_projections` authors split Q/K/V/O matrices from attention kind and geometry even when storage is unknown; `attention_region` then independently defaults unknown to MHA/split/scaled/RoPE. | Project only the canonical owner-bound region. Storage, score scaling, positional ops, and cache must not be reconstructed from kind/counts. A partial attention may expose dimensions without invented matrices. |
| J-04 | P0 | `expanded.ffn` inherits canonical FFN defaults and additionally constructs MoE expert templates with `gated=True` when gating is absent. | Expert and dense JSON graphs consume the exact FFN/expert mechanism facts. Undeclared internals produce an opaque expert/cell, not a conventional gated template. |
| J-05 | P0 | `expanded.residual` calls every topology without a left-lane FFN `sequential`; `expanded.block_graph` always connects all blocks in list order, regardless of forks, joins, residual sources, or external lanes. | Both are projections of the canonical cell DAG/topology fact. Absence of a recognized parallel marker is not proof of sequential flow. |
| J-06 | P0 | `expanded.sections._is_diffusion` reads `extras.render.family`, then uses that presentation field to select structural dimensions and I/O. `_diffusion_io` authors patchify, final norm, linear output, and noise-prediction semantics. | Render policy cannot select machine architecture. Root/component facts decide model kind and exact I/O regions; JSON formats those regions. Unknown denoiser bookends remain partial. |
| J-07 | P0 | ordinary `build_io` universally authors token IDs, embedding lookup, final norm, and LM head; `_final_norm_kind` borrows the last layer norm instead of consuming a final-norm fact. | Root I/O, embedding, final norm, and output head are separate typed facts. Encoder-only, headless, multimodal, tied/shared, and unresolved roots do not receive decoder-LM bookends by convention. |
| J-08 | P1 | `expanded.grouping.signature` omits structural fields such as cache, bias, RoPE/position application, score behavior, projection storage, cross-attention details, FFN storage/routing, and several block attributes. Structurally different layers can collapse into one group. | Generate grouping signatures from the registered structural fact/cell schema, not a hand-maintained tuple. Add a poison for every newly registered per-layer structural field. |
| J-09 | P1 | attention/FFN `code_finding_ids` are selected from global `(kind,value)` detection buckets rather than exact component owner and fact provenance. A sibling tower can receive another tower's finding IDs. | Trace fields carry the projection receipt of the exact owner-qualified fact. Remove semantic bucket matching from the serializer. |
| J-10 | P1 | `expanded.loop`, `modalities`, `pathways`, and `block_graph` faithfully copy raw `extras`, but this makes raw parallel structural IR look canonical and bypasses fact-projection accounting. | These modules project registered component/topology DTOs. Raw `extras` may remain only during a counted migration quarantine, with owner and deletion unit. |
| J-11 | P1 | boolean cleanup (`value or None`) removes proven `False` for fields such as bias/qk-norm/shared, conflating a negative fact with absence in the public schema. | Serialize tri-state fact status deliberately: proven false remains `false`; unknown remains `null` or a typed unresolved object; not-applicable is distinct. |
| J-12 | P1 | expanded JSON has no independent anti-fabrication frontier: current tests mainly pin familiar positive outputs, including the asserted cache and decoder stack. | Add JSON to every vertical fact's projection receipts, metamorphic matrix, reverse-fabrication audit, and preservation witnesses. Tests must include unknown, negative, sibling-owner, encoder, diffusion, and source-missing cases. |

---

## 17. Dependency order discovered by the whole-tree audit

The ledger changes the safe execution order. Domain conversion must not begin by
rewriting individual parsers while their targets still have architectural
defaults. That would create “new evidence plus old fallback forever.”

### Phase A — close the substrate before changing model behavior

1. Correct H3 alias/owner accounting (T-01/D-02).
2. Correct the structural-author census and actual projection receipts (C-09 to
   C-11).
3. Complete the H5 owner-bound program index and typed failure taxonomy (X-08,
   X-09, X-11).
4. Remove/tri-state object, assembly, label, and op-graph mechanism defaults
   (C-01 to C-07) behind preservation fixtures.
5. Enforce renderer/expanded-JSON/params/conformance consumer firewalls (X-01
   to X-04 and J-01 to J-12).

At the end of Phase A, an unresolved fact must remain visibly unresolved even if
no domain reader has yet been migrated. This prevents every later unit from
silently falling back to the old familiar architecture.

### Phase B — transformer and recursive-component facts

Migrate in this order because downstream components reuse the root machinery:

1. attention mechanism/storage/cache/scaling;
2. FFN mechanism/storage/activation;
3. norm and cell topology;
4. mask/position schedules;
5. MoE/router/per-layer selectors;
6. cross-attention and fusion;
7. recursive submodel contract;
8. vision/audio/video/projector paths;
9. MTP/codebooks/codec/other auxiliary components.

Each item is a vertical fact slice, not a parser-wide rewrite.

### Phase C — diffusion components

1. root denoiser/block ownership and stream topology (D-01 to D-10);
2. conditioning/projector/fusion (D-12 to D-14);
3. UNet stage graph, then each cell family (U-01 to U-10);
4. VAE/codec exact component graph (V-01 to V-06);
5. scheduler `step()` graph (D-16);
6. diffusion loop composition and typed component edges (D-18).

### Phase D — parameter/conformance closure and deletion

1. parameter graph coverage for every migrated owner;
2. exact-owner JSON projection, conformance, and typed abstraction receipts;
3. delete semantic YAML sections and legacy raw extras as their last consumers
   disappear;
4. delete manual renderer graphs, fallback spec builders, class-default structural
   paths, and identity-scoped conformance exceptions;
5. run H10 only after the deletion/stale-symbol gates prove those paths are gone.

---

## 18. Breakage map and mandatory witness families

Removing defaults will intentionally expose unresolved cases. That is safer than a
false familiar diagram, but it can also make existing views pale or incomplete if
a reader is not ready. Therefore each phase freezes both positive and negative
witnesses before editing.

| Change frontier | Positive witnesses | Mandatory counterexamples/preservation controls |
|---|---|---|
| attention kind/storage/cache | MHA, GQA, MQA, MLA, fused QKV, split QKV, cached decoder | bidirectional encoder, cross-attention, linear/recurrent mixer, source-missing owner |
| FFN | dense GELU, split SwiGLU, fused gate-up, Conv-GLU, MoE | dense RMSNorm model, gated sibling tower, ambiguous owner, source-missing FFN |
| masks/position | causal, bidirectional, sliding, chunked, RoPE, ALiBi, learned absolute, NoPE | parameters present but unused, partial RoPE, multimodal/vision positions, source missing |
| schedules | uniform, alternating, threshold, explicit list, nested/tiled | short/malformed list, unequal aliases, equivalent token spellings, unseen token |
| multimodal | placeholder replace, prefix concat, cross-attention, grid stream | unused declared component, width mismatch without projector, dense vision sibling, source-missing wrapper |
| DiT | dual stream, joined stream, cross-attention, single parallel, single sequential | dimension fields with a different forward, reader abstention, no AdaLN, non-softmax attention |
| UNet | SD/SDXL common cells, Kandinsky-like code-defined placement, temporal stages | no mid block, simple cross-attention without Transformer2D, non-SiLU ResNet, custom sampler |
| VAE/codec | KL, VQ, temporal, causal 3-D, audio 1-D | config fields present but unused, no mid attention, mixed stage types, unknown latent decoder |
| scheduler | Euler-like, multistep/history, stochastic, flow, consistency | class name containing “Flow” with different `step()`, prediction enum unused, source missing |
| expanded JSON | decoder, encoder, diffusion, cross-attention, cache/no-cache, parallel/sequential | unknown root, source-missing cell, sibling-owner evidence, bidirectional MHA without cache, partial topology |
| params | tied/untied, dense/gated, MHA/GQA/MLA, MoE/shared | unknown mechanism range, shared modules, fused storage, embedded component deduplication |

For every table row, use:

- the failing model;
- an equivalent mechanism with a different family/name;
- the same config signal whose code does **not** enact the mechanism;
- an unrelated renderer/control model;
- source-present, partial-source, source-missing, and rival-owner cases;
- summary image, deepest affected drill, cards, JSON, params, conformance, and
  click-routing comparison.

---

## 19. One-pass implementation rule

“Go at once” means execute one coherent campaign against this single ledger; it
does not mean one unsafe mega-commit. The working branch may carry the campaign,
but each vertical fact family lands as a separately reviewable commit that:

1. introduces the capability only if the shared interpreter lacks it;
2. migrates all producers of that fact across root and embedded contexts;
3. makes every consumer use the same typed fact/region;
4. deletes the config/default/renderer/conformance path it replaces;
5. runs the complete witness matrix and unchanged-tree gates;
6. updates the ledger row from `PENDING` to `DONE` with its receipt.

No row is complete merely because one known model looks correct. Completion means
the structural relation is encoded once, the same relation works for an equivalent
model of another identity, a counterexample does not receive it, and every old
authority path has disappeared.

---

## 20. Binding implementation runbook — exact work to perform

This section converts the audit ledger into executable work. It is authoritative
about sequence, interfaces, deletion targets, and achieved output. An implementer
may make a smaller internal refactor when the tree proves it necessary, but may
not change the authority boundary, preserve an old fallback indefinitely, or
declare a unit complete with a different output.

### 20.1 The four shared contracts to build once

Do not let each domain invent its own version of these objects.

#### Contract A — one config resolution

Implement in `model_unfolder/evidence/config_access.py`:

```text
ConfigResolution[T]
- component: owner path
- canonical: canonical field name
- selected_path: exact config path or null
- selected_alias: exact spelling or null
- value: T or null
- state: present | absent | ambiguous
- present_aliases: ordered tuple of (exact path, value)
- source_kind: checkpoint | class_default
- reason: typed explanation
```

Required behavior:

1. One call both returns the value decision and records the corresponding
   `ConfigAccessEvent`; callers must not manually duplicate the event.
2. No present aliases means `absent`; it does not create a fictional consumed
   config read.
3. Multiple unequal values means `ambiguous`; no value is selected and no
   structural fact may be emitted.
4. Multiple equal values select deterministically by declared alias order while
   retaining every exact spelling in `present_aliases`.
5. `bind(reader, fact_owner, fact_key)`, `consume(fact_owner, fact_key)`, and
   `ignore(reason)` are explicit transitions; merely inspecting a value cannot
   count as projection.
6. Class defaults use the same result type but remain distinguishable from
   checkpoint values.

`ConfigAccessEvent` remains the immutable audit event. It must not be used as a
value container. Delete parser-local first-hit resolution after all callers use
`ConfigResolution`.

#### Contract B — one exact component owner

Implement in `model_unfolder/evidence/models.py` (or a single adjacent module if
an import cycle requires it):

```text
ComponentOwner
- path: root / root.text / root.vision / root.audio / root.denoiser / root.vae / ...
- config_path: exact component-config path
- class_ref: resolved constructed class
- source_files: exact files containing the owner
- constructor: exact constructor callable
- forward: exact forward/call callable
- parent_path: owning component
- completeness: complete | presence_only | partial | oracle_missing | ambiguous
- failures: tuple[ReaderFailure]
```

Identity fields may help create this address, but are not part of its semantic
capabilities. All readers accept a `ComponentOwner`; they must not independently
rediscover classes or scan a union of same-role classes.

#### Contract C — one typed reader outcome

Implement in `model_unfolder/evidence/models.py`:

```text
ReaderResult[T]
- state: proven | proven_absent | partial | unsupported | ambiguous | oracle_missing
- value: T or null
- owner: ComponentOwner
- spans: tuple[SourceSpan]
- premises: tuple[fact id]
- completeness
- failure: ReaderFailure or null
```

Rules:

- `False` is emitted only by `proven_absent` with complete coverage.
- `partial`, `unsupported`, `ambiguous`, and `oracle_missing` never collapse to
  `None` that a caller can reinterpret as the common architecture.
- A result becomes an `EvidenceFact` exactly once. The parser may project it; it
  may not reinterpret the raw AST observations independently.
- Broad exception catches convert to typed `ReaderFailure` with owner, phase,
  file, syntax node, and cause. They cannot silently return a structural value.

#### Contract D — one projection receipt

Implement in `model_unfolder/evidence/projection.py` and store receipts call-locally
on `ParseContext`/`RenderContext`:

```text
ProjectionReceipt
- fact_id
- owner
- fact_key
- fact_value_hash
- fact_status
- surface: ir | opgraph | block | card | html | json | params | conformance
- structural_target
- projector symbol
- output hash or node ids
```

The actual projector emits the receipt. A manually maintained set saying a field
“is drawn” is not a receipt. The same fact may have multiple receipts, but every
receipt points to one exact owner-qualified fact.

### 20.2 Unit map and non-negotiable ordering

| Unit | Primary ledger rows owned | May begin when | Concrete achieved output |
|---|---|---|---|
| U0 | verification substrate | immediately | reproducible unchanged-tree baseline and preservation manifest |
| U1 | T-01, D-02 and H3 defects | U0 | every config decision has exact owner/path/alias and both audit nets are owner-tight |
| U2 | C-08 to C-13, H2/H6 defects | U1 | every structural writer and projection is registered or shrinking debt |
| U3 | X-08 to X-11 | U2 | one owner-bound program index and typed failure path |
| U4 | C-01 to C-07, C-12 | U3 | no IR/opgraph/label/card/JSON default can turn unknown into a familiar mechanism |
| U5 | X-01, X-06/X-07, J-09 to J-12 (firewall; final closure in U14/U15) | U4 | consumer firewalls and receipt-based reverse-fabrication nets |
| U6 | T-02, T-08 to T-11, D-05, M-14/M-15, J-02/J-03 | U5 | one attention fact family across every owner altitude |
| U7 | T-07, T-09, T-12, M-12/M-13, J-04/J-05/J-07 | U6 | one FFN/norm/cell-topology and root-bookend fact family across every owner altitude |
| U8 | T-03 to T-06, T-13 to T-18 | U7 | code-bound masks, positions, schedules, routers and auxiliary layer selectors |
| U9 | M-01 to M-16 | U8 | recursive multimodal owners, exact projector/fusion paths, no family fallback |
| U10 | D-01, D-03 to D-15, D-17/D-18 | U9 | exact diffusion root, stream, conditioning and component topology |
| U11 | U-01 to U-10 | U10 | source-derived UNet stage/cell graph; hand-authored interpreter deleted |
| U12 | V-01 to V-06 | U10 | source-derived VAE/codec graph; conventional VAE template deleted |
| U13 | D-16 | U10 | scheduler update graph from the exact `step()` implementation |
| U14 | J-01 to J-12, X-02 to X-05 | U6–U13 facts complete | JSON, parameters and conformance are projections of the same facts |
| U15 | C-14, D-12/D-13, X-12 and YAML/dead paths | all relevant owners migrated | semantic YAML/config/default authority removed; only lawful value/syntax/display data remains |

U11, U12, and U13 may be separate reviewed commits after U10, but only one may
modify shared evidence/program infrastructure at a time. U14 can migrate one
already-complete fact slice at a time; it cannot invent missing domain facts.

### 20.3 U0 — freeze the real baseline

**Edit/add:** `test_support/tree_state.py`, `tests/test_isolation.py`, and the
completion record in Section 21.

Perform exactly:

1. Add a fingerprint helper that hashes tracked content, untracked content,
   relative paths, and executable bits while excluding only declared test
   artifacts. `git diff` alone is insufficient because it omits untracked files.
2. Record collection count and the fingerprints of current Sable corpus files.
3. For every witness named below, save without re-blessing:
   - normalized `ModelIR`;
   - fact/config ledgers;
   - canonical regions and block ids;
   - expanded JSON;
   - parameter totals and assumptions;
   - HTML/SVG structural hash, view ids, and click targets;
   - conformance findings and existing PNG/gallery paths.
4. Run each hardening test file alone once, then the grouped hardening gate, full
   suite, and `tests/test_sable.py` alone on the same unchanged tree.
5. Add a test that deliberately mutates a temporary copied tree during a run and
   proves the fingerprint gate rejects the result.

**Do not:** modify corpus blessings, normalize an unexplained delta, or use a test
run that changed the source tree.

**Done means:** Section 21 contains the before/after fingerprint, collection
count, exact commands, results, duration, and frozen preservation witnesses.

### 20.4 U1 — replace every config decision with the exact resolver

**Primary edits:**

- `model_unfolder/evidence/config_access.py`
- `model_unfolder/evidence/context.py`
- `model_unfolder/parser.py`
- `model_unfolder/adapters/transformer/parser.py`
- `model_unfolder/adapters/diffusor/parser.py`
- `model_unfolder/adapters/transformer/special_parts/modalities/accessors.py`
- component readers under `special_parts/modalities/`

Perform exactly:

1. Add `ConfigResolution` and make it the only alias/default resolution API.
2. Give `ParseContext` the call-local config ledger directly; compatibility bare
   name lists must be derived read-only views.
3. Migrate root, text, vision, audio, denoiser, VAE, scheduler and nested encoder
   scopes. Every read must carry the correct component path.
4. Replace both parser-local `_resolve`/first-hit implementations. Do not retain a
   “fast path” that bypasses conflict detection.
5. Replace leaf-key `PENDING_PROJECTION_DEBT` matching with exact
   `(component/fact owner, canonical, fact key)` matching.
6. Make absent config values default/class-default premises rather than accessed
   or consumed config fields.
7. Reject conflicting aliases at the decision site. The parse may continue with
   an unknown fact, but may not choose either value.
8. Remove the three audit-clearing diffusion reads unless U10/U12 already gives
   them a real fact and receipt.

**Delete:** parser-local alias loops, manual touched/bound/consumed mutation, and
leaf-key pending-debt excusal.

**Tests:** `test_config_access.py`, `test_config_intents.py`,
`test_h7_diffusion.py`, `test_isolation.py`, plus counterexamples for equal and
unequal aliases, sibling `hidden_size`, nested capture, concurrency, absent
fields, class defaults, and source missing.

**Done means:** no structural parser contains a first-present alias loop; both
blocking nets report owner-qualified events and a sibling component cannot clear
another component's debt.

### 20.5 U2 — close every structural-author and projection escape

**Primary edits:**

- `model_unfolder/evidence/structural_writes.py`
- `model_unfolder/evidence/registry.py`
- `model_unfolder/evidence/context.py`
- `model_unfolder/renderers/html/fact_projection.py`
- `model_unfolder/renderers/html/render_context.py`
- `model_unfolder/expanded/`
- `model_unfolder/params.py`

Perform exactly:

1. Change `StructuralWrite.key` to
   `(module, enclosing_symbol, sink_kind, normalized_target)` and pin a
   line-insensitive multiset. A second writer for an existing target must grow
   the count and fail.
2. Statically census:
   - legacy and typed ledger writes;
   - dataclass/IR/spec construction and mutation;
   - every nested `extras` leaf;
   - `Region`, `Op`, block, card and view construction;
   - renderer structural phrases/kinds;
   - expanded JSON structural keys;
   - parameter formula selection and conformance assumptions.
3. Recursively census runtime values, including lists and nested dictionaries;
   retain owner, writer, type and exercised value.
4. Expand `FactDefinition` so projection policy is owner-qualified and covers
   `ir`, `opgraph`, `block`, `card`, `html`, `json`, `params`, and
   `conformance`.
5. Introduce `ProjectionReceipt`; run it in shadow mode beside the current
   `*_DRAWN` sets and prove parity for currently registered facts.
6. Replace `DRAWN_UNLEDGERED_DEBT` and raw-extras baselines with entries carrying
   exact owner, writer symbol, reason, last consumer, migration unit and intended
   deletion. No free-form allowlist row is accepted.
7. Add poison writers for a second writer of an existing target, nested extras,
   a spec field, opgraph default, card claim, JSON claim and parameter formula.

**Delete after parity:** nominal manual projection witnesses for migrated facts;
top-level-only runtime census logic.

**Tests:** `test_structural_writes.py`, `test_fact_registry.py`,
`test_projection_audit.py`, `test_projection_obligations.py`, and new JSON/params
poisons.

**Done means:** changing representation cannot bypass the registry, and every
drawn/serialized/counted structural claim has an actual owner-qualified receipt.

### 20.6 U3 — build the single raw program index and owner resolver

**Primary edits/additions:**

- add `model_unfolder/evidence/program_index.py`
- `model_unfolder/evidence/models.py`
- `model_unfolder/evidence/context.py`
- `model_unfolder/evidence/sources.py`
- `forward_ops.py`, `transitive.py`, `inspector.py`, `patterns.py`, `stacks.py`
- specialized `vision.py`, `audio.py`, `projector.py`, `fusion.py`, `position.py`,
  and `ffn.py`

Build one `ProgramIndex` per `SourceBundle` containing:

- files/modules/import aliases;
- classes, bases and class assignments;
- constructor fields and the expressions assigned to them;
- methods/functions, parameters, returns and source spans;
- direct calls, `self.method` calls and reachable call closures;
- static branches/loops/comprehensions and their controlling expressions;
- attribute reads/writes and config-path reads;
- constructed submodule class references;
- tensor/dataflow-relevant operation observations;
- unsupported syntax and parse failures.

Perform exactly:

1. Resolve source once in `ParseContext` using identity only as address.
2. Resolve a `ComponentOwner` from the parent's construction graph. Candidate
   role/name markers may nominate a class but cannot complete resolution.
3. Make every evidence reader query the same index and exact owner.
4. Replace broad class-role unions with rival-owner/ambiguous results.
5. Replace bare `None`/False failures with `ReaderResult`.
6. Add AST fixtures for alias imports, helper methods, inherited methods,
   factory functions, comprehensions, conditional construction, equivalent
   candidate classes, rival candidates and unsupported dynamic dispatch.

**Delete:** independent reparsing, same-role class union, broad “best candidate”
selection, and duplicated low-level call/field extraction once parity is proven.

**Tests:** `test_code_evidence.py`, `test_conformance.py`, all specialized
evidence tests, `test_reader_exceptions.py`, and `test_h9_frontier.py`.

**Done means (amended 2026-07-27 by
`docs/U3_COMPLETION_MASTER_PLAN.md`):** the neutral ProgramIndex, exact owner
graph, factory/config binding and typed failure path are authoritative for every
migrated or new reader. Surviving legacy semantic readers are held in an exact,
blocking, non-growing symbol/caller inventory assigned to U6–U13 and deleted
once in those units. U3 does not mechanically port known-wrong semantics merely
to delete them before U4. U15 requires the quarantine to reach zero.

### 20.7 U4 — make unknown safe before migrating mechanisms

**Primary edits:** `ir.py`, transformer `assembly.py`, `opgraph.py`, `labels.py`,
`renderers/html/metadata.py`, `cards.py`, and `expanded/`.

Perform exactly:

1. Make architectural defaults tri-state/typed unknown:
   - attention kind, RoPE, bias, QK norm, cache, storage and score scaling;
   - FFN kind, gating, activation and storage;
   - norm kind/placement and cell topology;
   - mask/causality and final/root bookends.
2. Remove RMS/pre/gated/MHA/split/cached/causal/RoPE/SiLU defaults from dataclasses
   and assembly factories.
3. Add explicit opaque/partial canonical regions. An unknown attention can carry
   known head geometry without receiving Q/K/V projections or SDPA. An unknown
   FFN can carry intermediate width without receiving gate/up/down.
4. Make `attention_region`, `ffn_region`, labels, cards, metadata, and expanded
   JSON distinguish true, false, unknown, and not-applicable.
5. Remove `_fallback_sub_inspect_children` reconstruction and the synthetic
   empty-model dominant layer.
6. Freeze positive preservation fixtures before each default is removed. If a
   known-good model becomes unknown, fix its evidence reader in the matching
   U6–U13 slice; do not restore the global default.

**Tests:** `test_opgraph.py`, `test_block_schema.py`, `test_expanded_json.py`,
`test_loud_miss.py`, cards/view tests, and explicit omitted-field poisons.

**Done means:** reader abstention, reader failure and missing source never produce
a familiar mechanism on any surface.

#### U4 execution ledger (binding decomposition, 2026-07-28)

U4 is implemented fact-by-fact across every consumer.  A slice is not complete
when only its dataclass or renderer changes: its producer fallback, canonical
region, labels/cards, expanded JSON, debt row, negative poisons, real-model
controls and preservation delta must close together.

| Slice | Exact boundary | State |
|---|---|---|
| U4-A | attention mechanism + mask vocabulary: missing/novel kind cannot become MHA/SDPA; missing/novel mask cannot become causal; known geometry may ride one opaque region | DONE — Soumil approved 2026-07-28; 14 inspected galleries/fixtures re-blessed; 26-witness preservation 46/46 green |
| U4-B | attention internals: position application, QK norm, projection bias, cache, projection storage and score scaling are independently true/false/unknown/not-applicable | DONE — Soumil approved 2026-07-29; guarded artifacts audited; commit `4857026`; detached-worktree receipt fully green |
| U4-C | FFN mechanism: kind, gating, activation, ordinary/expert storage and widths project independently; an unknown inner form is opaque | ACTIVE — implementation and 26-witness review complete; committed-tree acceptance pending |
| U4-D | layer cell: norm kind, placement, residual topology, parallel/sandwich structure and bookends require an owner-bound fact | PENDING |
| U4-E | empty/unresolved presentation: remove the synthetic dominant layer and card-side structural reconstruction | PENDING |
| U4-F | cross-surface closure: IR, canonical regions, HTML, cards, metadata, expanded JSON and params preserve unknown identically; full poisons and 26-witness acceptance | PENDING |

U4-A's semantic controls are deliberately asymmetric:

- Qwen3 remains GQA because its head geometry decides that known mechanism;
- Sana's self-attention remains linear because modeling code proves its
  processor; its separately-existing cross-attention remains present but its
  mechanism becomes unresolved because that exact sublayer has no reader yet;
- FLUX and SDXL become mechanism-unresolved because the old diffusion reader
  only detected the exceptional linear case and otherwise defaulted to MHA.

That loss of familiar detail is an intentional honesty delta, not evidence that
FLUX/SDXL lack attention.  U10 may restore MHA only through an exact denoiser
owner reader.  U4 must not preserve it with a family branch, asserted debt row
or renderer default.

Working-tree verification receipt:

- focused U4/opgraph/expanded/block-schema: **76 passed**;
- affected opgraph/expanded/block-schema after the final cross-role fix:
  **65 passed**;
- U2 identity/structural-writer/blocking/exception authority gates:
  **57 passed**;
- real Sana Sable: **mechanical PASS**, 13 distinct views / 13 PNGs;
- visual positive: Sana self-attention remains the detailed code-proven linear
  Q/K/V + kernel-feature-map graph;
- visual abstention: Sana cross-attention preserves its known role and K/V
  source but renders one `Cross-attention mechanism unresolved` block;
- visual abstention: FLUX preserves `Joint Attention` but renders its mechanism
  as one unresolved block;
- preservation partition: **32 passed / 14 intentional failures**.  Every
  transformer witness and SDXL's separately-authored UNet path stayed green.
  The failures are the diffusion DiT witnesses whose old MHA/cross-MHA detail
  came from the deleted defaults; Sana changes only on the cross-attention
  surface.
- changed-file pyflakes and `git diff --check`: **clean**.

Independent public-API example bracket (added after the first visual pass):

- `unfold(qwen3-8b)`: 36 causal GQA layers; expanded attention retains
  Q/K/V, cache, scaled dot product, softmax, value application, output
  projection and RoPE;
- `unfold(sana)`: 20 full linear-attention layers; expanded self-attention
  retains its code-proven linear Q/K/V + feature-map graph; cross-attention
  remains a separate text-K/V sublayer with unknown mechanism;
- `unfold(flux-2-dev)`: 56 full-mask layers, mechanism `None`, expanded
  attention contains only `opaque`;
- `unfold(sd3.5)`: 38 full-mask layers, mechanism `None`, expanded attention
  contains only `opaque`;
- `unfold(hunyuanvideo)`: 60 full-mask layers, mechanism `None`, expanded
  attention contains only `opaque`; the independently rendered FFN remains its
  existing Linear → GELU → Linear graph;
- `unfold(pixart)`: 28 layers; self- and separate text cross-attention are
  preserved, both mechanisms unresolved; expanded attention is opaque;
- `unfold(sdxl)`: remains a UNet with no fabricated transformer layer list.

Four independent Sable renders also pass mechanically: Qwen3 **4/4**,
HunyuanVideo **22/22**, PixArt **16/16**, SDXL **29/29**.  Visual inspection
caught two cross-surface issues before blessing: the PixArt parent
cross-attention card originally hid the unknown mechanism, and the first honest
label overflowed its compact box.  The shared typed label now renders
`Cross-Attention / (unresolved)` while its title, facts, description and drill
retain the full `Cross-attention mechanism unresolved` statement.  A permanent
real-PixArt control pins that agreement.

No manifest or durable gallery is changed until Soumil approves this semantic
delta.  A re-bless before that approval would hide the exact decision U4 exists
to make explicit.

Soumil approved the U4-A semantic/artifact delta on 2026-07-28.  The guarded
project `bless()` path then reproduced and replaced exactly the 14 inspected
diffusion galleries/fixtures.  Artifact audit results:

- every changed fixture preserves its exact model identity, source and config;
- every changed fixture records its exact former hash signature as superseded;
- exactly the expected 14 manifest witnesses changed and all 12 controls stayed
  byte-identical across every surface and view;
- the rebuilt 26-witness preservation gate passes **46/46**.

U4-A is therefore closed.  U4-B begins from the independently typed internal
facts; it must not infer one internal merely because another internal or the
overall attention mechanism is known.

#### U4-B implementation ledger (local, awaiting final gate)

U4-B removes the remaining cross-fact shortcuts without pretending that U6,
U8 or U10 have already supplied their exact readers:

- `AttentionSpec` carries QK norm, projection bias, cache, RoPE application,
  QKV storage and score scaling as independent tri-state values. Its grouping
  signature includes them, so unlike layers cannot collapse into one card.
- transformer and diffusion parsers no longer let a similarly named config
  field author QK normalization, projection bias, RoPE application or score
  scaling. A declared score constant becomes an operand only when source also
  proves that this attention applies a scale.
- the canonical attention region draws split/fused QKV storage only from the
  exact storage fact, cache ports only from `cached=True`, a scale formula only
  from a positive/negative scaling fact, and RoPE nodes only from the exact
  `(rope=True, kind=rope, application=qk_rotation)` conjunction.
- HTML child cards, summary chips, expanded JSON and metadata preserve the
  same true/false/unknown distinction. An unresolved QKV layout is one opaque
  projection stage; unknown cache/scale/position are stated explicitly.
- MLA's compressed latent path no longer fabricates a cache: the latent
  transform and an actual cache write/read are separate nodes/facts.
- config-computed NoPE schedules, the block-diffusion QK-norm override and
  UNet no-cache/no-RoPE assumptions were removed. A declared per-layer
  position schedule makes that schedule unknown until U8 proves its selector;
  it may not cause model-wide RoPE to be stamped onto the exceptional layers.
- the `attention_kind`, projection-storage and score-scaling asserted
  conventions are retired from the fact registry/census and their structural
  debt rows are deleted. The blessed-corpus asserted population shrinks from
  the historical 593 to 24, all of which are the separately owned
  MusicGen norm-placement debt.

Real-model controls already exercised on the local tree:

- Llama: split QKV, scaled scores and RoPE remain source-proven; bias/cache are
  unknown rather than inferred.
- BLOOM: fused QKV, biased projections, scaled scores and ALiBi remain.
- Qwen3: GQA, split QKV, QK norm, scaled scores and RoPE remain.
- T5: raw unscaled `QK^T` and relative-position bias remain; storage/cache stay
  unknown.
- FLUX: its separately proven QK norm, scaling and RoPE facts remain, while
  U4-A's unresolved overall attention mechanism keeps the canonical graph
  opaque.
- PixArt: unresolved mechanism and internals remain opaque.
- SDXL UNet: the existing hand-authored MHA/SDPA shell remains visible debt for
  U10, but U4-B no longer supplies QKV storage, cache, position or scaling
  details underneath it.

The intentional loss ledger is equally important:

- StableLM's config-gated per-head QK norm stays unknown because its norm
  wrapper executes a `ModuleList`; U6 must prove that container path.
- Llama-4's QK-norm selector remains proven per layer, but its interleaved
  RoPE/NoPE application is unknown until U8 proves the independent position
  selector.
- ordinary Llama/Qwen bias flags remain unknown until U6 binds each exact
  `Linear(..., bias=config.<field>)` construction.
- decoder KV-cache behavior remains unknown until U6 proves writes/reads;
  causality alone never supplies it.
- UNet's MHA/SDPA shell is not ratified by this slice; U10 must replace that
  template with the exact nested owner evidence.

These are not permission to restore a config fallback. They are the exact
reader worklist for later units.

The real-model pass also exposed one cross-unit substrate regression rather
than an attention-policy mistake.  U3 intentionally represents
`ForwardOps.forward_params` as an ordered tuple and `signature_tokens` as a
set-like census.  The diffusion rotary reader and its conformance twin still
used set union on those unlike containers.  The resulting `TypeError` was
swallowed by `_code_has_rope`, so FLUX's source-proven rotary application
silently became unknown.  U4-B closes the producer seam rather than restoring
a FLUX special case:

- both readers iterate the two observation collections without assuming a
  shared container type;
- a real installed-FLUX regression pins the exact record shape;
- the broad parser catch is deleted, because the lower reader already handles
  missing/unparseable files and programming-contract failures must remain loud;
- the broad-exception ratchet drops `adapters/diffusor/parser.py` from 18 to
  17;
- FLUX again carries independently proven RoPE, QK norm and score scaling while
  its still-unresolved overall attention mechanism remains one opaque region.

Current local verification, before any artifact blessing:

- the U4-B affected cross-surface bracket is **550 semantic passes**, with the
  only remaining failure being the intentionally stale Sable view lock;
- the corrected nested-conformance corpus is **17/17 green** and retires the
  obsolete self-conditioning attention unresolved pin;
- the focused unknown-safety plus U2 authority bracket is **224/224 green**;
- four fresh public `unfold` + Sable runs produced complete galleries for
  Llama, Qwen2-VL, FLUX and PixArt. Llama/FLUX/PixArt are mechanically clean;
  the synthetic Qwen2-VL fixture retains its pre-existing
  `temporal_patch_size` standing-consumption finding, unrelated to U4-B;
- visual inspection confirms: Llama retains split Q/K/V, RoPE and scaled
  scores while naming cache unknown; Qwen2-VL's vision drill retains fused QKV
  and source-proven RoPE while naming score scaling unknown; FLUX and PixArt
  remain honest opaque attention mechanisms rather than acquiring fabricated
  MHA internals;
- a ten-model parent/current comparison keeps parameter estimates identical
  and limits structural deltas to the attention/architecture surfaces this
  slice owns.

Soumil approved the U4-B honesty/artifact delta on 2026-07-29.  The guarded
project `sable()` + `bless()` path then reproduced every affected fixture with
all blocking mechanical checks green, a present code oracle, one PNG per
distinct view, and an offline-equal signature.  The complete artifact audit
records three distinct cases instead of laundering them into one “26 changed”
claim:

- **25 fixtures** changed signature and retain their exact former signature in
  `superseded_hash_signature`;
- **25 galleries / 42 PNG files / 46 named views** changed; every prior
  `her_eyes_review.md` remains byte-identical;
- **BLOOM** has no visual or fixture change, but its canonical IR/evidence
  surfaces changed and are therefore updated in the 26-witness manifest;
- **SDXL** grows from 29 to 30 distinct gallery views because its self-attention
  and cross-attention drills no longer collapse after their independently
  unknown storage/scale facts diverge; the added self-attention and retained
  encoded-text cross-attention views were inspected separately;
- every fixture's model identity, source and frozen config is byte-equivalent
  to its predecessor, and the corpus remains exactly 26 inputs plus 26
  galleries with no orphan.

The rebuilt manifest changes canonical surfaces for all 26 witnesses, as
expected from the fact/IR/expanded/metadata contract change.  Before the local
candidate commit:

- the non-preservation full suite is **2228 passed, 11 skipped, 2 xfailed,
  0 failed**;
- the focused unknown-safety plus U2 authority bracket is **224/224 green**;
- nested conformance is **17/17 green**;
- the combined Sable-regression plus preservation bracket is **47/47 green**
  in 1102.39 seconds;
- tree fingerprint
  `bdc22be4b20ee30bd5d9b3b449c18cb583e23e4c645384e505456d455434985a`
  and blessed-artifact fingerprint
  `67d95d446c6c34e7c10acb2f30d1d420625b85c2160e72f8762b2d525b09fe14`
  are each unchanged before/after that bracket;
- changed-file pyflakes and `git diff --check` are clean.

The exact staged implementation/artifact commit is `4857026`.  Its
detached-worktree coordinator receipt is
`/private/tmp/model-unfolder-verification/598777dc06`:

- focused affected semantics: **1010 passed**;
- U2 authority: **44 passed**;
- collection: **2289 tests**;
- preservation: **46 passed**;
- exhaustive non-preservation partition: **2186 passed, 11 skipped,
  2 xfailed, 0 failed**;
- changed Python static gate: **38 files clean**;
- every lane's complete source-tree and ignored blessed-artifact fingerprint
  is identical before/after.

U4-B is therefore `DONE`.  The next permitted slice is U4-C, which must apply
the same independent-fact and opaque-region law to FFN mechanism, gating,
activation, ordinary/expert storage and widths.  U4-B supplies no permission
to infer any FFN fact from attention or from a model/config identity.

#### U4-C implementation ledger (local candidate, final gate pending)

U4-C applies that law to the complete feed-forward fact without pretending that
U7 has already migrated every FFN reader:

- `FFNSpec` and the canonical FFN region treat mechanism, gating, activation,
  ordinary projection storage, expert projection storage, ordinary width and
  expert width as independent facts.  Missing gating or storage produces one
  explicitly named opaque stage; it never manufactures `gate/up/down`,
  `gate_up/down`, GELU, SiLU or a dense two-linear MLP.
- transformer, diffusion, vision, audio, refiner and nested-submodel producers
  may carry a mechanism only when the exact callable proves it.  A checkpoint
  activation or gating selector is an operand only after code binds that
  selector to the resolved FFN.  Config presence alone no longer authors the
  inner graph.
- ordinary and expert storage remain separate.  A model can prove dense
  `in/out`, split `gate/up/down` or fused `gate_up/down` for one owner while a
  sibling owner remains opaque; a union across sibling FFNs is not evidence.
- labels, cards, tower cells, canonical opgraphs, expanded JSON, metadata and
  parameter assumptions project the same fact.  Unknown is stated as
  `mechanism unresolved`, `gating unresolved`, `storage unresolved` or
  `unsupported`, rather than being translated into a familiar FFN.
- truth-bearing two-line labels use shared content-driven geometry in both the
  graph tower and the legacy top-level architecture view.  Geometry reacts to
  label shape only; it contains no model/family/mechanism branch and therefore
  supplies no architectural authority.
- the vision callable reader now follows the actual `CallableInfo` contract:
  only invoked linear fields contribute projection storage; `chunk`/`split`
  comes from its call census.  The former cross-DTO access could crash a
  secondary tower and is pinned by fused, split, dense and unused-linear
  counterexamples.

This slice also exposes one shared U4 substrate correction: the four existing
layer-grouping implementations are replaced by one schema-derived
`layer_signature`.  Every typed attention/FFN architectural field and structural
block-topology field participates, while presentation prose and provenance debt
do not.  This is required so different FFN mechanisms—and the attention
differences already made independent in U4-B—cannot collapse into one rendered
group.  It does **not** complete U4-D or U4-F: norm/cell facts are not made more
authoritative here, and final all-fact cross-surface closure remains its own
acceptance unit.

Real-model controls on the local candidate:

- DeepSeek-V3 retains its code-proven ordinary FFN and fused expert storage;
  an unbound expert activation stays generic rather than borrowing the sibling
  FFN's activation.
- GPT-OSS retains its MoE routing and code-proven fused expert storage without
  acquiring an unproven activation.
- Qwen2-VL retains its source-proven vision FFN; an independently unresolved
  vision attention mechanism remains opaque and cannot affect FFN detail.
- MusicGen retains its conditioning-tower dense FFN and attention topology;
  the activation remains absent where the frozen config spelling is not bound
  to that exact callable.
- AuraFlow, Qwen-Image, HunyuanVideo and SDXL preserve their known
  conditioning/attention topology while any unproven denoiser/UNet FFN storage
  becomes one readable opaque node.

The guarded visual review covered summary, denoiser/tower and deepest FFN/expert
views.  Lumina's 18 distinct views are an honest dedup result (18 fixture
hashes, 18 certified PNGs and 18 files), not a missing gallery item.

Verification completed before the final committed-tree bracket:

- affected U4-C cross-surface battery: **164 passed**;
- correction/geometry/corpus-identity poison bracket: **7 passed**;
- exact 26-fixture / 26-gallery census with no orphan or duplicate frozen
  config;
- fresh-process Sable regression corpus: **1 passed in 419.67s**;
- all 26 changed fixtures reproduce mechanically clean with a present oracle,
  one PNG per distinct view and an offline-equal signature;
- the final 26-witness preservation manifest has been regenerated from the
  reviewed corpus.

One scope-adjacent defect was discovered by this acceptance run: `bless()`
selected a fixture path from a reconstructed display name, so an offline
MusicGen config could create a second witness instead of updating
`musicgen-small`.  That is acceptance-infrastructure—not FFN semantics—and will
remain identified as such even though the already-running local completion step
amended it into the same unpushed candidate.  The repaired boundary selects an
existing exact config, preserves its reviewed name/gallery path and rejects
duplicate config witnesses.  This co-location grants the guard no architectural
authority: it neither reads facts nor changes parsing/rendering.  U4-C is not
`DONE` until this exact committed tree reproduces and the full
frozen-fingerprint bracket is green.

### 20.8 U5 — establish consumer firewalls

Perform exactly:

1. Renderers accept geometry/layout/display plus canonical blocks/regions; they
   may not import config loaders, evidence readers, source resolvers, family
   identity, or interpret raw config values.
2. Expanded JSON serializes canonical facts/regions and receipts; it may not
   infer from `render.family`, raw `extras`, head counts or absence.
3. Parameters consume registered facts and an owner-bound parameter graph; they
   may not choose a mechanism from a spec default.
4. Conformance compares a projected region to that exact owner's program graph;
   it may not choose split/MHA or union sibling candidates.
5. Add an AST dependency test defining allowed imports for each layer:

```text
source/address -> program index -> mechanism reader -> EvidenceFact
-> canonical IR/Region -> HTML | JSON | params | conformance
```

No arrow may point backward. Display/address wrappers cannot flow to structural
sinks. Raw `extras` access is a shrinking, exact-symbol quarantine until U15.

**Done means:** an attempted raw-config branch in a renderer, JSON serializer,
parameter formula or conformance rule fails a blocking static test.

### 20.9 U6 — migrate attention once for every altitude

**Reader:** create/centralize owner-bound attention interpretation in
`model_unfolder/evidence/attention.py`, querying U3 rather than rescanning.

Facts to emit independently:

- mixer/attention mechanism;
- projection storage and exact Q/K/V/O or latent paths;
- head-geometry field bindings;
- self versus cross attention and exact K/V source;
- cache writes/reads;
- score scaling/softcap/sinks;
- mask application;
- position application;
- Q/K normalization, bias and output gating.

Perform exactly:

1. Parse constructor expressions to bind count/dimension config paths.
2. Parse reachable forward operations and dataflow; do not infer storage from
   counts or mechanism from class names.
3. Emit each fact with owner/completeness/spans and config premises.
4. Make transformer, vision/audio tower, diffusion, UNet nested transformer and
   expanded JSON consume the same fact schema.
5. Make `attention_region` the only detailed attention graph.

**Delete:** transformer `_attention_kind`, diffusion MHA fallback, modality
head-count classification, cache conventions, split-projection default, and
JSON-local projection/cache construction.

**Witness matrix:** Llama MHA/GQA, an MQA model, DeepSeek MLA, fused GPT-style
QKV, split QKV, T5/raw-score attention, cached causal decoder, bidirectional
vision and diffusion no-cache attention, cross-attention, linear/recurrent mixer,
unknown source, and a sibling tower with a different mechanism.

### 20.10 U7 — migrate FFN, norm and cell topology once

**Readers:** `evidence/ffn.py` plus one owner-bound norm/cell reader backed by U3.

Perform exactly:

1. Prove dense, gated, fused-gate-up, Conv-GLU, expert and MoE paths from
   construction plus reachable forward dataflow.
2. Bind activation config only where the code dispatches/uses it. A token such
   as `silu` cannot prove gating.
3. Resolve routed versus shared experts, router/top-k/renormalization and expert
   storage under their exact callables.
4. Resolve norm implementation/math and pre/post/double/parallel topology from
   the exact cell graph.
5. Project one canonical FFN/expert/cell region to main tower, nested encoder,
   diffusion block, cards, HTML, JSON, params and conformance.

**Delete:** `FFNSpec.gated=True`, RMS/pre defaults, config activation-to-structure
logic, card reconstruction, tower dense fallback, diffusion assumed activation,
and JSON MoE gated template.

**Witness matrix:** Llama SwiGLU, GPT-style dense GELU, fused gate-up, Conv-GLU,
Mixtral and DeepSeek routed/shared MoE, Hunyuan text encoder, Qwen vision dense
versus gated text sibling, parallel single-stream diffusion, and source-missing
FFN.

### 20.11 U8 — migrate position, masks, schedules and layer selectors

**Primary readers:** `evidence/position.py`, `stacks.py`, `decoderness.py`, exact
constructor/forward selector readers added on U3.

Perform exactly:

1. Derive RoPE/partial RoPE/NoPE, ALiBi, learned/fixed absolute, relative bias,
   and no-position from operations and their actual application site.
2. Derive causal/bidirectional/sliding/chunked/compressed masks from mask creation
   and application, not architecture suffix or token substring.
3. Parse the source expression selecting per-layer modules. Bind its exact config
   field/list/threshold/frequency and resolve every candidate class.
4. Apply the same selector machinery to mixer schedules, MoE schedules,
   cross-attention schedules, MTP layers, per-layer embeddings and codebooks.
5. Retain YAML only for spelling normalization while a source-bound selector
   owns semantics. Delete a semantic row as soon as its final consumer migrates.

**Delete:** source-present positional table fallback, `rope_theta`-presence
mechanism inference, causal suffix authority, substring sliding logic,
`mixer_kinds`, semantic layer-token tables, and frequency/list convention paths.

**Witness matrix:** Llama RoPE, GPT/StarCoder learned absolute, BLOOM ALiBi,
partial RoPE, Llama-4 interleaved NoPE, T5 relative bias, causal and bidirectional
owners, Gemma alternating masks, hybrid recurrent/attention schedules, explicit
MoE lists, malformed/short lists, and unused config signals.

### 20.12 U9 — make every submodel recursive and owner-bound

**Primary edits:** `special_parts/modalities/{accessors,detect,vision,audio,
projector/fusion equivalents,builder,registry,schema}.py`, `evidence/{vision,
audio,projector,fusion}.py`, and `metadata_modalities.py`.

Perform exactly:

1. Resolve each component from the root construction graph into a
   `ComponentOwner`; a declared but unused config component is inventory only.
2. Run the same attention/FFN/norm/position readers used for the root against
   vision/audio/text/codec owners.
3. Prove patchification, tiling, pooling, pixel shuffle, feature selection,
   concatenation, projector internals and token emission from exact dataflow.
4. Prove placeholder replacement, prefix concat, grid interleave and
   cross-attention fusion from the root forward path.
5. Permit width/count/image-size/token-id values only after the owning operation
   is proven.
6. Replace `tower_submodel_spec` hardcoded MHA/dense/cache behavior with the
   recursive component result.

**Delete:** `is_unified_grid_stream` config predicate,
`has_cross_attention_adapter` topology inference, position/tiling/reduction/
projector config-presence branches, width-mismatch projector creation, fusion
fallbacks, and renderer-local encoder classification.

**Witness matrix:** Gemma multimodal, Qwen2-VL unified grid, Mllama
cross-attention, Qwen2-Audio, a prefix-concat model, placeholder replacement,
vision dense versus text gated siblings, declared-unused component, width
mismatch without projection, partial source and rival owner.

### 20.13 U10 — migrate diffusion root, streams and conditioning

**Primary edits:** `adapters/diffusor/{loader,parser,blocks,compound}.py`, U3
readers, and diffusion views only as consumers.

Perform exactly:

1. Resolve the exact denoiser class from the pipeline construction graph.
2. Derive transformer stack versus U-shaped stage graph from constructed
   submodules, not class markers/config signatures.
3. Derive dual/single stream, concat/join, parallel/sequential branches and
   self/cross attention from the block forward graph.
4. Derive AdaLN/modulation/gates only from positive code evidence. Reader unknown
   produces unresolved conditioning, never AdaLN.
5. Bind conditioning/projector enums and dimensions to the exact source branch.
6. Separate temporal geometry from proven temporal operations; a three-axis
   patch size alone cannot create a video mechanism.
7. Replace generic `config_facts.yaml` consumption with explicit registered
   facts or scoped non-architectural ignores.
8. Make loop/root blocks consume typed component/root facts.

**Delete:** `dit_class_markers` structural selection, `is_unet` config authority,
MHA/fused/AdaLN fallthroughs, conditioning structural templates, text-component
presence fallback, temporal-field video classification, and generic fact-to-chip
authoring.

**Witness matrix:** Flux double/single stream, SD3-like MMDiT, HunyuanVideo,
CogVideoX, Wan, Mochi, AuraFlow/HunyuanDiT, plain self-attention DiT, cross-
attention DiT, no-AdaLN counterexample, non-softmax mixer, image versus video,
and source missing.

### 20.14 U11 — replace the UNet interpreter

Perform exactly in `adapters/diffusor/unet.py` and
`renderers/html/block_views/unet.py`:

1. Resolve down/mid/up stage ownership and the exact class for every configured
   block token.
2. Parse constructor loops/list comprehensions to obtain block counts, widths,
   resnet/transformer depths and sampler placement.
3. Read each resolved ResNet, attention, transformer and sampler cell through U3
   and reuse U6/U7 facts.
4. Build one canonical stage DAG with exact skip sources/joins and conditioning
   edges.
5. Project that DAG to cards, HTML, JSON and params.

**Delete:** substring `Attn` semantics, `2`/`1`/`+1` defaults, head-dimension
reinterpretation, GroupNorm+SiLU+Conv template, 4× Transformer2D template,
all-but-final sampler placement and renderer-local UNet graphs.

**Witness matrix:** SD/SDXL-style UNet, Kandinsky variants, no-mid-block,
attention-free stage, custom activation, simple cross-attention without nested
Transformer2D, temporal UNet, custom down/up sampler and source missing.

### 20.15 U12 — replace the VAE/codec template

Perform exactly in diffusion parser/blocks and
`renderers/html/block_views/vae.py`:

1. Resolve encode, quantize, decode, mid and sampler owners.
2. Distinguish KL, VQ, deterministic/unknown latent decoder, temporal/causal 3-D
   and audio 1-D from actual operations.
3. Parse constructor stage expressions and exact resnet/sampler cells.
4. Bind channel/ratio/activation config values only to proven expressions.
5. Build one canonical component DAG and parameter graph.

**Delete:** VQ field-presence classification, neutral-KL wording, RGB/channel-3
default, power-of-two scale assumptions, layers-plus-one, all-but-last upsample,
hardcoded ResNet/nearest-upsample chains and inferred output domain.

**Witness matrix:** AutoencoderKL, VQ, temporal/causal video VAE, audio codec,
no-mid-attention, custom activation/sampler, unused VQ-like config field and
source missing.

### 20.16 U13 — derive the scheduler update graph

Perform exactly:

1. Resolve the exact scheduler class as a component address.
2. Index `step()` plus reachable helpers and state/history reads.
3. Derive predictor/corrector, multistep/history, stochastic/noise injection,
   flow/consistency target, coefficients and state updates from code.
4. Bind config values only to expressions used by that graph.
5. Project one scheduler region to the card, drill, JSON and conformance.

**Delete:** class-substring flow detection, identity-selected Euler-like prose,
and prediction-enum mechanism inference.

**Counterexample:** a class name containing `Flow` whose `step()` is not flow
matching is mandatory.

### 20.17 U14 — make JSON, parameters and conformance exact consumers

Perform exactly:

1. `expanded.stack` consumes root topology; remove `decoder_only` constant.
2. Remove expanded-local cache, projection, FFN expert, residual, block-order and
   root-I/O authorship. Serialize canonical facts/regions/receipts.
3. Generate layer grouping signatures from the registered structural schema;
   every registered per-layer fact automatically affects grouping.
4. Preserve proven false in JSON; distinguish false/unknown/not-applicable.
5. Build parameters from owner-bound module/parameter expressions. Unknown
   mechanisms produce partial totals or ranges with missing terms.
6. Deduplicate shared modules by exact owner/parameter identity.
7. Conformance compares exact owner program graph to exact projected region and
   records scoped abstraction receipts.
8. Remove CogVideoX/Flux identity exceptions after equivalent structural rules
   pass their witnesses.

**Tests:** `test_expanded_json.py`, params tests, `test_conformance.py`,
`test_projection_obligations.py`, `test_submodel_parity.py`, and JSON poison cases
for no-cache bidirectional MHA, encoder root, diffusion root, unknown topology and
sibling evidence.

### 20.18 U15 — delete semantic configuration authority and close release

Audit every YAML section, not merely every file:

| Resource | Final lawful content | Content to delete |
|---|---|---|
| transformer/diffusor `aliases.yaml` | spelling equivalence used by a source-bound config read | mechanism, topology or default meaning |
| `config_facts.yaml` | none; replace any surviving pure metadata with a typed display/value path | generic present-field-to-chip/silent consumption |
| `text_encoders.yaml` | friendly display title only, in a display-typed channel | pipeline, projection, fusion or encoder structure |
| `layer_types.yaml`, `layer_schedules.yaml`, `layer_topology.yaml` | temporary syntax normalization only; preferably empty after U8 | token-to-mechanism/mask/topology mappings |
| diffusion `conditioning.yaml` | display wording for an already-proven enum branch | modality/projector/topology templates |
| diffusion/transformer `typing.yaml` | schema/layout vocabulary and syntax only | class markers, topology fallbacks, scheduler semantics, norm/mechanism maps |
| conformance YAML | generic op tokens and scoped structural abstraction rules | family exceptions and role/name promotion without owner proof |
| `decoderness.yaml` | address/display hints at most | causal mask authority |
| `ignored_fields.yaml` | owner-scoped non-architectural ignores with reasons | global bare-key architectural ignores |

Then:

1. Delete dead loaders and stale manifest fingerprints.
2. Delete compatibility resolvers, raw-extras structural paths, manual renderer
   graphs, class-default structural hydration and legacy fact writers whose debt
   count reached zero.
3. Make identity/config semantic taint, structural-write census, projection
   receipts, config nets, metamorphic frontier and dependency firewall blocking.
4. Require every debt/allowlist count to be zero or smaller than the U0 baseline
   with an exact remaining owner and deletion unit.
5. Run the full witness matrix, full suite, Sable, corpus/conformance, visual
   galleries and unchanged-tree fingerprint.
6. Present every intentional diagram/card/JSON/parameter delta to Soumil. Only
   Soumil may bless or commit the final reviewed units.

**Final achieved output:** there is no configuration, identity, presentation,
serializer, parameter or conformance path capable of creating a mechanism. Code
proves mechanisms; config supplies only owner-bound checkpoint values; all
surfaces project the same facts; unknown stays unknown.

---

## 21. Required implementation receipt and live tracker

Append one row per unit/vertical slice. Never replace historical rows; corrections
append a superseding row.

| Unit | Status | Commit | Ledger IDs | Producers migrated | Consumers migrated | Old paths deleted | Positive/equivalent/counterexample witnesses | Intentional deltas | Tests and duration | Tree fingerprint before/after | Debt delta | Soumil decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U0 | PENDING | — | baseline | — | — | — | — | none | not run | not recorded | — | none |
| U1 | PENDING | — | T-01, D-02 | — | — | — | — | — | not run | not recorded | — | — |
| U2 | PENDING | — | C-08..C-13 | — | — | — | — | — | not run | not recorded | — | — |
| U3 | PENDING | — | X-08..X-11 | — | — | — | — | — | not run | not recorded | — | — |
| U4 | ACTIVE — U4-A/U4-B DONE; U4-C final gate | `4857026`, local U4-C candidate | C-01 to C-07/C-12 plus attention/FFN halves of T-05/T-08/T-10/D-08/U-07 | independent attention and FFN facts with canonical opaque/partial projection | IR, opgraph, labels/cards, metadata, expanded JSON, params, Sable/conformance, towers and nested submodels | norm/cell topology defaults, synthetic dominant layer/card reconstruction, UNet internal templates, final cross-surface closure | Llama/BLOOM/Qwen3/T5/DeepSeek-V3/GPT-OSS/Qwen2-VL/MusicGen/FLUX/PixArt/SDXL plus independent-detail poisons | U4-A 14 guarded re-blesses; U4-B 25 changed fixture signatures; U4-C 26 reviewed/reproduced galleries with exact corpus identity | U4-C affected 164p; fresh Sable corpus 1p/419.67s; final committed-tree bracket pending | U4-B `/private/tmp/model-unfolder-verification/598777dc06`; U4-C receipt pending | asserted attention/storage/scale debt retired; FFN config fallbacks removed or assigned exact U7 debt | Soumil approved U4-A/U4-B and U4-C honesty direction; close U4-C only after frozen committed-tree gate |
| U5 | PENDING | — | X-01, X-06/X-07, J-09..J-12 | — | — | — | — | — | not run | not recorded | — | — |
| U6 | PENDING | — | attention slice | — | — | — | — | — | not run | not recorded | — | — |
| U7 | PENDING | — | FFN/norm/cell slice | — | — | — | — | — | not run | not recorded | — | — |
| U8 | PENDING | — | positions/masks/schedules | — | — | — | — | — | not run | not recorded | — | — |
| U9 | PENDING | — | M-01..M-16 | — | — | — | — | — | not run | not recorded | — | — |
| U10 | PENDING | — | diffusion root/conditioning | — | — | — | — | — | not run | not recorded | — | — |
| U11 | PENDING | — | U-01..U-10 | — | — | — | — | — | not run | not recorded | — | — |
| U12 | PENDING | — | V-01..V-06 | — | — | — | — | — | not run | not recorded | — | — |
| U13 | PENDING | — | D-16 | — | — | — | — | — | not run | not recorded | — | — |
| U14 | PENDING | — | J-01..J-12, X-02..X-05 | — | — | — | — | — | not run | not recorded | — | — |
| U15 | PENDING | — | YAML/dead-authority deletion | — | — | — | — | — | not run | not recorded | — | — |
| U0 | BLOCKING GREEN | landed with this row (2026-07-13) | baseline + §5 probe pins P1–P13 | — (baseline unit) | — | none (no production behavior changed) | 25/25 corpus models frozen ×7 surfaces (`tests/preservation_baseline/`, hashes in committed `tests/preservation_manifest.json`, corpus input hash `5a93f1b0…`, 0 failures); 12 strict-xfail defect pins (each names its fixing unit) + 6 taint-shape controls + mutation-rejection gate test | none — TREE GATE UNCHANGED; no blessings touched | alone 19/19 green (717s) · grouped 649p+12xf (545s) · FULL 1100p+12xf (1811s) · sable-alone 33p (326s) · collection 1112 · gates total 3401s, started HEAD=d475d87 | `0067d0d8a9b3e59ad3358712c8a1d189f55693e1376e8c4c316a8a5f83841be8` == after (UNCHANGED) | +12 pinned defects (strict xfail); calibration: helper/bool/mapping/ternary/decorated-class taint shapes ALREADY caught (controls); true §5.6 holes = decorated identity-NAME branch + decorated unconditional structural return | pending review |

| U0 | SUPERSEDED (correction) | — | baseline | — | — | — | — | — | — | — | — | the prior BLOCKING GREEN over-claimed: baseline artifacts were local+gitignored with NO committed generator/verifier, html hashes were raw (mount-UUID nondeterministic), and the manifest was blind to U1's 15-witness structural drift (recovery plan §2.3/R-01). Repaired by REC-1 (deterministic canonical harness + 10-case poison matrix) and REC-7 (committed clean-checkout generator + BLOCKING zero-drift gate) | — |
| U0 | BLOCKING GREEN (recovered, REC-7) | 50d2408 + a14341b + REC-7 commit | baseline + R-01 | — | — | basename-wide .claude exclusion; raw-html hashing | 25/25 canonical witnesses enforced zero-drift; poison matrix; mount-normalization proven on real renders; corpus-input hashes pinned | none | §13.4 groups (alone ×8 → grouped → collect → FULL) — see REC-7 receipt log | recorded in REC-7 receipt | shadow debt retired (0/15 drift) | awaiting review → DONE |
| U1 | BLOCKING GREEN (recovered, REC-2..REC-6) | b2316f5 (Soumil) + 33ca12e + b7b859c + 4677c95 + d720a32 + 1f3316c | T-01, D-02, H3 defects, R-02..R-11 | BOTH first-hit resolvers DELETED; transformer tier-exact scopes w/ wrapper paths + 17 fact-linked consumes; diffusion denoiser+VAE consumed census; modality accessors honest | unread/excusal joins share ONE owner attribution over exact occurrences; net-1 ungated + audit_incomplete named; net-2 content-published; config_ambiguity BLOCKING | first-hit loops, _tiered-as-authority, width-comparison heuristic, U1 config_facts rows (8), absent-union, leaf-name truth, module-name owner guess, embed_dim/dim/inner_dim false synonyms | GPT-2 alias inversion fixed at event level; §9.6 Llama-conflict exact; R-04 same-owner nested counterexample; growth/static poisons | ZERO structural deltas — 25/25 witnesses at U0 parity; evidence-surface deltas documented (spelling-exact ledgers, owner-qualified events, tie tier upgrade) | per-unit receipts in REC-2..REC-6 commit messages | per-unit brackets VALID | +9 exact visible pending entries (deletion units U3/U9/U11/U12) | awaiting review → DONE (visual matrix §13.3 is Soumil's) |

Allowed status transitions are:

```text
PENDING -> ACTIVE -> SHADOW GREEN -> BLOCKING GREEN -> REVIEWED -> DONE
```

`BLOCKED` is reserved for a concrete missing capability or user decision. Test
failure, difficult implementation, or an incomplete migration is `ACTIVE`, not
`BLOCKED`. A unit cannot reach `DONE` unless its row names the deleted authority
path and contains all three witness classes.

---

## 22. Definition navigation, checkpoint corroboration, and visible truth audit

This section closes a subtle gap in the phrase “code proves mechanism; config
supplies value.” The implementation must not merely search modeling files for
field tokens. It must navigate from the selected root class through the actual
construction/forward graph, navigate the matching config class and its defaults,
and reconcile that interpretation with the checkpoint's physical tensor metadata
when available.

The system is reconstructing:

> the executable architecture implemented by exact source revision **S**, selected
> by checkpoint configuration **C**, and instantiated by checkpoint tensor set
> **W**.

It is not reconstructing an idealized paper architecture or silently correcting
Hugging Face code according to convention.

### 22.1 Can a class-definition prober recover `hidden_size` and similar values?

Partly, with an important boundary:

- The model/config class can prove that `config.hidden_size` controls a particular
  constructor argument, tensor width, reshape, or loop.
- The checkpoint config normally supplies the checkpoint's actual value.
- The config class definition can supply a default only when the checkpoint omits
  the field and the exact config class/revision is resolved.
- The constructor expression can derive a value such as
  `hidden_size // num_attention_heads` from evidenced premises.
- Tensor metadata can corroborate physical matrix/channel dimensions.

The modeling class generally cannot recover a checkpoint's actual
`hidden_size=4096` by itself because it receives that number at runtime. A class
probe that replaces the checkpoint value with the config class's default would
misrepresent checkpoints that override it.

### 22.2 Mandatory top-down definition navigator

Extend U3's `ProgramIndex` with a `DefinitionNavigator`. It begins at the real
entrypoint selected for the checkpoint, not at every class whose name looks
relevant.

Resolution order:

```text
checkpoint auto_map / architecture address
  -> exact root wrapper class
  -> matching config class and composition
  -> Python MRO and super() constructor chain
  -> constructed child modules / factories / ModuleList comprehensions
  -> selected child classes and their constructors
  -> root forward/call
  -> reachable helper/inherited call graph
  -> active child calls and tensor/dataflow edges
```

The navigator must understand or explicitly reject:

- import aliases and relative imports;
- inheritance, MRO and `super()` calls;
- class attributes, properties and `PretrainedConfig.attribute_map` aliases;
- config composition (`text_config`, `vision_config`, `denoiser`, `vae`, etc.);
- constructor helper/factory functions;
- `ModuleList`/`ModuleDict` construction, loops and comprehensions;
- conditional construction and conditional forward dispatch;
- wrapper delegation (`ForCausalLM.model`, pipeline components, encoder/decoder
  wrappers);
- functional operations that have no child module;
- attention/backend dispatch (`eager`, SDPA, Flash or custom implementation);
- unsupported dynamic imports, monkey-patching and generated code.

Starting at the top class is the ownership rule, but not a license to stop there.
Many HF top classes are wrappers whose forward delegates to an inner model; many
components are built in helpers; and multiple configured modules may exist while
only one is reachable. Ownership is confirmed by construction **and** use.

### 22.3 Exact config-class default policy

U1/U3 must resolve the exact config class alongside the model class.

Allowed class-default flow:

1. Source proves the owner reads a canonical config field.
2. Checkpoint config is checked through `ConfigResolution`.
3. If present, the checkpoint value wins.
4. If absent, navigate the exact config-class constructor/MRO/property expression.
5. If a stable default is statically evaluable, emit a `class_default` premise
   with source span and exact library/repository revision.
6. If the default is computed, derive it only from evidenced premises.
7. If revision, class, expression or prerequisites are unresolved, the value is
   unknown. Do not use a globally hydrated defaults dictionary.

Class defaults cannot prove that a mechanism exists. They only supply a value to
a mechanism/config read already bound by modeling code.

### 22.4 Optional tensor-manifest prober

Add a non-executing `CheckpointTensorIndex` as a corroboration channel. It should
read `safetensors` headers/indexes or equivalent state-dict metadata without
loading tensor payloads when available.

Record:

- exact tensor key, shape and dtype;
- shard/file and checkpoint revision/hash;
- module-owner candidate derived from the resolved construction tree;
- sharing/tie aliases when provable;
- unmatched expected and unexpected tensors.

Tensor metadata may strongly corroborate:

- embedding/vocabulary/hidden dimensions;
- split versus fused projection storage;
- intermediate/expert widths;
- convolution channels/kernel shapes;
- expert count/storage;
- tied/shared physical parameters.

Tensor shapes cannot by themselves prove forward semantics such as RoPE, masking,
activation order, routing renormalization, cache use, residual wiring or whether a
stored module is reachable. They are corroboration, not a replacement mechanism
reader.

This channel is optional because weights may be unavailable. Its absence lowers
corroboration; it must not cause a conventional fallback.

### 22.5 Runtime probing is opt-in and non-authoritative by default

An optional runtime probe may inspect an already-instantiated trusted model to
corroborate resolved modules, shapes and selected dispatch. It must never execute
downloaded remote code merely to make the diagram complete.

Requirements:

- explicit opt-in;
- sandbox/trust boundary;
- no training or mutation;
- record environment, package versions, device, dtype and selected backend;
- distinguish observed execution from unexecuted conditional alternatives;
- feed observations into typed facts/receipts, never directly into renderers.

Static source remains the normal path. Runtime evidence is useful for dynamic
dispatch that static analysis cannot safely resolve.

### 22.6 Final authority matrix

| Claim category | Primary authority | Permitted corroboration | Forbidden shortcut |
|---|---|---|---|
| mechanism and operation order | exact reachable modeling code | runtime trace; tensor storage where relevant | config field/token presence, class name, paper convention |
| root/component ownership | construction plus reachable use from exact root | tensor namespace, runtime module path | declared nested config alone |
| checkpoint numeric value | exact checkpoint config path | tensor shape, runtime module attribute | unrelated class default or common family value |
| omitted checkpoint value | exact config-class default expression at exact revision, after source-bound read | tensor shape | blanket defaults hydration |
| derived geometry | exact constructor/forward expression plus evidenced premises | tensor shape | library convention such as 4x FFN |
| storage layout | constructor/state declaration plus forward use | tensor keys/shapes | head counts or naming alone |
| runtime-selected backend | code dispatch plus known selector/environment | runtime trace | installed-package default without provenance |
| token/placeholder numeric id | checkpoint config | tokenizer metadata | token name alone |
| display title | typed display metadata | repository metadata | allowing title to select a mechanism |
| intended/paper architecture | annotation only | paper/docs | overriding executable evidence |

### 22.7 Conflict and Hugging Face defect policy

Never silently choose a preferred channel when they disagree.

Required outcomes:

| Situation | Required interpretation |
|---|---|
| checkpoint value differs from config-class default | checkpoint wins; record the override |
| code reads a missing field with no resolvable default | unknown value and typed failure |
| config dimension conflicts with checkpoint tensor shape | blocking `config_tensor_conflict`; do not fabricate a resolved dimension |
| source expects tensors absent from the checkpoint | blocking `source_checkpoint_mismatch` |
| checkpoint contains tensors not constructed/reachable in resolved source | declared `orphan_tensor`/revision mismatch; not active architecture |
| class is constructed but never reachable from forward | declared dormant inventory, not an active path |
| forward uses a functional operation with no stored tensor | code-proven operation; tensor absence is expected |
| reachable HF code contains an apparent bug | represent the executable behavior and attach an anomaly; do not “correct” it from a paper/family |
| bug/dead branch is unreachable for the checkpoint | omit from active architecture; retain in alternate/dead-code audit if useful |
| source revision cannot be matched to checkpoint | `source_revision_unresolved`; facts cannot claim complete source proof |
| static dispatch cannot resolve safely | conditional/unknown architecture, with alternatives and failure context |

Every source-derived fact must include source provenance sufficient to reproduce
the reading: repository/package, revision/version, file hash, class, callable and
spans.

### 22.8 How Soumil can identify mistakes after the transition

Make evidence inspection a product feature, not merely an internal test.

Add to `Diagram`:

```text
diagram.audit() -> ArchitectureAudit
diagram.explain(node_or_fact_path) -> ClaimExplanation
diagram.audit_report() -> serializable report
```

`ClaimExplanation` must show:

- what architectural claim was made;
- exact component owner;
- exact source class/callable/spans and revision;
- exact config path/alias/value and whether checkpoint/default supplied it;
- tensor/runtime corroboration when present;
- completeness and reader failures;
- premises/derivation chain;
- every HTML/card/JSON/params/conformance projection receipt;
- competing evidence and conflicts;
- abstraction receipts for raw operations intentionally omitted from the visual.

The UI should expose a “Why is this here?” evidence drawer from every structural
block/chip. A user must be able to trace a rendered box back to evidence without
reading parser code.

`ArchitectureAudit` must list at least:

- structural projections with no fact;
- facts with no projection;
- config fields read but neither consumed nor scoped-ignored;
- config values consumed without a source binding, except pure geometry;
- source-bound fields whose checkpoint/default value is unresolved;
- identity/display taint reaching structure;
- incomplete negative facts;
- sibling/rival-owner contamination;
- reachable architectural operations with no fact/abstraction receipt;
- config/source/tensor/runtime conflicts;
- source/checkpoint revision mismatches;
- raw structural `extras`, YAML or renderer fallbacks still used;
- legacy/asserted facts and conventional defaults reached;
- unknown facts and exactly why each remains unknown.

### 22.9 Blocking health counters

Expose these counts in every Sable/corpus report and completion receipt:

```text
unreceipted_structural_claims       == 0
identity_tainted_claims             == 0
unbound_structural_config_reads     == 0
incomplete_negative_projections     == 0
sibling_owner_evidence_leaks        == 0
silent_reader_failures              == 0
suppressed_evidence_conflicts        == 0
conventional_default_hits           == 0 after U15
raw_structural_renderer_reads        == 0 after U15
semantic_family_table_hits           == 0 after U15
```

Unknown counts are not required to be zero. They must be explained, owner-bound,
and non-fabricating. A falling unknown count is coverage progress; a zero unknown
count achieved through defaults is a regression.

### 22.10 Mandatory adversarial probes

Add these cases to U3/U5/U14/U15 acceptance:

1. Same source and checkpoint values under renamed model/class identity produces
   the same structure.
2. Same config signal exists but the resolved code never reads it; no mechanism
   is drawn.
3. Same config, different reachable implementation; the diagram follows code.
4. Same implementation, different bound config value; only the affected geometry
   or selected branch changes.
5. Checkpoint omits a field; exact config-class default resolves and carries its
   source span.
6. Config-class default changes between source revisions; unresolved/mismatched
   revision cannot silently pick either.
7. Tensor shape contradicts config; conflict is visible and blocking.
8. Root wrapper delegates through helper/inheritance; exact child owner is still
   resolved.
9. Module is constructed but unused; it remains dormant inventory.
10. Functional op is used without a module/tensor; it remains visible from code.
11. Dynamic dispatch is unsupported; architecture remains conditional/unknown.
12. A reachable source bug differs from the paper; executable behavior is shown
    with an anomaly rather than silently corrected.

These requirements extend U1, U3, U5, U14 and U15. None of those units can be
marked `DONE` without the corresponding definition-navigation, conflict and
visible-audit receipts.
