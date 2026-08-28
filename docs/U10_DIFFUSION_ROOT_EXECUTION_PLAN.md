# U10 — Diffusion Root, Stack, Stream, and Conditioning Execution Plan

Status: **DONE — source-derived diffusion root/stack/stream authority landed,
approved artifacts rebuilt, and committed-tree closure reproduced**

Authority: this document is the binding execution plan for U10. It refines
`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` §20.13 without changing the master
intent. If an older note conflicts with this document, stop and resolve the
conflict before editing production code.

---

## 1. The result U10 must deliver

U10 replaces the current diffusion denoiser decision funnel with one exact,
owner-bound evidence route:

```text
pipeline/config address metadata
        |
        v
exact root component + exact source occurrence
        |
        v
root topology proof
  transformer stack | U-shaped graph | unresolved
        |
        v
exact repeated-stack and block occurrences
        |
        +--> canonical U6 attention evidence
        +--> canonical U7 FFN/norm/cell evidence
        +--> canonical U8 position/mask evidence
        |
        v
block stream + conditioning execution graph
        |
        v
typed denoiser/root facts and projection DTO
        |
        +--> parser/IR
        +--> cards and drills
        +--> expanded JSON
        +--> parameter consumers
        +--> conformance/Sable
```

After U10, a config field can supply a number only after source proves what
operation consumes that number. Class names, field names, enum values, and
dimension presence may locate candidates or provide display text; they cannot
create a denoiser topology or operation.

U10 is complete when a new diffusion transformer can be rendered from its
construction and forward code without adding a model-family branch, and when
unresolved source produces an explicitly opaque denoiser rather than a
conventional DiT.

---

## 2. Why U10 is needed

The live parser currently combines six different jobs in
`model_unfolder/adapters/diffusor/parser.py`:

1. adapter routing;
2. config value resolution;
3. exact-source observation;
4. architecture interpretation;
5. IR/card construction;
6. explanatory prose.

That mixture creates the recurring failure mode: a weak input can enter through
one branch and emerge as a stronger drawn mechanism. Verified examples in the
current tree include:

- `_conditioning(...)` selecting dual stream, concat joint, KV join, cross
  attention, or plain self-attention from dimension-field presence;
- `matches(...)` using `dit_class_markers` and `is_unet(config)` for structural
  classification;
- `_dit_attention(...)` replacing missing KV-head evidence with query heads;
- `_dit_norm_kind(...)` selecting an implementation from config vocabulary;
- `_temporal_axis(...)` turning temporal-field presence or three patch axes into
  video computation when source proof is unavailable;
- `_resolve_conditioning(...)` turning config enums or text-component presence
  into modality/projector topology;
- `_secondary_stack_specs(...)` using whole-file unions and parameter-name lane
  vocabulary;
- `config_facts.yaml` mapping generic present fields directly to card chips;
- `blocks.py` authoring patch/timestep/conditioning/denoiser stories from a raw
  `geom` dictionary and conventional defaults;
- twelve quarantined whole-file diffusion readers and broad failure catches
  collapsing attribution failures into ordinary `None`/`False` values.

The problem is not that config values exist. Checkpoints must still supply
depths, widths, counts, patch sizes, and selected options. The problem is that
their spelling or presence currently decides the *meaning* of the architecture.

### 2.1 Verified starting inventory

At planning time the live tree contains:

- 12 quarantined U10 semantic readers (11 in `evidence/patterns.py`, one in
  `evidence/stacks.py`) feeding 14 parser wrapper/call paths;
- 14 U10 rows in the owner-qualified `StructuralDebt` register;
- 110 lines of generic diffusion config-fact authoring data;
- 63 lines of conditioning enum/template data;
- five U10-owned structural vocabularies in `typing.yaml`:
  `dit_class_markers`, `norm_type_kind`, `temporal_config_fields`,
  `stack_lane_params`, and `companion_denoiser_fields`;
- direct structural authors in `parser.py`, `blocks.py`, raw diffusion extras,
  renderer views, and parameter/conformance consumers.

These numbers are a starting receipt, not targets to preserve. U10 closure must
regenerate the inventories and explain every removed, retained, or reassigned
row.

---

## 3. Non-negotiable laws

### 3.1 Address is not mechanism

`_class_name`, pipeline slot names, `model_type`, import names, and framework
protocol fields may locate the diffusion component. They may not prove DiT,
UNet, attention kind, stream topology, fusion, AdaLN, temporal computation, or
projector structure.

`matches(cfg)` is allowed to route a pipeline/config to the diffusion adapter as
an address-level candidate. It is not allowed to decide the drawn denoiser
shape. Structural classification happens only after `ParseContext` supplies the
exact source bundle and owner graph.

### 3.2 Values parameterize proven operations

A checkpoint value is consumable only through an exact source binding:

```text
config occurrence -> constructor/forward expression -> typed fact -> consumer
```

The config may say `num_attention_heads=24`; source must prove which exact
attention construction reads it. The config may say `cross_attention_dim`; it
does not prove that a cross-attention sublayer exists.

### 3.3 Occurrences, never class unions

Every stack, block, attention site, FFN, norm, projector, and companion denoiser
is identified by its construction-site occurrence. Two instances of the same
class remain separate. A same-role or same-class union is forbidden.

### 3.4 Unknown never becomes conventional architecture

- unknown attention is not MHA;
- unknown single-stream fusion is not fused;
- unknown timestep conditioning is not AdaLN;
- unknown norm is not LayerNorm or RMSNorm;
- unknown position use is not RoPE or NoPE;
- unknown temporal computation is not image or video mechanism;
- unknown root topology is not a transformer;
- unknown companion equivalence is not “shares this architecture.”

### 3.5 One fact, every consumer

Parser, cards, drills, expanded JSON, parameters, Sable, and conformance must
consume the same typed fact or projection DTO. A renderer or parameter formula
may not independently inspect raw config or source and may not reconstruct a
mechanism from labels.

### 3.6 U10 must not become a new family classifier

No table or branch keyed by a model/repository/class family may select a drawn
fact. Mechanism-specific protocols are lawful only when defined by observable
code behavior and attacked by complete-renaming and same-name/different-code
controls.

### 3.7 Positive local proof is not whole-forward completeness

The U3 execution substrate intentionally publishes conservative local relations,
not a complete Python CFG. U10 may draw a relation that is positively proven.
It may not claim that unobserved calls/branches do not exist. If a requested
topology needs a new ProgramIndex record or completeness law, stop that sub-unit
and make the substrate change its own reviewed prerequisite; do not hide a new
parser inside U10.

---

## 4. Exact scope and explicit non-scope

### 4.1 U10 owns

- diffusion adapter address-versus-structure separation;
- exact denoiser root topology: transformer-stack, U-shaped, hybrid/ambiguous,
  or unresolved;
- exact repeated stack and block occurrences;
- primary versus secondary stack edges when source proves their role;
- block self/cross/joint attention sites and stream joins;
- dual/single stream, sequential/parallel, concat and KV-join relations;
- per-block timestep/AdaLN/modulation/gate evidence;
- denoiser attention, FFN, norm, and position cutover to the canonical U6/U7/U8
  readers at exact block occurrences;
- root patch/input/output and conditioning projection operations;
- temporal-operation evidence versus temporal-geometry declarations;
- companion denoiser presence and structural-equivalence status;
- removal of generic diffusion config-fact-to-chip authoring;
- typed diffusion-root/loop projection and U10 debt deletion.

### 4.2 U10 does not own

- the detailed UNet down/mid/up stage and cell interpreter: **U11**;
- the VAE/codec internal graph or audio/video output-domain proof: **U12**;
- scheduler `step()` semantics: **U13**;
- final cross-surface JSON/parameter/conformance cleanup outside the U10 facts:
  **U14**;
- deletion of all remaining semantic YAML and release closure: **U15**.

The U10 root classifier may hand a proven `u_shaped` root to the existing UNet
interpreter as an explicitly quarantined compatibility consumer. It must not
rebuild UNet internals. U11 deletes that interpreter.

Scheduler and VAE config readers must not be opportunistically refactored in an
U10 commit merely because they share `parser.py` or `blocks.py`.

---

## 5. Reuse before building anything new

U10 must start from the substrate already delivered:

- `ParseContext.program_index()` — one content-addressed program observation;
- `resolve_component_root(...)` — exact D0 component root with parse-failure and
  rival isolation;
- `OwnerGraph` and construction-site occurrences;
- `ContainerInventory` — neutral repeated-container address inventory;
- U3 execution-flow local relations — positive def/use, call, loop, guard, and
  unresolved evidence;
- `ReaderResult[T]` — typed resolved/absent/ambiguous/failed outcomes;
- canonical U6 attention evidence;
- canonical U7 FFN, norm, cell, and router evidence;
- canonical U8 position, mask, and schedule evidence;
- U9 recursive component inventory and operation routes;
- FactLedger, ProjectionRoute/Receipt, structural-write census, identity guard,
  and StructuralDebt.

No U10 reader may reopen source files, call `ast.parse`, use `ast.walk` over a
file union, run source regexes, or create another cache.

---

## 6. Frozen typed contracts

The exact Python names may change once implementation begins, but the separation
of responsibilities below may not.

### 6.1 `DiffusionRootTopology` (U10-A positive boundary)

Evidence module: `model_unfolder/evidence/diffusion_root.py`

Required fields:

- `ReaderResult` status: `incomplete | ambiguous | failed` in U10-A;
- component root occurrence and exact source symbol;
- strongest positive candidate: `repeated_stack | u_shaped`;
- exact construction/call evidence that supports the topology;
- every rival root/topology candidate with spans;
- typed unsupported/failure evidence;
- provenance and completeness limited to the classified boundary.

Rules:

- a repeated-stack candidate requires an exact root-owned container loop and an
  exact call through its loop-bound element; it is not yet a Transformer claim;
- a U-shape candidate requires two such stages plus an exact positive bypass
  route from an earlier invocation into a later one, not config block lists or
  a class/field substring;
- U10-A returns positive candidates as `incomplete`: U3 supplies conservative
  local relations but no whole-callable CFG coverage certificate, so observing
  one route cannot prove no additional route exists;
- a proved U-route plus an independent repeated stack is `ambiguous` in U10-A,
  never silently collapsed to either candidate; later stream/companion units
  may type an intentional hybrid only with exact ownership;
- U10-C may strengthen `repeated_stack` to a Transformer stack only when the
  exact element occurrence carries canonical attention/FFN evidence;
- source missing/unparseable is `failed`, not a config fallback;
- the result consumes a resolved component root and cannot bypass D0.

### 6.2 `DenoiserStackInventory`

Evidence module: `model_unfolder/evidence/diffusion_stack.py`

Each stack occurrence carries:

- exact owner occurrence;
- exact container address and authoritative construction records;
- exact block-construction alternatives and spans;
- count expression and, only when proven, its exact config occurrence;
- source order, never execution order by inference;
- positive execution relation when one exists;
- typed unresolved construction/execution relations;
- an optional source-proven role edge (`latent`, `text/context`, `joint`, or
  unknown), never a raw parameter-name label.

Symbolic repetition stays one template plus a count. U10 may not fabricate N
owner occurrences from a config number.

### 6.3 `DenoiserBlockEvidence`

Evidence module: `model_unfolder/evidence/diffusion_block.py`

This is an owner-bound composition, not a new mechanism classifier. It cites:

- exact U6 attention evidence for every attention site;
- exact U7 FFN/norm/cell evidence for every FFN/norm/cell site;
- exact U8 position/mask evidence for every application site;
- exact source/config operands consumed by those facts;
- unresolved sites and rival owners.

Self-attention and cross-attention get separate facts. One must never inherit
QK norm, heads, bias, position, processor, or projection geometry from the
other.

### 6.4 `DenoiserStreamGraph`

Evidence module: `model_unfolder/evidence/diffusion_stream.py`

U10-D is a **block-local positive boundary**, not a global modality classifier.
It starts from each exact U10-C block occurrence and retains:

- exact block formals and exact stack-call actuals;
- exact guarded return routes;
- exact U10-C attention and FFN call sites;
- exact imported concat calls;
- exact canonical norm invocations;
- U7 residual topology only when that exact block already proves it;
- unresolved attention/FFN/conditioning rows.

Its role vocabulary is deliberately local:

- `state`: an exact formal positively carried to an exact block return;
- `context`: an exact non-returned formal supplying the proven K/V side of an
  attention lane;
- `auxiliary`: an exact non-returned formal entering an explicit joined input;
- `conditioning`: a non-stream formal reaching an exact norm/gate application.

`single_state`, `contextual_single_state`, `dual_state`, and `joined_inputs`
describe only those local relations. They do **not** mean latent, image, text,
timestep, or guidance. Those global names require the root bookend proof in
U10-E. A numeric dependency is not a state identity, an explicit join is not
automatically a two-state join, and an attention/FFN call is classified only
when its output positively reaches an exact return.

No unproven relation is labelled “unordered,” “not present,” “sequential,”
`kv_join`, or a modality. The open U3 substrate keeps every aggregate
`incomplete` even when individual local relations are positively proven.

### 6.5 `DenoiserConditioningEvidence`

Evidence module: `model_unfolder/evidence/diffusion_conditioning.py`

U10-D carries exact block-local applications only:

- a non-stream formal entering an exact norm before a proven attention lane
  (`norm_modulation`);
- a non-stream-derived value multiplying an exact attention/FFN result that
  positively reaches a return (`bare_gate`);
- a non-stream-derived gate entering an exact canonical norm with the exact
  attention/FFN result, where the norm reaches a return (`gate_in_norm`).

Timestep/guidance naming, encoded-token routes, pooled conditioning,
pre-stack projectors/fusion, and bound projector dimensions belong to U10-E's
root bookend graph. This separation prevents a block parameter spelling from
authoring a global modality role.

The existence of `joint_attention_dim`, `cross_attention_dim`,
`encoder_hid_dim_type`, `addition_embed_type`, or a text-encoder component is
never enough to set one of these facts.

### 6.6 `DenoiserRootOperations`

Evidence module: `model_unfolder/evidence/diffusion_bookends.py`

Carries exact root operations:

- latent/token input adaptation;
- patch/embed operation and dimensionality;
- timestep/guidance embedding;
- pre-stack conditioning projection/fusion;
- final norm/projection;
- unpatchify/reshape/output operation;
- positive temporal operations.

Geometry values may annotate an already-proven operation. A patch tuple, frame
count, or temporal compression value alone cannot create a temporal operation.

### 6.7 `CompanionDenoiserEvidence`

Evidence module: `model_unfolder/evidence/diffusion_companion.py`

Pipeline keys may locate companion candidates. Each companion is independently
resolved. The result states:

- present and structurally equivalent;
- present and structurally different;
- present but equivalence unresolved;
- absent;
- failed/ambiguous.

“Shares this architecture” is legal only for proven equivalence over the U10
root/stack/block facts. Matching keys, class names, or dimensions are not an
equivalence proof.

### 6.8 Projection DTO

Adapter module: `model_unfolder/adapters/diffusor/schema.py`

The parser projects the evidence above into one typed `DenoiserSpec` (name may
change) whose unknowns are explicit. `blocks.py`, views, expanded JSON, params,
and conformance consume that DTO/facts. They do not read the old raw `geom`
dictionary for structural decisions.

Presentation-only values such as theme, canvas label, and friendly component
title must live in a typed presentation payload, not masquerade as evidence.

---

## 7. Execution units

No unit may be combined with the next one merely to make a failing tree green.
Each unit deletes only the old path it replaces and receives its own receipt.

### U10-0 — close U9 and freeze the diffusion baseline

Prerequisite, not a U10 production change.

1. Obtain Soumil's explicit approval for the U9 structural/evidence artifact
   delta.
2. Rebuild/re-bless only the approved U9 surfaces.
3. Run U9's full committed-tree bracket.
4. Commit/push U9 and require a clean tree except Soumil's known unrelated docs.
5. Record U10 starting HEAD, source fingerprint, collection count, 29-witness
   manifest, galleries, current U10 debt rows, legacy-reader inventory, identity
   resources, and broad-exception counts.
6. Generate the U10 qualification matrix before implementation.

Stop if U9 is not reproducible from its committed tree. U10 must not be used to
hide an unfinished U9 delta.

### U10-A — exact root topology boundary

Primary files:

- new `evidence/diffusion_root.py`;
- new `tests/test_diffusion_root.py`;
- `adapters/diffusor/parser.py` only for shadow publication;
- `docs/U10_DIFFUSION_ROOT_EXECUTION_PLAN.md` tracker.

Actions:

1. Consume the resolved D0 root and ProgramIndex/OwnerGraph.
2. Classify transformer-stack versus U-shaped evidence from constructed and
   invoked submodules.
3. Publish typed rivals/failures; do not change rendering yet.
4. Run old classification and new evidence side by side over every witness.
5. Explain every disagreement at the source level.

Required poisons:

- a class named `Transformer` that constructs a U-shape;
- a class named `UNet` that constructs a repeated transformer stack;
- config lists that claim UNet stages while source constructs none;
- both stack and U-shape candidates;
- same class constructed twice;
- helper/factory construction;
- broken sibling source;
- imported alias/rival;
- complete class/field/model rename;
- source missing;
- exact same config with different source topology.

Acceptance: no identity/config field can alter the typed root topology when the
source graph is held fixed.

### U10-B — repeated stack and block occurrence inventory

Primary files:

- new `evidence/diffusion_stack.py`;
- `evidence/container_inventory.py` only if an observed boundary defect is
  proven; otherwise unchanged;
- new `tests/test_diffusion_stack.py`.

Actions:

1. Enumerate every root-owned repeated container from B2.
2. Bind exact construction alternatives and exact count expressions.
3. Bind positive invocation/dataflow relations from the root forward.
4. Preserve main, secondary, refiner, and rival candidates as separate
   occurrences.
5. Derive lane/role only from dataflow; delete no legacy reader until the new
   inventory covers its exact responsibility.

Required controls include sliced containers, indexed loops, two stacks sharing
one block class, one stack with two block alternatives, guarded constructors,
symbolic counts, unknown count path, unrelated parameter-name decoys, and
source-order-versus-execution-order inversion.

Acceptance: the new inventory can express every current diffusion stack without
same-class union or fabricated repeated occurrences.

### U10-C — exact block facts through U6/U7/U8

Primary files:

- new `evidence/diffusion_block.py`;
- canonical U6/U7/U8 evidence modules only for genuinely altitude-neutral bugs;
- `adapters/diffusor/parser.py` shadow consumers;
- new `tests/test_diffusion_block.py`.

Actions:

1. Run canonical attention, FFN, norm/cell, and position readers on each exact
   block occurrence.
2. Keep self/cross/joint attention sites separate.
3. Bind config operands only through exact constructor/application expressions.
4. Publish a block fact bundle with no diagram changes.
5. Prove parity where old output was source-supported and list honesty deltas
   where old output was config/default/union-authored.

This unit owns replacement of the legacy reader responsibilities but not their
deletion yet. It must expose all old/new comparisons first.

Required matrix:

- softmax and non-softmax attention;
- self plus cross attention with different heads/QK norms;
- dense, gated, Conv-GLU, and unresolved FFN;
- AdaLN norm class versus plain norm sibling;
- RoPE applied, learned positions, no observed position, config-declared but
  unused RoPE;
- same block class at two occurrences with different constructor guards.

Acceptance: no whole-file vote or sibling fact can populate another occurrence.

### U10-D — stream and conditioning graph

Primary files:

- new `evidence/diffusion_stream.py`;
- new `evidence/diffusion_conditioning.py`;
- new `tests/test_diffusion_stream.py`;
- new `tests/test_diffusion_conditioning.py`.

Actions:

1. Build exact positive block-local relations among returned state slots,
   K/V-side context, explicit joined inputs, canonical attention/FFN/norm calls,
   gates, and returns.
2. Interpret local single/dual/contextual/joined-input relations only from
   sufficient exact graph proofs. Defer global latent/text/timestep/guidance,
   KV-join, projector, and bookend names to U10-E.
3. Publish unresolved relations when U3 cannot prove order or coverage.
4. Distinguish bare gate multiplication from gate-in-norm execution.
5. Keep config enum/dimension values unconsumed until their exact source branch
   is bound.

Required controls:

- Flux dual/single exact block occurrences without family transfer;
- SD3-style external/opaque MMDiT remaining typed unknown;
- HunyuanVideo heterogeneous stacks without sibling inheritance;
- CogVideoX dual attention kept distinct from its separate join/split FFN path;
- Wan/PixArt/Sana external or ambiguous lanes remaining typed unknown while LTX
  proves only its exact local self/contextual lanes;
- AuraFlow's exact per-call local relations without a global sequential claim;
- PRX's outer contextual single-state lane without inventing an inner KV join;
- plain self-attention DiT;
- no-AdaLN block;
- two candidate context routes;
- concat executed only on one branch;
- dimension-field decoy with no corresponding operation;
- text encoder present but unused;
- source partial/unsupported.

Acceptance: removing or renaming config topology fields while preserving source
does not change the mechanism; changing source while keeping config fixed does.

### U10-E — root bookends, temporal operations, and companions

Primary files:

- new `evidence/diffusion_bookends.py`;
- new `evidence/diffusion_companion.py`;
- new tests for each;
- `adapters/diffusor/loader.py` only if the exact pipeline component address is
  unavailable from the existing source bundle.

Actions:

1. Resolve root input/patch/embed/timestep/guidance/output operations.
2. Replace `_temporal_axis` fallback with separate facts:
   `declared_temporal_geometry` and `proven_temporal_operations`.
3. A temporal-mechanism label or 3-D operation is emitted only from the latter.
   A global output-domain label (`Frames`/`Image`/`Waveform`) additionally
   requires U12 codec/output evidence and is not decided by U10 alone.
4. Resolve companion denoisers independently and compare typed structures.
5. Keep source-missing companions unresolved and source-missing temporal config
   geometry-only.

Required controls include 2-D patch tuple of length three, video source with no
temporal config field, temporal config field unused by forward, reshape-only
geometry versus temporal convolution/attention, equivalent companions,
different companions with identical dimensions, and missing companion source.

Acceptance: no temporal field or companion key asserts a mechanism/equivalence.

### U10-F — typed projection and config-author dismantling

Primary files:

- new `adapters/diffusor/schema.py`;
- `adapters/diffusor/parser.py`;
- `adapters/diffusor/blocks.py`;
- `adapters/diffusor/compound.py`;
- diffusion renderer views as passive consumers;
- `expanded/`, params, Sable, conformance, registry, receipts, structural debt;
- `everchanging/diffusor/config_facts.yaml` and loaders;
- structural portions of `typing.yaml` and `conditioning.yaml`.

Actions:

1. Build the typed projection DTO solely from U10 evidence.
2. Cut the parser from hand-authored variants to that DTO.
3. Make every structural consumer use the same facts/DTO.
4. Replace every `config_facts.yaml` row with exactly one disposition:
   code-bound registered fact, presentation-only value on a proven owner, or
   owner/path/reason-scoped ignore.
5. Delete the generic present-field-to-chip author and loader.
6. Remove structural authority from:
   `dit_class_markers`, `norm_type_kind`, `stack_lane_params`,
   `companion_denoiser_fields`, `temporal_config_fields`, and conditioning enum
   templates. Keep only separately audited syntax/display entries.
7. Add projection receipts for every migrated structural surface.

#### Binding cut order inside U10-F

U10-F is one production migration, but it must land through four independently
reviewable cuts.  A later cut may not be started by silently weakening an
earlier cut's laws.

**U10-F1 — closed source projection (shadow-only).**  Add the typed projection
schema and assemble it exclusively from the canonical U10-A/B/C/D/E values.
The schema is not a second mechanism classifier: every normalized property is
computed from, and construction-time checked against, the exact evidence object
it carries.  It keeps exact stack/block/lane occurrences separate, preserves
unresolved stacks and branches, and keeps tensor rank separate from temporal
operations.  It consumes no raw config and has no parser, IR, renderer, params,
receipt, debt, or artifact authority.

**U10-F2 — exact operand binding (still shadow-only).**  Re-run the canonical
readers with the root `PreparedDocument` and the U1 exact selector.  A config
value may enter the projection only when source evidence names its exact path
and owning occurrence.  The same shadow selection must create an exact
owner/path/reader **binding** event, never a consumption: bind proves which
operand source code reads, while consume is reserved for F3 when a production
fact actually uses it.  Equal-head versus grouped-KV source protocols may
become a diagram kind only after their exact checkpoint operands are joined;
field presence, aliases, dimensions, and conventional defaults remain
powerless.

**U10-F3 — atomic production cutover.**  Move parser, IR, renderer, expanded
JSON, parameters, Sable, and conformance to the same typed projection in one
reviewed cut.  Unknown remains opaque.  This cut is expected to create honesty
deltas where legacy config/family fallbacks drew unsupported detail; every
delta is inspected before blessing.  No passive consumer may reopen source,
read raw config, or infer a mechanism from a projection label.  Each projected
config operand transitions from F2's exact binding to one owner-qualified
consumption here; F3 may not consume a path absent from the typed binding table.

**U10-F4 — authority deletion and closure.**  Delete every legacy/YAML author
made redundant by F3, disposition each remaining `config_facts.yaml` row, add
owner-qualified registry routes and real-consumer receipts, shrink structural
debt in the same commit, and make renderer/params raw-config or source reads
blocking.  Compatibility code cannot remain as a fallback behind the new DTO.

F1/F2 are allowed to be pixel-identical shadow foundations.  F3/F4 are not
complete merely because tests pass: the old authority must actually be removed,
all changed artifacts must be inspected, and preservation witnesses outside the
intended honesty delta must remain byte-identical.

The cutover must not introduce renderer conditionals keyed by a model/class,
fact label, or config field.

Acceptance: a renderer/params raw-config or source read is blocking; every drawn
U10 leaf has an owner-qualified fact or explicit debt assigned beyond U10.

### U10-G — delete legacy paths and quarantine the U11/U12/U13 handoffs

Delete in the same reviewed cutover:

- `secondary_stacks_from_files` U10 use and its raw parameter-name lane map;
- `diffusion_ffn_activation_from_files`;
- `diffusion_axes_dims_rope_from_files`;
- `diffusion_rope_from_files`;
- `diffusion_attn_kind_from_files`;
- `diffusion_ffn_kind_from_files`;
- `diffusion_qk_norm_from_files`;
- `diffusion_cross_qk_norm_from_files`;
- `diffusion_single_stream_fusion_from_files`;
- `diffusion_gate_via_norm_from_files`;
- `denoiser_block_timestep_conditioning_from_files`;
- `attention_score_scaling_from_files` diffusion call path if the canonical U6
  occurrence reader fully replaces it;
- `_conditioning`, `_dit_norm_kind`, `_resolve_conditioning`, `_temporal_axis`,
  `_secondary_stack_specs`, `_config_fact_chips`, and `_code_*` wrappers whose
  responsibilities have moved;
- MHA, KV-head, fused, AdaLN, transformer-style, patch, and position fallthroughs
  replaced by typed unknowns.

Do not delete the old UNet, VAE, or scheduler implementation here. Instead:

- the exact root topology gates the legacy UNet handoff and labels it U11 debt;
- typed component boundaries feed the current VAE and scheduler cards while
  their internals remain U12/U13 debt;
- no U10 fact is allowed to certify those legacy internal graphs.

Acceptance: generated reader quarantine and legacy parse inventory contain no
U10-owned semantic reader; structural debt contains no U10 row whose deletion
condition is already satisfied.

### U10-H — qualification, artifacts, and closure

1. Run every U10 test alone.
2. Run identity/config authority, structural-write, receipt, reverse-fabrication,
   config-access, reader-quarantine, and broad-exception gates.
3. Run the exact qualification matrix and source counterfactuals.
4. Run all diffusion tests and all U11/U12/U13 compatibility tests.
5. Run preservation and generate exact structural/gallery deltas.
6. Inspect at minimum the diffusion pipeline, denoiser map, every block variant,
   attention drill, FFN drill, position drill, conditioning rails, expanded JSON,
   and parameter surface for every changed witness.
7. Soumil approves all intentional artifact deltas before re-bless.
8. Rebuild only approved artifacts.
9. Run the full committed-tree parallel coordinator and isolated archive check on
   one unchanged fingerprint.
10. Update the master tracker, reader inventory, debt report, current-state docs,
    and this tracker with actual commits and receipts.

U10 is DONE only after the pushed committed tree reproduces the receipt.

---

## 8. Qualification matrix

### 8.1 Required real witnesses

| Mechanism | Required examples | What must be checked |
|---|---|---|
| dual + single stream | FLUX family witness | exact separate stack occurrences; mid-model join; no entry-join lie |
| dual-stream MMDiT | SD3.5 | separate latent/context projections and FFNs; joint attention |
| heterogeneous video denoiser | HunyuanVideo | primary/refiner ownership; no host-width borrowing; temporal proof |
| concatenated joint sequence | CogVideoX, Mochi | one upstream join; shared attention; correct gate dialect |
| cross-attention DiT | PixArt, Sana, Wan, LTX | distinct self/cross sites and geometry; no inherited QK norm |
| sequential joined stream | AuraFlow | no fused/parallel invention |
| nonstandard joint/KV route | PRX or available equivalent | exact K/V concat relation |
| plain/weakly conditioned DiT | available plain DiT/AuraFlow counterexample | no fabricated text rail or AdaLN |
| non-softmax mixer | available U6 diffusion control | opaque/correct mechanism, never MHA fallback |
| U-shaped root handoff | SDXL | exact positive `u_shaped` route; unchanged U11 compatibility handoff |
| source missing/partial | synthetic + corpus fixture | opaque root/block, config values remain declarations only |
| companion denoiser | Wan/Qwen-image available fixture | candidate presence separate from equivalence proof |

If a named witness is unavailable locally, the exact mechanism must be covered
by a synthetic HF-style source fixture and the plan must record the absent real
witness. Do not substitute a family name as proof.

### 8.2 Mandatory metamorphic controls

Every migrated fact family must demonstrate:

1. complete class, field, local-variable, and model-type rename;
2. same names/config, different source mechanism;
3. same source mechanism, alternative config spelling;
4. config field present but unused;
5. source uses a value through a different exact alias;
6. conflicting aliases;
7. same class at two construction occurrences;
8. two block variants in one model;
9. sibling component with the same leaf config key;
10. missing source;
11. partial/unparseable source;
12. rival import/construction;
13. unsupported expression/control-flow form;
14. clean-checkout import and parse.

---

## 9. Files and symbols expected to change

This is the surgical map, not permission to touch everything at once.

### New evidence/contract files

- `model_unfolder/evidence/diffusion_root.py`
- `model_unfolder/evidence/diffusion_stack.py`
- `model_unfolder/evidence/diffusion_block.py`
- `model_unfolder/evidence/diffusion_stream.py`
- `model_unfolder/evidence/diffusion_conditioning.py`
- `model_unfolder/evidence/diffusion_bookends.py`
- `model_unfolder/evidence/diffusion_companion.py`
- `model_unfolder/adapters/diffusor/schema.py`

These may be consolidated only when the resulting module still has one clear
evidence responsibility and does not become another `patterns.py`.

### Existing production files

- `model_unfolder/adapters/diffusor/parser.py`
  - dispatch/structure separation;
  - evidence orchestration;
  - old wrapper and fallback deletion;
  - typed DTO projection.
- `model_unfolder/adapters/diffusor/blocks.py`
  - passive typed projection only;
  - remove raw-geom structural decisions and conventional prose.
- `model_unfolder/adapters/diffusor/compound.py`
  - consume typed root/stage facts only.
- `model_unfolder/adapters/diffusor/loader.py`
  - only exact component-address preparation, if current SourceBundle lacks it.
- `model_unfolder/adapters/diffusor/unet.py`
  - no internal rewrite; only typed U10 handoff if needed.
- `model_unfolder/evidence/patterns.py`, `stacks.py`
  - delete U10 legacy readers after cutover.
- `model_unfolder/evidence/registry.py`, `receipts.py`,
  `structural_debt.py`, `structural_writes.py`,
  `legacy_reader_quarantine.py`, `identity_guard.py`
  - registry, receipts, shrink gates, and deletion pins.
- `model_unfolder/everchanging/diffusor/config_facts.yaml`
  - delete generic structural authoring.
- `model_unfolder/everchanging/diffusor/typing.yaml`
  - delete U10 structural identity/name maps; retain separately-owned U11/U12/U13
    syntax/display data until their units.
- `model_unfolder/everchanging/diffusor/conditioning.yaml`
  - remove topology/projector authority; optional display text must be visibly
    display-only.
- renderer/expanded/params/conformance files
  - consumers only; no architecture inference.

### Test files

Add one focused test file per evidence contract plus:

- a U10 integration matrix;
- renderer/expanded/params equality tests over the same facts;
- identity/config counterfactuals;
- preservation/gallery tests;
- generated inventory/debt/quarantine checks.

---

## 10. Exact old behavior to eradicate

The following are not acceptable temporary fallbacks after their replacement
unit lands:

- `dit_class_markers` or `is_unet(config)` selecting structure;
- conditioning dimension presence selecting the block variant;
- `None` attention selecting MHA;
- missing `num_kv_heads` silently selecting query-head count as a proven fact;
- unknown fusion selecting fused single-stream rendering;
- unknown conditioning selecting AdaLN/gates;
- config `norm_type`/epsilon spelling selecting norm implementation;
- config RoPE/QK-norm/bias presence creating the operation;
- temporal config fields/patch tuple creating video computation;
- text encoder presence creating text-conditioning topology;
- companion field presence claiming architectural equivalence;
- parameter-name vocabulary selecting secondary-stack lane;
- file-wide class/role unions selecting block facts;
- broad `except Exception` returning an ordinary semantic value;
- renderer prose or parameter formulas rebuilding architecture from raw `geom`.

Unknown/failed evidence may leave an opaque block with declared numeric metadata.
That is a valid U10 result, not a regression to patch with a convention.

---

## 11. What remains config-based after U10

Config is not eradicated as checkpoint data. It remains lawful for:

- exact numeric/string/list values consumed by source-proven expressions:
  depths, widths, head counts, patch/channel sizes, limits, selected activation
  operands, and similar parameters;
- pipeline/component addresses and exact config paths;
- sample canvas/display metadata that is explicitly presentation-only;
- user/checkpoint-selected modes after code proves what branch the selector
  controls;
- syntax aliases that locate the same concept without assigning semantics;
- display titles/labels that cannot affect structure;
- U11/U12/U13 debt not owned by U10.

It does not remain lawful for:

- family/class/config tables choosing topology;
- field presence proving an operation;
- enum-to-architecture templates without a source branch binding;
- defaults filling an unknown mechanism;
- renderer or parameter consumers reading raw config to infer structure.

Aliases may remain as syntax vocabulary. `config_facts.yaml` does not remain as a
generic architecture author.

---

## 12. Stop conditions

Stop the active sub-unit and report exact evidence when:

1. the required conclusion needs a ProgramIndex record or CFG completeness law
   that does not exist;
2. two exact owner occurrences remain rivals;
3. a config value cannot be bound to one source expression;
4. a renderer/params consumer asks for a structural fact the evidence layer did
   not produce;
5. a model/class/field spelling is needed to select a mechanism;
6. a legacy path must remain after the new path lands merely to preserve output;
7. a preservation delta cannot be explained as removal/correction of one exact
   unsupported claim;
8. a test would need blessing before Soumil reviews the artifact;
9. the tree changes during verification;
10. U10 begins modifying U11/U12/U13 internals.

A stop is not permission to add an allowlist or lower a gate. Record the
counterexample, the missing capability, the smallest neutral prerequisite, and
the honest output until that prerequisite exists.

---

## 13. Verification contract

Every implementation commit runs, in parallel where independent:

1. static/diff/pyflakes and forbidden-symbol gates;
2. its focused semantic and poison tests;
3. affected U2 authority/receipt/debt/identity gates;
4. diffusion affected tests;
5. preservation and targeted example unfolds;
6. full suite at the phase boundary, not redundantly after every mechanical
   follow-up;
7. fingerprint comparison;
8. isolated committed-worktree import/parse/focused test.

Fast per-commit lanes never replace the full phase-boundary bracket. A receipt
must name commit, command, count, duration, fingerprint, artifact delta, and
isolated-tree result.

Examples are part of verification, not a final cosmetic pass. At every semantic
cutover, unfold at least:

- one positive mechanism witness;
- one equivalent-control witness;
- one counterexample that must stay opaque/absent;
- one heterogeneous model containing two variants.

---

## 14. U10 definition of done

All must be true:

- exact root topology comes from source occurrence evidence;
- every repeated stack and block is occurrence-qualified;
- stream/conditioning terms retain their graph proof;
- U6/U7/U8 facts are reused at diffusion block altitude;
- config values are consumed only through source-bound expressions;
- all U10-owned whole-file readers and broad wrappers are deleted;
- generic diffusion config-fact chip authoring is deleted;
- structural identity/YAML selectors owned by U10 are gone;
- unknown stays opaque on every consumer surface;
- U10 StructuralDebt is zero or every residual is explicitly reassigned to a
  later unit because its consumer truly belongs there;
- the U10 qualification matrix and all counterfactuals pass;
- U11 UNet, U12 VAE, and U13 scheduler compatibility remain intact and visibly
  quarantined;
- every artifact delta is explained and approved;
- the committed, pushed tree reproduces all gates on an unchanged fingerprint.

Final achieved output:

> The diffusion adapter is only a router and projector. Exact denoiser source
> occurrences prove root shape, repeated stacks, block mechanisms, streams,
> conditioning, and root operations; checkpoint config supplies only bound
> values. No family table, field-presence heuristic, whole-file vote, renderer
> default, or source failure can manufacture a diffusion architecture.

---

## 15. Live tracker

| Unit | Status | Exit artifact |
|---|---|---|
| U10-0 | DONE | U9 commit `705f497`; tree `460aee5a374eb04034aea82624fd416cc7489710`; 3,327 collected; 29 witnesses; 14 U10 debt rows; 12 legacy readers; diffusion parser broad-exception baseline 17; exhaustive receipt `/private/tmp/model-unfolder-verification/256d238edd` fingerprint-identical |
| U10-A | DONE | commit `92200e1`; `evidence/diffusion_root.py`: exact repeated-container execution and exact bypass-route U-shape proof; parser shadow publication only; 23 synthetic poisons + 15 real witnesses; no IR/renderer consumer; committed-tree receipt `/private/tmp/model-unfolder-verification/0f1c9e9080` fingerprint-identical |
| U10-B | DONE | commit `98f1e96`; `evidence/diffusion_stack.py`: occurrence-exact container/block/call inventory over D0/B2/owner-graph rails; 25 synthetic poisons + 15 real witnesses; legacy semantic consumer intentionally retained until U10-C/F covers it; committed-tree receipt `/private/tmp/model-unfolder-verification/93a35b676f` fingerprint-identical |
| U10-C | DONE | commit `ee836b0`; occurrence-exact U6/U7/U8 composition in shadow mode; 33 synthetic controls + 15 real witnesses; 3,453 collected; 307 focused + 44 authority + 52 preservation + exhaustive 3,341 passed / 14 skipped / 2 xfailed; zero drift; interrupted coordinator transparently resumed in its intact committed worktree; receipt `/private/tmp/model-unfolder-verification/bb2795cf7c/continuation_receipt.md` |
| U10-D | DONE | commit `14fee0c`; exact block-local stream/conditioning graph; 39 synthetic/real controls in the two new test files; 3,492 collected; 324 focused + 44 authority + 52 preservation + exhaustive 3,380 passed / 14 skipped / 2 xfailed; zero drift; receipt `/private/tmp/model-unfolder-verification/8eddcda6a0` fingerprint-identical |
| U10-E | DONE | commit `a379cf0`; source-only bookends + geometry/mechanism separation + independently resolved companions; 3,535 collected; 350 focused + 44 authority + 52 preservation + exhaustive 3,423 passed / 14 skipped / 2 xfailed; zero drift; receipt `/private/tmp/model-unfolder-verification/e3b70ca7b0` fingerprint-identical |
| U10-F | DONE | F1 commit `e12c568`: closed source-only projection; F2 commit `5b65c20`: exact checkpoint-operand binding; closure commit `7ade5cf` cuts parser/IR/render/expanded/Sable to `DiffusionIRProjection`, adds owner-qualified facts/receipts, and deletes the generic config author |
| U10-G | DONE | commit `7ade5cf`: all U10 legacy semantic readers and their parser wrappers deleted; `config_facts.yaml`, `evidence/stacks.py`, and the obsolete refiner renderer deleted; exactly five raw readers remain and are U11-owned; U10 structural debt is zero |
| U10-H | DONE | Soumil approved the exact 29-witness honesty/geometry/external-KV delta on 2026-08-28; 28 changed galleries reproduced reviewed occurrence-order hashes before guarded re-bless, SDXL's byte-identical gallery was not rewritten, and all 29 baselines + expected manifest were rebuilt; committed-tree receipt `5564c3881a`: 3,607 collected, 391 focused, 44 U2 authority, 52 preservation, full 3,495 passed / 14 skipped / 2 expected xfailed, every lane fingerprint identical |

Do not change a row to DONE from focused tests alone. Record the commit and full
receipt beside each completed row.

### U10-0 frozen baseline (2026-08-18)

- starting committed HEAD: `705f497fee43be1874932936420ef1d9c2d22df1`;
- source tree: `460aee5a374eb04034aea82624fd416cc7489710`;
- collection: **3,327**;
- preservation corpus: **29 witnesses** and 29 gallery directories;
- U10 structural debt: **14 owner-qualified rows**;
- quarantined U10 semantic readers: **12**;
- `adapters/diffusor/parser.py` broad-exception ratchet: **17**;
- structural config inventory: `config_facts.yaml` **110 lines**,
  `conditioning.yaml` **63 lines**, `typing.yaml` **178 lines**;
- closing receipt: static pass, 317 focused, 44 U2 authority, 52
  preservation, and exhaustive **3,311 passed / 14 skipped / 2 expected
  xfailed**, with every lane fingerprint-identical.

This baseline is a worklist, not a target to preserve. Counts may only shrink
or move to a later explicitly-owned unit with a machine-checked reason. U10-A
must first publish exact root-topology evidence in shadow mode and explain every
old/new witness disagreement before any parser or renderer authority changes.

### U10-A qualification matrix (2026-08-18)

U10-A deliberately publishes `repeated_stack`, not `transformer_stack`. A loop
over a module container proves repeated execution; it does not prove that the
element implements attention. U10-C may strengthen that shape only after the
exact block occurrence passes canonical U6/U7/U8 mechanism readers.

The U-shape proof is stronger and fully local: an earlier repeated invocation
returns at least two values; one is accumulated as a bypass, a later binding
derives from that accumulator, and a later repeated invocation consumes the
derived bypass alongside the carried value. No name (`UNet`, `down`, `up`,
`skip`) participates in the predicate. Both positive candidates remain typed
`incomplete` because this substrate does not claim whole-forward coverage.

| Witness | Legacy branch | U10-A result | Exact repeated stages |
|---|---:|---:|---:|
| AuraFlow | DiT | `incomplete / repeated_stack` | 2 |
| CogVideoX | DiT | `incomplete / repeated_stack` | 1 |
| FLUX.2 | DiT | `incomplete / repeated_stack` | 2 |
| FluxTransformer2DModel | DiT | `incomplete / repeated_stack` | 2 |
| HunyuanVideo | DiT | `incomplete / repeated_stack` | 2 |
| LTX-Video | DiT | `incomplete / repeated_stack` | 1 |
| Lumina Image 2 | DiT | `incomplete / repeated_stack` | 3 |
| Mochi | DiT | `incomplete / repeated_stack` | 1 |
| PixArt Sigma | DiT | `incomplete / repeated_stack` | 1 |
| PRX Pixel | DiT | `incomplete / repeated_stack` | 1 |
| Qwen-Image | DiT | `incomplete / repeated_stack` | 1 |
| Sana | DiT | `incomplete / repeated_stack` | 1 |
| Stable Diffusion 3.5 | DiT | `incomplete / repeated_stack` | 1 |
| Stable Diffusion XL | U-Net | `incomplete / u_shaped` | 2 |
| Wan 2.2 | DiT | `incomplete / repeated_stack` | 1 |

All 15 results are unpatched source results. The 23 synthetic controls additionally
pin misleading class names, config-only stage lists, renamed classes/fields/
locals, no invocation, no bypass, an unconsumed bypass, same-class duplicate
elements, helper construction, guarded constructor rivals, imported aliases,
broken sibling source, missing source, same config path with different source,
and a U-shaped route plus an independent repeated stack (typed ambiguity).

Shadow publication is cached under
`("root.denoiser.topology", ())` on the call-local `ParseContext`. No ModelIR,
card, expanded JSON, parameter, conformance or renderer code reads it in U10-A.

### U10-A committed-tree receipt (2026-08-19)

- commit: `92200e1fbcd578a878ff51c7cb54395bf9b50123`;
- static: pass, 3 changed Python files;
- collection: **3,365** tests;
- focused U10/diffusion/owner substrate: **428 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive: **3,349 passed / 14 skipped / 2 expected xfailed**;
- every isolated lane fingerprint before/after: **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/0f1c9e9080`.

This closes U10-A only. Its typed result remains shadow evidence and deliberately
stays `incomplete` for positive routes because the U3 substrate does not prove
whole-forward coverage. U10-B is now the active unit.

### U10-B qualification matrix (2026-08-19)

U10-B carries one exact container construction address, one exact symbolic block
occurrence and every exact invocation/binding. It never expands a symbolic count
into fabricated layers, and it never derives `main`, `secondary`, `refiner`,
`text`, `latent` or any block mechanism from names or config. Positive results
remain `incomplete` because U3 has no whole-forward coverage certificate.

| Witness | Exact stack occurrences | Typed unresolved candidates |
|---|---:|---:|
| AuraFlow | 2 | 0 |
| CogVideoX | 1 | 0 |
| FLUX.2 | 2 | 0 |
| FluxTransformer2DModel | 2 | 2 |
| HunyuanVideo | 1 | 2 |
| LTX-Video | 1 | 0 |
| Lumina Image 2 | 3 | 7 |
| Mochi | 1 | 4 |
| PixArt Sigma | 0 | 1 |
| PRX Pixel | 1 | 1 |
| Qwen-Image | 1 | 3 |
| Sana | 1 | 2 |
| Stable Diffusion 3.5 | 0 | 1 |
| Stable Diffusion XL | 0 | 4 |
| Wan 2.2 | 1 | 1 |

The zero-positive rows are not omissions: PixArt and SD3 expose a root stack
whose element class is external/unresolved in the current source bundle; SDXL's
down/up factory alternatives belong to U11. Their exact fields, calls and source
spans remain typed unresolved evidence. Hunyuan's nested token-refiner resolves,
while its root dual/single containers preserve real config-guarded class rivals.
Lumina's three containers reuse one block class but remain three graph occurrences.

One neutral substrate defect was corrected: `CallArgumentBinding` used to require
an imported child class to share the root wrapper's source file. Exact graph
occurrences lawfully cross source files, so closure now checks the child callable
against the child symbol; the enclosing `CallBindingResolution` still proves the
occurrence-to-graph-node join. The imported-alias poison pins this boundary.

The legacy `secondary_stacks_from_files` reader remains quarantined and active.
U10-B does not yet provide block semantics, norm/FFN facts or projection, so
deleting the old semantic consumer here would be premature. U10-C must consume
the exact positive occurrences; U10-F/G may delete the old path only after its
remaining responsibilities are covered or explicitly unresolved.

### U10-B committed-tree receipt (2026-08-19)

- commit: `98f1e9647695848625e35f83c41e14e342a6dea9`;
- static: pass, 3 changed Python files;
- collection: **3,405** tests;
- focused U10/diffusion/owner/execution substrate: **447 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive: **3,389 passed / 14 skipped / 2 expected xfailed**;
- every isolated lane fingerprint before/after: **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/93a35b676f`.

This closes U10-B only. The inventory remains occurrence-exact, symbolic and
shadow-only: it does not assign stack roles, infer block mechanisms, expand a
symbolic layer count, or drive a renderer. U10-C is now active and must compose
canonical U6/U7/U8 facts independently for each exact positive block occurrence
while keeping every unresolved candidate opaque.

### U10-C qualification matrix and honesty deltas (2026-08-19)

U10-C composes already-proven U6/U7/U8 readers at each exact U10-B block
occurrence. It adds one altitude-neutral attention execution boundary for the
normal Diffusers container/processor protocol: an exact framework `Attention`
constructor is a positive attention lane; an indexed source container using the
exact framework processor mixin is positive only when its exact injected/default
processor proves attention compute. Exact `attention_dispatch` import calls are
also a closed framework primitive protocol. Class names, field names, processor
suffixes and model families never participate.

The parser publishes this inventory call-locally in shadow mode and deliberately
passes no raw config document or selector. Therefore a config-guarded constructor
remains ambiguous until U10-F joins the exact U1 operand. The reader also remains
globally `incomplete`: U3 proves positive local relations, not whole-forward
coverage.

The source-only stack/block pair is held in a bounded process cache keyed by the
immutable ProgramIndex and resolved D0 root. This is a performance cache only:
the key includes source content fingerprints and component ownership, a same-path
source edit misses the cache, and every ParseContext still exposes its own
call-local result. It prevents name-blind/corpus gates from recomputing identical
source evidence for different checkpoint dictionaries without letting config
participate in the answer.

| Witness | Positive stacks | Proven attention lanes | Observed compute protocol | Honest unresolved reason |
|---|---:|---:|---|---|
| AuraFlow | 2 | 2 | framework container | finer Q/K/storage facts are inside the external container |
| CogVideoX | 1 | 1 | framework container | finer Q/K/storage facts are inside the external container |
| FLUX.2 | 2 | 2 | attention dispatch | source does not prove every finer lane fact at this altitude |
| FluxTransformer2DModel | 2 | 2 | attention dispatch | 2 upstream U10-B candidates remain opaque |
| HunyuanVideo | 1 | 1 | framework container | root dual/single guarded rivals stay opaque; refiner is exact |
| LTX-Video | 1 | 2 | attention dispatch | two invocation lanes retained separately |
| Lumina Image 2 | 3 | 6 | scaled-dot-product attention | two lanes at each of three exact occurrences; never class-unioned |
| Mochi | 1 | 0 | — | exact external processor implementation is outside the indexed bundle |
| PixArt Sigma | 0 | 0 | — | U10-B external block candidate remains opaque |
| PRX Pixel | 1 | 1 | attention dispatch | 1 upstream U10-B candidate remains opaque |
| Qwen-Image | 1 | 1 | attention dispatch | 3 upstream U10-B candidates remain opaque |
| Sana | 1 | 0 | — | one unguarded and one config-guarded framework lane make the source-only census ambiguous; U10-F owns the operand join |
| Stable Diffusion 3.5 | 0 | 0 | — | U10-B external block candidate remains opaque |
| Stable Diffusion XL | 0 | 0 | — | U-shaped factory alternatives belong to U11 |
| Wan 2.2 | 1 | 2 | attention dispatch | two invocation lanes retained separately |

#### Old/new comparison

The legacy diagrams remain byte-identical in U10-C; no new fact is a renderer or
IR authority yet. Where the legacy card was already source-supported, the shadow
inventory now supplies an exact block occurrence and exact call. Where legacy
output depended on a config/default/whole-file answer, U10-C does **not** copy it:

- self/cross/joint roles are withheld until U10-D proves dataflow;
- internal head geometry, Q/K norm, projection storage and position use are not
  inferred from an external framework container;
- ordinary dense/gated FFNs resolve only from exact two/three-projection local
  dataflow; Conv-GLU and opaque FFNs remain unknown rather than inheriting the
  legacy class-construction label;
- plain LayerNorm/RMSNorm facts do not turn a modulated sibling into AdaLN;
- a positive half-turn application is retained as application evidence, while
  model-stage learned positions, no observed position, and config-declared but
  unused RoPE never become block rotation;
- repeated uses of the same block class remain distinct occurrences, including
  separately guarded container constructions.

Mochi is a named substrate debt, not a family exception: its exact imported
implementation address is retained as `external_unavailable`. A future neutral
external-import-closure unit may index that source with provenance. U10-C must
not infer its semantics from the `MochiAttention` spelling. Sana is different:
the source is present, but the exact constructor guard needs the checkpoint
operand, so U10-F—not an import crawler—owns its resolution.

#### U10-C committed-tree receipt

The first committed-tree attempt exposed a real shadow-boundary defect: a
source-less or unresolved D0 root was passed into the strict direct U10-C reader,
which correctly rejected it, but the parser shadow hook leaked that rejection as
an exception. The correction keeps the direct reader strict and converts the
shadow publication to the typed U10-B `missing_source`/conflict failure. The two
legacy regressions and a permanent source-less control pass; both affected legacy
files then passed in full (188 tests).

Final commit `ee836b0` passed static checks, collection (3,453), focused (307),
all 44 U2 authority checks, and all 52 preservation checks with zero structural
or pixel drift. The user interrupted the coordinator after preservation had
finished while the exhaustive lane was still running. Its clean detached
`ee836b0` worktree was retained; the exhaustive partition was resumed there and
passed 3,341 tests with 14 expected skips and 2 expected xfails. Tree fingerprint
`e39be8bc6a4197e131186b7a8a84d836fce1c80ed25ab12b71be426a6e43ecec`
and artifact fingerprint
`c6d07cc930511fbe4a00cdfc97757222c052200fc2b999ef18c8aa414253abb6`
were identical before/after. The transparent composite receipt is
`/private/tmp/model-unfolder-verification/bb2795cf7c/continuation_receipt.md`.

### U10-D implementation boundary and qualification matrix (2026-08-19)

U10-D implements the corrected block-local contract in §§6.4–6.5. It does not
name modalities and does not project into the legacy IR. The parser publishes
two call-local shadow results only:

- `root.denoiser.streams` — exact block formals, returns, U10-C attention/FFN
  calls, canonical norm calls, explicit imported joins, local relations, and
  unresolved rows;
- `root.denoiser.conditioning` — exact norm modulation, bare multiplication
  gates, gate-in-norm applications, and unresolved branch rows.

The implementation strengthened one canonical U7 boundary rather than
reverse-engineering norm call sites inside U10: `norm_invocations_at_owner`
publishes every positively classified exact norm invocation, and the existing
`norm_kind_at_owner` aggregate derives from that call-level census. Thus norm
kind and norm application topology share one source.

The lineage helper is callable-local and consumes only immutable ProgramIndex
observations. It is not a second source parser, cache, or CFG. It preserves
guarded rivals as unresolved; exact imported `torch.cat`/`concat` calls are the
only join protocol; every classified attention/FFN result must positively reach
an exact return. State identity is kept separate from arbitrary numeric
dependence: timestep/modulation inputs cannot become state streams merely
because they influence a returned tensor.

| Witness | Attention local relations | Unresolved attention | FFN local relations | Unresolved FFN | Exact conditioning applications |
|---|---|---:|---|---:|---|
| AuraFlow | dual + single | 0 | three single-state calls | 0 | 3 attention gates + 1 FFN gate |
| CogVideoX | dual | 0 | — | 0 | 2 attention gates on the two returned branches |
| FLUX.2 | dual | 1 | — | 2 | 3 attention gates + 2 FFN gates; rival/guarded routes remain unresolved |
| FluxTransformer2DModel | joined inputs | 1 | — | 0 | — |
| HunyuanVideo | single | 0 | — | 0 | 1 attention gate |
| LTX-Video | single + contextual single | 0 | — | 0 | 1 attention gate |
| Lumina Image 2 | — | 6 | — | 0 | — |
| Mochi | — | 0 | — | 0 | —; U10-C implementation remains external |
| PixArt Sigma | — | 0 | — | 0 | —; no positive U10-C block row |
| PRX Pixel | contextual single | 0 | — | 0 | 1 attention gate |
| Qwen-Image | — | 1 | — | 0 | —; an opaque helper leaves two rival non-stream gate roots |
| Sana | — | 0 | — | 0 | —; U10-C source-only constructor remains ambiguous |
| Stable Diffusion 3.5 | — | 0 | — | 0 | —; no positive U10-C block row |
| Stable Diffusion XL | — | 0 | — | 0 | —; U11 owns the U-shaped cells |
| Wan 2.2 | — | 2 | — | 0 | — |

These are source outcomes, not coverage targets. Important honesty deltas are
intentional:

- One FluxTransformer block proves an explicit joined-input route through the
  later split/return; its other lane and all Lumina lanes remain unresolved.
  An invoked lane is not a stream relation unless its result reaches a return.
- CogVideoX proves dual-state attention. Its separate FFN join/split path is not
  collapsed into the attention relation.
- PRX proves only the outer contextual lane. An inner KV join would require the
  exact inner attention implementation; the class/model is not used as a proxy.
- Mochi/PixArt/SD3 and other external/ambiguous implementations stay opaque.
- An explicit concat is called `joined_inputs`, not `joined_state`: a
  non-returned input does not become a second state merely by being joined.
- One call may have several exact gate applications; the inventory retains each
  application occurrence instead of collapsing them by branch call.

Permanent controls include full formal/class/field renaming, same config shape
with different source, dimension/text/guidance decoys, discarded attention and
discarded gated branches, guarded rival definitions, direct-return gates,
multi-output multi-gate calls, transparent state-preserving transforms,
state-slot reassignment, explicit joins, foreign-index refusal, source-missing
parser publication, DTO forgery, and the 15-witness matrix above.

#### U10-D committed-tree receipt

- commit: `14fee0cdc794a0be8b855910f7174b05f8abfb0c`;
- static: pass, 6 changed Python files;
- collection: **3,492** tests;
- focused U10-D + canonical norm/code evidence: **324 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive: **3,380 passed / 14 skipped / 2 expected xfailed**;
- every isolated lane fingerprint before/after: **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/8eddcda6a0`.

This closes U10-D. Its results remain shadow evidence and are not yet renderer
or IR authority. U10-E is now active and owns exact root bookends,
temporal-operation proof, and companion denoiser evidence. Global output-domain
naming remains U12 codec/VAE work: even a proven temporal denoiser operation is
not by itself proof that the decoded output is a video.

### U10-E implementation record

U10-E adds two shadow-only source boundaries. Neither is a production drawing
authority before U10-F moves all consumers together.

`evidence/diffusion_bookends.py` starts from the exact U10-B stack executions
and U10-D state/conditioning formals. It uses the shared U10-D local-lineage
engine and the shared U9 operation classifier to retain positive operations on
three separately typed routes: root state input, root conditioning input, and
root state output. It does not reopen source, parse another AST, search sibling
classes, or accept raw config.

The temporal split is deliberately stronger than the initial draft:

- `temporal_operations` contains only a positive temporal/3-D mechanism. The
  current closed kind is `three_dimensional_convolution`.
- `tensor_geometry` contains source-observed rank/shape only. A five-axis
  reshape is `rank_five_shape`; it is **not** temporal computation and cannot
  authorize a video label.
- A three-entry patch tuple, `num_frames`, or any other temporal config field
  cannot enter the source reader at all.
- A nested/folded reshape cannot borrow the outer invocation's five arguments;
  the observed rank and registered operation must cite the same exact call.

This is the source half of action E.2. The old `_temporal_axis` fallback and its
declared temporal geometry remain live compatibility code until U10-F creates
the typed config-side geometry fact and cuts every IR/card/renderer consumer in
one reviewed change. Therefore U10-E does **not** claim that production output
has already stopped consulting the legacy fallback. Deleting it in E would
either lose declared geometry or make a shadow reader a partial production
authority before receipts exist.

`evidence/diffusion_companion.py` consumes only exact companion component
addresses retained by `SourceBundle`. The loader establishes an additional
address only when a pipeline entry has the exact same framework component
declaration as the selected denoiser, then fetches that slot's own config. Each
address gets its own D0 owner graph and its own U10-A/B/C/D/E profile. Slot
spelling and equal dimensions never participate in comparison. Duplicate/root
addresses are typed conflicts.

The strongest comparison available in E is intentionally limited:

- identical content fingerprint and class-body address ->
  `same_source_contract`, not instantiated architecture equality;
- equal positive typed signatures across different sources ->
  `matching_partial_evidence`, still not equality;
- differing positive signatures -> `different_positive_evidence`;
- missing or incomplete source -> `unresolved`.

`architecture_equivalent` is always unknown in E. U10-F may strengthen it only
after every deciding config operand is exactly bound to the source branch it
selects.

#### U10-E real-witness lower-bound matrix

The following `(state input, state output, conditioning input, temporal op)`
counts are pinned from the frozen corpus. They are observations, not coverage
targets:

| Witness | Counts |
|---|---:|
| AuraFlow | 7, 0, 0, 0 |
| CogVideoX | 0, 0, 0, 0 |
| FLUX.2 | 2, 3, 3, 0 |
| FluxTransformer2DModel | 1, 2, 0, 0 |
| HunyuanVideo | 0, 0, 0, 0 |
| LTX-Video | 1, 2, 3, 0 |
| Lumina | 0, 0, 0, 0 |
| Mochi | 0, 1, 0, 0 |
| PixArt | 0, 0, 0, 0 |
| PRX | 1, 1, 1, 0 |
| Qwen-Image | 0, 1, 0, 0 |
| Sana | 0, 2, 0, 0 |
| SD3.5 | 0, 0, 0, 0 |
| SDXL | 0, 0, 0, 0 |
| Wan 2.2 | 0, 1, 0, 0 |

Zero temporal operations is honest: current real temporal implementations hide
their decisive work behind helper/control-flow routes the positive local-flow
substrate cannot yet close. Synthetic exact-source controls prove Conv3D and
rank-five separation, while real models remain unknown rather than inheriting
the legacy config-based video claim. The frozen configs also do not retain a
real fetched companion slot; loader/source-bundle controls exercise the exact
by-ID address path without patching a model witness.

Permanent controls cover unused input/output/conditioning projections, output
calls that do not reach the return, full class/field/local renaming, cross-root
dependency laundering, source-missing publication, rank-four versus rank-five
shape, geometry-to-temporal forgery, Conv3D without temporal config, temporal
config unable to enter the source API, same-source versus equal-partial versus
different companion profiles, missing companion source, slot rename, duplicate
addresses, and independently constructed owner graphs.

#### U10-E committed-tree receipt

- commit: `a379cf0cca79660d3933e06238a3f9690c60b772`;
- static: pass, 10 changed Python files;
- collection: **3,535** tests;
- focused diffusion/bookend/companion/stream/conditioning: **350 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive: **3,423 passed / 14 skipped / 2 expected xfailed**;
- every isolated lane tree fingerprint and artifact fingerprint before/after:
  **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/e3b70ca7b0`.

This closes the U10-E source boundary. Its cache entries remain shadow evidence:
U10-F is the one reviewed production cutover for typed declared geometry,
projection DTOs, config-author deletion, and passive consumers.

### U10-F1 implementation record

U10-F1 adds `adapters/diffusor/schema.py`, a closed passive projection over the
canonical U10-A/B/C/D/E evidence graph.  It does not reopen source or read
config.  Each projected block, attention lane, FFN, stream relation,
conditioning application, bookend, and companion relation construction-time
checks against the exact evidence object it retains.  Source protocols remain
below final diagram vocabulary: equal heads and grouped KV are not promoted to
MHA/GQA/MQA without checkpoint operands, and rank-five geometry remains
separate from temporal computation.

The source projection is cached call-locally by the diffusion parser but has no
IR, renderer, expanded-JSON, parameter, receipt, conformance, or structural-debt
authority.  A name-blind frontier control exposed an initial cache-boundary bug:
the projection had required Python object identity between equal immutable
evidence reconstructed in separate bounded caches.  The repaired closure joins
by the complete frozen evidence value and still checks every exact occurrence;
it neither accepts a weaker key nor loses cross-cache determinism.

#### U10-F1 committed-tree receipt

- commit: `e12c568d53ac95412903a26ccb922697b5d2bfb4`;
- static: pass, three changed Python files;
- collection: **3,548** tests;
- focused U10/F1: **400 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive: **3,436 passed / 14 skipped / 2 expected xfailed**;
- every isolated lane tree and artifact fingerprint before/after: **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/ae1df4a393`.

F1 is therefore DONE.  F2 is the next shadow boundary and may bind only exact
checkpoint-declared operands retained by the source evidence.  Class defaults,
aliases not named by source, familiar dimensions, and missing operands remain
unbound; F2 emits no consumption and cannot change a production diagram.

### U10-F2 implementation record

F2 adds `evidence/config_registration.py` and
`adapters/diffusor/config_binding.py`.  The former is a closed framework
address protocol for the exact imported Diffusers `register_to_config`
decorator: it maps ordinary root-constructor parameters to same-key checkpoint
paths but reads no value and assigns no architectural meaning.  A local or
multiply-bound function with the same spelling cannot activate the protocol.
The latter rebuilds the occurrence graph with those exact root address
bindings, re-runs the canonical U10 readers, and partitions every retained
config dependency into a bound checkpoint occurrence or a typed unresolved
operand.  Each row carries the exact source occurrence, decisive spans,
owner/fact/reader target and exact config path.

This framework-address step was required by a real negative control.  The
first F2 implementation was green synthetically but bound **zero** operands on
all fifteen Diffusers witnesses: the synthetic fixture passed one `config`
object, whereas real Diffusers roots use registered scalar constructor
parameters.  D0 correctly refused to guess a config path for those many
parameters.  The repair recognizes the imported framework execution protocol,
not a model/class/parameter spelling; a local same-name decorator remains
powerless.  A second real seam then showed repeated counts are retained as a
normalized expression such as `range(num_layers)`.  F2 now walks that frozen
expression tree and joins only identifier occurrences with one exact
owner-graph config binding; it never searches diagnostic source text or
interprets `range` by name.

The fifteen-witness matrix now yields the following exact bound-row counts:

| witness | projected blocks | bound operands |
|---|---:|---:|
| AuraFlow | 2 | 0 |
| CogVideoX | 1 | 1 |
| FLUX.2 | 2 | 2 |
| FluxTransformer2DModel | 2 | 2 |
| HunyuanVideo | 1 | 1 |
| LTX-Video | 1 | 1 |
| Lumina Image 2 | 3 | 3 |
| Mochi | 1 | 1 |
| PixArt Sigma | 0 | 0 |
| PRX Pixel | 1 | 1 |
| Qwen-Image | 1 | 1 |
| Sana | 1 | 1 |
| Stable Diffusion 3.5 | 0 | 0 |
| Stable Diffusion XL | 0 | 0 |
| Wan 2.2 | 1 | 1 |

The real positive rows are currently exact symbolic stack-count operands.  The
internal head/FFN operands remain unavailable for framework attention
containers whose implementation source is not in the component bundle; F2
does not borrow their conventional Diffusers semantics.  Synthetic exact-source
controls prove the full nested scalar route through query/KV head operands.
Zero-row witnesses are retained as negative controls, not treated as success
by vacuity.

F2 deliberately has no production parser call site.  Activating it in shadow
mode would create real bound-but-unconsumed U1 events and make the blocking
standing-debt net red for bookkeeping rather than architecture.  F3 must add
the call and convert each used row to one owner-qualified consumption in the
same atomic production cut.  No pending-debt or hidden ledger exception is
introduced to bridge the two phases.

#### U10-F2 committed-tree receipt

- commit: `5b65c20ea4dac50832be6bccc23267b30cfc6e5f`;
- static: pass, four changed Python files;
- collection: **3,583** tests;
- focused F2/U10: **309 passed**;
- affected U2 authority gates: **44 passed**;
- preservation: **52 passed**, zero structural or pixel drift across all 29
  witnesses;
- exhaustive lane: **3,471 passed / 14 skipped / 2 expected xfailed**;
- all lane tree and external-artifact fingerprints before/after: **identical**;
- receipt logs: `/private/tmp/model-unfolder-verification/9675296510`.

F2 is therefore DONE.  F3 is the first production-authority cut: it must
activate this boundary and consume only its typed operand rows atomically with
the parser/IR/consumer migration.  No F2 shadow event is left running in
production before that cut.

### U10-F3/F4/G local implementation record — awaiting artifact approval

The working tree now performs the atomic production cut through
`adapters/diffusor/projection_ir.py`.  It is the sole U10 bridge from the bound
source graph to layer templates, bookend blocks, registered facts and projection
receipts.  Parser, renderer, expanded JSON, Sable and conformance consume that
projection or the resulting typed IR; none reopens source or raw config to
reconstruct a denoiser mechanism.  Exact checkpoint operands are consumed only
through their F2 owner/path rows.

F4/G authority deletion is also present locally:

- the generic `everchanging/diffusor/config_facts.yaml` author is deleted;
- `evidence/stacks.py` and all U10-owned whole-file semantic readers are
  deleted, together with their parser wrappers;
- `dit_class_markers`, `norm_type_kind`, `stack_lane_params`,
  `temporal_config_fields`, and other U10 structural identity vocabulary no
  longer drive production structure;
- the obsolete diffusion refiner renderer is deleted;
- U10 `StructuralDebt` is zero and the legacy-reader quarantine reports zero
  U10 rows;
- exactly five raw diffusion readers remain, all assigned to U11's UNet
  internals (`unet_mid_block_present`, stage attention/cell/temporal readers,
  and transformer-FFN activation at a UNet cell).

#### Real counterexamples found during F3/F4

1. **Framework defaults are not source absence.**  A minimal AuraFlow config
   still resolves 4+32 layers because the installed Diffusers constructor
   supplies exact registered defaults.  A genuinely uninstalled transformer
   remains opaque.  The old test incorrectly expected both cases to be zero.
2. **A proven operation does not imply a proven mechanism.**  Sana source and
   its exact constructor/config guard prove self- and cross-attention
   applications plus the external context route, while the internal attention
   kind remains unknown.  The diagram now draws those applications and keeps
   both mechanisms opaque.
3. **Class-wide guarded calls are not runtime lane cardinality.**  Lumina reuses
   one block class for three exact construction occurrences.  Each class body
   contains two guarded attention calls, but each constructor literal activates
   one.  `UnresolvedStreamRelation.state` now distinguishes a source-proven
   inactive guard from an unresolved runtime lane.  F3 excludes only the former;
   any other unresolved rival still blocks materialization.  The real result is
   three templates with counts 2+2+26 and one active single-state lane each.
4. **Cross-domain preservation caught a graph/card coupling defect.**  MusicGen
   remains a transformer/composite model, but its already-proven cross-attention
   drill now carries the exact external encoded-prompt K/V source node and card.
   The shared renderer had briefly made every secondary input static, which
   removed legitimate ALiBi/position drills and would also have made this real
   K/V card unreachable.  The final rule is symmetric and card-driven: a
   secondary input is clickable iff the enclosing block has child cards.  An
   opaque cross-attention still gets the external-source card because placement
   and K/V provenance are known independently of the inner mixer; it does not
   gain fabricated Q/K/V or softmax internals.  Permanent controls pin the
   clickable and leaf-static cases, the same-spec card, Sana's opaque positive
   rail, and MusicGen's real rendering/coupling path.  Bloom's attention and FFN
   views are byte-identical to HEAD after this correction; its structural IR is
   also byte-identical.
5. **Parameter incompleteness remains visible, not solved by convention.**
   HunyuanVideo and CogVideoX can have exact depth/width while unresolved
   attention/FFN terms leave the current estimate incomplete (displayed with
   its assumptions).  Completing those formulas belongs to U14; U10 does not
   invent terms to avoid a zero subtotal.
6. **Honest labels exposed a generic sizing bug.**  Several truthful unresolved
   labels are longer than the old fixed boxes, and the pre-existing Bloom
   ``Feed-forward (FFN)`` label already overflowed its nominal box.  The shared
   block-layout rule now grows any single- or multi-line box to fit its actual
   label and treats an explicit width as a floor, never as permission to shrink
   below the node-kind minimum.  Representative Llama, DeepSeek-V3, Granite,
   Qwen2-VL, Gemma-2 and GPT-OSS architecture galleries were inspected; no
   collision, clipping or architectural change was found.  Their only visual
   delta is this generic geometry correction.

#### Local pre-approval receipt

- targeted guarded-lane/F3 controls: **6 passed**;
- stream + bound-operand/F3 lane: **82 passed**;
- diffusion + unknown-safety + cross-surface lane: **141 passed**;
- projection/receipt/debt/identity/quarantine lane: **131 passed**;
- fact/qualification/authority/Sable lane: **122 passed, 2 expected xfailed**;
- the sole failure is `test_sable_regression_corpus`, at the deliberately
  unblessed AuraFlow artifact delta;
- `git diff --check` and changed-file `pyflakes` are clean;
- representative galleries inspected: AuraFlow, HunyuanVideo, Sana, Lumina,
  CogVideoX, FLUX, FluxTransformer, LTX, Mochi, PixArt, PRX, Qwen-Image,
  SD3.5, SDXL, Wan and MusicGen; all inspected layouts are unclipped and their
  opaque/detail boundaries match the typed evidence.
- final 29-witness delta regenerated after the secondary-input correction at
  `/private/tmp/u10-preservation-delta.json`: all 29 have expected evidence/
  metadata movement from the closed fact/receipt inventory; all 15 diffusion
  witnesses have their intended U10 honesty projection; the 14 transformer or
  composite controls retain their mechanisms, with inspected view movement
  limited to generic label geometry plus MusicGen's exact external-K/V card;
  SDXL's 29 rendered views remain byte-identical despite its ledger/IR audit
  movement.

Soumil approved the exact delta on 2026-08-28.  The 28 changed gallery locks
match their reviewed occurrence-order hashes, SDXL's byte-identical gallery
remained untouched, and the 29 canonical baselines plus expected manifest were
rebuilt.  Closure commit `7ade5cf87a14af4ebb7cf6d8951f9ea3013ce1bf`
then reproduced from detached worktrees: static clean over 63 changed Python
files; 3,607 tests collected; 391 focused, 44 U2-authority and 52 preservation
tests passed; exhaustive result 3,495 passed / 14 skipped / 2 expected xfailed;
every lane source/artifact fingerprint identical.  Receipt:
`/private/tmp/model-unfolder-verification/5564c3881a`.  U10 is **DONE**.
