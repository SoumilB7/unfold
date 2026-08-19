# U10 — Diffusion Root, Stack, Stream, and Conditioning Execution Plan

Status: **ACTIVE — U10-A DONE; U10-B occurrence-exact stack inventory next**

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

Node kinds are mechanism-neutral execution facts:

- exact input/output parameter;
- exact attention/FFN/norm/projector/modulation call site;
- exact concat/stack/add/multiply/split operation;
- exact repeated stack invocation;
- exact condition/context/timestep/guidance source;
- unresolved operation/relation.

Edges cite producer and consumer spans and are classified only as:

- proven unconditional;
- proven conditional with guard path;
- unresolved.

Derived topology terms (`dual_stream`, `single_stream`, `concat_joint`,
`kv_join`, `cross_attention`, `parallel`, `sequential`) are permitted only as
typed interpretations of a sufficient set of graph relations. Each term needs
its own counterexample and must retain the underlying graph proof.

No unproven relation is labelled “unordered,” “not present,” or “sequential.”

### 6.5 `DenoiserConditioningEvidence`

Evidence module: `model_unfolder/evidence/diffusion_conditioning.py`

Carries independent, tri-state facts for:

- timestep embedding reaching a block;
- adaptive norm shift/scale;
- output gates and whether a gate is a multiplication or folded into a norm;
- encoded-token route to self/joint/cross attention;
- pooled conditioning route;
- pre-stack projection/addition/concatenation;
- guidance route;
- exact conditioning projector chain and bound dimensions.

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

1. Build exact positive local relations among latent, context/text, timestep,
   guidance, attention, FFN, projector, join, gate, and output sites.
2. Interpret dual/single, concat, KV-join, cross-attention, parallel, and
   sequential only from sufficient graph proofs.
3. Publish unresolved relations when U3 cannot prove order or coverage.
4. Distinguish bare gate multiplication from gate-in-norm execution.
5. Keep config enum/dimension values unconsumed until their exact source branch
   is bound.

Required controls:

- Flux dual and single stacks;
- SD3-style dual-stream MMDiT;
- HunyuanVideo heterogeneous stacks;
- CogVideoX/Mochi joined sequence;
- Wan/PixArt/Sana/LTX cross-attention;
- AuraFlow sequential joined stream;
- PRX-style KV join;
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
3. A video/temporal label or 3-D operation is emitted only from the latter.
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
| U10-B | ACTIVE | occurrence-exact stack inventory over the existing D0/B2/owner-graph rails; replace the responsibility of the whole-file `secondary_stacks_from_files` reader without adding another parallel detector |
| U10-C | NOT STARTED | exact U6/U7/U8 block fact composition |
| U10-D | NOT STARTED | stream + conditioning graph |
| U10-E | NOT STARTED | bookends + temporal + companions |
| U10-F | NOT STARTED | typed projection + config-author dismantling |
| U10-G | NOT STARTED | legacy deletion + later-unit handoffs |
| U10-H | NOT STARTED | artifact approval + committed-tree closure |

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
