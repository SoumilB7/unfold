# U11 — Source-Derived U-Net Stage and Cell Execution Plan

Status: **ACTIVE — U11-A1 through U11-E2c DONE; U11-F/G must close stage
selection and runtime-conditioning joins before the E3/H atomic cutover**

Authority: this document is the binding execution plan for U11. It refines
`EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` §§16.5 and 20.14 without weakening
their intent. If an older note conflicts with this document, stop and resolve
the conflict before changing production behavior.

---

## 1. The result U11 must deliver

U10 already proves the outer denoiser route:

```text
exact root component + source occurrence
        |
        v
positive root execution proof: u_shaped
```

U11 must continue from that exact handoff without using config or identity to
fill in the interior:

```text
U10 u_shaped root topology
        |
        v
exact imported-source closure
        |
        v
down / optional-mid / up construction occurrences
        |
        v
stage execution + skip-state DAG
        |
        +--> exact ResNet/temporal cell occurrences
        +--> exact attention/transformer occurrences
        +--> exact sampler occurrences
        +--> exact conditioning/bookend occurrences
        |
        v
canonical typed U-Net projection
        |
        +--> parser / IR
        +--> cards and drills
        +--> expanded JSON
        +--> parameter graph
        +--> Sable / conformance
```

The final diagram must be a projection of the same owner-qualified stage/cell
graph checked by conformance and parameter consumers. A renderer may choose
coordinates, spacing, colors and wording; it may not invent stage order, cell
kind, repeat count, sampler placement, skip joins, conditioning edges or
mechanism internals.

U11 is complete when a new U-shaped denoiser can become accurate by exposing
its construction and forward code—not by adding a family branch, block-name
substring, config template or renderer special case. Unsupported or incomplete
source must remain an explicit opaque stage/cell at every drill depth.

---

## 2. Verified starting state

The live tree currently has two very different authorities:

1. U10's `read_diffusion_root_topology(...)` proves a U-shaped root from exact
   down/up execution and a real skip accumulator route.
2. `adapters/diffusor/unet.py::parse_unet(...)` then reconstructs almost the
   entire interior from config lists and Diffusers conventions.

That compatibility interpreter presently authors:

- stage count and widths from `block_out_channels`;
- attention placement from the substring `"Attn"` in block tokens;
- ResNet repeat defaults (`2`), transformer-depth defaults (`1`) and the up-path
  `+1` rule;
- all-but-final down/up sampler placement;
- legacy reinterpretation of `attention_head_dim` as a head count;
- a width-mismatch encoder projection;
- a universal ResNet cell story (GroupNorm, activation, two Conv3x3 operations,
  timestep add, residual and optional 1x1 shortcut);
- a universal nested Transformer2D story (self-attention, cross-attention and
  FFN); and
- descriptions and child cards later restated by
  `renderers/html/block_views/unet.py`.

Five quarantined whole-file readers remain and are all owned by U11:

- `unet_code_attention_placement_from_files`;
- `unet_mid_block_present_from_files`;
- `unet_stage_attn_cell_from_files`;
- `unet_stage_temporal_from_files`;
- `unet_transformer_ffn_activation_from_files`.

The real SDXL corpus witness proves the U-shaped root, but its root source file
constructs resolution stages through imported `get_down_block` /
`get_up_block` factories. The current `ParseContext.program_index()` indexes
the exact root file but not that imported Diffusers construction closure. The
legacy `_augment_diffusion_files(...)` helper compensates by reopening and
reparsing a bounded package subtree. U11 must remove that second source universe
before it can claim exact stage ownership.

The blessed corpus currently contains one proven U-shaped root (SDXL). Synthetic
counterexamples and exact installed-source controls are therefore mandatory;
one real witness cannot establish generality.

---

## 3. Non-negotiable laws

### 3.1 U10's handoff is necessary but not sufficient

Only a positive `DiffusionRootTopology(kind="u_shaped")` may enter U11. The
U-shaped verdict proves the outer skip route; it does not prove the number,
classes or mechanisms of interior stages.

### 3.2 Address is not mechanism

The following may locate candidates only:

- `_class_name`, architecture, import aliases and pipeline slots;
- `down_block_types`, `mid_block_type`, `up_block_types`;
- factory string tokens and class names;
- field names such as `down_blocks`, `mid_block`, `up_blocks`, `resnets`,
  `attentions`, `downsamplers` and `upsamplers`.

No spelling or substring may prove an attention cell, ResNet cell, temporal
operation, sampler, stage role or execution order. The mechanism comes from the
resolved class's construction and callable graph.

### 3.3 One ProgramIndex universe

U11 may not revive `_augment_diffusion_files`, `ast.parse`, source reopening or
a private registry. Imported source needed for an exact construction must enter
the same call-local `SourceBundle` / `ProgramIndex` with explicit provenance and
component ownership. Parse failures and unresolved imports remain typed.

### 3.4 Occurrences, not classes

The unit of truth is an exact construction occurrence. The same block class may
be constructed at several stages with different config operands, guards or
execution positions. Same-class unions and best-candidate votes are forbidden.

### 3.5 Construction/storage order is not execution order

`ModuleList`, list-comprehension order and factory-call order are construction
facts only. Stage execution order, mid placement, skip production/consumption
and sampler placement require positive callable/dataflow proof.

### 3.6 Config values parameterize proven source expressions

A checkpoint value is consumable only through:

```text
exact config occurrence
  -> exact constructor/forward expression
  -> exact owner occurrence
  -> typed fact
  -> projection receipt
```

Class defaults use `class_default`, checkpoint values use
`checkpoint_declared`, and a code-bound operand uses `code_and_config`. Missing
or conflicting aliases never become a conventional number.

### 3.7 Unknown stays unknown at every depth

Unresolved stage count, cell class, temporal branch, sampler, attention
mechanism, FFN or residual equation produces an opaque/partial node. It must not
fall through to common Diffusers defaults such as two ResNets, one transformer,
SiLU, GroupNorm, Conv3x3, MHA, four-times-width FFN or factor-two sampling.

### 3.8 Existing canonical evidence is reused, never copied

Once an exact nested occurrence is resolved, U11 calls the existing U6/U7/U8
attention, FFN, norm, cell and position readers. It must not implement a UNet-
specific copy of those mechanisms.

### 3.9 One structural author

One canonical `UNetSourceProjection` (name provisional until U11-F) owns the
stage/cell DAG. Parser, renderer, JSON, params and conformance consume it or the
typed IR it creates. They cannot inspect raw config/source and reconstruct a
parallel graph.

### 3.10 Migration and deletion are atomic

For each migrated fact family, the exact legacy reader/branch/default and its
quarantine/debt row are deleted in the same commit. “New reader plus old
fallback” is not completion.

---

## 4. Boundaries that must remain separate

| Boundary | It may prove | It may not prove |
|---|---|---|
| source closure | exact imported file/symbol address | architecture semantics |
| root topology | positive U-shaped outer route | interior stage structure |
| stage construction | exact constructed occurrences and operand expressions | runtime order |
| stage execution | ordered calls, optional branches, skip sources/joins | cell internals |
| cell evidence | exact operations/dataflow inside one occurrence | sibling-stage facts |
| config binding | value used by one proven expression | meaning from field presence |
| projection | normalized typed graph from carried evidence | new source/config inference |
| renderer | layout and presentation of the graph | architecture decisions |
| params | owner-bound terms from constructed modules | conventional hidden terms |

No convenience function may cross more than one of these boundaries invisibly.

---

## 5. Execution phases

### U11-A — exact imported-source closure

Goal: make the one call-local `ProgramIndex` contain every source required to
resolve a root construction, without a second parser.

Implementation:

1. Move the general import-closure address logic out of
   `evidence/conformance.py::_augment_diffusion_files` into the source-resolution
   layer.
2. Follow only exact imports used by construction/factory calls, stay inside the
   installed package/revision root, retain component ownership, and publish the
   files as `SourceBundle.supporting_files` (or equivalently typed external
   `SourceFileNode`s in the same index).
3. Preserve every unresolved, cyclic, out-of-root, dynamic and parse-failed edge
   as a typed record. Never skip it and never choose by basename/class substring.
4. Make `ParseContext.program_index()` the sole index consumed by U11.
5. Migrate conformance callers from `_augment_diffusion_files` only when their
   exact import closure is available through the same bundle; do not leave two
   closure implementations.

Exit:

- SDXL's exact root index contains the called down/up factory definitions and
  the exact imported candidate classes needed for stage resolution;
- a same-name class in an unimported file cannot enter;
- imported alias, relative import, re-export, cycle, parse failure, dynamic
  import, path escape and source-missing controls are typed;
- content-fingerprint invalidation and component isolation remain green;
- no parser/render output changes.

Implementation is split deliberately:

- **U11-A1 — demand-driven address boundary.** `SourceBundle` declares exact
  package roots; `resolve_called_import_source(...)` expands one immutable
  ProgramIndex through one exact called import/re-export chain. It never walks
  arbitrary imported calls or a whole package. This bounded unit lands before
  any production UNet reader consumes it.
- **U11-A2 — consumer cutover.** U11-B/E stage and nested-cell readers use A1;
  once every old caller is covered, `_augment_diffusion_files(...)` and the
  remaining whole-file registries are deleted. A1 is not permission to leave
  the legacy closure indefinitely.

### U11-B — exact stage construction inventory

Goal: identify every constructed candidate that feeds the exact repeated
containers proved by U10, without assigning a mechanism from its name.

Build closed DTOs for:

- exact root occurrence and U10 topology proof;
- exact container/factory construction site;
- source/config guard chain;
- candidate class symbol(s) and rival/unresolved alternatives;
- constructor argument expressions;
- symbolic repeat/count expressions; and
- exact config paths occurring inside those expressions, when provable.

Factory string tokens are address operands only. `get_down_block("X")` may
resolve the exact branch/class selected by `"X"`; the token `"X"` cannot prove
what that class does. A comprehension remains one symbolic template, not N
fabricated occurrences.

A direct root field is not a mid/bottleneck stage merely because of its field
or factory spelling. U11-C must first prove that the exact constructed field is
invoked between the two repeated sides. Only that positively selected address
is expanded as the direct stage candidate. This sequencing is intentional: it
prevents `mid_block`, `get_mid_block`, or any renamed equivalent from becoming
semantic evidence.

Exit: exact stage candidates are resolved/ambiguous/incomplete/failed with no
config values consumed and no structural output changed.

### U11-C — stage execution and skip DAG

Goal: derive the actual down/optional-mid/up execution graph.

Prove separately:

- which constructed containers/cells are invoked;
- execution order and guarded alternatives;
- whether and where a mid occurrence executes;
- skip values produced, accumulated, sliced/popped and consumed;
- concatenation/addition/other join operation;
- conditioning/timestep/class/additional-condition inputs; and
- sampler calls and their exact position.

The U10 `SkipRoute` is the outer seed, not a complete interior DAG. Symmetry,
all-but-final sampling and “up count = down count + one” are forbidden
assumptions.

Exit: one typed `UNetStageGraph` carries proven nodes/edges plus unresolved
relations. Source-missing and unsupported control flow stay partial/opaque.

Implementation boundary: U11-C is the root-stage graph, not a claim that the
whole callable or every child cell has been interpreted. It may publish only:

- the exact repeated call occurrences already proved by U10;
- exact root-owned constructed-field call occurrences inside the U10
  inter-loop address interval, retaining each guard and call site separately;
- the one positive U10 skip-dataflow edge; and
- typed unresolved relations for every unproved ordering/construction/coverage
  claim.

Textual position is only an address filter. A call inside an unsupported region
or after an unguarded return cannot become a stage node. The boundary remains
partial/open even when no known gap is observed. Child-cell execution,
operation roles, joins, conditioning semantics and sampler mechanisms remain
owned by U11-D/E/F; U11-C retains their raw addressed call evidence but cannot
name them early.

### U11-D — exact ResNet and temporal cells

Goal: replace the universal `_unet_resnet_ops(...)` template.

For each exact cell occurrence, derive ordered operations and residual/time
conditioning dataflow. Reuse canonical norm/activation/residual readers where
possible. Prove Conv1d/2d/3d, kernel/stride/padding, shortcut, output scaling,
temporal branches and blending independently.

Root-level frames evidence may label the root geometry; it cannot stamp Conv3d,
temporal attention or AlphaBlender into a child. Image, video and source-missing
counterexamples are required.

Exit: the ResNet drill is source-derived or opaque; GroupNorm/SiLU/two-Conv/
temporal prose is never conventional.

#### U11-D1 — neutral stage-child occurrence inventory

Before any residual/attention/sampler classification, enumerate every exact
constructed child invocation under every U11-C stage-class candidate. Support
direct, sliced/reversed/enumerated and zip/enumerate(zip) container iteration,
plus direct constructed fields. Preserve every factory/import rival, call site,
guard, loop binding and construction site. Uncalled containers do not enter;
unsupported/unreachable calls remain typed unresolved. This inventory assigns
no child role and remains whole-callable-open.

Status: **IMPLEMENTED, TARGETED-VERIFIED** — synthetic controls cover direct,
plain-loop and paired-container routes, renaming, repeated calls, uncalled
containers, unsupported/unreachable code and cross-construction laundering.
The real installed SDXL source proves exact residual/attention/sampler child
addresses while the DTO exposes no role field. U11-D is not complete until D2
derives the actual per-cell mechanisms and the phase gate is green.

#### U11-D2a — exact local cell mechanisms

Status: **IMPLEMENTED, TARGETED-VERIFIED; EVIDENCE-ONLY** — every exact D1
child-construction occurrence now receives an independent positive mechanism
row. The row derives return-path and local-call operations through the shared
primitive/operation protocols, proves a return-level residual add and output
scaling expression independently, proves additive versus scale/shift side-input
injection from local definition lineage, and retains exact Conv1d/2d/3d
constructor operands. The canonical primitive protocol now recognizes exact
external GroupNorm. D2 applies exact GroupNorm/Dropout only over its own proven
return/local call census; it deliberately does not widen existing projector
consumers before U11-G cutover.

Counterexamples cover a conventional two-convolution residual cell, a one-conv
non-residual cell, additive and scale/shift conditioning, guarded input-branch
projection, Conv3d, full field/class renaming and DTO forgeries. The installed
SDXL witness proves GroupNorm + Conv2d + Dropout + residual/scale evidence from
its exact child occurrences. No parser, renderer, fact, debt, manifest or
gallery surface consumes this evidence yet.

Conv3d remains only a three-dimensional convolution. D2a always publishes
`temporal_axis_unproven`; a later D2 boundary must prove exact frame-axis
lineage through the parent call before any temporal label or AlphaBlender claim
may be projected. Whole-callable CFG coverage also remains open, so missing
operations are not negative facts.

#### U11-D2b — neutral repeated-axis mixing proof

Status: **IMPLEMENTED, TARGETED-VERIFIED; EVIDENCE-ONLY** — a cell may now
publish a structural repeated-axis mix only when one exact side-input-derived
axis count reaches an exact reshape, that reshaped value reaches an exact
internal Conv3d path, and a later exact internal callable arithmetically blends
the dimensional result with the preserved branch. The DTO closes every call,
binding, reshape, Conv3d and blend-return span. A standalone Conv3d and a class
whose name contains both `Temporal` and `AlphaBlender` prove nothing.

The installed `UNetSpatioTemporalConditionModel` witness yields three exact
repeated-axis mix rows for structurally discovered cells. Production still
labels all three `temporal_axis_proven=False`: the structural row must be joined
to U10/root frame-axis lineage during U11-G before temporal language is legal.

#### U11-D committed-tree receipt

Status: **DONE; EVIDENCE-ONLY** — commits `be64cce`, `5efd851`, `bdf2204`
and containment correction `43e4e53` were verified together from committed
tree `43e4e53ab0688ddd55e9e8a822b6c8d23671d478`.

The first phase gate caught a real cross-domain widening: adding GroupNorm and
Dropout to the generic projector-chain protocol changed CogVideoX and MusicGen
evidence surfaces even though U11 was the only intended consumer. The
correction restored the generic protocol's prior contract, retained neutral
return/local-call observations, and made the U11 cell reader interpret those
operations only inside exact U11-owned occurrences. Direct CogVideoX and
MusicGen preservation controls passed after the correction, and the final
phase gate found no structural, pixel or artifact drift.

Final coordinator receipt `00a4c43004`:

- collection: 3,686 tests;
- focused U11-D/primitive/projector lane: 265 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,574 passed, 14 skipped, 2 expected xfails; and
- every lane's tree fingerprint and source-artifact fingerprint was identical
  before and after.

U11-D therefore closes only the exact cell-evidence boundary. Its neutral
repeated-axis rows remain semantically non-temporal until U11-G joins exact
root frame-axis lineage; no renderer/parser cutover has happened early.

### U11-E — attention, nested transformer and FFN cells

Goal: delete the three U11 attention/FFN whole-file readers and the universal
Transformer2D template.

For every exact cell occurrence:

1. prove attention call sites and self/cross input lineage;
2. resolve each exact nested attention/transformer/FFN occurrence;
3. reuse U6 attention, U7 FFN/norm/cell and U8 position/mask facts;
4. preserve multiple attention/FFN kinds within one model by occurrence; and
5. leave simple attention, non-transformer attention and unresolved cells
   distinct.

Exit: block-name `Attn` semantics, same-role unions, hand-authored MHA/no-RoPE/
no-cache/4x-FFN claims and `unet_transformer_ffn_activation_from_files` are
deleted.

Implementation is split so an address proof cannot silently become a role or
selector proof:

- **U11-E1 — exact positive nested-mechanism inventory.** Starting only from
  U11-D2's exact cell occurrences, recursively walk their exact owner graphs,
  retain caller-formal input lineages, and reuse U6/U7 positive mechanism
  readers. Imported framework containers require an exact lexical import,
  declared package root, resolved source symbol and graph occurrence. Rival
  construction routes are preserved and inspected independently; no route is
  selected because it looks familiar. E1 assigns neither self/cross roles nor
  config-selected FFN variants.
- **U11-E2 — selector and interface semantics.** Bind each exact constructor
  operand/default through the construction chain before selecting an FFN
  activation/gating variant or an attention input role. A class default may
  fill an absent operand but may not override a checkpoint declaration. Mixed
  attention/FFN kinds remain occurrence-qualified. Unknown selectors or input
  interfaces remain unknown.
- **U11-E3 — consumer cutover and deletion.** Project the E1/E2 evidence from
  one typed source into parser/IR/cards/expanded/params/conformance, then delete
  the three whole-file readers and the universal Transformer2D compatibility
  template in the same commit series. E1/E2 evidence may not become another
  permanent parallel interpretation path.

  **Binding sequencing correction (post-E2c audit):** E3 is an atomic output
  cutover, not the next substrate unit. U11-B's carried contract still marks
  stage factory/config branch selection open, and E2c's `context_slot` is not a
  runtime external-conditioning proof. The compatibility parser's visible
  config-created stage rows therefore cannot yet be joined occurrence-exactly
  to E1/E2. Cutting over now would either guess that join or erase known output
  into opaque cards merely to satisfy phase numbering. Both violate §§3.4,
  3.6, 3.7 and 3.9. Execute U11-F's exact selector/conditioning/sampler joins
  and U11-G's canonical occurrence projection first; then land E3 and H as one
  migration series that projects the typed graph and deletes the old readers,
  templates and satisfied debt atomically. No legacy reader is permitted to
  gain new behavior during this interval.

#### U11-E1 implementation state

Status: **DONE; EVIDENCE-ONLY** — commit `6b0c979`.

The new boundary begins at exact U11-D2 cell occurrences and adds no
parser/IR/renderer/parameter consumer. It preserves the three guarded
`Transformer2DModel` construction routes exercised by installed SDXL rather
than selecting one. Every route independently reaches exact nested attention
occurrences. Diffusers `Attention` is qualified only through a closed framework
container protocol joined to its exact import source and owner-graph
occurrence; a matching short name or a same-shaped foreign import is powerless.

The same installed source exposes a deliberate E2 dependency: `FeedForward`
selects dense/gated activation implementations from an `activation_fn`
operand. E1 therefore emits no FFN mechanism for SDXL. It does not manufacture
GEGLU, gating, a four-times expansion or a familiar default from class
spelling. E2 must prove the exact checkpoint/default-to-constructor-to-selected-
implementation chain before that mechanism becomes drawable.

#### U11-E1 committed-tree receipt

Commit `6b0c9790e2345c240256fade5fcd60843116caf8` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, eight changed production Python files clean;
- collection: 3,695 tests;
- focused U11 nested-mechanism and prerequisite lane: 378 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,583 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint and source-artifact fingerprint was
  identical before and after.

Logs: `/private/tmp/model-unfolder-verification/92ba7aaac7`.

E1 therefore establishes only exact positive nested mechanism inventory. It
does not claim that a constructor selector is architectural evidence, does not
assign self/cross attention roles, and does not project any new structure.

#### U11-E2a implementation state and committed-tree receipt

Status: **DONE; VALUE-TRANSPORT ONLY** — commit `eb9e2f8`.

E2a adds the mechanism-neutral constructor-value boundary required before any
selector can author architecture. It validates exact Python call binding and
transports only a literal actual, omitted literal class default, exact parent
formal, or exact registered `self.config.<formal>` access through an explicit
construction-frame chain. The registration reader is occurrence-qualified;
the old duplicate private path recognizer is deleted. Positional-only formals
are preserved by ProgramIndex. No selector token is interpreted and no
mechanism, fact, IR field, card, parameter estimate or rendered pixel changes.

The installed SDXL witness preserves all three rival
`Transformer2DModel -> BasicTransformerBlock -> FeedForward` construction
routes. Each independently resolves the effective operand to `"geglu"` through
the same exact three-step route: child-formal forward, registered-config
forward, then the Transformer's literal class default. This proves a runtime
operand, not GEGLU semantics; E2b must still prove the selected implementation
from code.

Commit `eb9e2f8fdf20f17be3d140f3243f742af209335d` was verified from
isolated worktrees by `scripts/verify_commit.py`:

- static: PASS, seven changed production Python files clean;
- collection: 3,713 tests;
- focused constructor/config/index/nested-mechanism lane: 255 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,601 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint and source-artifact fingerprint was
  identical before and after.

Logs: `/private/tmp/model-unfolder-verification/06f4f53eef`.

#### U11-E2b implementation state and committed-tree receipt

Status: **DONE; SELECTED FFN MECHANISM EVIDENCE ONLY** — commit `980a9fe`.

E2b joins E2a's exact runtime operand to the complete selected implementation
expression and then proves the implementation's operation protocol from source.
It does not assign semantics from the selector token, implementation class name,
or model identity.  Every installed-SDXL rival route independently selects the
same fused gate/up projection, proves an exact two-way last-axis split, proves
the activation is applied to exactly one half, proves elementwise gating, and
proves the down projection.  A final self-review found that accepting
``chunk(2)`` without its axis could falsely certify a non-channel split; the
reader now requires exact ``chunk(2, dim=-1)`` (keyword or positional), with
four wrong-axis/missing-axis controls permanently pinning the correction.

No parser, fact, IR, renderer, parameter, debt-register, manifest or gallery
consumer changed.  The selected mechanism remains evidence-only until E3's
single-source consumer cutover.

Commit `980a9fe6a1c7a8965059b7d7291ee8160a5890f2` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, five changed production Python files clean;
- collection: 3,752 tests;
- focused selector/FFN/nested-mechanism lane: 305 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,640 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint and source-artifact fingerprint was
  identical before and after.

Logs: `/private/tmp/model-unfolder-verification/419a9c2663`.

#### U11-E2c implementation state and committed-tree receipt

Status: **DONE; ATTENTION INPUT-ROLE EVIDENCE ONLY** — commit `c6f0ae8`.

E2c proves an attention implementation's input interface from its exact source
instead of from a familiar container/class/field spelling.  Every default
implementation alternative independently proves its Q/K/V producer boundary,
with Q descending from one callable formal, K/V from the same distinct optional
formal, and an exact `None -> primary` fallback.  The container's exact
constructor/default/installer/delegate route must preserve that interface
unanimously.  A second callable on the same class is not treated as a
transparent wrapper: if the positive compute proof and entry interface belong
to different callables, the result remains typed unknown until a future exact
wrapper-binding boundary joins them.

At the exact parent occurrence, the lane invocation is then classified only as
`self`, `context_slot` or `conditional`.  `context_slot` means that the
source-level K/V interface can consume a distinct caller formal; it deliberately
does **not** claim that non-`None` external conditioning reaches that slot at
runtime.  U11-F owns that later component/config/execution join, after which a
projected cross-attention label may become legal.  Constructor-decidable branch
and guard values are occurrence-qualified, sibling field guards cannot clear a
lane, and every exact constructor alternative for a lineage-transparent norm
call must independently classify as LayerNorm or RMSNorm.  One opaque, dynamic,
non-norm or unknown rival blocks transparency.

The installed SDXL source provides the permanent real control.  Its three exact
`BasicTransformerBlock` construction routes each prove the direct first
attention lane as `conditional` (self versus distinct context slot).  The three
direct second lanes remain typed `incomplete_graph`: an optional runtime
GLIGEN/fuser transform may replace their primary state before invocation.  The
nested fuser attention belongs to its own child occurrence and cannot be
laundered through the parent frame.  This 3-proven / 3-unknown partition is the
honest E2c output; no conventional self+cross template is inferred.

The first committed-tree gate caught a real containment defect.  E2c initially
added tuple/membership/boolean evaluation to the shared config-expression
evaluator by default, which widened an unrelated production reader and changed
Flux-2 IR, ledgers and HTML metadata.  No re-bless was accepted.  The capability
is now explicitly opt-in for E2c constructor evidence, while the default shared
evaluator retains its former conservative contract.  A poison pins that default
refusal, and the exact Flux witness returned to zero drift before the final
gate.

Commit `c6f0ae8c97015635522c7aa47d929ab18c01c050` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, 17 changed production/test Python files clean;
- collection: 3,830 tests;
- focused attention-interface/role/constructor/norm/SDXL lane: 282 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with all 29 witnesses at zero structural,
  evidence, HTML, gallery and pixel drift;
- full suite: 3,718 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint and source-artifact fingerprint was
  identical before and after.

Logs: `/private/tmp/model-unfolder-verification/c6834d87b3`.

E2c changes no parser, fact, IR, renderer, parameter, debt, manifest or gallery
consumer.  E1/E2a/E2b/E2c are now a closed evidence substrate; E3 must project
them through one typed consumer path and delete the three legacy whole-file
readers plus the universal nested Transformer2D template atomically after F/G
make the occurrence joins provable.  Until then those legacy paths are frozen:
they may not be extended, patched for another model or treated as evidence for
the new projection.

### U11-F — exact samplers, bookends and conditioning

Goal: bind the remaining UNet-specific structural/config debt.

Derive exact input/output projections, centering, timestep/class/additional
embedding graph, encoder projection, per-stage external K/V routes and exact
down/up sampler operations. Config values supply channels, dimensions, selected
options and scale factors only after the source use is proven.

Do not absorb scheduler or VAE semantics (U13/U12), and do not reinterpret U10
transformer-root semantics as UNet internals.

Implementation order is binding:

1. **U11-F1 — occurrence-exact stage selection.** For every U11-B factory or
   constructor alternative, bind the exact root config occurrence to the exact
   constructor/formal/loop/factory guard that selects it. Preserve each stage
   position as an occurrence; a symbolic repeated template stays symbolic until
   the exact checkpoint count/list occurrence instantiates it. A block token or
   candidate class spelling remains an address operand only—the selected class's
   source proves its mechanism. Unequal aliases, short/long lists, rival factory
   branches, dynamic tokens and source-missing paths remain typed ambiguous or
   incomplete. This unit is the prerequisite for mapping E1/E2 mechanisms onto
   visible down/direct/up occurrences.
2. **U11-F2 — runtime attention-source join.** Starting from E2c's exact lane
   call and `self`/`context_slot`/`conditional` role, prove whether a non-`None`
   external value reaches that exact context formal for that selected stage
   occurrence. Join the exact parent/root formal lineage, component slot,
   checkpoint/config guard and bookend application. Only this join may project
   `cross_attention=True` and a K/V source. A context-capable lane with no proven
   runtime source remains `context_slot`/conditional, never conventional cross.
3. **U11-F3 — exact sampler and cell-count selection.** Bind down/up sampler
   calls and per-stage child repetitions from execution plus constructor/config
   operands. Do not infer all-but-final sampling, mirrored counts or the legacy
   up-path `+1` rule. Preserve symbolic count expressions and partial lists.
4. **U11-F4 — bookends and conditioning operations.** Prove input/output
   projections, centering, timestep/class/additional embedding applications,
   encoder projections and other conditioning arithmetic from exact calls and
   dataflow. Reuse U10 bookend/conditioning evidence where it already closes
   the same occurrence; do not create a U-Net copy.

F1–F4 are evidence/config-binding units only. They may publish typed results
and projection obligations, but parser/renderer/parameter output remains frozen
until U11-G constructs the one canonical graph. Each unit needs renamed-class,
same-class-two-occurrences, conflicting-selector, truncated-list,
missing-source, runtime-`None`, sibling-conditioning and image/video controls.
The real SDXL matrix must retain all exact rivals until F1 selects them and must
not upgrade E2c's three unresolved second lanes before F2 proves their runtime
source path.

#### U11-F1 implementation state and committed-tree receipt

Status: **DONE; EVIDENCE-ONLY** — commit `e3aa738`.

`read_unet_stage_selection` now closes the exact chain from an imported
`register_to_config` constructor formal, through the checkpoint-declared list
occurrence, exact `for`/`enumerate` target, exact factory actual/formal binding,
and exact guarded factory return.  Every list position is retained as its own
construction occurrence.  Repeating the same class twice therefore produces
two positions rather than one class-level fact; short and long lists remain
their actual lengths rather than being mirrored or filled.  Tokens and selected
class symbols remain address evidence only.

The installed SDXL control resolves exactly six rows: three checkpoint-ordered
`down_blocks` positions and three checkpoint-ordered `up_blocks` positions,
including repeated cross-attention block classes as distinct occurrences.  It
retains no unresolved candidate for that witness.  Missing checkpoint fields,
class defaults, scalar selectors, local selector transforms, nested unrelated
loops, dynamic guards, rival live returns, incomplete factory censuses and
direct multi-template containers remain explicit partial evidence.  Full
class/formal/container/local renaming preserves the structural join.

The one neutral source normalization needed by installed Diffusers—literal
`str.startswith` plus exact integer/slice access—is separately opt-in on the
shared expression evaluator.  Its default contract remains unchanged and is
pinned by a refusal control, so F1 cannot widen existing readers as E2c's first
draft did.

Coordinator receipt `3e338d2b29` on
`e3aa7388f30a274efc3111cbecf8791d2f37ce2e`:

- static: PASS, four changed Python files clean;
- collection: 3,849 tests;
- focused F1/expression/U11 prerequisite lane: 262 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with all 29 witnesses at zero drift;
- full suite: 3,737 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree and source-artifact fingerprints were identical.

F1 has no parser, fact, IR, renderer, parameter, debt, manifest or gallery
consumer.  U11-F2 must now prove the non-`None` runtime source reaching each
E2c context-capable lane; F1 selection alone cannot label any lane cross-attention.

#### U11-F2a implementation state and sequencing correction

Status: **DONE; EVIDENCE-ONLY FOUNDATION** — commit `e0fb168`; runtime
projection remains forbidden pending F3 and F2b.

The first F2 implementation pass established the exact neutral rail rather
than forcing SDXL through a familiar cross-attention template:

- `FormalBindingEdge` proves one exact caller formal → call actual → callee
  formal edge with complete call/dataflow provenance.  `FormalSourceRoute`
  composes only contiguous edges and cannot skip a callable or merge roots.
  This is deliberately a source-template relation inside one addressed
  component; the surrounding F1/D/E evidence, not this neutral rail, supplies
  runtime construction-occurrence identity.
- Expanded `**kwargs` may coexist with an explicitly supplied target formal
  because Python raises on a duplicate; an expanded mapping alone cannot prove
  an omitted target.
- A non-`None` external interface contract requires both a required formal and
  a non-optional source annotation.  A missing annotation, `Optional`,
  `T | None`, `=None`, literal `None`, multiple roots or unresolved guard stays
  unproven.
- E1's `AlternativeCellRoot` now retains the authoritative runtime invocation
  that produced its rival constructor-container route.  Previously it retained
  the constructor sites but omitted the exact Transformer2D→block call needed
  by F2.
- Constructor fields now have a typed neutral derived-expression proof for one
  unguarded expression over exact constructor formals.  This closes literal
  boolean formulas without adding field-name or model-family semantics.

The real SDXL counterexample exposed two genuine prerequisites, so the original
F2→F3 order is corrected to **F2a rail → F3 occurrence/config operands → F2b
closure → F4**:

1. The selected Transformer2D occurrence declares
   `is_input_continuous = (in_channels is not None) and (patch_size is None)`.
   Its exact `in_channels` value is forwarded from a selected stage position;
   F1 binds only the stage-class selector, not the parallel per-position
   constructor operands.  Without F3's occurrence/config binding, the patched
   input branch can still rewrite `encoder_hidden_states`, so F2 must not clear
   that rival.
2. The down path iterates `list(zip(self.resnets, self.attentions))`.  U11-D1
   does not yet bind that derived iterable to the two source containers, so its
   nested attention occurrence is absent.  F3 must add a neutral exact zip
   binding; F2 may not infer the missing down-path lane from the selected class
   or its `has_cross_attention` field.

Accordingly the installed SDXL control currently emits **zero** runtime-source
claims and 18 typed `lane_route_unresolved` rows for the source-visible up-path
alternatives.  The optional fuser's own lane is separately rejected by exact
occurrence mismatch.  This zero is an anti-laundering pin, not the desired final
architecture.  F2b may turn rows into external K/V routes only after F3 closes
the two prerequisites above.  No parser, fact, IR, renderer, parameter, debt,
manifest or gallery consumer may use F2a directly.

Coordinator receipt `7f327519ac` on
`e0fb1680204eaf7511a4c92038e386f688d5d53c`:

- static: PASS, eight changed Python files clean;
- collection: 3,865 tests;
- focused F2a/formal-route/constructor/nested-mechanism lane: 235 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,753 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree and source-artifact fingerprints were identical.

#### U11-F3a implementation state

Status: **ACTIVE; SELECTED-OPERAND AND ITERATION-ADDRESS FOUNDATION** — no
sampler/cell-count architectural claim and no production consumer yet.

F3a closes the two neutral prerequisites exposed by F2a without turning either
one into architecture:

- a callable-local iterable such as
  `pairs = list(zip(self.left, self.right)); for left, right in pairs` is now
  bound only through the exact reaching assignment, exact loop target and
  lexically unshadowed Python wrapper names. Conditional, reassigned, cyclic,
  stale, unused or shadowed routes remain `unsupported_iteration`. The carried
  aliases and wrapper spellings recompute against the final ProgramIndex.
- every F1-selected factory call now has an occurrence-exact operand inventory.
  F1 retains the prepared root document that authored the selection, and F3
  accepts no second independently supplied document.
  Values are evaluated from the registered checkpoint document, exact loop
  position/value, positively selected guards and exact local definition
  lineage. A short parallel list is never filled; later loop iterations are
  never inferred from a single lexical pass; class/default/config provenance
  remains explicit. Builtin sequence protocols are a closed opt-in set and
  every callable **and** `isinstance` type spelling must be lexically
  unshadowed.

The installed SDXL control currently proves, independently for all six selected
stages, exact output channels `(320, 640, 1280)` down and `(1280, 640, 320)`
up, cell-count operands `(2, 2, 2)` down and `(3, 3, 3)` up, external context
width `2048`, and sampler-construction flags `(true, true, false)` on both
paths. Later down-path input channels remain typed unresolved because source
updates the previous-channel local across loop iterations; F3a does not yet
provide a recurrence proof. These are constructor operands only. F3b must join
them to the selected stage constructor and its exact executed child/container
evidence before a cell count or sampler-presence fact exists. F2b may then use
that same selected-constructor environment to close the nested attention route.

### U11-G — canonical projection and consumer cutover

Build one closed typed projection over U11-A..F. It must carry exact owners,
statuses, provenance, unresolved relations and fact routes. Project it to:

- `ModelIR` / `extras` compatibility view;
- cards and drill DTOs;
- HTML/SVG layout input;
- expanded JSON;
- owner-bound parameter terms; and
- Sable/conformance expectations.

Every projected structural leaf needs a registered fact/debt disposition and a
real-consumer receipt. Reverse fabrication is owner-qualified.

### U11-H — delete the compatibility interpreter

Delete or reduce to layout-only:

- `adapters/diffusor/unet.py::is_unet` and `parse_unet`;
- all config/default structural construction in that module;
- all five U11 quarantined readers and their raw AST/import-closure helpers;
- renderer-local stage/cell graph authorship in
  `renderers/html/block_views/unet.py`;
- U11-owned legacy parse-authority and `StructuralDebt` rows whose conditions
  are satisfied; and
- any obsolete config vocabulary/table that selects UNet structure.

No old fallback may remain reachable when the new projection is incomplete.

### U11-I — qualification, artifacts and closure

Run the qualification and artifact bracket in §8, inspect every intentional
delta, rebuild only approved baselines/galleries, run the committed-tree
coordinator, and update this plan plus the master tracker with exact commits and
receipts.

---

## 6. Required counterexample matrix

Each mechanism needs positive, negative, ambiguity, source-missing and
equivalent-control coverage. At minimum:

| Axis | Required controls |
|---|---|
| outer shape | U-shaped execution with neutral names; class named UNet with a flat stack; config stage lists with no U-shape |
| import closure | relative import; alias; re-export; same-name unimported class; cycle; broken imported file; package escape; content change with preserved mtime |
| stage construction | direct class; imported factory; helper return; guarded rivals; same class at two sites; symbolic comprehension; dynamic token |
| mid | executed mid; constructed-but-never-called mid; no mid; guarded/ambiguous mid; config declares a mid absent from code |
| counts | scalar, list, symbolic expression, class default, conflicting config aliases, missing operand |
| attention | attention-free; simple cross-attention; nested transformer; self+cross; multiple kinds in one model; unresolved imported cell |
| ResNet | conventional 2-D; custom activation/norm; one convolution; no time injection; learned shortcut; different residual scaling; temporal/3-D |
| sampling | sampler in each stage; sparse/custom placement; custom scale; no sampler; constructed-but-not-executed sampler |
| skips | tuple accumulation+slicing; stack push/pop; concat; add; multiple skip lanes; unmatched producer/consumer; guarded rival |
| conditioning | text K/V; image K/V; class; timestep; added conditioning; absent conditioning; unused config declaration |
| ownership | sibling component with same class/field names; same class at multiple occurrences; embedded U-Net; source-file order reversal |
| unknown | missing source; partial import closure; parse failure; unsupported control flow; unresolved factory; renderer cannot strengthen |

The real matrix must include SDXL plus installed-source examples where available.
Kandinsky-like, temporal and custom-source cases may be synthetic until they are
added as frozen real witnesses; synthetic coverage never substitutes for a
blessed real regression once one exists.

---

## 7. Foresight: likely failure modes

1. **Import closure becomes a new whole-package union.** Prevent with exact used-
   import edges, component-qualified source IDs and an unimported-same-name poison.
2. **Factory token becomes semantics by indirection.** It may select a factory
   branch/class only; the selected class still needs its own operation proof.
3. **One class erases occurrence differences.** Every fact key and config binding
   includes the construction occurrence.
4. **Storage order is mistaken for execution order.** Construction and execution
   DTOs stay separate; only forward/dataflow edges create the DAG.
5. **One SDXL witness overfits the reader.** Synthetic renaming/collision/rival
   controls are mandatory before production cutover.
6. **Unknown cells inherit renderer defaults.** Opaque nodes carry no conventional
   child cards or prose; cross-surface unknown-safety tests cover every consumer.
7. **Nested transformer evidence is unioned across cells.** U6/U7/U8 readers take
   exact nested occurrences, never a list of class names/files.
8. **Config defaults outrank checkpoint declarations.** Winning operands retain
   event/fact provenance; arbitration is exact-path and owner-qualified.
9. **Skip symmetry hides unequal architectures.** Producers and consumers are
   joined by explicit value lineage, never mirrored indices.
10. **Params reconstruct what drawing left unknown.** Parameter terms require the
    same owner/fact route; unresolved terms remain named incompleteness for U14.
11. **Renderer remains a second graph author.** Static gates forbid raw config,
    source reads and structural conditionals over legacy UNet keys in renderer
    modules.
12. **The migration grows forever.** Each phase has a finite deletion receipt;
    new capability work outside the phase stops and is recorded as a prerequisite.

---

## 8. Verification discipline

Every phase commit must pass, on its committed tree:

1. focused unit/poison tests;
2. affected U2 authority and debt ratchets;
3. U3 owner/index/execution boundary tests;
4. U6/U7/U8 canonical mechanism tests for every reused reader;
5. all diffusion and conformance tests;
6. the 29-witness preservation bracket;
7. representative real unfolds (at least SDXL plus a flat diffusion negative and
   a transformer/composite cross-domain control);
8. full suite;
9. static identity/config-author/source-reopen/broad-exception gates;
10. isolated committed-worktree verification; and
11. identical source/artifact fingerprints before and after every read-only gate.

Use the parallel verification coordinator for independent lanes. Reserve a
serial full-suite run for phase boundaries and final closure rather than every
small implementation edit. A green test is invalid if the tree changed while it
ran.

Artifact changes require:

- exact before/after semantic-view and PNG occurrence hashes;
- named explanation for every changed view;
- byte-identical confirmation for unaffected witnesses;
- Soumil's explicit approval before re-blessing; and
- committed-tree rerun after the approved rebuild.

---

## 9. Stop conditions

Stop and report before proceeding if:

- an exact stage/cell requires model/family/class substring semantics;
- the one ProgramIndex cannot express a required source address and a new neutral
  observation record would be needed;
- a weak config/class-default fact would be projected as a stronger mechanism;
- a renderer or parameter path needs raw config/source access;
- a phase requires modifying U12/U13/U14 semantics;
- a real model contradicts the proposed general rule;
- a preservation delta has no exact evidence-level explanation;
- an allowlist/debt row would grow without owner, reason and deletion phase; or
- the verification tree changes during a gate.

---

## 10. Completion tracker

| Phase | Status | Receipt / outcome |
|---|---|---|
| reconnaissance | DONE | U10 handoff, compatibility interpreter, five readers, debt and SDXL source-boundary gap verified |
| U11-A1 demand-driven source address | DONE | `51d016e`; 15 exact-boundary controls including real installed SDXL `get_down_block`; committed-tree coordinator: focused 297, U2 authority 44, preservation 52, full 3,510 passed / 14 skipped / 2 xfailed; every lane fingerprint-identical; no consumer/output change |
| U11-A2 legacy closure cutover | PENDING | lands with the U11-B/E consumers; deletes `_augment_diffusion_files` only after exact coverage |
| U11-B stage construction | DONE | `0e5f71a`; exact U10 container→producer→append relation; exact U11-A1 factory expansion; all guarded returned class candidates preserved; 19 synthetic/closure controls + real installed SDXL; committed-tree coordinator: focused 269, U2 authority 44, preservation 52, full 3,530 passed / 14 skipped / 2 xfailed; every lane fingerprint-identical; no consumer/output change |
| U11-C execution/skip DAG | DONE | `d7f7bae`; exact repeated-stage nodes; exact guarded inter-loop constructed-call occurrences; one U10 skip edge; unsupported/unreachable calls and all non-proven order remain typed unresolved; real SDXL retains both guarded direct invocations without using `mid_block` semantics; committed-tree coordinator: focused 362, U2 authority 44, preservation 52, full 3,553 passed / 14 skipped / 2 xfailed; every lane fingerprint-identical; no consumer/output change |
| U11-D ResNet/temporal cells | DONE — EVIDENCE-ONLY | `be64cce` + `5efd851` + `bdf2204` + containment correction `43e4e53`; exact child inventory, local mechanisms and neutral reshape→Conv3d→blend proof; real SDXL and spatio-temporal UNet qualify without role spellings; initial gate caught and removed cross-domain projector widening; final coordinator: focused 265, U2 authority 44, preservation 52, full 3,574 passed / 14 skipped / 2 xfailed, every fingerprint and source-artifact digest identical; root frame-axis semantic join remains correctly deferred to U11-G |
| U11-E attention/transformer/FFN | ACTIVE — E1 + E2a + E2b DONE, E2c IN PROGRESS | `6b0c979` exact nested-mechanism inventory; `eb9e2f8` mechanism-neutral exact constructor-value transport; `980a9fe` code-selected fused gated FFN proof with exact last-axis split; all three installed-SDXL transformer rivals independently resolve the same operand and mechanism without token/class/model semantics; E2b coordinator: focused 305, U2 authority 44, preservation 52, full 3,640 passed / 14 skipped / 2 xfailed, every tree/artifact fingerprint identical; attention input roles and all production consumers remain unprojected |
| U11-F sampler/bookend/conditioning | ACTIVE | F1 DONE (`e3aa738`); F2a rail built, F3 prerequisites then F2b/F4 pending |
| U11-G projection/cutover | PENDING | — |
| U11-H legacy deletion | PENDING | — |
| U11-I qualification/artifacts | PENDING | — |

U11 remains ACTIVE until all phases are complete, U11-owned quarantine/debt is
zero (or reassigned with explicit later-unit ownership), approved artifacts are
rebuilt, and the committed-tree acceptance bracket is green.

### U11-A1 committed-tree receipt

Commit `51d016ef916e8f827d780e8e68160f3f7c3ccc08` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, four changed production Python files clean;
- collection: 3,622 tests;
- focused import/root/conformance lane: 297 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,510 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint was identical before and after.

Logs: `/private/tmp/model-unfolder-verification/3954fc72d0`.

### U11-B committed-tree receipt

Commit `0e5f71abab7da44ec7afd2b55717f105c47889d7` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, two changed production Python files clean;
- collection: 3,642 tests;
- focused owner/import/container/topology/construction lane: 269 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,530 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint was identical before and after.

Logs: `/private/tmp/model-unfolder-verification/2324e8e02a`.

### U11-C committed-tree receipt

Commit `d7f7baebd9dc491ac9ae7ce0d7c4af70459039b2` was verified from isolated
worktrees by `scripts/verify_commit.py`:

- static: PASS, four changed production Python files clean;
- collection: 3,665 tests;
- focused U10/U11 ownership, construction and execution lane: 362 passed;
- affected U2 authority lane: 44 passed;
- preservation lane: 52 passed with zero structural/pixel drift;
- full suite: 3,553 passed, 14 skipped, 2 expected xfails; and
- every lane's complete-tree fingerprint was identical before and after.

The gate also pins the two audit corrections found before commit: a call after
an unconditional return and a call inside an unsupported expression remain
typed unresolved evidence and can never be promoted to a stage node. Calls
nested inside the return expression itself remain lawful local call evidence.

Logs: `/private/tmp/model-unfolder-verification/7e60324efc`.
