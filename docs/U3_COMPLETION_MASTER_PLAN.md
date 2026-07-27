# U3 Completion Master Plan — Program Authority Without Semantic Scope Creep

> **Status:** Ratified by Soumil on 2026-07-27; binding for U3 completion.
>
> **Authority:** This is the unit-level completion plan for U3. It is subordinate
> to `docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md` and does not replace the
> U0→U15 sequence. Once Soumil ratifies it, it supersedes conflicting U3 status,
> sequencing, and completion claims in `docs/U3_RUNBOOK.md` and
> `docs/U3_D_TO_H_EXECUTION_AND_VET_PLAN.md`.
>
> **Purpose:** Finish the one source-program/owner substrate, stop expanding U3
> into mechanism implementation, classify the useful semantic work already
> produced, and create a finite handoff to U4–U15.

---

## 1. One-line finding

The recurring defect was not a shortage of recognizers; it was that multiple
readers independently scanned unions of files/classes, selected their own
plausible owner, interpreted a mechanism, and allowed parser, renderer,
parameters, and conformance to reconstruct different architecture from that
unshared decision.

The correction is:

```text
one source bundle
  -> one neutral ProgramIndex
  -> one exact construction-occurrence owner
  -> one typed mechanism fact
  -> every structural consumer projects that same fact
```

U3 owns the first three steps and the typed failure boundary. U6–U13 own
mechanism meaning. U14 owns JSON/parameters/conformance projection. U15 deletes
the final legacy/config authority.

---

## 2. Critical verdict on the current direction

### 2.1 What is architecturally right

The following direction is correct and should remain:

- `ProgramIndex` is observation-only and is built once per `SourceBundle`;
- file content, not mtime, identifies indexed source;
- source/class names can locate an address but cannot prove mechanism meaning;
- ownership is a construction-site occurrence chain, not a class name;
- two instances of one class remain different owners;
- parse failure, unsupported syntax, absence, ambiguity, and incompleteness are
  distinct typed states;
- unresolved and rival producer candidates are retained rather than discarded;
- a specialized reader starts from an exact owner and may only consume evidence
  belonging to that owner;
- parser and conformance may share evidence, but conformance must compare that
  evidence independently against the projected artifact;
- renderers and parameter estimators never inspect source, config, owner graphs,
  or reader results directly.

### 2.2 What drifted

The recent campaign used the label `U3-F` for two different jobs:

1. **U3 infrastructure:** owner resolution, exact occurrences, local dataflow,
   factory/wrapper descent, typed ambiguity and shared program observations.
2. **Mechanism semantics:** attention storage, QK norm, FFN shape, expert
   storage, normalization topology, projection bias, learned sinks, schedules,
   codebook aggregation, weight tying, and conformance conclusions.

The project master plan assigns the second job to U6–U9 and U14. The code can be
useful and correct while its unit classification is wrong. Continuing under the
current label would make U3 indefinitely absorb every architecture problem.

### 2.3 The sequencing conflict in the existing plans

The old U3 completion wording requires every legacy reader to be migrated and
deleted before U4. Many of those readers currently author semantics that U6–U13
are explicitly supposed to redesign. Mechanically porting their old semantics
in U3 and then replacing them in U6–U13 would implement the same surface twice.

This plan resolves the conflict:

- U3 finishes the neutral kernel, exact ownership, typed failures, and a
  non-growing quarantine of surviving legacy semantic readers.
- U4 makes unknown safe.
- U6–U13 replace each quarantined semantic reader once, using the U3 substrate.
- U14 cuts over JSON, parameters, and conformance.
- U15 proves that the quarantine and semantic configuration authority are zero.

This is a proposed amendment to the U3 `Done means` clause in the project master
plan. It requires Soumil's ratification before U3 is marked complete.

---

## 3. Target architecture and dependency firewall

### Layer A — source acquisition

**Owns:** model ID resolution, downloaded/local source bundle, component slots,
external-file provenance.

**May use identity for:** locating a declared class/module/file.

**May not author:** attention, FFN, norm, position, topology, fusion, scheduler,
or any other mechanism fact.

Primary surfaces:

- `model_unfolder/evidence/sources.py`
- `model_unfolder/evidence/context.py`
- adapter loaders

### Layer B — neutral program observation

**Owns:** syntax and address observations only:

- imports and exact symbol bindings;
- classes, bases, assignments and construction sites;
- callables, calls, returns, branches, loops and comprehensions;
- exact spans and callable-local ordering;
- config-path reads without reading checkpoint values;
- producer candidates, unresolved relations and unsupported regions.

Primary surface:

- `model_unfolder/evidence/program_index.py`

`ProgramIndex` must not contain `is_attention`, `is_ffn`, `is_rope`,
`is_decoder`, family names, renderer labels, or conventional defaults.

### Layer C — exact ownership and occurrence resolution

**Owns:** component root, declared stage, construction graph, repeated-container
addresses, exact child occurrences, wrapper/factory return addresses and
config-prefix bindings.

Primary surfaces:

- `model_unfolder/evidence/component_owner.py`
- `model_unfolder/evidence/container_inventory.py`
- `model_unfolder/evidence/repeated_child.py`
- `model_unfolder/evidence/delegated_stage.py`
- `model_unfolder/evidence/decoder_block.py`
- `model_unfolder/evidence/output_repeated_stage.py`
- `model_unfolder/evidence/config_scoped_owner.py`

An address result answers **which occurrence**. It cannot answer **what mechanism
that occurrence implements**.

### Layer D — mechanism interpretation

**Owns:** attention, FFN, norm, position, selector, modality, diffusion, UNet,
VAE and scheduler facts.

This is primarily U6–U13 work. Each reader:

- accepts an exact owner/occurrence;
- consumes ProgramIndex observations and exact config premises;
- returns a closed `ReaderResult[T]`;
- emits an owner-qualified fact with provenance and completeness;
- never searches sibling classes or the full file bundle for a vote.

### Layer E — canonical facts and structure

**Owns:** arbitration, fact registry, canonical regions/IR and projection
obligations.

The parser may coordinate facts. It may not independently rediscover a mechanism
already owned by a reader.

### Layer F — consumers

Renderers, expanded JSON, parameters and conformance consume canonical
facts/regions. They do not:

- inspect config or model source;
- infer architecture from missing fields;
- select a family/class;
- manufacture a default mechanism;
- certify a value they authored themselves.

---

## 4. Classification of the 35 local commits

The current branch is 35 commits ahead of its upstream baseline. Before pushing,
classify them by the project unit whose acceptance criteria they must satisfy.

### Ratified commit attribution

| Commit(s) | Credited unit | Truthful scope |
|---|---|---|
| `3b1c003`, `c194042` | U3 | immutable ProgramIndex observation/address caches |
| `4bb8abe`, `887578a`, `ed86068` | U3 | shared exact block address, return delegation and ambiguity |
| `4bf5c56`, `c47e25b`, `facc5ef`, `b171d80` | U3 | neutral comprehension and producer-candidate observations |
| `bb1b4b1`, `243f4a2` | U3 | output-reaching repeated stage and wrapper address boundaries |
| `50a115b` | U3 support | migration-debt reduction lock |
| `78ab271`, `4958a8f`, `d4d8b2f`, `306fd8e` | U6 candidate | attention storage, QK norm and learned sinks |
| `43df77c`, `c37bbb5`, `80fc2c3`, `ddbd419` | U7 candidate | exact ordinary FFN mechanism and cutover |
| `d96c411`, `027a1da` | U7 candidate | routed-expert storage and registry correction |
| `6effc34`, `d9740c3`, `1ea4036` | U7 candidate | norm/parallel topology and retired FFN unions |
| `a788288` | U6/U7 correction | attention/FFN projection bias plus norm/expert corrections |
| `14b096f` | U7 root-bookend candidate | exact manual weight tying |
| `0bbcbb7` | U8 candidate | exact additive cross-attention schedule |
| `b848808` | U9 candidate | codebook stream aggregation |
| `703274d` | mixed, held | U3 output-owner routing plus unblessed U6/U7/U9 CLIP projection |
| `4102e1c` | U14 candidate | nested storage conformance cutover |
| `10fec07`, `a4b2da1`, `2382dd1`, `03eb48e` | documentation | receipts/status for their immediately preceding slices |

The mixed `703274d` commit is intentionally not rewritten, but its semantic
delta cannot satisfy U3 acceptance. It remains held until the relevant future
unit and visual receipt accept it.

### U3-proper infrastructure or migration substrate

- ProgramIndex observation/address caches;
- shared selected decoder-block address path;
- exact decoder return delegation;
- preservation of stack ambiguity;
- comprehension binding;
- unresolved-call producer invalidation;
- transformed and rival producer-candidate retention;
- output-reaching repeated-stage address;
- unique output-contributing wrapper descent.

These are U3 when they remain neutral and do not author mechanism meaning.

### U6 attention candidates

- attention storage;
- Q/K normalization;
- attention projection bias;
- learned attention sinks;
- attention score/dataflow facts.

These require U6 witness and fact-family acceptance even if implemented while
building U3.

### U7 FFN/norm/cell candidates

- ordinary FFN shape and activation;
- routed-expert storage;
- decoder norm primitive and placement;
- ordinary-FFN projection bias;
- parallel-branch normalization.

These require U7 acceptance.

### U8 selector/schedule candidates

- additive cross-attention scheduling;
- any per-layer/mask/position/router schedule conclusion.

These require U8 acceptance.

### U9 recursive-modality candidates

- codebook stream aggregation;
- recursive text/audio/vision owner use;
- any modality-specific projection of shared mechanism facts.

Manual weight tying is a valid semantic candidate but does not yet have an
unambiguous unit assignment in the current master map. Assign it explicitly
before it is credited as complete; do not hide it under U3.

### U14 consumer/conformance candidates

- nested text-encoder storage conformance cutover.

Sharing exact readers is correct, but final conformance authority is U14.

### Decision on the current CLIP delta

The exact structured-output path exposed source-backed `LayerNorm` and
attention-bias facts in embedded CLIP towers. That is a semantic/visible change,
not a neutral U3 migration.

The neutral output-owner boundary remains U3. The source-backed `LayerNorm` and
attention-bias projection is retained as a U6/U7/U9 candidate, remains
unblessed, and cannot be credited to the U3 completion receipt. U3 alone may not
bless the delta.

---

## 5. Exact remaining U3 work

### U3-C0 — reconcile and freeze the current local chain

Files:

- `docs/U3_RUNBOOK.md`
- this document
- commit history from upstream baseline through current `HEAD`

Actions:

1. Produce a commit-to-unit table for all 35 local commits.
2. Separate neutral U3 infrastructure from U6–U9/U14 semantic candidates.
3. Record every intentional output delta and its owning future unit.
4. Do not push or bless artifacts until the classification is approved.
5. Preserve unrelated user-owned documentation changes outside the chain.

Achieved output:

> Every local commit has truthful attribution (including an explicit mixed
> classification where history is intentionally retained), and no semantic
> change is smuggled into U3 as infrastructure.

### U3-C1 — close exact factory/constructor binding

Known blocker:

`_from_config`/factory-created components can be addressed as returned children,
but constructor-parameter/config-prefix binding is not always proven. This is
why an exact nested tower can remain unable to prove its FFN activation.

Primary files:

- `model_unfolder/evidence/program_index.py`
- `model_unfolder/evidence/component_owner.py`
- `model_unfolder/evidence/construction_calls.py`
- `model_unfolder/evidence/config_scoped_owner.py`
- focused tests for those modules

Required behavior:

1. Bind actual arguments to exact formal parameters through one indexed helper
   or factory call.
2. Carry the returned construction occurrence and its config-prefix chain.
3. Preserve defaults, keyword arguments, aliases, `*args`/`**kwargs`,
   unsupported dynamic arguments and rival returns as distinct typed states.
4. Never infer a parameter binding from a field name, class name, suffix, or
   value equality.
5. Two returned construction candidates remain ambiguous.

Mandatory poisons:

- positional versus keyword binding;
- default not supplied;
- renamed formal and local variables;
- imported factory alias;
- helper returning two rival constructors;
- `**kwargs` with unknown keys;
- same class constructed twice;
- config object forwarded under a different formal;
- external/unindexed factory;
- factory result used nowhere;
- two identical values from different config paths.

No renderer, fact schema, mechanism reader or architecture output may change in
this unit.

Achieved output:

> Exact factory-created occurrences can cite their actual constructor/config
> premises, or return typed ambiguity/incompleteness without guessing.

### U3-C2 — centralize the remaining neutral raw extraction

Primary candidates:

- `model_unfolder/evidence/forward_ops.py`
- `model_unfolder/evidence/transitive.py`
- `model_unfolder/evidence/sources.py`
- any shared low-level source/callable extraction still reparsing model files

Actions:

1. Inventory every model-source `ast.parse` and distinguish:
   - production model-source interpretation;
   - repository static audit/lint scanning;
   - test-only fixture parsing.
2. Move shared model-source observation onto `ProgramIndex`.
3. Delete duplicate caches and file reopening once parity is proven.
4. Add only the smallest neutral observation needed by a demonstrated
   counterexample.
5. Do not create a general CFG, SSA engine, symbolic executor or framework
   interpreter unless a named required boundary cannot be represented without
   it and Soumil approves that new kernel.

Achieved output:

> Migrated and new production readers cannot parse or cache model source outside
> the one ProgramIndex.

### U3-C3 — freeze the legacy semantic-reader quarantine

Generate a line-insensitive, symbol-based inventory with:

- definition symbol;
- exact production callers;
- raw parse/cache dependency;
- current semantic authority;
- assigned future unit;
- deletion condition;
- current witness coverage.

Rules:

- the inventory is blocking on growth;
- no new caller may use a quarantined reader;
- a reader may change only in its assigned semantic unit;
- deleting or migrating a reader must shrink the inventory in the same commit;
- a renamed/moved equivalent reader counts as growth;
- a generic helper called only by quarantined readers is quarantined too.

Achieved output:

> U3 can end without mechanically rebuilding known-wrong semantics, while the
> remaining authority is exact, visible, non-growing and scheduled for deletion.

### U3-C4 — regenerate the U3 reader inventory

The current `docs/U3_READER_INVENTORY.md` is stale: it still lists readers that
have been deleted. Regenerate it from the current tree and make generation
checkable.

The generated inventory must distinguish:

- migrated exact-owner reader;
- neutral infrastructure reader;
- quarantined semantic reader;
- repository audit scanner;
- test-only parser;
- dead symbol.

Achieved output:

> The tracker describes the code that exists, not historical line numbers or
> deleted functions.

### U3-C5 — final U3 receipt and handoff

Required on one unchanged committed tree:

1. focused ProgramIndex/owner/ReaderResult/factory-binding tests;
2. U2 authority and fabrication gates;
3. legacy-quarantine growth poisons;
4. identity/name-blind and rename/collision controls;
5. collection;
6. full suite;
7. preservation across all blessed witnesses;
8. isolated committed-checkout import and parse;
9. fingerprint before equals fingerprint after;
10. no unexplained architecture, IR, parameter, JSON or pixel delta.

If retained semantic candidates cause deltas, they require their future-unit
receipts and Soumil's visual decision; they cannot pass under a U3-only receipt.

Achieved output:

> U3 provides one reusable program/ownership truth substrate, and every surviving
> old semantic authority is frozen for one later deletion rather than being
> reimplemented twice.

---

## 6. Current legacy semantic worklist and correct future owner

The current tree contains 25 `*_from_files` production readers. Their correct
unit assignment is:

### U6 — attention

- `attention_score_scaling_from_files`

### U7 — FFN, norm, topology and router mechanism

- `decoder_layer_topology_from_files`
- `layer_class_count_from_files`
- `decoder_ffn_activation_from_files`
- `decoder_router_evidence_from_files`
- `decoder_intermediate_size_from_files`

### U8 — position, masks and schedules

- `decoder_rope_dim_from_files`
- `decoder_moe_schedule_from_files`
- `attention_causality_from_files`

### U10 — diffusion root/stream/conditioning

- `secondary_stacks_from_files`
- `diffusion_ffn_activation_from_files`
- `diffusion_axes_dims_rope_from_files`
- `diffusion_rope_from_files`
- `diffusion_attn_kind_from_files`
- `diffusion_ffn_kind_from_files`
- `diffusion_qk_norm_from_files`
- `diffusion_cross_qk_norm_from_files`
- `diffusion_single_stream_fusion_from_files`
- `diffusion_gate_via_norm_from_files`
- `denoiser_block_timestep_conditioning_from_files`

### U11 — UNet

- `unet_transformer_ffn_activation_from_files`
- `unet_stage_temporal_from_files`
- `unet_stage_attn_cell_from_files`
- `unet_code_attention_placement_from_files`
- `unet_mid_block_present_from_files`

Additional raw AST surfaces are assigned by module:

- `evidence/position.py` → U8;
- `evidence/vision.py`, `audio.py`, `projector.py`, `fusion.py` → U9;
- remaining diffusion parser/pattern scans → U10/U11/U12;
- scheduler source graph → U13;
- conformance-local model-source interpretation → U14;
- repository self-audit scanners remain lawful static tooling when they do not
  interpret a model's architecture.

These assignments must be generated and verified rather than maintained only as
this prose list.

---

## 7. What U3 must not build next

Do not add:

- another attention/FFN/norm/position recognizer merely to support one witness;
- a family/class/model table;
- a role inferred from class or field spelling;
- a general CFG or symbolic executor without a blocking counterexample;
- a renderer branch for a newly discovered mechanism;
- a new fact schema while claiming an ownership-only change;
- a second source cache or AST walk;
- `matched or fallback`, first/sorted/shortest candidate selection;
- a legacy-reader compatibility wrapper that becomes permanent;
- a semantic conclusion inside `ProgramIndex` or owner resolution.

When a model exposes missing semantic evidence, report:

1. exact owner that was resolved;
2. exact operation or relation that remains unproven;
3. whether the gap is observation, ownership, interpretation, fact arbitration,
   or projection;
4. the future unit that owns it;
5. the smallest counterexample test required before implementation.

---

## 8. Ratified architectural decisions

Soumil's 2026-07-27 instruction, “Do it,” ratifies the recommended route:

1. **U3 completion amendment:** U3 may close with an exact, blocking,
   non-growing legacy semantic quarantine. Each row is deleted once in its
   U6–U15 owner unit.
2. **Current local semantic commits:** keep the existing commit chain intact,
   but credit each semantic change only to its true future unit.
3. **Embedded CLIP delta:** retain as an unblessed U6/U7/U9 candidate. It is not
   part of U3's neutral acceptance.
4. **Manual weight tying:** assign the canonical root-bookend fact to a U7
   root-bookend slice; U14 parameters consume it for physical sharing and
   deduplication.
5. **History:** do not rewrite or split the 35 local commits. Correct their
   attribution in trackers and receipts before push.

No further architectural choice is required to complete the neutral U3 work.

---

## 9. U3 definition of done under this proposed amendment

U3 is complete when:

- every source bundle has one content-identified ProgramIndex;
- every migrated/new reader starts from one exact owner occurrence;
- factory/helper construction and config binding are exact or typed unresolved;
- failure, absence, ambiguity and incompleteness cannot collapse into a default;
- no migrated/new reader reparses or independently caches model source;
- ProgramIndex and owner resolution contain no mechanism semantics;
- the remaining legacy semantic readers form an exact, blocking, non-growing
  quarantine assigned to U6–U13;
- the reader inventory is generated from the current tree;
- U3 itself causes no unexplained structural or visual delta;
- the full unchanged-tree and clean-checkout receipt passes.

The next unit is then U4: make unknown safe everywhere before further semantic
fact migration.

---

## 10. Implementation and acceptance record (2026-07-27)

### Completed neutral work

| Slice | Status | Receipt |
|---|---|---|
| U3-C0 — reconcile/reclassify the local chain | **DONE** | `7530467` |
| U3-C1 — exact conservative factory/config binding | **DONE** | `77aec7e` |
| U3-C2 — classify remaining raw parse authority | **DONE** | 33 exact evidence-layer parse sites: 1 central ProgramIndex, 1 address bootstrap, 7 repository audits, 1 test guard and 23 legacy model-source sites |
| U3-C3 — freeze the semantic-reader quarantine | **DONE** | 25 exact readers; exact definitions/callers; normalized reader + same-module helper implementation digest; exact legacy-parse caller digest; growth/body/alias/move poisons blocking |
| U3-C4 — regenerate the current inventory | **DONE** | `docs/U3_CURRENT_READER_INVENTORY.md`; generator `--check` is blocking |
| U3-C5 — project-wide release gate | **HELD** | neutral U3 parity is exact, but the inherited unblessed semantic chain keeps the project-wide preservation/full gate red |

The factory proof accepts only an exact directly indexed, unshadowed
`@classmethod`, one unguarded `return cls(...)`, no unsupported execution
region, no rebinding of a forwarded formal, and no `*args`/`**kwargs`
expansion. Defaults, aliases, guarded/rival returns, external factories,
class-name calls, dynamic forwarding and unsupported control flow remain
typed opaque. No name, position or value-equality fallback was added.

The quarantine blocks:

- new/moved `*_from_files` definitions, including nested definitions;
- direct, qualified, imported-alias and assigned-alias consumers;
- a body or same-module helper-closure change;
- a new/aliased evidence-layer `ast.parse`;
- a new caller of an already-existing legacy parse authority;
- stale/dead rows and generated-inventory drift.

### Committed-tree receipt

Receipt: `/private/tmp/model-unfolder-verification/d2467f9b90`, commit
`77aec7e04d878ae58e15a271fead2b6aff83158e`.

- static: **PASS**, 7 changed Python files clean;
- collection: **PASS**, 2,252 collected;
- focused ProgramIndex/owner/quarantine: **PASS**, 182;
- U2 authority: **PASS**, 44;
- exhaustive non-preservation partition: **2,147 passed, 11 skipped,
  2 xfailed, 2 failed**;
- preservation: **27 passed, 19 failed**;
- every lane's source-tree and external-artifact fingerprints: **unchanged**.

The two exhaustive failures are inherited semantic outputs:

1. `test_text_encoder_shows_real_config_dims` — the retained embedded-CLIP
   `norm: LayerNorm` candidate is not in the old expected structure;
2. `test_sable_regression_corpus` — the existing unblessed gallery delta is
   still deliberately blocking.

The 19 preservation failures are likewise inherited from the semantic commits
reclassified to U6–U9/U14. They are not accepted or re-blessed by U3.

### Exact neutral-delta proof

The parent `7530467` and U3 implementation `77aec7e` were each rendered in an
isolated worktree across all 26 witnesses, four workers per tree. Both runs
passed 26/26. Every canonical surface and every view hash compared byte-for-byte
equal; the combined filename+content digest on both sides was:

`df775f8a9a61509023e9c13eacd189f8bfad6077f2ec18794971913a05f65d5d`

Therefore the U3 completion implementation itself caused **zero** architecture,
IR, expanded JSON, parameter, evidence-ledger, HTML-metadata, gallery or pixel
delta.

### Binding status

The neutral U3 implementation is complete. **U3 release acceptance is not
marked DONE**, because this plan explicitly forbids using U3 to bless the
retained semantic deltas. The next action is not more U3 substrate code: each
inherited semantic delta must be either reverted to the blessed behavior or
accepted under its assigned U6–U9/U14 receipt and Soumil visual decision. Only
then may the project-wide gate turn green and U4 formally unlock.
