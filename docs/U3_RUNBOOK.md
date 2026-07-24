# U3 runbook — the single raw program index and owner resolver (LIVE)

Spec: docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md §20.6 (lines 1457-1503).
Started 2026-07-19 immediately after U2 closure (1d0c72b). Soumil: "go ahead".

## ⚠ SOUMIL'S PRE-IMPLEMENTATION BOUNDARIES (2026-07-19, BINDING)
1. ProgramIndex is OBSERVATION-ONLY: AST facts/expressions/calls/spans/
   control structure. It never decides "this is the FFN" / gated / best
   candidate — those belong to owner-bound readers.
2. Index CONSTRUCTION-SITE OCCURRENCES, not just classes: ComponentOwner
   resolves an occurrence chain parent owner → construction site →
   field/slot → child class. Class identity alone is insufficient.
3. CONTENT identity, not mtime: deterministic source-snapshot fingerprint
   over file CONTENT. Poison: content change with preserved mtime — a
   stale index must be impossible.
4. Config values stay on the U1 rail: the index reports THAT source reads
   config.layer_types; the VALUE resolves through the owner-qualified
   ledger and is consumed into a fact. The index never reads checkpoints.
5. Conflicts become RECORDS: ReaderResult.ambiguous carries rival owner
   chains, conflicting config prefixes, exact construction sites, spans.
   Drop-on-conflict loses evidence; never silent.
6. ReaderResult[T] WRAPS the EIGHT domain evidence dataclasses (Positional, FFNStructure, VisionTower, AudioTower, Projector, Fusion, QKNormCode, RouterCode; ConformanceProblem is an additional typed failure surface) (common
   status/owner/completeness/failures/provenance) — never replaces them.
7. Cache ownership is CALL-LOCAL: one immutable index on ParseContext/
   SourceBundle; optional cross-parse cache = content-fingerprint keyed +
   immutable. Never module-global mutable truth.
8. External import closure is EXPLICIT: shared diffusers files enter as
   provenance-bearing EXTERNAL source nodes, never blended into the
   model bundle.
9. Parity alone is insufficient: every migrated reader ALSO needs
   class-renaming metamorphic control; same class under two roles;
   equivalent candidates; rival candidates; missing/partial source;
   unsupported dynamic dispatch; old reader DELETED in the same commit.
10. FIRST MIGRATION IS NARROW: core + nine AST fixture families, then ONE
    low-risk reader end-to-end. Never all 60 in the core commit.
Process: the runbook + inventory are versioned IN unfold-pkg/docs (this
z-docs copy is the working mirror; the repo copy is authoritative once
receipts depend on it).

## The contract (verbatim distillation)
ONE `ProgramIndex` per SourceBundle (evidence/program_index.py): files/
modules/import aliases; classes/bases/class assignments; constructor fields +
assigned expressions; methods/functions/params/returns/spans; direct calls +
self.method calls + reachable closures; static branches/loops/comprehensions
+ controlling expressions; attribute reads/writes + config-path reads;
constructed submodule class refs; tensor-op observations; unsupported syntax
+ parse failures (typed, never silent).
Steps: (1) resolve source ONCE in ParseContext (identity as address only);
(2) ComponentOwner from the PARENT'S construction graph — markers may
nominate, never complete; (3) every evidence reader queries the same index +
exact owner; (4) broad class-role unions → rival-owner/ambiguous results;
(5) bare None/False failures → ReaderResult; (6) AST fixtures: alias imports,
helper methods, inherited methods, factory functions, comprehensions,
conditional construction, equivalent candidates, rivals, unsupported dynamic
dispatch.
DELETE (after parity): independent reparsing, same-role class union, broad
best-candidate selection, duplicated call/field extraction.
Tests: test_code_evidence.py, test_conformance.py, specialized evidence
tests, test_reader_exceptions.py, test_h9_frontier.py.
DONE = all mechanism readers receive the same exact owner graph and cannot
disagree because they scanned different files/classes/closures.

## Folded-in debts
- position.py raw `_config_value`/`_config_scopes` walks (funnel-invisible;
  U2 carry-forward) die when position.py queries the index under its exact
  owner + funnel-located config reads.
- U2 gates stay green throughout: writer census, debt register, receipts,
  name-blind (the index must resolve by ADDRESS only), zero-drift manifest
  (pixels byte-identical; ledger drift = new bless, avoid until U3 closes or
  document evidence-only delta per the U2 precedent).


## PHASE LADDER (Soumil, 2026-07-19 — REVISED, supersedes the earlier split)
The earlier ladder (A=walker, B=assembly, C=fixtures) is WRONG and deleted:
it would have committed a walker before its fixtures existed. The single
unambiguous sequence is:

- **U3-A** — the ProgramIndex CORE, ONE atomic independently-receipted commit:
  types + walker + immutable assembly + call-local ParseContext attachment +
  ALL 13 fixture families. A walker is NEVER committed before its fixture
  families exist. No reader migration, no rendering change, no old-extraction
  deletion, no architectural role inference, no name-based selection.
- **U3-B** — the ComponentOwner resolver + construction-graph resolution
  tests. Emits typed rival-owner / config-prefix conflicts when resolution
  cannot prove uniqueness (the raw index only RECORDS candidates). No reader
  migration.
- **U3-C** — the generic `ReaderResult[T]` substrate + failure-law tests.
  No reader migration.
- **U3-D** — the first narrow reader pilot (mechanically selected: single
  caller; typed evidence already; smallest deletion surface; no cross-owner
  reach; corpus witnesses exercise it).
- **U3-E..H** — bounded migration clusters, conformance migration, and
  old-parser eradication. Each migrated reader wraps `ReaderResult[T]`, ships
  its six adversarial controls + class-renaming metamorphic control, and
  DELETES the old reader in the same commit.

Every phase = its own independently verified commit (six-point receipt:
focused tests, U2 nets green, full suite, fingerprint unchanged, pixels
byte-identical, isolated committed-tree pass) + push. Proceed WITHOUT asking
for routine implementation choices; after U3-A is pushed and green, continue
autonomously into U3-B. STOP ONLY FOR: unresolved semantic conflict;
unexplained artifact delta; preservation regression; product-judgment change.

### Parallel committed-tree receipt (binding from U3-D2 onward)

Use the repository coordinator instead of serial scratch scripts:

```bash
python3 scripts/verify_commit.py --focus tests/test_<reader>.py \
  --forbid <deleted_legacy_symbol>
```

The coordinator does not remove a gate. It runs focused/kernel, U2 authority,
preservation, collection, static/eradication, and an exhaustive partition of
the complete suite concurrently, with every lane in its own detached worktree.
Host-aware allocation reserves one process each for preservation, focused and
U2, assigning the remaining cores to the measured full-core long pole.
Preservation owns exactly `tests/test_preservation.py`; the full-core lane owns
every other test file, so no test is omitted and expensive fixtures are not
duplicated across process workers. Every worktree is
fingerprinted before and after. Gitignored blessed galleries/baselines are
copied from one content-hashed snapshot and independently hashed in every lane;
they are never mistaken for committed files. A missing focus path, missing lane, non-zero
lane, collection error, forbidden symbol, static failure, or changed
fingerprint makes the whole receipt fail.

Run one serial full-suite bracket at each U3 phase boundary and before final U3
closure with `--serial-full` as an additional order-dependence control. It does
not replace the parallel full suite required for every migration commit. Each
run writes a machine-readable `receipt.json` beside its lane logs.
NEVER stop merely because a phase finished.

## U3-A SCHEMA CONTRACT (Soumil, 2026-07-19 — BINDING, precedes any visitor)
Correct the type layer BEFORE writing visitors. Never use a bare class or
function name as authoritative identity.

Stable qualified identities:
- **SourceId** — canonical path, content fingerprint, component key, external
  flag, external provenance.
- **SymbolId** — source identity + qualified name.
- **SourceSpan** — source identity/path, line, column, end line, end column.
- **ConstructionSiteId** — owner SymbolId, callable SymbolId, source span,
  assignment/slot ordinal.

Records:
- **ModuleRecord** — module/package identity + its source node (was missing).
- **ParseFailure** — read / decode / AST-parse failures ONLY.
- **UnsupportedSyntaxRecord** — separate: a HEALTHY file with unsupported
  dynamic syntax stays indexed AND carries an unsupported-syntax observation.
- **SourceFileNode.__post_init__ invariants** — internal node requires a
  component key; external node requires non-empty provenance; fingerprint must
  match supplied source content during assembly; contradictory internal/
  external metadata raises.

Query-bearing expressions are NEVER stored as strings:
- **ExprNode** (frozen, normalized) — kind, value/name/operator, children,
  keyword children, span, optional source segment for DIAGNOSTICS ONLY.
  Structurally represents names, attributes, constants, calls, subscripts,
  slices, collections, boolean/binop/comparison, IfExp, comprehensions.
  Replaces every expression-string field in assignments, returns, kwargs,
  guards, controls, calls, dataflow. Readers query the STRUCTURE — never
  ast.unparse() or substring search over the diagnostic segment.
- Defaults distinguish three cases: no default; literal None; unsupported/
  dynamic default.

Honest construction sites — ConstructionSite records: exact site identity;
owner + enclosing callable as qualified SymbolIds; target kind + target
field/slot; raw constructor expr (ExprNode); positional args; keyword args;
guard/control references; ZERO, ONE, or SEVERAL statically-resolved child
candidates; resolution provenance. Zero candidates = dynamic construction;
multiple = rivals. The walker NEVER chooses one child_class. Factory/helper
resolution creates PROOF-BEARING candidate edges — "factory-resolved" is
never a guessed single name.

Observation separate from resolution:
- CallObservation.order → **lexical_order** (+ enclosing control/guard path).
  Runtime execution order is a reader's job on a proven branch, not the index.
- Dataflow records carry def-use edges + structural expression relationships
  ONLY — never an attention/FFN/gating/projection role label.
- Config observations carry: syntactic config-root binding; exact typed path
  segments; dynamic/unresolved segments made explicit; owner callable + span.
  PATHS only; checkpoint values stay exclusively on the U1 ledger.
- ConflictRecord is NOT emitted merely because the walker sees multiple
  candidates. The raw index records candidates; U3-B's resolver emits the
  typed rival-owner / config-prefix conflicts.

U3-A FIXTURES = the nine spec families PLUS: content-changed-mtime-preserved;
same qualified class at two distinct sites/roles; conflicting owner/config-
prefix candidates retained as RIVALS; one malformed + one healthy file
(partial usability). PLUS: two modules defining the same class name; literal
None default vs no default; two construction calls on the same source line;
dynamic factory returning no proven candidate; multiple factory candidates
with no winner selection; branch calls proving lexical_order ≠ runtime order;
external node without provenance MUST raise; expression-source renaming that
leaves the structural observations equivalent.

U3-A ACCEPTANCE — do not modify any reader, renderer, architectural output,
or old extraction path. Complete ONLY when: all types + walker + assembly +
SourceBundle integration + ParseContext call-local ownership present; exactly
ONE immutable index per bundle/context; the aggregate bundle fingerprint
canonically includes source identity, component ownership, external
provenance, and content fingerprints — NOT file contents alone or iteration
order; every required fixture passes; no query API performs name/substring
architectural selection; focused tests + all U2 nets + full suite + unchanged
fingerprint + byte-identical preservation + isolated committed-tree
verification all pass. Commit + push U3-A only after that complete receipt,
then proceed autonomously to U3-B.

## Working state
- [x] Reader inventory recon DONE → u3-reader-inventory.md (43KB, verbatim).
      Headlines: ~60 readers / 12 modules; 34 model-source ast.parse sites
      (3 package self-audit sites are OUT of scope); 4 competing parse
      caches (2 without mtime keys) + ≥10 uncached sites — one sable run
      parses a modeling file up to ~8 ways; 10 duplication clusters
      (ctor-assign walks, ModuleList ×3, ~14 BFS reachability copies,
      6+ config-expr readers, 3 raw config-scope walks outside the U1
      ledger, 3 execution-order visitor copies); ~24 same-role-union/
      best-candidate deletion sites (worst: patterns whole-file unions +
      votes, _find_decoder_layer[0], vision/audio max-field-count owner
      picks, conformance or-fallbacks, resolve_view_code shortest-name);
      2 name-completes-resolution sites (sources._looks_like_diffusion_
      class :354; conformance domain/view name-marker filters) + TP
      _source_files :119 throwing the owner away (flat bundle.files);
      position.py raw walks read alibi/layer_types/_attn_implementation/
      rotary_dim/rotary_pct/partial_rotary_factor via a hardcoded
      5-wrapper chain (siblings: conformance :1727, projector :761);
      8 typed evidence dataclasses already exist to converge on (+ ConformanceProblem as a typed failure surface);
      projector._config_param_chains (:319) = the ComponentOwner +
      config-prefix prototype to seed the index design.
- [x] U3-A ProgramIndex CORE — DONE, pushed 44294f0. Types (SourceId/SymbolId/
      SourceSpan/ConstructionSiteId, ModuleRecord, UnsupportedSyntaxRecord,
      ExprNode, ChildCandidate edges) + observation-only walker + immutable
      assembly + aggregate fingerprint (identity/ownership/provenance/content,
      order-independent) + ParseContext.program_index() lazy call-local + 28
      fixtures. Vocab in everchanging/evidence/program_index_vocab.yaml (config
      roots, ACT2FN/get_activation, container classes) with container_classes
      registered in the identity-guard lawful manifest. Six-point receipt green
      (full 1521 passed, fingerprint identical, preservation 20, identity 26,
      ratchet 4, isolated bracket PASS).
- [x] U3-B ComponentOwner resolver — corrected in the independent-audit baseline
      after audit of 602b133. The first version still identified nodes by class
      SymbolId at lookup (two same-class occurrences could collapse), resolved
      an unimported cross-file class by bundle-wide name uniqueness, and recorded
      a rival config-prefix conflict while publishing the parent's plausible
      prefix. The corrected evidence/component_owner.py uses
      OwnerOccurrenceId(root + complete ConstructionSiteId chain), retains both
      helper-call and helper-return sites, exposes occurrence-only node lookup
      plus plural nodes_for_symbol, resolves cross-file classes only through an
      exact import binding, preserves guarded same-field constructions as typed
      rivals, carries rival prefixes through descendants without fallback,
      records cycles/depth limits explicitly, and requires explicit root
      bindings when several constructor parameters are viable. Factory inputs
      are recorded as factory inputs—not falsely mapped onto __init__ params.
      18 resolution-law tests. No reader migrated.
- [x] U3-C generic ReaderResult[T] substrate — corrected in the independent-audit
      baseline after audit of a02c95d. ReaderResult now owns an exact
      OwnerOccurrenceId rather than a class SymbolId; successful/partial values
      require structured ReaderProvenance; provenance kinds enforce their real
      channels (source span, config path, or both); failure kinds are closed;
      incomplete results must explain their missing evidence; ambiguity accepts
      only exact typed rival records; resolved/failed/absent/ambiguous states are
      mutually exclusive. The value_or default escape hatch is deleted and
      require_value raises on every non-value state. 23 failure/provenance-law
      tests. No reader migrated (that begins at U3-D).
      Committed-tree receipt (independent-audit baseline): 41 focused B/C + 69
      U3-A/B/C green; affected U2 (identity/structural-write/registry/projection)
      gates green; preservation green; full suite green; tree fingerprint
      identical before/after. Marked complete on that green committed-tree gate.
- [x] U3-D0 component-root address boundary — DONE, pushed 343aa7d (V1
      approved). evidence/component_owner.py resolve_component_root(index,
      bundle, component_key) bridges a SourceBundle component address to the
      exact root OwnerOccurrenceId: address from component_architectures[key]
      (bundle.architecture = root compat only); exact-identity class match in the
      requested component only; hidden-rival law (any component parse failure ->
      failed before the candidate count, canonically sorted); self-verifying
      ComponentRootCandidate; closed ComponentRootResolution (incl. graph-root-
      symbol == occurrence-root, empty root site-chain); canonical rival sort;
      address_resolved (an address claim, binding ambiguity stays in
      graph.root.unresolved). 34 poisons. Committed-tree receipt: focused 103,
      affected U2 44, preservation 20 zero-drift, full 1596/0, fingerprint
      identical, isolated worktree green.
- [~] U3-A1 identifier-observation completeness — CODEX-OWNED KERNEL CORRECTION,
      implemented in the working tree and awaiting its own committed-tree
      receipt. The D1 stop condition proved that reconstructing names from
      selected call/dataflow/control records could not support a general
      negative. ProgramIndex now records neutral IdentifierObservation rows for
      every exact callable-scoped ast.Name/ast.arg with its SourceSpan and
      syntactic context (parameter, load/store/delete, annotation, default,
      decorator). Nested defs/classes/lambdas are lexical boundaries and are
      published as exact-callable unsupported regions rather than contaminating
      the parent. Bare unsupported expressions are also published, so a negative
      cannot skip opaque syntax. No identifier is interpreted by ProgramIndex.
- [x] U3-D1 first production-reader pilot — denoiser temporal-axis, DONE (V2
      approved, pushed 795c3ff). evidence/denoiser.py denoiser_temporal_axis(
      index, owner) -> ReaderResult[bool] queries the exact forward's
      IdentifierObservation census (index.identifiers_in), rejects child/non-root
      occurrences, matches the temporal marker vocabulary against that complete
      exact-span census, cites the matching identifier's exact span for a
      positive, and returns a negative ONLY when the exact forward is completely
      observable (no marker AND no unsupported/nested region — else a typed
      failure, never False). adapters/diffusor/parser.py::_temporal_axis rewired
      onto program_index() + resolve_component_root("root") + address_resolved +
      status==resolved, checkpoint-declared fallback kept, broad except removed;
      legacy denoiser_temporal_axis_from_files + AST loop deleted same commit; the
      U2 broad-except baseline for adapters/diffusor/parser.py lowered 19->18 in
      the same atomic unit. Committed-tree receipt PASS (full 1633/0, fp
      identical, preservation 20 zero-drift, isolated 29, eradication clean).
- [x] U3-E leaf-reader migration cluster — DONE, EMPTY (no lawful candidates).
      Exhaustive inventory (docs/U3_READER_INVENTORY.md, "U3-E owner-access
      classification"): of all 40 *_from_files readers, exactly ONE is ROOT-ONLY
      (unet_mid_block_present), and it fails Section-6 rule 5 — SDXL is the
      corpus's only UNet, so it has no negative witness. Every other reader is a
      whole-file / per-class UNION or a nested/block traversal, so none has an
      already-available exact owner AND a corpus positive+negative AND full
      answerability from current observations. The Section-6 selection rules were
      NOT weakened to force a candidate through. U3-E requires no reader commit.
- [x] U3-B1 declared model-stage address boundary — DONE (commit 933bb90, Codex
      V4-approved; receipt `/private/tmp/model-unfolder-verification/0a36445ce2`:
      static/focused(168)/collect(1741)/u2-authority(44)/full(1681)/preservation(46)
      all green, every lane fingerprint identical). Replaces the rejected
      execution/dataflow proposal ("primary sequence body"/"output head"/"main
      hidden state" were semantic classifications, not address evidence). Resolver
      uses ONLY the closed-code FrameworkAddressProtocol (base_model_prefix)
      resolved by LAZY EXACT precedence (root-direct / first-exactly-bound-base-
      direct / exact C3 when the closure is fully indexed; unresolved earlier base
      -> failed(mro_incomplete), never skipped; decisive dynamic -> failed) with a
      proof_trace (root -> declaring_class, MRO-prefix/precedence), matched
      EXHAUSTIVELY against the authoritative OwnerGraph: one resolved child ->
      resolved; >=2 -> ambiguous(real occurrences); a matching unresolved entry
      never degrades to absent (rival_owner/ambiguous_import -> ambiguous with
      authoritative OwnerRival records; dynamic/external/unknown -> typed failed);
      no fabricated occurrences. B1 consumes a RESOLVED ComponentRootResolution
      (D0), inheriting component isolation + the hidden-rival/parse-failure law
      (a broken component file -> D0 failed -> B1 refuses). Every
      DeclaredModelStageResolution status is mutually closed; provenance and the
      framework protocol are invariant-guarded. 57 poisons. Corpus 10 resolved /
      15 failed(mro_incomplete) / 1 absent, unpatched. NO class names / model types
      / role vocab / embedding-layer-norm evidence / call ordering / return-flow /
      most-plausible-child / YAML family tables; no new ProgramIndex record family.
      This boundary does not itself classify a norm or a repeated child; those
      separate F2/F3 mechanisms are now present below.
- [ ] U3-F..H nested-mechanism / modality / conformance clusters + old-parser
      eradication (per-reader commits, U2 receipt discipline each)
      - [x] U3-F0: occurrence-exact repeated-child reconnaissance
        (`docs/U3_F0_REPEATED_CHILD_RECON.md`).
      - [x] U3-F1: exact direct/sliced/builtin-enumerated iteration observations.
      - [x] U3-F2: exact repeated-child occurrence resolution; Qwen2-VL's nested
        text stack remains explicitly outside this B1 model-stage boundary.
      - [x] U3-F3a: exact external construction-call addresses integrated into
        the one execution-flow graph—never fabricated OwnerOccurrenceIds.
      - [x] U3-F3b: embedding/LayerNorm/RMSNorm semantics from exact external
        framework protocols or indexed implementation math, never class spelling.
      - [ ] U3-F4: embedding-stage norm production cutover implemented; focused
        gates green.  Its original receipt was stopped after exposing an
        eight-hour worker-allocation pathology; F4 is included in the next
        resource-balanced committed-tree campaign receipt rather than accepting
        an incomplete run.
      - [ ] U3-F5a: exact attention-child positive classification implemented,
        pending the combined committed-tree receipt.  It starts at the exact F2
        repeated-child occurrence and follows only graph-authoritative invoked
        children.  Attention is proven by exact SDPA protocol or unguarded
        dot-product+softmax implementation in one callable.  A bound local
        fallback is followed only through the exact
        ``target = resolver(..., fallback); target(...)`` relationship.  Class,
        field and local spellings never classify the child; softmax-only,
        dot-only, guarded, unused-helper and sibling-owner controls abstain.
        Real controls: BLOOM, DeepSeek-V3, Gemma-2, GPT-OSS, Llama, OLMo-2,
        Qwen3 and StableLM resolve; Qwen2-VL remains honestly outside the F2
        model-stage boundary for U3-G.
      - [ ] U3-F5b: exact Q/K/V projection-storage reader implemented as a
        boundary, not yet a production cutover.  It consumes F5a's exact
        attention occurrence and versioned local dataflow: three independent
        exact Linear occurrences reaching the compute input prove `split`; one
        exact Linear feeding a code-proven three-lane unpack proves
        `fused_qkv`.  Conditional producer creation, dishonest unpack helpers,
        low-rank/chained projections and unrelated Linear calls abstain.  Real
        controls: BLOOM fused; Gemma-2/GPT-OSS/Llama/OLMo-2/Qwen3/StableLM
        split; DeepSeek-V3 unknown.  Falcon deliberately remains unknown
        because its attention child is selected through a config-keyed dispatch
        registry.  Production cutover is forbidden until U3-F5c proves either
        the exact selected candidate or unanimous candidate-equivalent storage;
        the legacy reader remains intact until that same cutover commit.

The binding execution order, exact first pilot, review ownership and stop gates
for U3-D through U3-H are specified in
`docs/U3_D_TO_H_EXECUTION_AND_VET_PLAN.md`. In particular, production migration
must not begin until its U3-D0 component-root address boundary is present; the
implementation agent may not invent a local root-class chooser.

## U3 Execution-Substrate Campaign — foundation receipt

This was the boundary-only, no-production-consumer foundation. Nothing in
parser/renderer/facts/IR/config/debt/manifest/gallery changed in that foundation
chain; the later F4 row above is the first bounded production consumer.

- [x] Phase 0 — U3-B2 final closure (owner-symbol cross-index check + typo).
- [x] Phase 1 — U3-A2 recon corrected + FROZEN with an occurrence-exact 26-witness
      appendix (`docs/U3_A2_RECON.md`). Diffusion root is diffusion-adapter
      authorized, never inferred from B1 failure; P11 is an exact set + labelled
      lower bound, never a zero.
- [x] Phase 2 — neutral execution records in `program_index.py` (StatementId,
      CallSiteId, BindingObservation, LoopObservation, ReturnObservation,
      ControlTransferObservation, UnsupportedExecutionRegion). Known unmodelled
      executable forms (IfExp/BoolOp/comprehension/lambda/unknown statement) are
      published as `UnsupportedExecutionRegion` records, but the inventory is
      deliberately non-exhaustive: absence of such a record never proves callable
      completeness. No SSA / roles / resolved owners / happens-before /
      layer-stack labels / semantic callee.
- [x] Phase 3 — `execution_flow.py` addressed invocation resolver
      (AddressedInvocation / RepeatedInvocationTemplate / UnresolvedInvocation);
      consumes a resolved D0 + EXPLICIT owner + B2 inventory; never selects the
      owner.
- [x] Phase 4 — conservative versioned def-use execution-flow resolver
      (InvocationNodeId / HappensBeforeEdge / ExecutionFlowResolution) + the
      permanent adversarial matrix. An OPEN, conservative LOCAL-RELATION substrate:
      NO resolved/complete status and NO closed-world coverage certificate (there
      is no CFG coverage unit yet), so every callable result is `partial`. Local
      proven/conditional edges are valid local relations; loops + unmodelled forms
      are PUBLISHED as non-exhaustive coverage gaps. Unresolved is never "unordered";
      cycles are a blocking failure; failed/ambiguous aliases are preserved as typed
      unresolved state on the target.
- [x] Phase 5 — corpus evaluation over 26 witnesses (no patching):
      `docs/U3_EXECUTION_FLOW_CORPUS.md`. 26 partial (all open) · 3 proven edges ·
      16 conditional · 7 templates — a local-relation lower bound, not a completeness
      result. Do NOT migrate a reader requiring whole-callable completeness; a reader
      needing a particular positively-proven local relation may be proposed
      separately.

The campaign has been approved and the bounded U3-F sequence is executing above.
Whole-callable negative claims remain forbidden: F4 consumes only a positively
proven local def-use relation into the exact repeated child.
