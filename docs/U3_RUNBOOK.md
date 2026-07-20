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
- [~] U3-D1 first production-reader pilot — denoiser temporal-axis remains
      BLOCKED until the separate U3-A1 committed-tree receipt and Codex kernel
      review pass. When resumed it must query the exact forward's
      IdentifierObservation census directly, reject child occurrences, cite the
      matching identifier's exact span, and return a negative only with no marker
      and no unsupported/nested region. None of the parser cutover, replacement
      reader or legacy-reader deletion may enter the U3-A1 commit.
- [ ] U3-E..H migration clusters + conformance migration + old-parser
      eradication (per-reader commits, U2 receipt discipline each)

The binding execution order, exact first pilot, review ownership and stop gates
for U3-D through U3-H are specified in
`docs/U3_D_TO_H_EXECUTION_AND_VET_PLAN.md`. In particular, production migration
must not begin until its U3-D0 component-root address boundary is present; the
implementation agent may not invent a local root-class chooser.
