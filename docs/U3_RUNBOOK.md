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
- [ ] ProgramIndex core (files/classes/ctor fields/calls/spans) + fixtures
- [ ] ComponentOwner resolver from parent construction graph
- [ ] ReaderResult type + reader migrations (one reader at a time, parity
      fixtures frozen before each)
- [ ] Deletions + gates + commit ladder (per-reader commits, U2 receipt
      discipline: focused + FULL suite + isolated receipt each)
