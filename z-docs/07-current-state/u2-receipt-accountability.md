# U2 — projection receipts and claim accountability (live)

Live status area. Rewrite as state changes; do not append session history.

## What U2 is for (intent — read this first)

**U2's job is to make every EXISTING structural claim accountable. It is not to
solve future architecture semantics.**

Every claim the product draws, serializes, counts, or asserts must resolve to
one of exactly two things:

1. a **registered fact** with a **real consumer receipt**, or
2. **exact, owner-qualified, shrinking migration debt** assigned to a later
   unit.

If a mechanism needs interpretation that only U6–U13 can supply, U2 records the
debt. It does **not** fabricate a receipt, and it does **not** implement the
later unit early. A receipt asserted for a claim whose evidence does not exist
is a lie with a hash attached — worse than an honest unknown.

Corollary that has already bitten once: the scheduler currently says "Euler"
because of a **class name**. U2 may not receipt that. It becomes exact U13 debt.

## The binding doctrine (producer-first, one-way evidence)

Evidence flows one way:

```
source/address → exact component ownership & reachability → mechanism reader
→ typed fact → canonical IR/region → renderer/JSON/params/conformance
→ projection receipt → audit
```

A **downstream artifact may never decide whether an upstream fact, owner,
mechanism, or obligation was applicable.**

When a gate fires, presume the finding is real. Locate the **earliest** point
where an invalid fact/owner/consumption/default/projection was authored, fix
**that** producer, and delete the invalid path. Forbidden ways to green a gate:
blocking→advisory; conditioning obligations on whether the renderer drew
something; allowlist/ignore/exception; broadening exact owner/path/mechanism to
a key or family; treating absence of output as proof of non-applicability;
model/family-specific production branches; keeping the false producer and
adding a second diagnostic for it.

Applicability is decided **upstream**: proven-active → fact may be consumed and
consumers owe receipts; proven-inactive → no positive fact; unresolved →
unknown/opaque, never a conventional fallback.

Model names may appear only in tests and reports. Production fixes are reusable
mechanism/ownership predicates.

Report before editing, for every defect: (1) invariant that fired, (2) full
evidence chain, (3) earliest incorrect author, (4) why the current fact/owner/
consumption is invalid, (5) the general predicate that replaces it, (6) the old
path being deleted, (7) effects on equivalent models.

## Where U2 stands

**Done: receipt core + projector-width pilot (U2.1/U2.1a), and the U2.2a path
producer fix (awaiting a bless — see below).**

- `7c6a298` — receipt rail + 2-scope pilot + producer ownership fix
- `81317d6` — shipped-contract docs + regenerated census
- `49d0d93`, `c061c92`, `86f03ba`, `dff4b14` — U2.1a: the four vet corrections
  (upstream expectation · missing-expectation blocks · node_path validated · ONE
  registry) + the registry-bypass close. Valid receipt on fingerprint
  `77e40e67`: all-25 = 0, grouped 252p/5x, full suite 1235p/5x/0f,
  clean-checkout 6 passed.

### U2.2a — a recorded config path must be TRUE (NOT complete; bless withheld)

**Soumil's vet of `28a66ad` (2026-07-17) found the first pass repaired the
CORPUS but not the PRIMITIVE.** Three proof holes, all now closed:

1. **A path could certify itself.** Passing `config_path=` set `path_exact` by
   fiat, so any producer could author a dotted string addressing nothing.
2. **A real path could be BORROWED by an unrelated object.** The first fix
   proved only that the path *exists* in the document — so a read of some other
   `{"real": 999}` certified itself as the document's `real`. The path resolves,
   the value is foreign, and the occurrence key is a fiction that passes.
3. **A qualified container trusted an unidentified read.** The prefix applied
   when the read never said which object it came from — prefixing "because
   nothing contradicted us", which is how the fabrications were authored.

**Exactness is now a PROOF, and it is two claims, both discharged:** the path
addresses a real location in the document, **and** the object the document
places there IS the object the reader accessed. Identity, never resemblance. An
explicit path the document disproves **raises**; an unidentified read stays
honestly inexact.

**Occurrence identity now survives the whole lifecycle.** `ConfigResolution`
carries the object it observed on, so `consume()`/`bind()`/`ignore()` prove
exactly what the inspection proved. This was a live bug, not a hypothetical:
`vision_encoder_hidden_size` probed for a value and then hand-emitted a second
`consumed` event carrying neither path nor object — so the observation proved a
location and the consumption, *which is what a claim binding actually joins on*,
proved nothing. Fixed with a new primitive rather than at the call site:
`resolve_priority` runs a PRIORITY chain over DISTINCT fields (`embed_dim`
beside a merger-out `hidden_size` legitimately differ, so their disagreement is
not ambiguity and alias-resolution is the wrong tool) and returns an occurrence
the caller CONSUMES. A value-only handoff between finding and consuming will
always drop the proof.

### U2.2a — the earlier pass (still true, superseded above)

The occurrence key is defined as "what was actually supplied, where". Nothing
enforced the *where*: a container scope glued its prefix onto every enclosed
read, including reads of a **different object**, so the ledger asserted exact
dotted paths (`vision_config.image_token_id`) that resolve in no document. This
silently defeated every exact-path join — the join the whole receipt rail is
built on.

Four producer laws now hold, each with a poison that fails when it is removed:

1. **A container applies only to reads OF the object it names**
   (`config_container(path, obj=)`). A builder legitimately reads both its
   sub-config and its host; the host read keeps its own true path.
2. **`container_scoped` resolves the object it names** from the call's own first
   argument — so a read of `cfg` is outside `cfg._vae_config` by construction.
   The hand-placed `config_container(())` escape hatch is **deleted**.
3. **A recursively-parsed slot is a DOCUMENT, not a container**
   (`document_scope`). Paths stay document-relative; the address travels beside
   them.
4. **A document root is an object too.** Its fields live at its top, so a read of
   it is exactly pathed at the bare leaf. Naming it (`document_scope((),
   obj=cfg)`) is what makes that provable — and it dropped honest-but-unlocatable
   reads from **305 → 14**.

Plus `present_spelling` (a container names the spelling the document *supplies*,
never one chosen by an alias-resolving read) — correct and unit-tested, but
**today's corpus does not falsify it**: hydration makes the canonical spelling
literally present everywhere, so both choices agree. It is a defensive fix, not
a corrective one. Do not credit it with the 14 rows below.

**Why paths are document-relative, not absolute.** `claims_audit` matches a
declared binding to an event by exact equality on `(owner, config_path)`, and
every binding is written relative to the model's own document
(`vision_config.hidden_size`). Absolute paths would make the *same mechanism*
match standalone and miss once embedded — accountability depending on where a
model is hosted, the identity-dependence this project exists to delete. So the
document address is recorded **beside** the path (`document_path`, published as
`config_access.document_roots`, rendered per-owner in the census), never glued
into it.

**Measured, corpus-wide (25 witnesses):** exact rows 624 · **fabricated paths
0** · honest-inexact 14. Of the 624, **610 resolve in the raw source** and 14
resolve only after class hydration (see the gap below).

**Zero behaviour change.** `ir`, `expanded`, `params`, `html_meta`, `gallery`
are byte-identical across all 25. Only `ledgers` (25) and `sable` (18) drift —
the evidence surfaces this fix corrects. **The preservation manifest therefore
needs a re-bless, which is Soumil's alone.** Until then
`test_expected_manifest_zero_drift_zero_skip` is red by construction, and that
is the honest state — not a gate to weaken.

Three findings the fix un-hid (all fixed at the producer, not papered over):

- **A test asserting the defect.** `test_cor4_ce1_qwen2vl…` required every
  modality path to *contain a dot* — a proxy for exactness that the fabricated
  prefix satisfied and a true top-level leaf cannot (its `{"vision_config"}`
  carve-out was the first honest bare leaf hitting it). Replaced with the real,
  strictly stronger predicate: the path **resolves** in the witness.
- **A silent first-match.** `partial_rotary_factor` is declared twice by some
  configs (legacy top-level + nested in `rope_parameters`); the parser read only
  the first and never looked at its rival, so the nested one was cleared by the
  transitional bare-leaf fallback rather than read. Now both are read and
  COMPARED under COR-4's alias law (equal = redundant evidence, unequal =
  structured ambiguity authoring nothing). No corpus witness disagrees, which is
  why `ir` is unchanged.
- **A poison riding on incidental state.** `test_cor5_poison_bare_funnel_read…`
  reached the inexact-read law only because llama's `rms_norm_eps` happened to
  be unpathed, and accepted *either* law's message via an `or` — so it could
  never prove which one fired, and it retired itself the moment that reader was
  fixed. Split into two poisons, each against a construction that cannot rot:
  unconsumed-ness (llama, integration) and inexactness (a constructed event,
  with an exactly-pathed control that must be lawful). The inexact-read law in
  `claims_audit` was never broken — verified by deleting it and watching the new
  poison fail.

**Lesson worth keeping:** each of the three was a green check resting on the
fabrication. A gate that asserts a *proxy* (the path has a dot; some message
matched) rather than the *predicate* (the path resolves) will bless the very
defect it was written to stop.

### What exists now

- `evidence/receipts.py` — typed `ProjectionReceipt` (fact key, owner,
  mechanism, surface, node path, projection kind, value/status hash);
  `RECEIPTED_SCOPES`; the occurrence→target→receipt join; reverse-fabrication.
- `RenderEvent.receipts` is the authoritative channel, emitted by the real
  declared-ops drill consumer. `facts_projected` remains as compat.
- The global `projection_receipts_available` flag is **retired**. Coverage is
  owner/mechanism-scoped: `ir.extras.config_access.projection_coverage
  .receipted_scopes`.
- Net 2 = `config_consumed_unreceipted`: blocks **unconditionally** inside a
  receipted scope; every other scope stays the advisory census.
- `ParseContext.component_namespace` carries ownership through recursive slots;
  modality owners are `<namespace>.<modality>`.

### Receipted scopes — the entirety of U2's coverage so far

```
('root.vision', 'projector_out_width')
('root.video',  'projector_out_width')
```

Everything else in the corpus is advisory and unreceipted. That is the honest
size of the pilot.

## The census is a DISCOVERY list, not a task list

`unfold-pkg/docs/COR5_NET1_MIGRATION_DEBT.md` — **279 occurrence-exact rows**,
2 owners with no consumption, 3 live claims. Regenerate with
`python3 scripts/census.py`; `--check` fails when the committed doc is stale.

Each section now names the DOCUMENT its paths are rooted in — a row is a
`(component, path)` pair and the path is document-relative, so without that a
reader cannot locate the value at all.

It grew 267 → 281 and that is **precision, not new debt**. The occurrence key
is `(component, path, spelling)`, so before the ownership fix a nested VLM text
encoder's `vision_config.*` reads and a genuine top-level VLM's reads were
**the same rows**. Un-merging them gave:

| owner | before | after |
|---|---|---|
| `root.vision` | 25 | 15 |
| `root.text_encoder.vision` | — | 24 (new true owner) |
| all others | — | unchanged |

The old 267 was an **undercount caused by the producer bug**. No registered
pending-debt entry grew. **267 must never be presented as current.**

U2.2a then moved 281 → **279**, and this too is precision, not payment. Two rows
did not get resolved — they got *joined*: once an inspected read and a consumed
read of the same field finally shared one true path, the consumption legitimately
excused the inspection. That join is what the fabricated prefixes had been
defeating. Sibling rows also un-merged and re-split as their real paths emerged.

Most of these 279 rows will not need interpretation. They must each get exactly
one disposition (U2.2 below).

## Plan (U2.1 → U2.9)

- **U2.1 — close the pilot. DONE.** (Both commits above; valid receipt;
  clean-checkout verified.)
- **U2.2 — classify every occurrence.** A row cannot be classified until its
  path is true and locatable, so this splits:
  - **U2.2a — make every path true. CODE DONE, BLESS PENDING.** The four
    producer laws above. Fabricated paths 0/25; behaviour byte-identical;
    `tests/test_config_paths.py` locks both the resolution invariant and the
    class-provenance one. Blocked only on Soumil's manifest re-bless.
  - **U2.2b — quarantine the honest-inexact.** 14 remain (was 305). Each is a
    reader touching a nested object without naming where it lives; the fix is
    per-reader (`wrapper_path`), not a census filter. `unconsumed_occurrences()`
    must stop publishing `path_exact=False` rows as occurrence-EXACT, and render
    them as their own not-yet-classifiable producer backlog.
  - **U2.2c — classify the genuinely-exact rows.** Exactly one disposition each:
    structural+already projected · structural but not projected · geometry-only ·
    display-only (typed display channel, cannot reach structural sinks) ·
    address/source-selection only (typed address channel) · non-architectural
    metadata (scoped-ignore with owner+reason) · unused/phantom (delete the read
    or fix ownership) · ambiguous/unsupported (preserve unknown, record typed
    failure — never choose a conventional value).
- **U2.3 — structural-author census.** Static + runtime coverage of every place
  architecture is authored: typed and legacy facts, IR/spec fields, nested
  extras, opgraph regions/ops, blocks/cards, HTML claims, expanded JSON,
  parameter formulas, conformance assumptions. Key is line-insensitive:
  `(module, enclosing symbol, sink kind, normalized target)`. Poisons proving a
  new author is detected in each of those 8 representations — so nobody bypasses
  receipts by moving a claim into another representation.
- **U2.4 — exact receipt contract.** Join: occurrence → consumed target →
  registered fact → canonical region/spec → actual consumer → receipt. A receipt
  is invalid if no registered fact/declared debt owns it, the owner differs, the
  mechanism differs, the value/status hash differs, or it names a consumer that
  did not really render/serialize/count the fact. No nominal "is drawn" booleans.
- **U2.5 — reusable emission at canonical boundaries** (opgraph/block
  projection, cards/prose, HTML drills, expanded JSON, parameters, conformance).
  One canonical fact projected to four surfaces yields four surface-specific
  receipts from shared infrastructure — never four model-family branches.
- **U2.6 — migrate by MECHANISM, not by model.** Order: projector dims (finish
  past the two pilot scopes) → scheduler surface → embedding/positional dims →
  attention geometry → FFN dims → norm facts → vision/audio tower geometry →
  denoiser geometry → VAE/UNet geometry → JSON/params/conformance. For each:
  pick one registered fact → find every consumer → emit receipts from those real
  consumers → positive/negative/ownership-collision controls → declare the
  (owner, mechanism) scope receipted → Net 2 blocking for it → regenerate the
  census and prove the expected shrink → commit independently.
- **U2.7 — quarantine later-semantics facts as exact debt.** Required fields:
  exact owner; config occurrence or evidence source; fact/mechanism; current
  writer; current consumers; why it cannot yet be proven; assigned future unit;
  deletion criterion. A generic "scheduler pending" allowlist is not sufficient.
- **U2.8 — scheduler is U2's first domain, with a hard boundary.**
  **U2**: register its structural claims, connect the genuinely sound ones to
  consumers, emit receipts, expose unsafe claims as exact debt.
  **U3**: resolve and index the exact `step()`/helper graph.
  **U13**: derive scheduler semantics and delete the class/config-selected
  fallbacks. Do **not** receipt an Euler/flow/multistep claim during U2 unless
  it is already genuinely evidence-backed.
- **U2.9 — flip blocking incrementally.** Migrated scope → missing receipt
  blocks, fabricated receipt blocks. Unmigrated → findings stay visible, each
  with an exact classified disposition; the pending-debt count only shrinks; no
  new pending entry without owner, reason and assigned unit. Never mass-register
  the remaining rows to flip the corpus blocking.

## U2 acceptance (all must hold)

- every structural-author surface is covered by the static/runtime census;
- every current structural claim is either a registered fact with a real
  consumer receipt, or exact owner-qualified shrinking debt;
- every config occurrence has a classified disposition;
- every migrated (owner, mechanism) scope has blocking Net 2;
- reverse fabrication is blocking everywhere;
- renderers, JSON, parameters and conformance cannot invent unregistered claims;
- new structural writers fail poison tests;
- no family table or class-name shortcut was added to obtain receipt coverage;
- the regenerated census contains no unclassified row;
- preservation, corpus, isolated-audit, full-suite and clean-checkout gates all
  pass on one unchanged tree.

## Known gaps / follow-ups

- **Class-hydrated provenance — DECIDED (Soumil, 2026-07-17): required, and
  LANDED.** `model_type` may LOCATE the installed config class — identity as
  *address* is lawful — but what that class supplies may not masquerade as what
  the checkpoint declared.

  My first framing of this was wrong and worth remembering: I measured "0 class
  keys are CONSUMED into a structural fact" and called the doctrine safe. That
  metric was blind. A class-supplied `layer_types` is recorded as merely
  `inspected` and still decides the mask of **every layer** — the influence was
  structural, the ledger's inspected-vs-consumed distinction is exactly the gap
  U2 exists to repair, so measuring through it proved nothing.

  Now: `hydrate_with_provenance` returns the document AND `{path: kind}`, which
  travels with `document_scope` onto every event. Vocabulary:
  `checkpoint_declared` · `class_default` · `class_normalized_alias` ·
  `loader_metadata`. **`class_normalized_alias` is deliberately unpopulated**:
  it needs an explicit normalization trace naming the raw path, and inferring it
  from a class-added key *resembling* a removed raw one (`rope_scaling` →
  `rope_parameters`) would invent the very source relationship the map exists to
  record. Class-derived is the honest answer until a trace exists.

  Class-supplied fields are **out of the checkpoint census** (asking "classify
  this declaration" is incoherent for one the checkpoint never made) and
  **visible** in their own section — 12 corpus-wide, all `class_default`.

- **⭐ STILL OWED — fact-level provenance.** The event now carries provenance;
  the FACT it authors does not yet. `facts.py` already ranks `class_default`
  beside `config_declared` and `registry.py:222` already allows both, so the
  vocabulary exists — what is missing is deriving a fact's status from the
  provenance of the read that decided it, so no class-added key can be reported
  as `config_declared`. Until then the Gemma-2 regression proves both halves at
  the EVENT/document level (the class supplies the schedule → `class_default`;
  the schedule really builds the alternating stack; the checkpoint alone is
  uniformly sliding), which is the honest limit of what is wired.

- **⭐ NOTED — hydration breaks embedded ≡ standalone.** Hydration runs only on
  the encoder-slot path, so the SAME Gemma-2 config renders a heterogeneous
  stack when embedded and a uniformly-sliding one when parsed standalone. The
  counterfactual is asserted in the regression rather than left implicit. This
  is a real parity question, not a U2.2a defect — flagged, not absorbed.
- **Clean-checkout import guard** — CLOSED in `86f03ba` (the archive now imports
  the package and parses two models).
- `facts_projected` (key-level) and the typed receipt channel overlap; the
  former is temporary compat.
- **14 honest-inexact reads remain** (`path_exact=False`): a reader touched a
  nested object without naming where it lives. `detect.cross_attention_layers`
  was the first, fixed by sharing `common.wrapper_path` (promoted out of a
  closure in `transformer/parser`, so one implementation serves both). The rest
  are U2.2b's quarantine work.
- Two owners still have zero consumption (`root.scheduler`, `root.denoiser` on
  one witness).

## Operating notes

- The working tree is shared; a concurrent actor has edited it mid-gate before
  (`sources.py`/`test_code_evidence.py` at 10:51 during a run), which invalidates
  a fingerprint receipt. Check `git status` and file mtimes before trusting a
  gate, and re-run on a quiescent tree.
- A commit must contain **every file the green tree used**, including previously
  untracked production modules and tests — otherwise the commit does not
  reproduce. Soumil's `U0_U1_*` plan docs stay untracked deliberately.
- Gate shape, every unit: manifest+all-25 → each audit file alone → grouped →
  full suite → fingerprint unchanged → clean-checkout. `test_support/tree_state
  .fingerprint` hashes the whole tree (content + path + exec bit), so any edit
  during a run invalidates it.
