# U3 accomplishments and current-procedure audit

Status: authoritative current audit guide as of local commit `4bd1395`.

This document answers two questions:

1. What did U3 actually accomplish?
2. What must Soumil check before accepting future parser, evidence or rendering
   work as correct?

It does not replace the historical execution record in `docs/U3_RUNBOOK.md`.
It interprets that record against the current code and defines the review
procedure going forward.

---

## 1. The shortest accurate explanation of U3

Before U3, many readers independently reopened Hugging Face files, searched for
classes or field patterns, and returned a model-level answer. A correct answer
could therefore come from the wrong class, a sibling component, an unused
module, or a union of several different layer implementations.

U3 built a shared source and ownership substrate:

```text
source bundle
  → one neutral ProgramIndex
  → exact component root
  → exact constructed occurrence
  → mechanism reader
  → typed ReaderResult
  → owner-qualified fact
  → parser/IR
  → renderer, params and conformance
```

The essential improvement is:

> A mechanism reader must start from the exact occurrence that the model
> constructs and executes. It may not search the whole source bundle for a
> convenient class-level vote.

U3 answers **where the relevant code is** and gives readers neutral observations
of that code. It does not, by itself, finish the semantic interpretation of
every architecture.

---

## 2. What U3 concretely added

### 2.1 One neutral source index

Primary implementation:

- `model_unfolder/evidence/program_index.py`

`ProgramIndex` records source syntax once per `SourceBundle`:

- exact source identities and content fingerprints;
- imports and exact symbol bindings;
- classes and bases;
- callables and parameters;
- construction sites;
- assignments and exact target patterns;
- calls, returns, loops, branches and comprehensions;
- attribute and config-path observations;
- exact spans, guards and lexical positions;
- transformed, ambiguous and unsupported observations.

The index is deliberately observation-only. It must not contain semantic labels
such as `is_attention`, `is_ffn`, `is_rope`, `is_decoder`, model families or
renderer terminology.

Content fingerprinting prevents a stale index from being reused after source
content changes even when file modification time does not.

### 2.2 Exact component and occurrence ownership

Primary implementations:

- `model_unfolder/evidence/component_owner.py`
- `model_unfolder/evidence/container_inventory.py`
- `model_unfolder/evidence/repeated_child.py`
- `model_unfolder/evidence/construction_calls.py`
- `model_unfolder/evidence/config_scoped_owner.py`
- `model_unfolder/evidence/decoder_block.py`
- `model_unfolder/evidence/output_repeated_stage.py`

These boundaries prove addresses such as:

- the exact component root;
- the declared base-model stage;
- a constructed child occurrence;
- a repeated container and its symbolic element template;
- an exact decoder-block occurrence;
- an external primitive construction;
- an exact helper/factory return;
- the config-prefix chain passed into that occurrence.

The ownership layer does not decide what a child means. An occurrence address
cannot become “attention” or “FFN” merely because its class or field name looks
familiar.

### 2.3 Neutral local execution relations

Primary implementation:

- `model_unfolder/evidence/execution_flow.py`

U3 added:

- stable call-site identity;
- addressed invocations;
- loop/container invocation templates;
- versioned local definitions;
- conservative producer-to-consumer relations;
- conditional relations;
- unresolved calls and relations;
- unsupported execution regions.

This is an **open, conservative local-relation substrate**, not a complete
Python control-flow graph or symbolic executor. Every corpus result remains
partial/open unless a later bounded unit proves a stronger coverage contract.

No consumer may interpret “no unresolved relation was noticed” as proof that a
callable was completely understood.

### 2.4 Closed reader outcomes

Primary implementation:

- `model_unfolder/evidence/reader_result.py`

Readers return typed states instead of overloading `None`:

- `resolved`;
- `absent`;
- `ambiguous`;
- `failed`;
- where applicable, explicit partial/incomplete evidence.

Each result retains exact provenance, failures or rivals. Ambiguity cannot carry
a structural value, and failure cannot silently become a conventional
architecture.

### 2.5 Exact factory and config binding

U3 closes only conservative, directly provable factory paths:

- an exactly indexed classmethod;
- one unguarded return;
- exact positional/keyword formal binding;
- no unsupported control flow;
- no unknown `*args`/`**kwargs`;
- no rival returned constructors;
- no inferred binding from names or equal values.

Anything outside that envelope remains typed unresolved.

### 2.6 A non-growing legacy quarantine

Current generated inventory:

- `docs/U3_CURRENT_READER_INVENTORY.md`

Current measured debt:

- 25 quarantined semantic readers;
- 33 classified evidence-layer `ast.parse` sites;
- 23 legacy model-source parse sites.

The quarantine blocks:

- a new or moved `*_from_files` reader;
- a new production caller;
- a renamed equivalent reader;
- a body or helper-closure change;
- a new evidence-layer model-source parse;
- movement of an existing parse authority;
- stale generated inventory.

The inventory is current under:

```bash
python3 scripts/generate_u3_reader_inventory.py --check
```

Quarantine is not completion of those readers. It is a guarantee that old
authority cannot silently grow while U6–U14 delete it mechanism by mechanism.

### 2.7 Reproducible verification

Primary implementation:

- `scripts/verify_commit.py`

The verifier runs independent committed-tree lanes for:

- static checks;
- collection;
- focused affected tests;
- U2 authority gates;
- preservation;
- the exhaustive suite;
- complete before/after tree fingerprints.

Each lane runs in a detached worktree. Passing in the developer’s dirty working
tree is not sufficient.

---

## 3. Semantic readers migrated during the U3 campaign

The U3 campaign also implemented several semantic readers on top of the new
substrate. They demonstrate that U3 works, but their semantic ownership belongs
to later units.

Examples include:

- embedding-stage normalization;
- exact attention-child and Q/K/V storage;
- dispatch-selected attention equivalence;
- Q/K normalization;
- ordinary FFN mechanism;
- routed-expert storage;
- exact norm primitives and placement evidence;
- projection bias;
- learned attention sinks;
- additive cross-attention schedule;
- codebook stream aggregation;
- manual weight tying;
- parallel attention/FFN normalization;
- structured encoder output-stage ownership;
- nested storage conformance.

Correct attribution:

- U3 owns the shared indexing, addressing and result substrate.
- U6 owns attention semantics.
- U7 owns FFN, norm, router and cell semantics.
- U8 owns position, masks and layer schedules.
- U9 owns recursive modality towers, projectors and fusion.
- U10 owns diffusion transformer semantics.
- U11 owns UNet semantics.
- U14 owns downstream consumer/conformance closure.
- U15 proves that legacy semantic authority and quarantine debt reach zero.

This distinction prevents “built during U3” from being mistaken for “all
semantic work is complete.”

---

## 4. What U3 did not accomplish

U3 does **not** mean:

- every architecture mechanism is now understood;
- every model-source file is parsed only by `ProgramIndex`;
- all 25 quarantined readers are deleted;
- the local execution substrate is a complete CFG;
- every repeated container is proven to execute in storage order;
- all model schedules are occurrence-exact;
- every nested modality owner is resolved;
- all config has been removed;
- renderers and parameter estimators are fully fact-only;
- unknown is safe at every remaining legacy consumer;
- U6–U15 are complete.

U3 also does not justify turning a reader failure into a generic diagram without
examining whether the missing detail can be re-proven through a missing exact
boundary.

---

## 5. The DeepSeek/GLM lesson: output parity is not proof parity

The old reader already drew a detailed FFN for DeepSeek-V3 and GLM-4.5.
It reached that answer through a broad source scan and class-level behavioral
union. For these models the answer happened to be correct, but the reader could
mix:

- different layer alternatives;
- shared FFNs and routed experts;
- reachable and unused children;
- owners from different scopes.

The first exact-owner U3 reader refused to choose between the dense and MoE
`self.mlp` construction sites. That was safer, but incomplete, and temporarily
made the diagram generic.

The review then made the wrong procedural decision: it accepted the lost detail
as an honest unknown before asking whether the old answer could be re-proven on
the exact substrate.

The current U7 correction proves the common ordinary/shared FFN mechanism across
every exhaustive alternative. It restores the same visual result through a
materially stronger proof.

Therefore:

> Detailed → generic → detailed is a real product-output loop. The final proof
> is progress, but the detour should have been prevented.

The permanent prevention rule is in Section 7.

---

## 6. Current end-to-end procedure

For every new or corrected architectural claim, review the following chain.

### Gate A — Define the exact claim

Write the claim before coding:

- owner/component;
- mechanism;
- value;
- scope;
- completeness;
- intended consumer.

Examples of different claims that must not be conflated:

- “this exact layer is MoE”;
- “all scheduled layers are MoE”;
- “both dense and MoE alternatives contain a split-gated ordinary/shared FFN”;
- “routed experts use fused gate/up storage.”

Specify the quantifier:

- one exact occurrence;
- every exhaustive alternative;
- some layers;
- every scheduled layer;
- model-level stage.

Stop if the quantifier is unclear.

### Gate B — Establish source and document provenance

Check:

- the correct component source bundle was resolved;
- the source is checkpoint/library appropriate;
- content fingerprints match;
- external files carry explicit provenance;
- parse failures are typed;
- source-missing does not consult identity/config fallback.

Identity may locate a file or declared class. It may not decide the mechanism.

### Gate C — Resolve the exact owner

Require:

- resolved component root;
- exact construction occurrence;
- exact config-prefix chain when config participates;
- owner graph round-trip;
- no sibling search;
- no bare class-name collision;
- rivals and unsupported paths retained.

If there are multiple occurrences, either:

- prove which occurrence the config/schedule selects; or
- inspect every exact exhaustive alternative and require unanimity for only the
  common fact.

Never select the first, shortest, best-looking or most model-like candidate.

### Gate D — Prove the mechanism locally

Require evidence from the exact owner:

- construction storage;
- invocation;
- producer/consumer dataflow;
- return/output reachability;
- guard and path conditions;
- exact config premise where code dispatches through config.

Observation is not execution. Construction is not invocation. Invocation is not
output influence. A field name is not a mechanism.

### Gate E — Close the typed evidence object

The result object must reject forged combinations:

- wrong owner;
- wrong source/span;
- wrong construction site;
- wrong config path;
- rival omission;
- semantic disagreement;
- non-exhaustive alternatives;
- duplicate or self-certified evidence.

A downstream consumer must not be able to trust a stronger statement than the
DTO itself enforces.

### Gate F — Publish one canonical fact

Check:

- fact key is registered;
- owner is exact;
- status matches the evidence strength;
- code evidence carries source spans;
- config evidence carries exact config paths;
- derived evidence carries premises;
- unknown remains unknown;
- parser and conformance consume the same cached reader result.

Do not publish the same architectural fact independently through specs, extras,
labels and parameters.

### Gate G — Project through real consumers

Inspect every consumer:

- architecture view;
- detailed block/card view;
- expanded JSON;
- parameter estimates;
- HTML metadata;
- conformance;
- Sable/Dable findings.

Renderers and parameter estimators may format or calculate from canonical facts.
They may not inspect source/config or invent mechanism defaults.

Where a projection receipt is required, it must come from the real consumer and
join to the exact fact and value—not to an expectation manufactured by that
same consumer.

### Gate H — Run adversarial controls

Every mechanism migration needs:

- real positive witness;
- real close negative;
- synthetic rename control;
- same class constructed twice;
- sibling component with the same field spelling;
- conflicting alternatives;
- non-exhaustive branch;
- dynamic/unsupported construction;
- source-missing or partial-source case;
- equivalent implementation with different class/field names;
- mixed architecture containing both mechanisms;
- routed/shared/dense separation where applicable.

Do not accept corpus parity alone. The corpus might not exercise the unsafe
branch.

### Gate I — Inspect the semantic delta

Classify every changed artifact:

- expected architectural correction;
- evidence/provenance-only change;
- parameter correction;
- metadata-only change;
- unintended visual/structural regression;
- stale manifest/input correction.

No blessing may change merely because tests are red.

### Gate J — Verify the committed tree

Minimum:

```bash
git diff --check
python3 -m pyflakes <changed-python-files>
python3 scripts/generate_u3_reader_inventory.py --check
python3 scripts/verify_commit.py \
  --commit <commit> \
  --workers 4 \
  --focus <affected-test-file>
```

Acceptance requires:

- static clean;
- collection clean;
- focused controls green;
- all U2 authority gates green;
- preservation green;
- exhaustive suite green;
- isolated committed checkout;
- before/after fingerprints identical.

---

## 7. Mandatory lost-detail procedure

This gate is required whenever a migration makes a diagram more generic,
removes a block, changes a label to unknown, or lowers parameter specificity.

### Step 1 — Reproduce both versions

Capture:

- last blessed output;
- candidate output;
- exact changed surfaces and views;
- introducing commit.

### Step 2 — Identify the old authority

Determine whether the old detail came from:

- exact source evidence;
- broad whole-file scan;
- config declaration;
- class default;
- family/model table;
- renderer default;
- parameter convention.

Correct output from weak authority is still weak authority, but it is a
re-proving obligation—not permission to discard the detail.

### Step 3 — Inspect the real source

Check the exact selected component and occurrence, plus every genuine scheduled
or conditional variant.

Answer:

- Is the detail explicitly present?
- Does it execute?
- Does it influence the returned value?
- Is it common to every relevant alternative?
- Is it layer-specific rather than model-global?

### Step 4 — Attempt exact re-proof

Before accepting generic output, ask:

> Is the missing result caused by absent source evidence, or by a missing
> ProgramIndex/owner/mechanism capability?

If the source is specific but the reader cannot express it, record a substrate
gap and build the smallest neutral/exact prerequisite. Do not call the output an
honest unknown yet.

### Step 5 — Attack the proposed proof

Add counterexamples where:

- alternatives disagree;
- one shared-looking child is never invoked;
- a child executes but does not reach the return;
- the conditional is non-exhaustive;
- the same field belongs to attention or another mechanism;
- routed experts exist without a shared FFN;
- two layers genuinely have different structures.

### Step 6 — Only then request Soumil’s decision

The review packet must say one of:

- **Source-proven correction:** detail should remain/be restored.
- **Source genuinely incomplete:** generic is the honest output.
- **Substrate gap:** implementation must stop; generic is not yet approved.
- **Presentation decision:** evidence is complete, but diagram depth is a
  product choice.

“The new reader returned `None`” is never an acceptable reason by itself.

---

## 8. Soumil’s review checklist

Use this checklist for every proposed unit.

### Architecture and scope

- [ ] What exact architectural claim is changing?
- [ ] What owner and component does it belong to?
- [ ] Is the claim per occurrence, per layer schedule, per alternative or
      model-global?
- [ ] Is this neutral U3 infrastructure or semantic U6–U14 work?
- [ ] Is the proposed change the smallest boundary that can prove the claim?

### Source and ownership

- [ ] Was the exact Hugging Face source inspected?
- [ ] Was the exact configured class/occurrence resolved?
- [ ] Are nested component paths preserved?
- [ ] Are every rival and parse failure retained?
- [ ] Is any class name, model type, family name or field substring selecting
      architecture?

### Evidence quality

- [ ] Does construction evidence prove execution?
- [ ] Does execution evidence prove output influence?
- [ ] Does a negative claim have complete-enough evidence?
- [ ] Are conditional alternatives exhaustive?
- [ ] If alternatives agree, is only their common fact published?
- [ ] If they disagree, does the result remain ambiguous?

### Fact and consumer integrity

- [ ] Is there exactly one canonical fact?
- [ ] Does the parser use the reader result instead of reinterpreting?
- [ ] Does conformance reuse that same result?
- [ ] Do renderer and params consume the fact rather than raw config/source?
- [ ] Is every strengthened visual claim receipted?
- [ ] Can a sibling owner launder evidence or debt?

### Regression protection

- [ ] Positive real model?
- [ ] Close real negative?
- [ ] Renamed synthetic equivalent?
- [ ] Mixed/scheduled model?
- [ ] Dynamic and incomplete-source cases?
- [ ] Existing blessed models unchanged except named deltas?
- [ ] Every affected detailed view manually inspected?

### Release receipt

- [ ] Commit contains only intended files?
- [ ] No unrelated working-tree changes staged?
- [ ] Inventory `--check` green?
- [ ] Static and pyflakes clean?
- [ ] Focused and authority gates green?
- [ ] Full suite green?
- [ ] Preservation green?
- [ ] Fingerprints identical?
- [ ] Exact expected deltas written down?
- [ ] No blessing changed without Soumil’s explicit decision?

---

## 9. Current verified state

Current local commit:

```text
4bd1395 prove conditional shared FFN mechanisms from source
```

Current committed-tree receipt:

```text
/private/tmp/model-unfolder-verification/7b368e2bd5
```

Results:

- static: PASS, 15 changed Python files clean;
- collection: 2,265 tests;
- focused: 290 passed;
- U2 authority: 44 passed;
- preservation: 46 passed across 26 witnesses;
- exhaustive: 2,162 passed, 11 skipped, 2 expected xfailed;
- every lane fingerprint: identical before/after.

The current DeepSeek-V3/GLM correction:

- restores the detailed split-gated ordinary/shared FFN;
- preserves dense-versus-MoE layer scheduling;
- keeps routed experts separate;
- leaves GPT-OSS routed-only/ordinary-unknown;
- adds no family/model production branch;
- changes no gallery or pixel hash;
- retires the final six `ffn_storage` asserted-convention rows.

---

## 10. Authoritative document map

Use the documents in this order:

1. **This document** — current accomplishments and audit procedure.
2. `docs/U3_COMPLETION_MASTER_PLAN.md` — finite U3 boundary and unit
   attribution.
3. `docs/U3_RUNBOOK.md` — historical execution ledger and individual receipts.
4. `docs/U3_CURRENT_READER_INVENTORY.md` — generated current quarantine and
   parse-authority worklist.
5. `docs/U3_SEMANTIC_DELTA_ADJUDICATION.md` — historical semantic-delta audit,
   including superseding corrections.
6. `docs/U7_CONDITIONAL_SHARED_FFN_PROOF.md` — exact DeepSeek/GLM correction.

Historical documents may describe the state at the time they were written.
Later explicitly marked superseding sections and this current audit guide govern
the present procedure.

---

## 11. Final U3 verdict

U3 successfully established the reusable, identity-blind program/ownership
substrate and froze the remaining legacy authority. That is genuine shared
infrastructure, not per-model support.

U3 did not finish architecture understanding. The project now has a lawful way
to migrate each remaining mechanism without returning to family tables,
whole-file voting or renderer inference.

The current procedure is correct only if it preserves this separation:

```text
neutral observation
  ≠ owner resolution
  ≠ mechanism interpretation
  ≠ fact arbitration
  ≠ rendering
```

The most important review rule is:

> Never accept lost architectural detail as “more honest” until the exact
> source has been checked and the old result has been given a fair attempt to be
> re-proven through the new substrate.
