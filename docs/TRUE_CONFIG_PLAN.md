# TRUE_CONFIG — implementation plan

> **Status: `#TODO` — not implemented.** No code reads or produces a true config
> today. This is the worklist only.
>
> **Supersedes** the first version of this document, which proposed a grouped
> schema with `_provenance`/`_unresolved`/`_coverage` nested inside the returned
> object. That design was rejected: it was a bloated dictionary of our own
> invention, not a config.

Durable product intent lives in `z-docs/01-product/true-config.md`. The field
vocabulary lives in `z-docs/08-reference/canonical-config.md`. This document is the
detailed worklist and holds no authority over either.

---

## 1. The contract

`true_config(x)` returns **the config.json this checkpoint should have shipped**:
the model's own field spellings and nesting, made complete and correct by filling in
every field the modeling code reads but the checkpoint left implicit.

Diffed against the repository's own `config.json`, every added line must be a field
that legitimately belongs there. The returned object *is* a config — no provenance
keys, no regrouping, no invented structural annotations.

## 2. The field-belonging rule

A field belongs when, and only when, **the modeling code reads it from config**.

This yields both properties at once:
- **complete** — nothing the architecture needs is missing;
- **minimal** — nothing the code does not read is added.

Bounded further by the canonical reference: a true config is the intersection of
that curated vocabulary with the fields this model's code actually reads. A code-read
field absent from the reference is surfaced as a reference-extension signal, never
silently emitted.

## 3. Evidence base (harvested, not assumed)

Established against real declarations on the development machine:

| Source | Supplies | Extent |
|---|---|---|
| installed `transformers` config classes | declared field set + defaults per family | 37 families of 656 registered |
| installed `diffusers` model/scheduler `__init__` signatures | the diffusers config contract | 21 classes |
| blessed corpus fixtures | real serialized checkpoint values | 25 checkpoints, 229 distinct fields |

### Findings that constrain the implementation

1. **RoPE is one object, not a top-level pair.** 28 harvested families declare
   `rope_parameters`; `rope_theta`/`rope_scaling` are not top-level class fields in
   the installed version. The real DeepSeek-V3 checkpoint serializes
   `rope_parameters` and carries no `rope_scaling`. The assembler must emit the
   model's actual spelling, not a remembered one.
2. **Some structural facts are declared fields in some families.** `is_gated_act`
   (T5/UMT5), `apply_residual_connection_post_layernorm` (BLOOM), and
   `num_ln_in_parallel_attn` (Falcon) are genuine config fields for facts that are
   code-proven elsewhere. The belonging rule handles this without special-casing:
   the code reads them, so they belong.
3. **Class defaults and real checkpoints each miss what the other has.**
   `scoring_func`/`topk_method` appear in the real DeepSeek-V3 config but are
   declared by almost no config class; conversely, unserialized class defaults never
   appear in a checkpoint. Neither source alone is sufficient — and the third
   source, *what the code actually reads*, is precisely what this feature adds.

## 4. Stages

**Stage 1 — Field-demand scan.** AST-scan the resolved source, per component and
scope, for every config read (`config.x`, `getattr(config, "x", …)`, `config["x"]`)
across the construction and forward closure. Output: the exact demand set. This is
the one genuinely new extractor; it is also reusable as the basis of an
unclaimed-signal check (a field the code reads that no fact consumes).

**Stage 2 — Value resolution.** For each demanded field, in order: serialized config
value → installed class default → code-derived default expression → unresolved.
Reuses the existing hydration channel, the class-default tier, and the config
expression evaluator. Never guesses.

**Stage 3 — Config-shaped assembly.** Merge into the model's native shape: keep
present fields, add resolved-missing ones, recurse sub-configs under their own
component source, preserve identity fields as addresses. No grouping, no sidecar.

**Stage 4 — Companion (separate object).** Provenance `{field → status, source}` and
the config-versus-true diff `{kept | filled-from-class-default | derived-from-code |
normalized | unresolved}`. Returned only on request; never inside the config.

## 5. Surface

- `Diagram.true_config()` → the clean config dict.
- `true_config(x)` top level.
- `true_config(x, with_provenance=True)` → `(config, companion)`.
- `.save("model.true.json")` writes the clean config only.

## 6. Rules the implementation must not break

- **Serializer only.** It reads already-resolved model representation and evidence.
  It never recomputes a fact. Guarded by a forbidden-import test, in the manner of
  the renderer firewall.
- **No invented fields.** Anything outside the canonical vocabulary is surfaced, not
  emitted.
- **No guessed values.** A demanded field that cannot resolve is omitted from the
  config and reported in the companion as a real gap.
- **Structural facts with no config field stay out.** They belong to the diagram and
  facts.
- **Dead flags normalize to the code's effective value**, with the checkpoint's
  stated value recorded in the companion.
- **Source-missing degrades honestly.** Without source, the demand set is unknown:
  return the present config plus an explicit note that completeness is unverified.
  Never fill from convention.

## 7. Verification

- every demanded field is resolved or explicitly unresolved;
- **round trip** — re-running the product on a true config reproduces the original
  architecture signature;
- **anti-bloat** — no field the code never reads appears in the output;
- **no upgrade** — a field with no recorded provenance channel is never presented as
  code-proven;
- **unknown never filled** (poison) — an unresolved demand must not acquire a value;
- **values stay config** — geometry fields remain checkpoint-declared.

## 8. Build order

1. field-demand scan (per component, scope-qualified);
2. value-resolution ladder over the demand set;
3. config-shaped assembler with sub-config recursion;
4. `Diagram.true_config()` / top-level surface / `.save`;
5. companion (provenance + diff);
6. the verification set in §7;
7. widen the canonical reference harvest beyond 37 families as a completeness net.

## 9. Known limits carried into the work

- 37 of 656 families harvested; the namelist was chosen for mechanism coverage, not
  popularity. The remainder are unaudited.
- Diffusers schemas are `__init__` signatures — the config contract, but version
  bound. Harvest ran against `transformers 5.12.1` / `diffusers 0.38.0`.
- The demand scan requires resolvable source. Coverage of true config is therefore
  bounded by source resolution, and says so rather than compensating.
