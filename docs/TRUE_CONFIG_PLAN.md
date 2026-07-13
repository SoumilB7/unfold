# TRUE_CONFIG — the complete config a checkpoint *should* have shipped

*A design plan for `true_config()`: reconstruct, from the partial config + the code
evidence we already resolve, the full structural config — every field a config
would have carried if configs declared structure — with provenance on every field
and honest holes where nothing is proven.*

*Committed on `audio-composite-support` as a separate DESIGN-ONLY change (no code).
Written 2026-07-13. Companion to `CONFIG_VS_CODE_CONVERTIBILITY.md` (workspace root)
and `EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md`.*

---

## 1. What it is (one sentence)

> `true_config(x)` returns the **config-shaped, provenance-carrying dict of the
> fully-resolved architecture** — the checkpoint's declared values *plus* every
> structural fact the code proves, each tagged with the evidence channel that
> decided it, and every still-unproven fact listed honestly as unresolved.

It answers a question the raw `config.json` cannot: *"given this partial config and
the modeling source, what is the complete structural specification of this model?"*

## 2. The core realization — it is a PROJECTION, not new acquisition

The parsed `ModelIR` + `FactLedger` (`ir.extras["fact_provenance"]`) is already the
fully-resolved structural truth. `true_config()` is a **fourth projection of the one
declaration** (alongside SVG, JSON, cards — Law 2, "one declaration, N projections"):
it serializes what is *already decided* into a config-shaped view. It never
re-derives a fact. This is the single most important constraint on the design.

```text
              ┌─ SVG (diagram)
              ├─ to_json() (traceable schema)
ModelIR ──────┼─ cards (prose)
+ FactLedger  └─ true_config()  ← NEW: config-shaped + provenance
```

## 3. The four guarantees it inherits from the laws

| # | Guarantee | Why | Mechanism |
|---|---|---|---|
| G-a | **Provenance never dropped** | a flat merge hiding channel = defaults-as-facts, the sin we killed | every field has a `_provenance` entry: `{status, source}` |
| G-b | **Unknown stays unknown** | Law 5 — honesty outranks completeness | holes go in `_unresolved` with reason (`oracle_missing` vs `ambiguous`), never filled |
| G-c | **Values vs shapes honored** | the binding law (§1.2) | geometry = checkpoint value; structure = code value; a code-named gate shows both |
| G-d | **One author (G-8) — serializer only** | true_config must not become a 2nd interpreter | reads decided IR/ledger values ONLY; unattributed IR fields tagged `ir_spec`, never upgraded |

## 4. Output shape

A nested, config-shaped dict grouped by owner/mechanism, with two sidecars:
`_provenance` (per field) and `_unresolved` (honest holes). Illustrative
(decoder LLM, e.g. a Llama-4-shaped model):

```jsonc
{
  "_true_config_version": "0.1",
  "_identity": { "model_type": "llama4_text", "note": "address/label only — not structural" },

  "geometry": {
    "hidden_size": 5120, "num_hidden_layers": 48,
    "num_attention_heads": 40, "num_key_value_heads": 8,
    "intermediate_size": 8192, "head_dim": 128, "vocab_size": 202048
  },
  "attention": {
    "kind": "gqa",
    "qk_norm": true,
    "rope": { "applied": true, "dim": 128, "theta": 500000, "nope_layers": [3,7,11] },
    "bias": { "q": false, "k": false, "v": false, "o": false },
    "score_scale": "1/sqrt(head_dim)"
  },
  "ffn":  { "gated": true, "activation": "silu", "storage": "fused_gate_up" },
  "norm": { "kind": "rmsnorm", "placement": "pre", "eps": 1e-5 },
  "moe":  {
    "per_layer": [true, true, "…"], "num_experts": 16, "experts_per_tok": 1,
    "expert_storage": "fused_gate_up",
    "router": { "scoring": "sigmoid", "aux_bias": true, "score_before_topk": true }
  },

  "_provenance": {
    "geometry.hidden_size": { "status": "config_declared", "source": "config.hidden_size" },
    "ffn.gated":            { "status": "code_proven",     "source": "decoder_ffn_gated_from_files @ modeling_llama4.py" },
    "norm.placement":       { "status": "code_proven",     "source": "norm dataflow @ modeling_llama4.py" },
    "moe.per_layer":        { "status": "code_and_config", "source": "is_moe_layer gate @ modeling_llama4.py; config.moe_layers" },
    "attention.qk_norm":    { "status": "code_proven",     "source": "decoder_qk_norm_from_files" }
  },

  "_unresolved": [
    { "field": "attention.chunked", "status": "ambiguous",     "reason": "attention_chunk_size present but reader not wired (bucket B)" },
    { "field": "scheduler",         "status": "oracle_missing", "reason": "not a diffusion model" }
  ],

  "_coverage": { "resolved": 23, "unresolved": 2, "unattributed": 1 }
}
```

### Access modes
- `true_config(x)` → the full object above (values + `_provenance` + `_unresolved`).
- `true_config(x, provenance=False)` → just the clean nested values (quick read).
- `true_config(x, diff=True)` → **the killer view**: a 3-way split
  `{declared_in_config, added_by_code, still_unknown}` — literally *"what your config
  was silent about and what code recovered."* This is the direct visual companion to
  `CONFIG_VS_CODE_CONVERTIBILITY.md`'s buckets.
- Surfaces: `Diagram.true_config()` + top-level `true_config(x)` + `.save("m.true.json")`.

## 5. Where every field comes from (no new work at read time)

All inputs are already produced during `config_to_ir`:

| Source | Supplies |
|---|---|
| `ir` typed specs (`LayerSpec`/`AttentionSpec`/`FFNSpec`, extras) | the resolved values |
| `ir.extras["fact_provenance"]` (FactLedger) | status + source per recorded fact |
| `ir.extras["config_audit"]` (`accessed`/`unread`) | which config fields were read; unread → candidate `_unresolved` |
| `ir.extras["source_provenance"]` | which files/architecture backed each component |
| `ParseContext.class_defaults` tier | `class_default` provenance for hydrated fields |

`true_config()` is therefore a **pure, cheap projection** over an already-parsed
`Diagram`. No source re-resolution, no re-parse.

## 6. Bounded honesty — true_config IS the campaign's coverage meter

The FactLedger is incremental today (H2 closed-registry census is still open), so
`true_config()` today is **partial and must say so**:

- fields with a ledger record → full provenance;
- IR spec fields with a value but no ledger record → emitted, tagged
  `status: "ir_spec"` (unattributed) — honest that the channel is not yet recorded;
- unread/ambiguous/oracle-missing facts → `_unresolved`.

`_coverage` makes the gap a number. As H2/H10 register every structural fact, the
`unattributed` count trends to zero and `_unresolved` shrinks — **true_config
becomes complete exactly when the config→code campaign is done.** It is the natural
"definition of done" artifact and a standing progress dashboard.

## 7. The round-trip property (metamorphic test + real value)

Feeding `true_config(x)` back through the parser should reproduce the same diagram —
because it is a superset of the original config with structure made explicit:

```text
unfold(config)          → diagram A, signature S_A
unfold(true_config(A))  → diagram B, signature S_B      assert S_A == S_B
```

This is both a **test** (any drift = a projection that added or lost a fact — a bug)
and a **feature** (a portable, complete, source-free structural spec that renders
identically without needing the modeling `.py` again). Requires a thin true-config
reader path so `unfold` can consume canonical field names; v1 may ship the
diff/inspection value first and the closed round-trip second.

## 8. Implementation sketch (for this worktree)

- New module `model_unfolder/true_config.py` — pure projection over `ModelIR`;
  zero fact re-derivation (enforced by a "no evidence-reader imports" test, like the
  renderer firewall).
- Field-name vocabulary in `everchanging/` (YAML, per the config-vocab-in-YAML law):
  maps IR spec fields → canonical true-config field names. Naming stays out of Python.
- Surfaces: `Diagram.true_config(...)`, top-level `true_config(x)`, `.save(*.true.json)`.
- Tests (metamorphic, per H9):
  - **provenance-complete**: every emitted value has a `_provenance` entry;
  - **unknown-never-filled** (poison): an `oracle_missing`/`ambiguous` fact must
    appear in `_unresolved`, never as a value;
  - **no-upgrade** (poison): an unattributed IR field must not claim `code_proven`;
  - **round-trip**: `unfold(true_config(x))` signature == `unfold(x)` signature;
  - **values-stay-config**: geometry fields keep `config_declared`.

## 9. Relationship to existing outputs (distinct audiences)

| Output | Shape | Audience |
|---|---|---|
| `to_ir()` | raw typed IR dict | internal |
| `to_json()` | expanded traceable schema (renderer-neutral, verbose) | tools/LLMs re-tracing |
| **`true_config()`** | **config-shaped, provenance-carrying, complete** | someone who wants *the config the model should have had* |

All three are projections of one IR — no divergence risk as long as true_config
stays a serializer.

## 10. Non-goals / risks

- **Not** a second interpreter — it must never recompute a fact (the top risk;
  guarded by the firewall test).
- **Not** a fabricated-complete config — holes are explicit, never guessed.
- **Not** an HF-loadable config by default — it uses canonical (owner-qualified)
  field names; HF-spelling emission is a possible later mode, not v1.
- Provenance for not-yet-ledgered facts is honest (`ir_spec`), so early adopters
  aren't misled about how strongly a field is proven.

## 11. Build steps

1. `true_config.py` projector reading IR + `fact_provenance` + `config_audit`;
   emit values + `_provenance` + `_unresolved` + `_coverage`. (serializer only)
2. `everchanging/true_config_fields.yaml` — IR-field → canonical-name vocabulary.
3. `Diagram.true_config()` + top-level `true_config()` + `.save(*.true.json)`.
4. `diff=True` 3-way view (declared / added-by-code / unknown).
5. Metamorphic + poison tests (§8).
6. Round-trip reader path (`unfold` consumes a true_config) — after the diff value ships.
7. Wire `_coverage.unattributed` into the campaign metrics (H2/H10 dashboard).
