# CODEBASE COHERENCE AUDIT — model-unfolder (evidence campaign U1/U2/U3/U6)
*Read-only architecture-coherence pass, 2026-07-12. Scope: ef9046c (U1 + U2 P0–P4,
committed) + the uncommitted U3/U6 working tree. No edits made. Baseline references:
PROJECT_CONTEXT Parts 1–4, SURGICAL_PLAN_EVIDENCE.md, MASTER_PLAN §3/§7/§8,
docs/EVIDENCE_ARCHITECTURE_HARDENING_PLAN.md.*

---

## TOP-LINE VERDICT (Soumil's question, answered straight)

**Not spaghetti at the core — but the worry is warranted at two seams, and one
of them is the very thing this campaign was supposed to fix.**

The spine is intact and coherent: evidence → typed IR → projection; one
`everchanging.load(domain, name)` loader convention; a uniform `SableCheck`
net registry; `detect-from-evidence-never-identity` genuinely held (the flagged
identity-table YAMLs `unet_blocks/vae_classes/schedulers.yaml` were **never
committed**, and the identity guard was hardened this campaign to catch
class-name lists by *behavior*, not a fixed key list). The campaign's own new
serializer fields (`position_declared`, `rope_theta_declared`) were landed in
**both** attention projections — disciplined. U3's `layer_schedules.yaml` and
U6's `conditioning.yaml` are lawful config-value/spelling vocabulary with an
honest-unknown fallback. That is not the profile of a drifting codebase.

**Where it IS drifting** is concentrated and nameable, not diffuse mush:

1. **Provenance is now three parallel channels, not the one the FactLedger
   promised.** P0's stated goal was to *unify* the scattered bools; instead the
   ledger was layered on top of them. Renderers read one channel, the new nets
   read another, and the two can silently disagree — the exact `Part 4 §3`
   failure class, now at the provenance layer.
2. **The two hazards `Part 4` explicitly named — blanket-except `_code_*`
   wrappers and the twin attention serializer — were GROWN, not paid down.**
   The campaign added ~15 more `except Exception: return None` wrappers and
   left `rope_3d` still missing from one of the two serializers.
3. **Two pieces of scaffolding shipped without their use** (a net that can
   never fire; a loader nobody calls) — a direct violation of "ship the drawing
   with the reader or don't ship."

None of these is fatal; all are localized and have a clear consolidation home
(HARDENING_PLAN H2/H5/H6 and MASTER_PLAN A3/A5). But #1 and #2 mean the
campaign added epistemic machinery faster than it retired the machinery that
machinery was meant to replace. That is the drift to arrest before the next
unit.

---

## SCORECARD (9 axes)

| # | Axis | Verdict |
|---|------|---------|
| 1 | Provenance unification vs proliferation | **DRIFT (HIGH)** — 3 parallel channels; ledger is transformer-only |
| 2 | Serializer fan-out consistency | **DRIFT (MED)** — campaign fields OK in both; `rope_3d`/`cached` still split; hazard unclosed |
| 3 | YAML vocabulary discipline | **DRIFT (MED)** — config_facts catch-all; `dit_class_markers` identity→arch; aliases untyped |
| 4 | Dead/dormant code | **DRIFT (MED)** — `consumed` rail inert; `load_layer_topology` dead; empty table file |
| 5 | Exception discipline | **DRIFT (MED-HIGH)** — `_code_*` blanket-except 28 → 49; consolidation not done |
| 6 | API surface + size | **DRIFT (MED)** — 35,783 → 42,182 pkg LOC (+17.9%) vs ≤30k target; `__all__` incomplete |
| 7 | Net coherence | **CLEAN** — uniform `SableCheck` list + `blocking` flag; new nets follow shape |
| 8 | Test coherence | **MOSTLY CLEAN** — one-behavior-one-test; test_code_evidence.py 3,483 LOC gravity-well |
| 9 | Concurrent-builder scars | **CLEAN** — deviations disclosed in the plan doc, not scarred into code |

---

## DRIFT ITEMS — worst first

### D1 · Provenance: three parallel channels that can disagree (HIGH) — Axis 1
FactLedger (`evidence/context.py:49`) was specced (SURGICAL_PLAN U2 P0) to
*"unify the scattered provenance bools (activation_from_class / activation_assumed
/ scores_scaled / projection_mode / the binary asserted tuple)."* It did not. All
three still exist and carry the same facts:

- **Channel A — typed spec fields** (`ir.py:88-136`): `scores_scaled`,
  `projection_mode`, `activation_from_class`, `activation_assumed`,
  `expert_projection_mode`. **This is what the RENDERERS read** — `labels.py:489`,
  `opgraph.py:133/436/473`, `cards.py:255`, `expanded/ffn.py:26`.
- **Channel B — the per-spec `asserted` tuple** (`ir.py:106`, `ir.py:144`),
  folded into the ledger at serialize time (`parser.py:100-109`).
- **Channel C — the FactLedger** (`ctx.facts` → `ir.extras["fact_provenance"]`),
  written at **25 `_note_fact` sites, all in `adapters/transformer/parser.py`**
  (self-declared "deliberately incremental," `parser.py:584`). **This is what the
  two new NETS read** — `sable.py:219` (projection-audit), `sable.py:289` (census).

The consequence is structural, not stylistic: **the picture and the nets read
different channels.** The parser decides a fact once, writes the value into the
spec field (Channel A, → pixels) *and* separately calls `_note_fact` with a
status (Channel C, → nets). If those two ever disagree — a refactor updates the
spec but not the `_note_fact`, or vice versa — the census/projection net passes
while the pixel lies, or fails while the pixel is right. This is precisely the
`Part 4 §3` "found only because the pixel still said sqrt(dim) after two of
three" failure, reproduced at the provenance layer.

Two aggravators:
- **The diffusor parser has ZERO ledger integration** (`grep _note_fact
  adapters/diffusor/parser.py` = 0). So net #13/#14 are **inert on all diffusion
  models** — "sable clean on flux/sd3.5" means "nothing to check," not "proven."
- Ledger coverage is 14 families on the transformer side (`decoder.attention.*`,
  `decoder.ffn.*`, `decoder.layer.*`, `model.tie_word_embeddings`) — good, but
  incomplete and single-adapter.

**Fix → HARDENING_PLAN H2 (closed fact registry) + I-4/I-5.** One write path:
make the spec field itself carry `(value, status, source)` (a `FactRecord`), and
derive the ledger *mechanically* from the specs at serialize time instead of at
25 independent `_note_fact` calls. Then Channel A ≡ Channel C by construction and
Channel B disappears. Until then there is no net that detects a spec/ledger
divergence — add one as an interim guard.

### D2 · `_code_*` blanket-except wrappers nearly doubled (MED-HIGH) — Axis 5
`Part 4 §2` named this the marquee bug-swallowing hazard ("28 across both parsers…
an AttributeError typo was silently swallowed; glm-4 looked correct only because
the fallback happened to agree"). The prescribed cure was *"one `_code_evidence()`
helper, narrow exceptions."* That was not done. The count grew:

- `adapters/transformer/parser.py`: **30** `except Exception: return None`
- `adapters/diffusor/parser.py`: **19**
- 49 total (was ~28). Every one re-imports the reader *inside* the try
  (`parser.py:141-146` is the template) and swallows **all** exceptions.

Each new code reader (P2a–d bias/tie/activation/mask, U3 schedules) added another
wrapper. A reader typo still degrades silently to "honest unknown" — undetectable
until someone calls the reader directly (the lived glm-4 failure).

**Fix → MASTER_PLAN A5 (wrappers → one table/helper).** Collapse to a single
`_code_evidence(reader, cfg, ctx)` that catches only `(ImportError,
AttributeError, SyntaxError, OSError, KeyError)` and **re-raises anything else**
(or re-raises all in a `MODEL_UNFOLDER_STRICT` debug mode). ~60–100 LOC deleted,
and reader bugs become loud.

### D3 · Scaffolding shipped without its use — dormant reader + dead loader (MED) — Axis 4/7
Two shipped-but-inert mechanisms violate the "no dormant readers" rule:

- **The `consumed` rail is inert.** `debug.py:64` supports `intent="consumed"`,
  `consumed_fields()` exists, `parser.py:110` publishes `config_consumed`, and
  `sable.py:470` registers the `config_accessed_unprojected` net — but **no
  production decision site ever marks a field consumed** (`grep intent="consumed"`
  finds only the plumbing + a comment admitting it, `parser.py:112`). So
  `consumed_fields()` is always empty, `config_consumed` is never published, and
  the net **can never produce a finding**. A registered net that is structurally
  incapable of firing.
- **`load_layer_topology()` is dead.** Defined (`everchanging/__init__.py:111`)
  but **never called** (`grep load_layer_topology(` = def only). It reads
  `layer_topology.yaml`, whose two tables are both `[]` (empty). The file's own
  header claims "the parser uses a row here" as an offline fallback — but nothing
  calls the loader, so the fallback is fictional. ~13 LOC loader + 23 LOC empty
  file, pure weight.

**Fix:** either wire the consumed marks at the ~14 transformer decision sites
(one day; then the net earns its keep) or **remove the net + `intent="consumed"`
rail until wired** — do not ship a reader whose drawing does not exist. **Delete
`layer_topology.yaml` + `load_layer_topology`** (deletion-ledger; the code reader
`decoder_layer_topology_from_files` fully replaced it).

### D4 · Attention serializer fan-out unclosed; `rope_3d` still proves it (MED) — Axis 2
The campaign was disciplined about ITS OWN new fields — `position_declared` and
`rope_theta_declared` land in **both** `ir.py:_attention_to_dict` (`:331-333`) and
`blocks/attention.py:attention_detail` (`:61-64`). Credit where due. But the
*structural hazard* `Part 4 §3` flagged is unchanged, and its exhibit is still
live:

- `rope_3d`: present in `attention_detail` (`:43`), **absent from
  `_attention_to_dict`** (verified: `sed -n 296,346 ir.py | grep rope_3d` = 0).
  The documented serializer drift *persists*.
- `cached`: same split — in `attention_detail:47`, not in the JSON projection.

Two hand-maintained attention→dict serializers with no shared source mean every
new attention fact costs two edits and one will eventually be missed. The
campaign paid the tax correctly this time; it did not remove the tollbooth.

**Fix → MASTER_PLAN 3b.** One serializer: `_attention_to_dict` is the single
source; `attention_detail` derives from it (adds only render-local keys). Emit
`rope_3d`/`cached` from the one place.

### D5 · YAML doctrine (Soumil's 4 binding rules) — per-file migration plan (MED) — Axis 3
Full inventory + migration plan in the next section. Headline: `config_facts.yaml`
is a 110-row catch-all that launders structural facts as `silent:true` chips;
`dit_class_markers` decides architecture from class names while its sibling
`temporal_forward_markers` explicitly refuses to — an internal inconsistency the
new identity guard will now surface.

### D6 · Size + API surface (MED) — Axis 6
- **Package LOC: 35,783 (2026-07-05 baseline) → 42,182 now (+6,399, +17.9%),
  136 files.** Target is **≤30k** (MASTER_PLAN §7). The package is ~40% over
  target and has been *growing* while Phase A (the deletion campaign) has not
  started. The evidence campaign's own net contribution is ≈ +2,600 package LOC
  (ef9046c +~1,550 pkg, uncommitted +~1,050 pkg) + ≈ +1,500 test LOC. Size
  governance rule §7 (no net-positive LOC without a mechanism) is *technically*
  met — nets/readers are mechanisms — but the ratchet is loose; D2/D3/D4
  consolidation would return ~250–400 LOC.
- **`everchanging/__init__.py` `__all__` lists 9 of 21 public `load_*`
  functions** (missing all `load_conformance_*`, `load_decoderness`,
  `load_composite_slots`, `load_constructor_classmethods`,
  `load_diffusion_config_facts`, `load_mistral_params_map`). The module docstring
  names 5 of the ~21 YAML files. Cosmetic, but it is exactly the "ad hoc export"
  smell — pick a rule (export all public loaders or none) and regenerate.

---

## YAML DOCTRINE — per-file inventory & migration plan (Soumil's 4 rules)

Rules: **(1)** aliases KEEP but narrow + TYPE every row (geometry-value /
declared-flag / schedule-values / address); no row gates structure by presence.
**(2)** config_facts: dismantle catch-all; structural rows → fact families.
**(3)** text_encoders: display vocabulary only, zero structural role. **(4)** any
class/model/family identity → architecture: REMOVE.

| File | Rows | Classification | Action |
|------|------|----------------|--------|
| `transformer/aliases.yaml` | 26 | **R1** — lawful but UNTYPED | TYPE each row (below) |
| `diffusor/aliases.yaml` | ~30 | **R1** — lawful but UNTYPED | TYPE each row |
| `diffusor/config_facts.yaml` | 110 | **R2** — catch-all; structural rows laundered | SPLIT (below) |
| `diffusor/text_encoders.yaml` | 10 | **R3** — display, but a structural fallback consumes presence | keep display; relocate the presence-branch |
| `transformer/layer_topology.yaml` | empty | **R4** — killed identity table, dead loader | **DELETE file + loader** |
| `diffusor/typing.yaml → dit_class_markers` | 6 | **R4** — class-name → "is a DiT" | migrate to construction/forward evidence; class name = address only |
| `diffusor/typing.yaml → scheduler_flow_matching_markers` | 1 | **R4-adjacent** — scheduler class → integrator | migrate to `step()` semantics (U6) or pin as reviewed self-declaration |
| `diffusor/typing.yaml → companion_denoiser_fields` | 2 | **R1-risk** — config-key presence → structural note | keep, but gate on the slot declaring a `model_type` (composite_slots pattern) |
| `diffusor/typing.yaml → scheduler_display / norm_type_kind / temporal_* / stack_lane_params / audio_vae_fields / stages / block_ids / part_kinds` | — | **LAWFUL** — display / config-value / forward-marker / taxonomy | KEEP |
| `transformer/decoderness.yaml` | 3 | **LAWFUL** — role-suffix vocabulary (U2) | KEEP |
| `transformer/layer_schedules.yaml` | 8 groups | **LAWFUL** — schedule-value spellings (U3) | KEEP |
| `diffusor/conditioning.yaml` | 9 | **LAWFUL** — config-enum-value → modality (U6) | KEEP |
| `transformer/composite_slots.yaml` | — | **LAWFUL** — slot vocab, evidence-gated | KEEP |
| `transformer/{typing,layer_types,mistral_params,ignored_fields}.yaml` | — | **LAWFUL** — taxonomy / spellings / normalizer / ignore-audit | KEEP |
| `conformance/{op_tokens,type_roles,fact_markers,conformance_map,abstractions,wiring_roles,transitive}.yaml` | — | **LAWFUL (exempt)** — net-internal op-role interpretation of an ALREADY-resolved class; identity guard blesses this domain | KEEP; keep the exemption narrow |

**R1 typing — `transformer/aliases.yaml` rows, by kind (the tag to add):**
- *geometry-value:* num_hidden_layers, num_attention_heads, num_key_value_heads,
  hidden_size, intermediate_size, vocab_size, max_position_embeddings, head_dim,
  mlp_ratio, sliding_window, num_experts, num_experts_per_tok, num_shared_experts,
  moe_intermediate_size, compress_ratios
- *declared-flag:* attention_bias, mlp_bias, tie_word_embeddings
- *declared-value:* hidden_act, norm_type, rope_theta, rope_scaling
- *declared-scalar:* residual_multiplier, embedding_multiplier, attention_multiplier, logits_scaling

  Migration: restructure into typed sections (or per-row `# kind:` tag) + a
  loader assertion. Then audit the 3 declared-flag call sites (`get("attention_bias")`,
  `get("mlp_bias")`, `get("tie_word_embeddings")`) to prove none gates STRUCTURE
  by presence — they must resolve a VALUE, not switch a branch on `is not None`.

**R2 split — `diffusor/config_facts.yaml`:**
- *KEEP (genuine numeric presentation chips):* time_embed_dim, freq_dim,
  time_factor, time_max_period, sample_size, default_sample_size, eps,
  resnet_norm_eps, batch_norm_eps, encoder_channels, cross_attention_input_dim,
  and the other pure dims/scales.
- *MIGRATE → denoiser attention topology fact family* (they change which
  sublayers/ops exist): `only_cross_attention`, `dual_cross_attention`,
  `double_self_attention`, `attention_type`, `use_linear_projection`,
  `upcast_attention`.
- *MIGRATE → VAE construction/stage evidence* (the `silent:true` per-stage
  tables — "read but never drawn" is the accessed-but-unprojected anti-pattern
  institutionalized): `add_attention_block`, `attn_scales`,
  `spatio_temporal_scaling`, `temperal_downsample`, `decoder_block_types`,
  `encoder_block_types`, `decoder_norm_types`, `decoder_act_fns`,
  `decoder_layers_per_block`, `encoder_layers_per_block`.
- *DEDUPE → conditioning fact family:* `encoder_hid_dim_type` is **already** in
  `conditioning.yaml` — remove the config_facts copy; `addition_embed_type_num_heads`
  joins it.
  Result: config_facts.yaml shrinks to non-structural chips only; every migrated
  row gains evidence + projection + a net witness (H6/I-7).

**R3 — `text_encoders.yaml`:** the file is display-only, but
`_resolve_conditioning` step 3 (`parser.py:2036`, `elif has_text: kv_modality =
"text"`) makes text-encoder *presence* decide the cross-attention K/V modality.
That is a structural role. It is *defensible* IF `_detect_text_encoders` decides
presence from pipeline *component construction* (a real fact) and uses the YAML
only for the display label — verify that; if a text encoder whose class is absent
from the YAML fails to be detected, the YAML is gating structure by membership →
move presence-detection to construction, keep the YAML for the label alone.

---

## CLEAN / POSITIVE FINDINGS (verified, worth preserving)

- **Net registry coherence (Axis 7):** all ~15 nets are `SableCheck(name,
  findings, blocking=?, note=?)` in one list (`sable.py:400-475`); staging is a
  uniform `blocking` field (module const or inline `False`). The 3 new nets
  (#13 projection_audit, #14 census, config_accessed_unprojected) follow the
  exact shape. No divergence. The census `_numbers_only` strip is implemented
  ONCE (`sable.py:237`); the doc is prose, not a code twin.
- **Identity-guard hardening (uncommitted, U6):** `scan_identity_yaml_source`
  now flags any non-approved list of ≥2 PascalCase class-name-like values by
  *behavior* (`identity_guard.py:283+`, `_looks_like_class_name`), closing the
  "fixed key-name list is a hole" gap the external review found — and it will
  correctly surface `dit_class_markers` for the R4 decision. Exactly the right
  mechanism. Note: the `conformance/` domain is consciously exempted, coherently.
- **Flagged identity tables never landed:** `unet_blocks.yaml`,
  `vae_classes.yaml`, `schedulers.yaml` do not exist — the U6 builder complied
  with the review's binding correction.
- **U3/U6 vocab is lawful:** `layer_schedules.yaml` (spellings grouped by input
  FORM + value/mixer maps), `conditioning.yaml` (config-enum-value → modality
  story, unmapped → honest-unknown). Neither keys on identity.
- **No concurrent-builder scars in code (Axis 9):** `grep -i
  reconcile|collided|mid-flight|TODO|FIXME|HACK` over the package returns only
  substantive domain comments (id-collision handling, palette note). The P1/P2
  collision and the silu-tier deviation are disclosed in SURGICAL_PLAN_EVIDENCE,
  not left as scars in the source.
- **Test coherence (Axis 8):** `test_fact_ledger.py` (457 LOC) is one-behavior-
  per-test with section headers and no contradictory pins; provenance behavior
  splits across test_fact_ledger / test_projection_audit / test_code_evidence
  along **net boundaries** (foundation vs net vs reader) — acceptable. Watch:
  `test_code_evidence.py` at **3,483 LOC** is becoming a gravity-well; split by
  fact family when next touched.

---

## CONSOLIDATION LEDGER (what to pull, and where it lives)

| Item | Fix | Home | Est. |
|------|-----|------|------|
| D1 provenance 3-channel | spec fields carry FactRecord; derive ledger mechanically; add spec↔ledger guard | HARDENING H2 (+ diffusor into ledger) | new unit |
| D2 blanket-except | one `_code_evidence()` helper, narrow catch, strict re-raise | MASTER_PLAN A5 | −60–100 LOC |
| D3 consumed rail inert | wire consumed marks OR remove net+rail | 1-day cleanup | ± |
| D3 dead layer_topology | delete file + `load_layer_topology` | deletion-ledger | −36 LOC |
| D4 twin serializer | one `_attention_to_dict`; detail derives; emit rope_3d/cached once | MASTER_PLAN 3b | −? |
| D5 config_facts catch-all | split per R2; structural rows → fact families | HARDENING H7 (+ H6) | restructure |
| D5 dit_class_markers | construction/forward evidence; class=address | HARDENING I-10 / U6 | ± |
| D5 aliases untyped | type every row; audit flag call sites | 1-day cleanup | ± |
| D6 `__all__` incomplete | export all public loaders (or none) + refresh docstring | 1-day cleanup | small |

*Recommended sequencing: D3 (delete/decide, frees mental load) → D2 (loud
readers, low risk) → D1 (the real one; blocks honest diffusion nets) → D4/D5
(fold into the H6/H7 verticals).*
