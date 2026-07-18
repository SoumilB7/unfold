# Net-1 migration-debt census (authoritative, generated)

Regenerate with `python3 scripts/census.py`; `--check` fails when this
file is stale. Never hand-edit — the generator is the producer.

Source: the parser's own owner-scoped OCCURRENCE-EXACT view
(`config_access.accessed_unconsumed_exact`: component + exact dotted path +
actual spelling) across the committed corpus.

## What this list is

A **discovery list**, not a list of receipts owed. No row carries a unit
assignment yet: an owner is not a unit (`root` alone spans U6 attention,
U7 FFN/norm and U8 position/mask; vision is mainly U9; denoiser U10/U11;
vae U12; scheduler U13), so assignment is per OCCURRENCE/mechanism and
belongs to U2.2.

Every row must receive
exactly ONE disposition (U2.2): structural+already projected · structural
but not projected · geometry-only · display-only · address/source-selection
only · non-architectural metadata · unused/phantom (delete the read or fix
ownership) · ambiguous/unsupported (preserve unknown). Only
mechanism-driving rows need new interpretation code.

## The law

- Migration is claimed at exact **(owner, mechanism)** scope — never an adapter or file.
- A claim is valid only when every present read in scope has an exact path and owner,
  and each read is consumed, scoped-ignored, or precisely classified.
- Ambiguities remain blocking regardless of claims.
- Net 1 blocks each claimed scope immediately (`config_migration_claims`).
- Net 2 (`config_consumed_unreceipted`) joins exact occurrence -> exact fact target ->
  exact render RECEIPT, and validates the receipt's value/status fingerprint against
  the fingerprint recorded AT the consumption plus the registered surface/kind policy.
  Coverage is owner/mechanism-SCOPED (`projection_coverage.receipted_scopes`), never a
  global flag: inside a receipted scope a missing or invalid receipt BLOCKS
  unconditionally; every other scope stays this advisory census.
- Rows below are VISIBLE debt: the count must shrink only as rows are genuinely
  resolved. Mass registration is debt-laundering and is rejected.

## Live claims (source-to-target bound; observed and matched corpus-wide)

| scope | claimed by | bindings (path -> target) | observed | target-matched |
|---|---|---|---|---|
| root.video/projector_out_width | COR-4 | vision_config.hidden_size -> root.video.projector_out_features | 2 | 1 |
| root.vision/encoder_width | COR-5 | vision_config.embed_dim -> root.vision.hidden_size; vision_config.vision_hidden_size -> root.vision.hidden_size; vision_config.width -> root.vision.hidden_size; vision_config.hidden_size -> root.vision.hidden_size | 6 | 2 |
| root.vision/projector_out_width | COR-4 | vision_config.hidden_size -> root.vision.projector_out_features | 2 | 1 |

## Owners with NO consumption at all (audit_incomplete)

- none

## Standing accessed-but-unconsumed occurrences: 0

Format: `exact.dotted.path (witness count)`, with `(as spelling)` when the
supplying alias differs. The row key is the FULL occurrence, so two paths
sharing a canonical leaf are two rows.

Paths are relative to each owner's DOCUMENT (named per section below):
that keeps the key host-independent, so a claim binding matches the same
mechanism whether a model is parsed standalone or embedded in a pipeline.
Prefix a row with its document to address the value in the witness file.

## PENDING occurrences (dispositioned, exact debt rows): 36

Each row here is EXCUSED by one exact config_read StructuralDebt row
(evidence/structural_debt.py): owner + exact path + U3-U14 unit +
checkable deletion condition + the excusal writer/consumer, all
gate-enforced.  These are visible debt with an assigned unit — not
standing UNCLASSIFIED reads, and never a family-wide excuse.

### `root` — 10 pending rows

`apply_residual_connection_post_layernorm` (1w, U7), `rope_parameters.factor` (2w, U8), `rope_parameters.original_max_position_embeddings` (2w, U8), `rope_parameters.partial_rotary_factor` (2w, U8), `rope_parameters.rope_theta` (8w, U8), `rope_parameters.rope_type` (8w, U8), `rope_parameters.type` (1w, U8), `text_config.rope_parameters.rope_theta` (1w, U8), `text_config.rope_parameters.rope_type` (1w, U8), `text_config.rope_parameters.type` (1w, U8)

### `root.denoiser` — 9 pending rows

`addition_embed_type` (1w, U11), `addition_time_embed_dim` (1w, U11), `block_out_channels` (1w, U11), `down_block_types` (1w, U11), `layers_per_block` (1w, U11), `mid_block_type` (1w, U11), `projection_class_embeddings_input_dim` (1w, U11), `transformer_layers_per_block` (1w, U11), `up_block_types` (1w, U11)

### `root.text_encoder` — 11 pending rows

`rope_parameters.rope_theta` (3w, U8), `rope_parameters.rope_type` (3w, U8), `rope_scaling.rope_theta` (1w, U8), `rope_scaling.rope_type` (1w, U8), `rope_theta` (1w, U8), `text_config.rope_parameters.rope_theta` (1w, U8), `text_config.rope_parameters.rope_type` (1w, U8), `text_config.rope_scaling.rope_theta` (1w, U8), `text_config.rope_scaling.rope_type` (1w, U8), `text_config.rope_scaling.type` (1w, U8), `text_config.rope_theta` (1w, U8)

### `root.text_encoder.vision` — 4 pending rows

`num_hidden_layers` (1w, U9), `rope_scaling` (1w, U9), `text_config.num_hidden_layers` (2w, U9), `vision_config.in_channels` (1w, U9)

### `root.vision` — 2 pending rows

`text_config.num_hidden_layers` (1w, U9), `vision_config.in_channels` (1w, U9)

## Reads whose LOCATION is unknown: 0

NOT classifiable, and NOT part of the census above: the read is real
and the value is real, but the reader touched a nested object without
naming which, so the ledger recorded an honest bare leaf. Asking for a
disposition here would be asking to classify a location nobody
established.

A **producer** backlog (U2.2b): each shrinks where its READER names the
object it read (`wrapper_path` / `config_container(obj=)`) — never by a
census filter, and never by deciding what an unlocatable row means.

- none

## Reads whose ORIGIN is unknown: 0

BLOCKING debt, and NOT part of the census above. The document these
were read from was never prepared, so nobody can say whether the
checkpoint declared them or a config class supplied them. Unestablished
is not a synonym for declared — letting it default into the checkpoint
census is what made the class's words look like the file's.

These are a few lost DOCUMENT BOUNDARIES multiplied across many reads,
not one problem per row: they collapse when preparation is centralized
(one prepared document per boundary), not by classifying them.

- none

## Fields the CHECKPOINT never declared: 5

The installed config class supplied these (located by `model_type` —
identity-as-ADDRESS, which is lawful). They are excluded from the
checkpoint census because they are not the checkpoint's words, and
listed here because they are real and often STRUCTURAL: a
class-supplied `layer_types` IS a mask schedule. The open question for
each is not "what does this declaration mean" but "may the class decide
this, and does the fact it authors say so".

- `root.denoiser` — 3: `_repo_id` [loader_metadata] (14w), `_text_encoder_configs` [loader_metadata] (15w), `_vae_config` [loader_metadata] (14w)
- `root.scheduler` — 1: `_scheduler_config` [loader_metadata] (15w)
- `root.vae` — 1: `_vae_config` [loader_metadata] (14w)

