# Independent broad failure triage — a83d704

Read-only source and preserved actual batch logs. No pytest, model construction, rendering or baseline update. This is a diagnosis, not acceptance of a corrected suite or lost drawing.

## Diffusion: 25 failed tests

`classification.json` lists all 25 exact test IDs, input-level cause, and the semantic obligation that must survive migration. Twenty use the same incomplete SDXL fixture: text_time is selected while projection_class_embeddings_input_dim is omitted (None), then installed UNet2DConditionModel creates TimestepEmbedding(None, ...), whose first projection is nn.Linear(None, ...). Two other fixtures explicitly pass num_attention_heads, which the installed root rejects. Source establishes these incompatibilities; broad logs do not print the worker exception, so an exact recorded failure should accompany the correction.

Three cases cannot be dismissed by those shared input defects: SVD constructs successfully; the encoder_hid_dim variant removes text_time; and the second DeepFloyd variant lacks the explicit rejected head field. Their concrete remaining evidence requirements are listed individually. No blanket move to the legacy flag and no whole-file test replacement is justified.

SVD's prior temporal root reader proves a lexical identifier, and its stage reader proves reachable construction of Conv3d/AlphaBlender. Those are meaningful bounded witnesses. They do not prove the fixed sequential temporal template, alpha formula, or runtime frame count. Actual constructed temporal occurrences and shapes must remain visible; this review has not seen that variant's full returned HTML.

## FFN wording failure

Batch 004 reaches the final assertion after all three source helper controls pass. The missing phrase is authored only by legacy unet.py:449–453. New production authors occurrence-qualified FFN facts and runtime_ffn cards. Retain the helper controls; replace the prose assertion with one actual runtime_ffn card's own fact citation, matching gated GELU/fused_gate_up value, and real operation nodes/GEGLU gate in that card's SVG. Merely asserting GEGLU anywhere is insufficient. Serialized fact_provenance has status/value, not typed claim proof; do not invent a serialized proof assertion.

## Release examples freshness failure

Batch 003 did not invoke the forbidden rasterizer. check() rendered deterministic HTML, compared it, and returned 1 for six of eight examples. The combined manifest message is also triggered by HTML hashes, so it does not alone establish changed hero SVG. Exact temporary candidates are removed by check() before returning; the log cannot establish the causes of all six byte deltas. Capture reviewed candidates after the active lane, enumerate changes, and retain the approval boundary before replacing reviewed outputs. Do not relax the freshness checker or assert only that no exception was raised.

Source snapshots come from exact git a83d704. Installed source snapshots and source/log hashes are retained in classification.json. Existing owner per-witness semantic ledgers can support matching changes, but they cannot approve an unseen candidate by association.
