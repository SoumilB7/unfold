# S4 pre-bless receipt and closed delta ledger

Status: **mechanically implemented; independent S4 review required before any
manifest or gallery bless**.

Committed-tree pre-bless receipt: **`d7f78f17ca`**, implementation tree
**`8f9862ed2a0937f34f862c0564a22bf59542a914`**.  The machine receipt is
[`receipt.json`](receipt.json), with all six original lane logs under
[`lanes/`](lanes/).  The anti-vacuous mutation outcomes are indexed in
[`poisons.md`](poisons.md).

This directory records the S4 boundary before approval.  The authoritative
product denominator is [`coverage.json`](../../../coverage.json): 29 reviewed
corpus witnesses plus the 15 frozen unseen-model inputs.  Its current totals
are **621 proven / 241 visibly flagged / 0 silent**.

## Existing-witness delta

The change is closed and has two layers:

1. All 29 witnesses change only on the canonical `sable` evidence surface:
   `asserted_facts` and `config_accessed_unprojected` change from advisory to
   blocking.  On 27 witnesses both were already empty.  No fact, label, view,
   parameter estimate, or pixel changed.
2. Exactly two witnesses also change on `ir`, `expanded`, and `html_meta`,
   because an existing audit finding is now visible on ordinary `unfold()`:

| witness | visible additions | exact evidence-level cause | SVG/gallery delta |
|---|---:|---|---|
| `granite-3-0-8b-instruct` | 2 | accessed but not structurally consumed: `root.embedding_multiplier`, `root.logits_scaling` | none |
| `stable-diffusion-xl-base-1-0` | 11 | accessed but not structurally consumed under `root.denoiser`: `addition_embed_type`, `addition_time_embed_dim`, `block_out_channels`, `down_block_types`, `encoder_hid_dim`, `encoder_hid_dim_type`, `layers_per_block`, `mid_block_type`, `projection_class_embeddings_input_dim`, `transformer_layers_per_block`, `up_block_types` | none |

The findings are not reclassified as facts.  Each remains unresolved, retains
its exact owner/path/spelling, and is merely transported to the product warning
surface.  An altered or missing `(check, message)` receipt turns the blocking
gate red.

The warning transport uses the explicit `Unresolved evidence —` prefix.  The
HTML consumer maps that producer-authored class to the “unresolved evidence”
badge; it does not inspect evidence internals or relabel these findings as
“partial config”.  This distinction is pinned by a poison test.

## Unseen-model outcomes

| model | proven | flagged | silent | decisive visible disposition |
|---|---:|---:|---:|---|
| `CohereLabs/c4ai-command-a-03-2025` | 20 | 9 | 0 | config audit/use findings |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` | 17 | 17 | 0 | `fact_conformance` plus config/qualification findings |
| `THUDM/glm-4-9b` | 1 | 8 | 0 | config audit/use findings |
| `EleutherAI/gpt-neox-20b` | 22 | 7 | 0 | unreceipted config and qualification findings |
| `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | 3 | 8 | 0 | config-field findings |
| `ai21labs/Jamba-v0.1` | 1 | 15 | 0 | `op_conformance` plus config-field findings |
| `LiquidAI/LFM2-1.2B` | 20 | 20 | 0 | config audit/use and qualification findings |
| `MiniMaxAI/MiniMax-M2` | 25 | 14 | 0 | config-field findings |
| `Qwen/Qwen3.5-27B` | 21 | 16 | 0 | rendered main stack; projector evidence unresolved |
| `Qwen/Qwen3.6-35B-A3B` | 19 | 19 | 0 | rendered main stack; projector evidence unresolved |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 1 | 78 | 0 | explicit config/document/fact findings |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | 17 | 10 | 0 | rendered main stack; projector evidence unresolved |
| `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers` | 7 | 0 | 0 | no unresolved finding on this snapshot |
| `CompVis/stable-diffusion-v1-4` | 0 | 6 | 0 | explicit config audit/use findings |
| `ByteDance-Seed/Seed-OSS-36B-Instruct` | 23 | 1 | 0 | config-field finding |

`flagged` is not completeness and does not certify a mechanism.  It is the
count of exact unresolved receipts visible in the product.  In particular,
the honest low-information rows for Jamba, Qwen3-Omni, and SD-v1.4 are not
presented as successful architecture coverage.

## Harness-only migration

The 29 old fixture documents now carry two locks generated from exact pre-S4
commit `ba33557`: labeled `view_signature` and the set of `proven_facts`.
Their legacy view hashes were reproduced before the fields were added.  The
preservation input hash is correspondingly narrowed to the executable fixture
input (`config` plus `source`), so adding lock/review metadata is not falsely
reported as a model-config mutation.  A config or source mutation still turns
the poison red.

## Approval boundary

No preservation surface hash, view hash, gallery, or manifest surface hash is
updated in this pre-bless change.  After an independent reviewer accepts this
ledger, the executor may rebuild the 29-witness manifest, verify that its delta
is exactly the 29 Sable evidence changes plus the two visible-warning rows
above, and run the final isolated zero-failure bracket.

The first isolated pre-bless bracket correctly failed preservation on the
unblessed ledger and also exposed seven stale tests whose old contract required
an unresolved finding to remain both visibly receipted and unsurfaced.  Those
tests now retain their no-fabrication assertions while joining the exact
`ship_findings` receipt; no architecture fact, support-set row, or coverage
count changed.  The corrected bracket collected **4,057** tests, passed **208**
focused and **44** U2-authority tests, and passed the **3,961-test** bounded
full bracket.  All coordinator, artifact, and lane fingerprints were identical
before/after.  Preservation was the sole red lane: all 29 witnesses changed on
`sable`; only Granite and SDXL additionally changed `ir`, `expanded`, and
`html_meta`, exactly as enumerated above.  This is the required independent
review stop, not a blessing failure to suppress.
