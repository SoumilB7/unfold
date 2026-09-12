# Supplementary initial claim-contract review

Reviewer: `/root/independent_review`. Source inspected at implementation `bb48a31`, reproduced from immutable checkout HEAD `349c02d2fcc41139ed770a1fe72233da4d5f6250`. This supplements the initial RETURN; it does not grade full C-8 completion. No production or test edits, pytest lane, commit, or blessing.

## R5 — FFN applied-function proof ignores a recorded callable override

Earliest false producer: `model_unfolder/evidence/unet_claims.py:117–122`, also the runtime selected-transform address check in `unet_nested_mechanism.py:650–657`. They establish exact class addresses but do not reject a recorded instance-specific `forward` replacement or hooks on the FFN/selected transform. In contrast, the initial primitive and direct-cell-connection readers explicitly check those fields.

The poison adds the inventory's existing serialized `forward` replacement marker to `down_blocks.1.attentions.0.transformer_blocks.0.ff`. The exact same GELU fused-gated FFN fact survives. Class source is insufficient to certify that instance's invoked computation once its callable has been replaced. This is distinct from R1's escaped-parent mutation and R2's spoofed framework class name.

`reproduce_ffn_override.py` reconstructs static reader evidence from the persisted ordinary SDXL inventory and current installed source while running the immutable initial code. It then alters only the callable marker in a DTO copy. `ffn-override-results.json` records the result and source fingerprints. This is a qualification-boundary poison, not a claim that the original SDXL contains such an override, and it constructs or executes no model. The required correction belongs at the exact runtime callable/source binding boundary and must apply to every mechanism claim that depends on that callable, including the selected transform inside an FFN.

Command, from the immutable checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/reproduce_ffn_override.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/ffn-override-results.json
```

## R6 — a spatial proof is broadcast to another same-class occurrence

Earliest false producer: `UNetSpatialClaimProof.value()`, `model_unfolder/evidence/unet_claims.py:411–425`. Its join retains the selected stage and field, but binds the operation to every matching direct child class in that field. It drops the exact construction site/index carried by the source proof.

The source fixture constructs:

```python
self.spatial = ModuleList([
    Spatial(width, convolution, stride=2),
    Spatial(width, convolution, stride=1),
])
```

The existing F3 readers establish two constructions and exactly one positive spatial reduction operation, for the stride-2 construction. They produce no issues. The new fact projects `effect=reduce, operand=2` to both `down.0.spatial.0` and `down.0.spatial.1`.

`reproduce_spatial_occurrence.py` builds the real typed static reader chain from a small source fixture and uses an explicit synthetic instance-inventory DTO at the boundary under test. This is not represented as a real model construction. The source, source hash, positive reader count, and incorrectly projected fact are persisted in `spatial-occurrence-results.json`. The command exited 0, confirming the defect:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/reproduce_spatial_occurrence.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-owner-review-initial/spatial-occurrence-results.json
```

Bind each source construction to its exact runtime member occurrence before projecting the operation. If that address cannot be uniquely established, keep the affected occurrence unresolved. Class equality within a field is not an occurrence join. The minimum paired control is two same-class children with different proven operands (including one which has no reduction), preserving only each child's own result. Apply the same audit to other field-wide joins, without assuming they necessarily share this defect.

## Supported conclusions and limits

- `UNetContextConnectionClaimProof` includes an explicit selected-stage path filter, retains the external-context versus cross-attention distinction, and checks the addressed lane's exact class. No separate cross-stage false positive was established in this audit. That is a bounded observation, not a blanket validation of every nested same-field join.
- Shape claims count shared parameter identities once and project values separately from connections. Their stated denoiser scope is accurate. No independent shape-count false positive was established. The inventory is config-dependent and does not independently verify checkpoint tensor metadata.
- Default declarations are labelled `class_default` and remain distinct from applied mechanisms and checkpoint declarations. No independent default-to-mechanism promotion was established in that fact path.
- The open execution rows and cross-attention chips are permitted within the user's stated exit boundaries. They are not RETURN items merely because complete execution or Q/K/V proofs are absent.

R5 and R6 require executor correction and independent re-review before these claim paths can be accepted. The initial review's R1–R4 and the remaining C-8 work retain their own dispositions.
