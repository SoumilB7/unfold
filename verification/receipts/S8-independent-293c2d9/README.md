# Independent narrow acceptance — 293c2d9

**ACCEPT the named-write correction and observation mapping canonicalization.** This does not mark S8 complete or remove the product limitations below.

Immutable checkpoint `293c2d98796c915f73e8422ddbf00b6fbc7492d4` at `/private/tmp/unfold-s8-final-boundary-review`. No model run, pytest, production edit, commit or blessing. Two standalone scripts exited 0.

`replay_ifexp.py` independently replays the previous branch-local named-write counterexample and the guard-write poison. Both are now unresolved. The ordinary conditional addition and opaque call alternatives remain available. The changed reader refuses an expression containing a named write before assigning source identities to its sibling operands; it does not invent evaluation-order semantics.

`replay_mapping.py` confirms dictionary-key insertion order does not change the observation after JSON roundtrip, while reversing a meaningful list changes both block structure and its signature. Source review confirms only dictionary traversal in `blocks()` is canonicalized; list order is preserved. It does not normalize away output values or semantic list changes.

The positive real SDXL primary proof → summary → fact ledger → citation → archive check from `S8-independent-9fe0868/` remains applicable to the unchanged registration/summary boundary. Its 197 references, exact root archive and claim-kind/root-forward refusal controls are not re-described as a fresh model execution here. Earlier 3693545 conclusions retain their explicitly limited source-port scope.

## Remaining product gap is separate

The actual 9fe ordinary IR retained at `../S8-final-9fe0868-report-stop/ordinary/ir.json.gz` contains these eight visible bookends in `other_block_ids`:

- `conv_in`
- `time_proj`
- `time_embedding`
- `add_time_proj`
- `add_embedding`
- `conv_norm_out`
- `conv_act`
- `conv_out`

They have real occurrence cards and shapes where available. They are not missing inventory. Their source call ports are not yet bound to those cards: the input assignment, output-normalization conditional and final output assignment still have no bound module IDs in their primary-region metadata. Likewise the three down stages, mid stage and three up stages are explicitly contained by repeat/conditional regions; that membership is not an individual invocation-target proof.

The wrapper-region route is valid within its stated source-port scope, and the unbound cards are honest limitations. Neither means the architecture's actual module connections are finished. A green check of existing graph edges cannot establish that required edges are present.

The user's S8 item 6 requires skip concats and conditioning connected. Plan `12-execution-order.md:418–451` and C-8 at `14-confirmation-checklist.md:157–158` require implementation → established connection → actual overview/drill and judge the HTML during this step. On the current evidence, the owner should not describe the task's full architectural connection intent as completed merely because the wrapper regions connect.

The bounded remaining work is to establish direct call-target/member identity for the input, conditioning and output components, establish the supported repeated-stage target-to-occurrence correspondence, and project those links onto the actual occurrence cards. A known invocation target can still have opaque internal computation; this does not require a general helper interpreter. Preserve optional guards, unknown helper semantics and the allowed cross-attention chips. The original 240 execution-unknown allowance remains separate from this product connection gap.

If a particular target or connection remains unsupported after the bounded investigation, show that precise limitation and carry it to the sheet for owner review. Do not silently turn containment into dataflow or declare that missing connection complete. This is not a demand to draw every inventory leaf or solve universal execution closure.

## Reproduce

Run from the immutable checkout using these receipt scripts:

```sh
s8_review_receipt=/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-independent-293c2d9
PYTHONDONTWRITEBYTECODE=1 python3 "$s8_review_receipt/replay_ifexp.py" > "$s8_review_receipt/ifexp-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$s8_review_receipt/replay_mapping.py" > "$s8_review_receipt/mapping-results.json"
```

`verification.json` records the checkpoint, source hashes, clean checkout and scoped verdict.
