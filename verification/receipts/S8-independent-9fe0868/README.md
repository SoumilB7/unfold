# Independent narrow review — 9fe0868

**RETURN on one conditional-expression branch-write false-positive.** Primary registration, real proof qualification/archive, ordinary conditional operands and opaque operator rails pass the reviewed checks. The unchanged source-authority conclusions from `S8-independent-3693545/` remain applicable; this is not a replay or acceptance of the full S8 campaign.

Immutable reviewed checkpoint: `9fe08684f53feafed01735d328586ca369905772`, `/private/tmp/unfold-s8-final-operator-review`. No pytest, model construction/execution, production edits or blessing. Three standalone scripts completed with exit 0; the expression script records the negative counterexample rather than treating command success as acceptance.

## Real primary proof boundary — PASS

`replay_qualification.py` rebuilds the actual installed SDXL root topology, construction and stage-execution source readers. It uses the saved typed ordinary inventory only for source identity and member selection, then runs the real primary reader, `proof.summary()`, `FactLedger.record_typed`, `ProjectionFactCitation`, and the demonstration's `_qualified_facts` archive function.

Results:

- Root static and saved runtime source SHA-256 both `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`.
- Nine primary regions and 197 sorted, unique evidence references.
- Required and authored claim kind both `connection`; the fact enters the registered ledger and citation boundary.
- The exact root bytes are archived, hash-checked, with zero unmatched source hashes.
- Changing the claim kind to `existence` is rejected. A recorded root-forward override suppresses the primary proof.

The matching source archive and proof/ledger artifacts are in `actual-primary-archive/`; `qualification-results.json` records the exact summary, inventory hash and refusal reasons. This saved inventory is not substituted for the fresh family execution receipt.

## Conditional expression — ordinary positive passes, branch write returns

The ordinary SDXL-shaped `emb + aug if enabled else emb` retains the source addition's two operands, the bypass value and an unresolved guard. Opaque call alternatives remain opaque. A walrus assignment in the guard is refused.

A walrus in a selected branch is not refused before subsequent values in that branch are attributed:

```python
selected = ((emb := aug), emb) if enabled else (emb, emb)
```

The true route is a sequence whose first item is unresolved and second item is `formal(emb)`. With `emb=11`, `aug=23`, and `enabled=True`, the actual value is `(23, 23)`: the second item consumes the updated value, not the original formal 11. The synthetic execution is recorded in `ifexp-results.json`.

Earliest producer: `model_unfolder/evidence/local_port_routes.py:291–298` checks named writes only inside the guard span, then visits both branch operands with the unchanged pre-expression cutoff. The sequence reader at 285–286 therefore restores the original formal for an operand evaluated after the write. An unknown first item does not repair the falsely attributed second item.

Minimum closure is a bounded refusal when a conditional expression contains a relevant unsupported named write, including in either branch. No general expression interpreter or source schema change is needed. This does not claim that ordinary installed SDXL uses the poison; it checks the authority boundary of the newly supported conditional-expression form.

## Operator projection/rails — PASS

The two source operands are distinct input cards connected to one explicit unresolved operator boundary, with a separate result port. Both ordinary `+` and augmented `+=` controls produce three arrows. The boundary declares operand dispatch, internal dependence and mutation behavior unresolved. No direct semantic-dependence or primitive-dispatch claim is introduced. Exact projected blocks and standalone SVGs are retained beside `operator-results.json`.

## Reproduce

Working directory: `/private/tmp/unfold-s8-final-operator-review`. Use the receipt's absolute path for the scripts and outputs. `replay_qualification.py` creates `actual-primary-archive/` and intentionally requires a new empty archive directory on a repeat run; preserve the existing result rather than overwrite a receipt inadvertently.

```sh
s8_review_receipt=/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-independent-9fe0868
PYTHONDONTWRITEBYTECODE=1 python3 "$s8_review_receipt/replay_qualification.py" > "$s8_review_receipt/qualification-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$s8_review_receipt/replay_ifexp.py" > "$s8_review_receipt/ifexp-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$s8_review_receipt/replay_operator.py" > "$s8_review_receipt/operator-results.json"
```

All original commands exited 0. The immutable checkout remained clean; `verification.json` records reviewed file hashes and the scoped verdict.
