# Independent authority followup — RETURN on R1

Reviewer: `/root/independent_review`, independent of the implementation and correction authors. Checkpoint `d3a4d61732233a14ee67917f48fe5ec11d41f10e`, clean detached checkout `/private/tmp/unfold-s8-followup-review`.

**R5 and R7 pass the reviewed correction controls. R1 remains partly open.** This is a scoped authority verdict, not full S8 acceptance or permission to bless. No production/test edits, pytest lane, commit, blessing, or checkpoint-model execution were performed by this reviewer. Tiny canonical/custom primitive calls are used in the inherited R2 control.

## R1 — remaining explicit attribute cases

The original direct alias and list-held-parent cases now correctly refuse stale member identity. Two cases still return `member_stays_bound=True`:

```python
holder = Namespace()
holder.parent = self
helper(holder.parent)
value = self.first(value)
return self.second(value)
```

```python
alias = self
alias.second = replacement
value = self.first(value)
return self.second(value)
```

Earliest false producer: `_member_stays_bound` in `model_unfolder/evidence/unet_cell_connections.py`. The new local target handling notices parent references stored into holder attributes, but `parent_escapes()` still rejects every attribute read as a possible escape. Attribute-write invalidation still delegates to `_member()`, which recognizes only the literal `self` receiver despite the known alias set.

These are explicit, indexed local assignments. Refusing affected bindings through the already known aliases/holders closes the cases without a general Python alias engine. Full executable sources and outputs are in `binding-results.json`. Return this boundary to the executor; no new model-name exception is needed.

## R5 — positive FFNs preserved and altered premises rejected

The fresh ordinary SDXL run supplied 1,930 module occurrences with **1,297 actual framework primitive witnesses**. Its raw inventory and config are persisted under `fresh-evidence/`; the source implementation manifest and ordinary result accompany them. This evidence was produced by the visual/executor lane on `d3a4d61`, not fabricated by the reviewer.

The reviewer reconstructed the static reader chain from that evidence on the immutable checkpoint. **All 70 ordinary FFN positives survive.** For the selected FFN, each of these recorded callable changes removes exactly its fact while preserving the other 69:

- FFN owner `forward` replacement;
- selected input-transform `forward` replacement;
- actual input affine projection `forward` replacement;
- transparent dropout `forward` replacement;
- actual output affine projection `forward` replacement.

The input projection, dropout and output projection were also replaced individually by coherent ReLU class/module/MRO records with an actual captured canonical ReLU witness, regenerating reconciliation each time. Each affected FFN fact is correctly rejected; 69 others remain. The old no-worker-witness condition now yields a limited FFN fact set, rather than borrowing type authority from names.

These are explicit typed-boundary poisons applied to copies of the fresh inventory. They do not claim that SDXL actually constructed those replacements. Existing parameter registrations can remain unused on a replacement module; no contradictory Linear witness was retained. The complete results are in `ffn-results.json`.

Static and runtime root source hashes match: `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`. This establishes the fresh ordinary positive control and the bounded descendant-premise corrections. It is not a blanket validation of arbitrary Python mutation or a full product demonstration verdict.

## R7 and retained controls

The reviewed loop-carried local, loop-target and `with`-target cases are now limited correctly both inside and after their regions. Starred result unpacking remains unresolved. Ordinary aliases and guarded direct assignments retain their positive routes. Opaque helpers still separate input ports from result ports and keep internal computation unresolved.

The replay also retains the reviewed R2 exact-type and R3 wrong-concat controls. `binding-results.json` records their results. R6's prior immutable correction verdict is unchanged and was not repeated here.

## Reproduction and limits

From `/private/tmp/unfold-s8-followup-review`:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/replay_bindings.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/binding-results.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/replay_ffn_fresh.py --inventory /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/fresh-evidence/inventory.json.gz --config /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/fresh-evidence/input.json > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-followup/independent/ffn-results.json
```

Both commands exited 0. Their assertions distinguish repaired controls from the remaining R1 defects; a zero exit is not a green implementation verdict. Binding replay fingerprints are identical: `2f06be744a94dfaccfd0dfef7edec10db2a772607833ea0f1ab146626ea0fd0c`; the final tracked-file check and receipt hashes are in `verification.json`. The checkout remained clean.

[execution-scope.md](execution-scope.md) separately records the bounded interpretation of recipe execution: production need not run recipes on every parse, while the family exit comparison must preserve compatible accepted observations and cannot expand the original 240-occurrence allowance. This is an authority interpretation, not an implementation approval.

Recipe/differential/visual/C-8 completion, preservation and broad-gate acceptance remain separate. No output-delta approval or re-bless is recorded here.
