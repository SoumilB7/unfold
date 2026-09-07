# Independent bounded checkpoint review — 4637ef4

**RETURN: one R7 loop-else false-positive remains. R1 and R5 pass the reviewed controls.** This is a correction checkpoint verdict, not S8 acceptance or output approval.

Reviewed immutable checkpoint `4637ef4b77a8139326a8643b9f7eb68ab9d0f93c` at `/private/tmp/unfold-s8-ports-review`. No production changes, pytest, checkpoint model execution, blessing, commit, or push by this reviewer. Four standalone scripts completed with exit 0; `replay_routes.py` records rather than rejects the established counterexample.

## R1 — corrected

The original direct alias, holder-container and holder-attribute escape poisons now refuse the init-time member identity. So do writes through `alias.second`, `self._modules["second"]`, `self.__dict__["second"]`, and `del self.second`. The no-write control still accepts. Exact sources and values are in `binding-results.json` and `storage-results.json`.

## R5 — preserved after caching

The fresh d3a4d61 worker inventory was independently re-investigated using this checkpoint's static readers. Its actual installed root source still matches the runtime source hash `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`. There are 1,297 exact primitive witnesses and 70 positive ordinary FFN proofs.

Forward replacement at the FFN owner, activation owner, input/output affine descendants or dropout removes the selected FFN claim. Coherent canonical ReLU type/witness replacement at claimed affine/dropout slots also removes it. Each isolated poison leaves the other 69 positives. Removing every primitive witness produces limited FFN facts. The cache diff changes evaluation reuse; these reconstructed-proof poisons found no weakened authority. This is not a general audit of mutation of already-created cached dictionaries.

Inputs are the previously persisted `../S8-authority-followup/independent/fresh-evidence/inventory.json.gz` and `input.json`. Their exact hashes are recorded in `ffn-results.json`; no new model run was needed.

## R7 — useful new boundaries, one false-positive

The new `loop_result` boundary lawfully retains the zero-iteration input separately from the end-of-body value. The tested body input remains `loop_carried`. Branch alternatives stay conditional; augmented updates retain both operands and `+` without claiming dispatch semantics. Break and continue make the end-of-body result unresolved. Original loop/with target and starred-unpack controls remain limited.

However, the following source produces an unconditional `call_result` for the `finish` assignment:

```python
class Cell:
    def forward(self, state, items, enabled, extra):
        for item in items:
            state = step(state)
            if enabled:
                break
        else:
            state = finish(state)
        consume(state)
```

On `state=0`, `items=[1]`, `enabled=True`, `step(x)=x+1`, and `finish(x)=x+100`, the actual consumed value is 1; `finish` does not run. The review executes precisely this small synthetic control and records the result. The returned route nevertheless identifies the consuming value as a call result from `finish`, merely leaving that call's input unresolved. An unresolved argument does not repair a false producer attribution.

Earliest positive producer: `model_unfolder/evidence/local_port_routes.py:100` selects the loop-else binding as guaranteed because its indexed guard is empty. Lines 164–171 return that assignment before the completed-loop/control-transfer checks at 172–202. Existing `LoopObservation.else_span` identifies the source region; a bounded refusal or explicit loop-exit conditional can close this without a new index schema. The exact source, binding guards, route and concrete value are recorded under `loop_else_break` in `route-results.json`.

The receiver control also confirms that `state.flatten()` currently has no incoming receiver port. This is the omission the executor had already identified, not an additional false semantic-dependence claim: the call remains explicitly opaque. Connecting preprocessing through method calls requires a separately evidenced receiver route or an explicit missing-input boundary. A callee attribute's receiver syntax is available in the existing index; it must remain distinct from target binding and internal dependence.

## Reproduce

Working directory: `/private/tmp/unfold-s8-ports-review`. Let `receipt` below denote `/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-ports-review-independent` and `fresh` denote the sibling `S8-authority-followup/independent/fresh-evidence` directory.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 "$receipt/replay_bindings.py" > "$receipt/binding-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$receipt/replay_storage.py" > "$receipt/storage-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$receipt/replay_ffn_fresh.py" --inventory "$fresh/inventory.json.gz" --config "$fresh/input.json" > "$receipt/ffn-results.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$receipt/replay_routes.py" > "$receipt/route-results.json"
```

Tracked-file fingerprint before/after: `b8ef9928edf6dc59c3260dbf4ffbbc27feb8b127d0792549193c425eb1283302`. Checkout remained clean. See `verification.json` for checkpoint, statuses and scope.
