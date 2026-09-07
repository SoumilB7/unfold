# Working local-port reader review — RETURN on two bounded items

Reviewer: `/root/independent_review`. This review concerns the working `model_unfolder/evidence/local_port_routes.py`, identified by the before/after SHA-256 `d901282e6abebf9494406e3e99871fb5f7718d8e94a949445fd02c7528965943` in `local-port-results.json`. It is not an immutable whole-tree verdict. No production edit, pytest lane, model execution, commit, or blessing.

Command from the primary checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verification/receipts/S8-owner-review-initial/reproduce_local_ports.py > verification/receipts/S8-owner-review-initial/local-port-results.json
```

The command exited 0 and source fingerprints remained equal during the probe. Its assertions reproduce the defects; they should fail after correction. The executor changed the working reader afterward, before a source snapshot could be copied; the snapshot-copy hash check rejected that changed file. No whole-file snapshot is claimed. Results describe only the named source hash, and the corrected reader requires re-review.

## R7 — incomplete reaching-definition coverage presents an earlier formal as the actual value

Earliest false producer: `seed()` at lines 96–111. An earlier guaranteed assignment returns before the loop-carried check. This body is incorrectly routed as the unconditional original formal `seed`:

```python
state = seed
for item in items:
    consume(state)
    state = update(item)
```

On later iterations `state` is the prior iteration's result, not necessarily `seed`. The reader's advertised loop boundary must cover locals initialized outside the loop, including writes textually after the queried call. This does not require deriving every iteration: an explicit loop-carried/unknown boundary is sufficient.

The reader also overlooks writes outside ordinary `BindingObservation` assignment records:

```python
def forward(self, state, items):
    for state in items:
        consume(state)
```

It routes `state` as the incoming formal rather than the iteration target. Likewise, `with manager() as seed: consume(seed)` incorrectly routes the original formal `seed`. Use the existing loop observations and unsupported-execution coverage to account for these writes or refuse the affected route. Unsupported control is allowed to remain unresolved; it must not preserve a stale formal as a positive connection.

## R8 — starred unpacking produces an incorrect fixed result slot

Earliest false producer: `_slot()` at lines 26–34. It counts tuple positions while ignoring a starred sibling:

```python
first, *middle, last = helper(seed)
consume(last)
```

The route claims `result_slot=[2]`. For a helper returning four items, `last` is slot 3; in general its position depends on result length. Either establish supported unpacking semantics and return-slot meaning, or mark this route unresolved. An unknown helper's result shape cannot justify a fixed index after a starred target.

## Supported behavior

The probes confirm that an ordinary local alias reaches its formal and a guarded exact assignment reaches its call result. Opaque helper results retain `mechanism=unresolved`; the result records separate argument ports without asserting argument-to-result semantic dependency. That separation is lawful and is not a RETURN item.

Returning unresolved for multiple optional assignments or unsupported selections is also a support limitation, not itself an authority defect. R7/R8 concern positive wiring that the supplied source does not establish.
