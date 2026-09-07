# Early independent primary-port producer review

**RETURN on three boundary-history defects.** This is a review of uncommitted source snapshots, not the final integrated product or S8 acceptance. No production edits, pytest or checkpoint execution. The standalone probe creates only temporary synthetic source and review artifacts.

Exact reviewed bytes, saved beside this receipt:

- `unet_primary_ports.py`: SHA-256 `4275ed21b63bccb5574ab73a9b47e4f2f5073c739d191f42d1d961a1d3c18991`.
- `local_port_routes.py`: SHA-256 `6ed2805017f30761904a0adc835bc44d2af3d9a107e5dea8514dff208d32713c`.

The script verifies both files are unchanged throughout its probe. All following line numbers refer to those exact snapshots. `results.json` records each complete synthetic source and produced region route.

## 1. Clipping the primary region also erases other locals' definitions

`local_port_routes.py:98` drops every earlier binding, regardless of which local is being routed. The final formal fallback at 187–189 then claims the original argument:

```python
def forward(self, state, conditioning):
    conditioning = normalize(conditioning)
    state = block(state, conditioning)
    return state
```

The block's second argument is reported as `formal(conditioning)`. Its actual source is the result of `normalize(conditioning)`. Leaving the block's internal mechanism unresolved does not correct this false argument producer. The same result occurs when the conditioning assignment is between two state regions.

This matters to the real root: SDXL assigns `encoder_hidden_states = self.process_encoder_hidden_states(...)` before the down loop. Partitioning only `sample` writes must not restore the original encoder-hidden-state argument at later stage calls. It may preserve the opaque preprocessing call boundary or explicitly limit that auxiliary route.

Minimum closure: apply the primary boundary only to the primary local; retain independently reaching definitions for other arguments, or refuse unresolved outside-region histories. No general dependence analysis is required.

## 2. Inter-region rebinding can disappear entirely

`unet_primary_ports.py:29–34` partitions only `BindingObservation` writes. The existing loop-target record and unsupported with-target region do not enter that partition:

```python
state = first(state)
for state in items:
    pass
state = block(state)
```

Likewise with `with manager() as state: pass` in the middle. Both produce two regions, each reporting `receives_previous_state: true`; the second region contains an unqualified `region_input`, and no region records the intervening rebind or unknown boundary.

A region input can mean the value at that region's entry. It cannot establish that value is the preceding listed region's result when unaccounted source exists between them. The advertised previous-state connection at lines 42–43 therefore needs the history check. Existing loop target and unsupported-region observations are sufficient to insert an explicit limited boundary or withhold that predecessor connection. This is a bounded completeness requirement for the selected local, not a request to interpret the unsupported construct.

## 3. The repeated body's seed disagrees with the loop boundary seed

```python
state = first(state)
for item in items:
    state = block(state)
```

The outer `loop_result.initial_route` correctly becomes `region_input`, representing the preceding region's result. The body call's `loop_carried.initial_route` instead becomes `formal(state)` at `local_port_routes.py:178–181`. The initial carried value is the result of `first`, not the original argument. Use the same applicable region seed in both positions, or leave the inner seed limited. This does not require resolving the iteration count.

## Supported direction

The producer explicitly keeps module target binding unresolved and treats constructed stage membership as containment. That separation is appropriate. Explicit conditional, loop carry, zero-iteration, opaque receiver and augmented-operation boundaries are also the right bounded representation. The defects above concern the correctness of inputs supplied to those boundaries, not a demand to prove opaque input-to-output semantics.

## Reproduce

Working directory: primary `unfold-pkg`. Replaying against a later working file is a new review, so compare the reported hashes; use the saved source snapshots to identify the version covered by this receipt.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verification/receipts/S8-primary-ports-early-independent/reproduce.py > verification/receipts/S8-primary-ports-early-independent/results.json
```

Original run exit 0; all five counterexample outputs collected. No files outside the receipt and temporary scratch source were changed by the reviewer.
