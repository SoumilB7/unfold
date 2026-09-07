# Early primary-port followup

Original three controls now corrected on the saved working snapshots: other-formal normalization remains an opaque call result; with/loop-target rebinding gets an explicit unresolved region; the loop body's initial carried value uses the region input.

**One boundary-version false-positive remains:**

```python
def forward(self, state):
    conditioning = normalize(state)
    state = first(state)
    state = block(state, conditioning)
    return state
```

In the block region, the reader correctly preserves `normalize` as a call boundary, but its argument becomes `region_input`, meaning the current region's incoming value, the result of `first`. The actual normalization ran on the original argument before `first`. Opaque internal computation does not make these argument sources interchangeable.

Earliest producer: `local_port_routes.py`'s early name case, `cutoff <= region_input.start`, returns the current region input even when the query is for a historical primary-local reference inside another local's earlier definition. Either preserve that earlier reaching definition or expose the historical input as unresolved. No general interpreter or schema extension is needed.

Exact saved snapshot hashes:

- `unet_primary_ports.py`: `626e861e1d51a3df06c9363e7bd22c08b10a65ce3484f0fa2ec13535180e7e42`.
- `local_port_routes.py`: `5af8f49c0d22604a5941b3e6db223213aa25881e05102e77ac2e0cf1e8676e32`.

`reproduce.py` retains the five original sources and adds `auxiliary_reads_historical_primary`; `results.json` contains complete routes. Original run exit 0, source unchanged during probe. Source snapshots saved as `.py.txt`. This is a read-only early review, no pytest/model runs or production changes; final acceptance waits an immutable integrated checkpoint.
