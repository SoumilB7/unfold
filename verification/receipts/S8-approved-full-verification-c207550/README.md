# Completed red verification — c207550

**COMPLETED_RED.** The whole coordinator exited 1 on commit `c2075501e4b0cf7e97cc004c3d913011e64e3632`. All 26 full-test batches completed: **4,415 passed, 30 failed, 14 skipped, 2 xfailed** (4,461 outcomes). The full lane failed. This receipt preserves that result without combining it with later diagnostics or reruns.

Other recorded lanes completed: focused 511 passed, authority 44 passed, collection 4,557 tests, static checks clean (four changed Python files), and preservation 52 passed in 927.54 seconds of pytest time. Lane-wrapper durations remain separately recorded in the original inner receipt. Outer elapsed time was 5,302.565 seconds; the inner coordinator recorded 5,302.097 seconds.

The coordinator, source-artifact, and each lane's before/after fingerprints compare equal. The outer selected external-input maps cover 59 installed source files and one cached task config and also compare equal. These equality checks do not turn the failed tests into a passing gate.

The 42 lossless streams preserve the complete released outer wrapper, inner coordinator, all 26 bracket logs and schedule, and late host telemetry. `summary.json` derives full-test counts directly from each batch's terminal pytest summary. Logs and warnings are retained exactly.

The `late-host-context/` records contain raw `vm_stat` output, exact observation times, and measured counter deltas during a late interval. They are **not original failure telemetry, per-worker memory measurements, peak measurements, or proof of a specific failure cause**. No causal diagnosis is inferred from them here.

Each complete logical path has `.gzip` appended. The map is injective; every compressed/raw size and hash and every decompressed byte was checked against the released source. Original files remained unchanged. From a clean checkout:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-full-red-restored
```

The destination must not exist. Outputs restore under separate `outer/`, `inner/`, `bracket/`, and `late-host-context/` directories. Historical paths inside original receipts remain provenance, not dependencies for archive verification.
