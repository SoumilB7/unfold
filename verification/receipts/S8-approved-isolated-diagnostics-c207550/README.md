# Isolated diagnostics — c207550

Three isolated original tests passed: FFN activation 1 passed in 270.72 seconds, output-drain/capture 1 passed in 3.34 seconds, and constructor memory cap 1 passed in 2.09 seconds. Wrapper durations remain separately recorded in each actual result. All recorded source/input and selected external-input before/finally maps compare equal.

The archive preserves the exact original commands, diagnostic scripts, logs, requests and results, observed call counts, full captured ModelIR and reader metadata for the FFN case, and supervisor observations for the physics cases. Scripts match the SHA-256 values recorded by their actual runs. No tests, models, or renderers were rerun for this packaging.

These isolated passes do **not** recover the unmeasured cause of the prior broad failures or replace the completed red gate. Passive observation has overhead and may alter scheduling. Retained supervisor RSS samples are actual completed samples, not a true process peak; no child import/construction phase instrumentation or historical resource state is inferred.

`artifact-map.json` preserves complete logical paths with an appended `.gzip` suffix. Raw and stored bytes/sizes/hashes, injective mapping, source stability, and full restoration are verified. The original released diagnostic directories and logs remain unchanged.

From a clean checkout, run `python3 verify_restore.py`, or restore into a new directory with `python3 verify_restore.py --out /private/tmp/s8-isolated-diagnostics-restored`. Each diagnostic has its own namespace; shared physics script bytes are stored once under `executed/`.
