# Immutable committed-input supplement

Three exact files were recovered from Git commit `c2075501e4b0cf7e97cc004c3d913011e64e3632`: the executed `run_broad.py`, its sibling `manifest.json`, and `selected-external-inputs.json`. The original 42-stream archive and map remain unchanged.

The recovered runner SHA matches the original command record's `runner_sha256`. Expanding the committed manifest's commit placeholder yields the exact recorded command. The committed selected-source/config union equals the actual before map. These bindings are recorded in `binding.json`.

The original run did not separately persist runtime byte hashes for the two input JSON files. This supplement supplies immutable committed bytes and the explicit command/value correspondence; it does not invent missing historical runtime hashes or substitute today's files.

All three streams use the same injective appended-`.gzip` format. Verify with `python3 verify_restore.py`, or restore to a new directory with `python3 verify_restore.py --out /private/tmp/s8-c207-committed-inputs`.
