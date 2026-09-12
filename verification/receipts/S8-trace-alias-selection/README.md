# Trace-stage alias selection — RETURN, reporting only

Actual 4560449 outputs contain both canonical occurrence cards and call-target aliases. `claim_traces` builds an occurrence-to-last-block dictionary, so an alias overwrites the canonical stage card. All three selected claims then report a missing overview stage: they select a nested call-slot card absent from the denoiser overview, although the exact `instance_down_blocks__0` or `instance_down_blocks__1` card is present there.

The replay preserves the reporter source, exact hashes of the actual IR/facts/HTML observation, and the selected/alternative stage IDs. Reporter bytes stayed unchanged. Root also inspected the actual overview and target drills. This is a false report gap, not a lost product stage or missing mechanism proof.

Returned to executor: select the actual visible stage card and the appropriate fact-citing mechanism card from the occurrence's available cards. Keep exact-card citation and number checks; do not weaken them. Continue frozen model generation. Run corrected reports with their own explicit reporter hash, preserving the original script pins/results. No model rerun or baseline change is needed for this report-only correction.

Replay from `unfold-pkg`: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 verification/receipts/S8-trace-alias-selection/replay.py`. Changed reporter source is rejected; retain this receipt and save the corrected result separately.
