# Bounded full-lane scheduling change

Prepared after the actual c207550 red gate and three separately accepted diagnostics. The only runner behavior change from `S8-final-lanes-fb1f871/run_broad.py` appends the existing coordinator argument `--workers 2`. All targets, assertions, child limits and six serial lanes remain unchanged. The original command manifest and selected external input bytes are copied exactly.

This preparation is not a green verification result. The original full failures and missing failure-time resource samples remain in their exact receipts. See `scheduling-rationale.json` for measured context and limits.
