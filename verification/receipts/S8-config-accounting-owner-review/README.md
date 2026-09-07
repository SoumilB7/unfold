# Owner disposition: config accounting remains visibly limited

This is a **named unresolved accounting limitation**, not a claim that configuration consumption is complete. It does not qualify any architectural fact and does not close a missing reader investigation. No production code, source evidence, original artifact, or baseline was changed.

The old UNet author explicitly called `unet._consume` (`unet.py:23–34`) and assigned reads to its `unet_geometry` mechanism. For example, line161 consumed `act_fn` into the old `denoiser.ffn.activation` target. The new production path records twelve separately qualified canonical facts through `unet_cutover.py`; it does not pretend those facts automatically consume every config occurrence in the older access ledger. That old author remains comparison-only.

The persisted new access census has present denoiser reads and **zero denoiser consumed events**. `model_unfolder/parser.py:253–286` therefore correctly emits `audit_incomplete=['root.denoiser']`: the owner neither has consumption records nor a complete occurrence-exact registered pending census. The two standing occurrences are `act_fn` and `norm_eps`. Their observed access is not evidence that their values have been projected. We do not retrospectively invent reader call stacks or consumption receipts for these accesses.

| Actual witness | Accessed but unprojected findings | Standing findings | Incomplete owner findings |
| --- | ---: | ---: | ---: |
| SDXL | 45 | 2 | 1 |
| SD-v1-4 | 17 | 2 | 1 |

The independent replay verifies every exact finding against both its persisted ship receipt and the actual HTML diagnostic text (excluding script/style text), and finds zero migration-claim violations. Artifact hashes remain unchanged. The Sable boundary in `sable.py:299–308` explicitly accepts exact surfaced limitations as a no-silence disposition; its docstring also says a chip is not a proof. `ship_findings.py:122–128` joins exact check/message pairs, so a generic warning cannot clear a different finding.

**S8 disposition:** retain these precise visible findings, including `audit_incomplete`; accept their output deltas as a named limited result when retiring the old author. Do not manufacture config consumption, suppress the audit, or claim it is clean. This is separate from S8's required qualification of every fact actually projected by the UNet cutover. Final family qualification and the broad gate still have to run.

**Continuing debt:** the config-consumption integration needs its own exact occurrence-to-fact/inspection/debt decisions during the remaining configuration-accounting work. It must be listed in the S8 sheet and carried into S9 review, rather than erased by this disposition. This receipt does not authorize further mechanism expansion in S8 or baseline blessing.

Reproduce from `unfold-pkg` with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 verification/receipts/S8-config-accounting-owner-review/replay.py`. `result.json` retains all exact messages, old consumed occurrences, actual output hashes and the source pins used for this review.
