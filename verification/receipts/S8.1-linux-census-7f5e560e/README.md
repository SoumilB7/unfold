# S8.1 Linux census freshness correction

The first pushed closure `7f5e560e` failed [Linux run34452333662](https://github.com/SoumilB7/unfold/actions/runs/34452333662) at config-consumption census freshness. Native sandbox proof passed; all later gates were skipped.

The unchanged producer reproduced the complete Linux diff in a committed isolated checkout. The census document dated from S4; S8.md already recorded the SDXL accounting limitations. The new document preserves two UNCLASSIFIED reads (`act_fn`, `norm_eps`),32 additional pending occurrences and incomplete denoiser accounting. No new consumption claim, disposition, source or checker change was made.

Independent exact-delta approval is in `independent/actual-candidate-review.json`. Generated-document commit `8d7bcdea` changed only `docs/COR5_NET1_MIGRATION_DEBT.md`. The unchanged committed-tree `census.py --check` passed, with matching source/blessed-artifact fingerprints; see `check/result.json`. The original Linux failure and complete candidate/diff are retained.

This is a generated discovery-report refresh, not S9 migration or a product/preservation re-bless. The full Linux workflow must still pass at the next pushed head. Original S8.1 local output/preservation/budget checks remain in [the closure receipt](../S8.1-approved-closure-b3624f2a/README.md).
