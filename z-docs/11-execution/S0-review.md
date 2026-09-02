# S0 review — reviewer verdict (Claude, 2026-09-02)

**Verdict: ACCEPT** (conditional on the reviewer's own preservation-lane
reproduction, running at the time of writing; the executor's lane passed
52/52 and all other checks reproduced. If the reviewer's lane fails, this
flips to RETURN before S1 closes.)

Reviewed tree: `4a8b338` (implementation) + `dbacd66` (receipt/sheet).
Method: `15` §7 — isolated worktrees at both commits, every C-0 check walked,
every poison run by the reviewer, receipt reproduced.

## What reproduced

| check | result | how |
|---|---|---|
| tree grep for `hf_…` literals | empty | `git grep` on `dbacd66` |
| notebooks scrubbed | clean | both files re-read on disk; cell-aware scanner |
| settings scrubbed | clean | `.claude`, `~/.claude/settings*.json` grepped |
| notebooks / `done/` untracked | 0 tracked | `git ls-files` |
| hub-side revocation | not independently re-verifiable (credential not recorded, by design); receipt records 202 then 401 | accepted on receipt; the only token on the machine is the live CLI login (200), not the revoked one |
| staged-notebook scanner poison | RED, literal not echoed | run by reviewer in worktree |
| output / metadata poisons | RED | `tests/test_repository_hygiene.py` 6 passed |
| commit-body hook | subject-only rejected; body accepted | real commits in worktree |
| z-docs + PROTOCOL tracked | 96 files | `git ls-files`; workspace `z-docs` is a symlink into the repo (one copy) |
| archive | 12 files match `06` §6 retire list | `archive/superseded-docs/2026-09-02/README.md` |
| blessed-artifact fingerprint | **exact match** `1b5e63…` | recomputed over `tests/preservation_baseline` + galleries |
| tree fingerprint | **reconstructed match** `fc45ee…` | equals pristine `4a8b338` content + the `.git` pointer of one serial worktree `verify-s0-4a8b338`; identical across all six lanes |
| full suite | 3,981 passed / 14 skipped / 2 xfailed | lane logs in `/private/tmp/model-unfolder-verification/s0-4a8b338/` |
| preservation | 52 passed, 29/29 (executor) | reviewer reproduction: in progress |

## Carry-forwards (required, no re-review; include in the S1 commit)

1. **Machine lane evidence in-repo.** The coordinator's `receipt.json` was not
   produced (lanes ran serially through a driver). Copy `lanes.tsv` and the six
   lane logs into `verification/receipts/S0-4a8b338/lanes/` so the receipt is
   not a hand-authored restatement of files that live in `/private/tmp`.
2. **Fingerprint primitive.** `test_support/tree_state.py` hashes the `.git`
   worktree pointer *file*, so `tree_fingerprint` is worktree-path-dependent
   and irreproducible elsewhere. Exclude the `.git` file (the `.git` dir is
   already excluded) — a producer fix, owned by S4 "receipts in-repo"; record
   a content-only fingerprint from then on.
3. **Retire-list remainder.** `docs/models_supported.md` (empty),
   `docs/architecture_atlas.html` (untracked), `docs/TRUE_CONFIG_PLAN.md`
   (modified, "until code exists") are still in `docs/`; archive or decide in
   S3's hygiene pass.
4. **Stale worktrees.** Six `verify-bb2795cf7c-*` worktrees from 2026-08-19
   (`ee836b0`) are still registered under `.claude/worktrees/`; prune them.

## Notes
- The sheet's "arbiter approval not required" is correct: zero deltas.
- The correction that the two notebook cells held one duplicated credential
  is accepted; the register's D9 wording is updated by this note.
