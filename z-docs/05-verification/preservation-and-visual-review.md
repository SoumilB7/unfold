# Preservation and visual review

## Preservation contract

The preservation harness is implemented in:

- `unfold-pkg/test_support/preservation.py`;
- `unfold-pkg/test_support/tree_state.py`;
- `unfold-pkg/tests/test_preservation.py`;
- `unfold-pkg/tests/test_isolation.py`;
- the committed preservation manifests and expected artifacts under
  `unfold-pkg/tests/`.

It must regenerate current surfaces, validate every input/expected hash, compare
the exact witness set, and pass from a clean exported checkout. Missing baseline
material is a failure, not a skip.

## Surfaces

Preservation should cover architectural IR, evidence ledgers, expanded JSON,
parameter output, HTML metadata, gallery/view identities, and the required
per-view visual hashes. A single high-level HTML hash is insufficient.

## Tree quiescence

Record the tree fingerprint before and after a verification run. A test result
is invalid if the run mutates tracked or relevant untracked source, fixture, or
expected material.

## Visual review

Mechanical consistency cannot prove that connectors are readable, blocks are
properly grouped, or a visually plausible diagram communicates the wrong
sequence. Generate every distinct current view into a fresh directory and
inspect all of them.

The image pass checks factual rendering and connection integrity. A separate UX
pass may judge pacing, labels, hierarchy, bundling, and newcomer clarity, but it
cannot approve unsupported structure.

## Blessing rule

Never refresh expected artifacts to make an unexplained delta disappear. An
intentional delta requires:

- the affected model and view;
- old/new structural and visual evidence;
- the exact fact or projection change;
- why the old output was false;
- explicit user approval.

## Release sequence

1. Unit tests and poisons.
2. Audit files alone.
3. Grouped verification/conformance checks.
4. Preservation in the working tree and clean checkout.
5. Full suite on the same unchanged tree.
6. Fresh exhaustive visual matrix.
7. Approval of every intentional delta.
