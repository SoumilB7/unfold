# S5 poison index

All items are exercised by the focused or static release checks; removing the
named protection makes its test/gate red.

1. Change the README total/order from `621 / 241 / 0`, omit a reviewed model,
   or place an unseen model in the reviewed section:
   `test_readme_support_set_and_counts_are_exactly_coverage_json`.
2. Add a second numeric `__version__` authority or change the project version:
   `test_release_has_one_authoritative_package_version`.
3. Mislabel the DeepSeek example, remove generation provenance, or drift an
   example byte: `test_generated_examples_are_reviewed_and_deepseek_is_really_deepseek`
   plus `scripts/generate_examples.py --check` in CI.
4. Rewrite/drop an exact audit receipt or show its summary more than once:
   `test_ship_warning_groups_keep_exact_receipts_under_one_friendly_summary`.
5. Submit the same finding twice: `test_one_check_cannot_launder_or_rewrite_an_exact_receipt`
   proves it remains one exact receipt.
6. Construct empty, duplicate, or foreign warning metadata:
   `test_warning_group_closure_rejects_duplicate_or_empty_details`.
7. Deep-copy or mutate typed warning metadata:
   `test_typed_warning_remains_exact_and_reconstructible`.
8. Repeat a side-reader failure fourteen times:
   `test_projector_failure_detail_is_deduplicated_at_the_producer`.
9. Attempt overlapping coordinator lanes:
   `test_coordinator_runs_lanes_strictly_one_at_a_time`.
10. Install a non-0.3.0 wheel or render fewer than 29 reviewed witnesses:
    `scripts/verify_release_install.py` exits non-zero.
11. Feed the Space a typed parse refusal containing HTML:
    sibling-repo `test_app.py::test_typed_refusal_is_visible_and_escaped` pins
    its persistent typed status, escaping, and no-fallback message.
