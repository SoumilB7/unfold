# S4 poison outcomes

These are the violations injected by the focused committed-tree lane.  Each
test is green because the corresponding production gate returned the stated
red result; none is a fixture that merely asserts a helper exists.

| poison | production result asserted by the test |
|---|---|
| Change only a frozen view label | `labeled view drift — a view label or its exact SVG changed` |
| Remove `decoder.attention.mechanism` from the current proven set | `recall regression: proven fact 'decoder.attention.mechanism' became unresolved or disappeared` |
| Add an asserted `future_default` without a receipt | `asserted_facts` returns the exact default finding |
| Replace that receipt with the same message under `some_other_check` | `config_migration_claims` returns the original violation; cross-check laundering fails |
| Set one coverage row to `silent=1` | `coverage.json: <model> has silent=1` |
| Remove one model from the frozen denominator | `<cohort> denominator changed` |
| Raise `RuntimeError('poison parser failure')` in the zero-asserted parse | the same `RuntimeError` escapes; it is not converted to an empty census |
| Set `visual_review='CLEAN'` without a verdict | `independent persisted review verdict required` |
| Persist `reviewer == implementer` | `reviewer must be independent from the implementer` |
| Return a Qwen projector caller outside the owner graph | main layers remain; `projector evidence unresolved` is present in IR and HTML |
| Return the same side-reader failure when no projector exists | no projector key is created |
| Render a typed ship finding alongside other warnings | badge contains `unresolved evidence` and not `partial config` |

The exact tests and their source assertions are:

- `tests/test_sable.py::test_regression_gate_rejects_a_label_only_change`
- `tests/test_sable.py::test_regression_gate_rejects_proven_to_unresolved`
- `tests/test_sable.py::test_asserted_fact_gate_requires_its_exact_visible_receipt`
- `tests/test_projection_audit.py::test_ship_receipt_poison_missing_or_wrong_receipt_keeps_gate_red`
- `tests/test_s4_recall_and_coverage.py::test_coverage_poison_one_silent_model_turns_the_gate_red`
- `tests/test_s4_recall_and_coverage.py::test_coverage_poison_cannot_shrink_the_declared_support_set`
- `tests/test_sable.py::test_zero_asserted_census_reraises_unexpected_parser_failures`
- `tests/test_sable.py::test_bless_refuses_self_set_clean_and_missing_visual_artifacts`
- `tests/test_sable.py::test_bless_refuses_a_non_independent_verdict`
- `tests/test_s3_consumer_honesty.py::test_qwen3_conditional_generation_plain_unfold_surfaces_side_reader_failure`
- `tests/test_s3_consumer_honesty.py::test_side_reader_failure_does_not_fabricate_a_projector_block`
- `tests/test_s4_recall_and_coverage.py::test_typed_ship_finding_is_never_mislabeled_as_partial_config`
