# Independent L1/L2 correction review — L2 ACCEPT; L1 temporal closure RETURN

Reviewed source bytes are retained in source/. fingerprints.json records hashes and identical before/after state. No pytest, full model runs, production edits, commits or blessings. Tests execute only tiny synthetic fixtures or replay the fresh saved actual inventory.

Original controls now PASS:

- replay_lookup_receiver.py: original ConfigProbe custom fallback is unresolved because its stored receiver effect lacks a witness. No future target is invented from the initial successful lookup.
- replay_helper_conditions.py: the target inside the inherited false helper guard is suppressed. The compatible root helper and subsequent root child stay conditionally bound.
- replay_saved_helpers.py: fresh S8-actual-lookup-positive-v2 inventory SHA256 d5ac60a14617eeae31cc4dacbeb54ea1c93a3a66915d3736b3980952f75e867c, 1,930 occurrences, current exact source files checked against saved hashes. All four actual root helpers remain conditionally closed; same-source guard contradictions are zero. This is a static saved-evidence replay, not another actual constructor run.
- replay_missing_nested_premise.py: removing only the _internal_dict / encoder_hid_dim_type scalar premise makes process_encoder_hidden_states unresolved while retaining the other three helper positives. The source reader requires the actual nested storage premise.

The worker records exact requested storage attributes, default lookup/no-fallback/no-descriptor selection, constructor identity and current namespace agreement, and scalar/missing/opaque result kinds. It does not rely on a FrozenDict name whitelist. Literal storage-address and requested-attribute matches are required by the source consumer. These changes close the original L1 stored-receiver counterexample and L2 inherited-guard counterexample.

## Residual L1: a storage alias write invalidates a borrowed scalar premise

The temporal closure remains incomplete. In unet_lookup_closure.py, close_parent_helpers builds its aliases from parent_value(property_storage) and uses them to reject escapes in call arguments and returns. It does not reject writes through those same aliases. Its final _member_stays_bound call builds a separate alias set which does not treat self.config as the parent, so a storage alias write passes both checks.

replay_storage_temporal.py is a tiny actual-worker fixture using the already supported alias, assignment and if forms:

    alias = self.config
    alias.flag = POISON
    if self.config.flag:
        pass

Construction proves the exact stored receiver's flag is a scalar False. At runtime the alias write replaces it with POISON. Testing its truth value invokes POISON.__bool__, which replaces owner.child from Linear(2,2) to Linear(3,3). The reader nevertheless returns helper in closed_helpers, with no conditions or unresolved finding. This is a false parent-stability claim caused by using the constructor scalar premise after a known storage write.

Minimum correction: reject attribute/subscript writes rooted in the already tracked storage/property aliases before qualifying later scalar reads or closing the helper. The direct nested-write/parent alias law already exists; carry that refusal through these new storage aliases. No descriptor evaluator, new syntax, new IR or architectural special case is needed. The original alias/property/storage proof scope requires this temporal check.

Commands from unfold-pkg:

    python3 verification/receipts/S8-l1-l2-independent/replay_lookup_receiver.py
    python3 verification/receipts/S8-l1-l2-independent/replay_helper_conditions.py
    python3 verification/receipts/S8-l1-l2-independent/replay_saved_helpers.py
    python3 verification/receipts/S8-l1-l2-independent/replay_missing_nested_premise.py
    python3 verification/receipts/S8-l1-l2-independent/replay_storage_temporal.py

The temporal replay deliberately records the current false positive. Positive ordinary replay does not cancel that counterexample. New iteration integration and actual HTML rendering remain outside this bounded correction review. Historical receipts were not overwritten.
