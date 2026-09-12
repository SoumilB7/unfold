# Retained-log focused repeat

Actual result: **57 passed in 0.91 seconds**. The nine exact source hashes, including the registry, agree before and after execution. No model construction ran. This repeats the earlier tool-reported 42 + 15 controls because that run did not retain redirected logs or an execution-time registry pin; it does not invent those missing historical records.

Executed in the working tree based on a83d704, with the reviewed conditional declaration and public DTO changes:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_conditional_projection_contracts.py tests/test_projection_obligations.py tests/test_s6_physics.py::test_production_can_request_isolated_inventory_but_cannot_construct_in_parent tests/test_s6_physics.py::test_lookup_boundary_reexports_only_the_same_frozen_request_result_types tests/test_s6_physics.py::test_parent_import_boundary_still_rejects_implementation_helpers
```

`pytest.log` is the actual redirected output. The conditional declaration is a representative capability census, not automatic fact emission. Legacy stamping and consumption permissions remain unchanged.
