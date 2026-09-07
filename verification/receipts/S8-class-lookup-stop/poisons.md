# S8 stop receipt: poison controls and limits

The isolated focused lane passed 104 tests (2 deselected), including the controls below. These are rejection/limited-result assertions run against the current candidate. **Separate pre-fix failing red transcripts were not captured**; this is not the completed C-8 poison inventory.

| Input/control | Test module / exact test | Required outcome |
|---|---|---|
| Same class name, different source | `test_s8_runtime_source.py::test_same_name_with_different_source_never_binds` | Source disagreement cannot bind |
| Different checkpoint runtime | `test_s8_runtime_source.py::test_other_checkpoint_cannot_supply_the_runtime_class` | Rejected |
| Population existence used for connection | `test_s8_runtime_source.py::test_constructed_population_cannot_qualify_a_connection` | Rejected |
| Drawn occurrence with unstamped fact | `test_s8_runtime_source.py::test_explicit_occurrence_placement_keeps_fact_qualification_separate` | Placement retained, qualification separate |
| Primitive bare-name imitation | `test_s8_runtime_source.py::test_framework_primitive_is_exact_type_not_bare_class_name` | No primitive authority from name |
| Omitted constructor value | `test_s8_runtime_source.py::test_omitted_defaults_are_labelled_declarations_never_mechanisms` | Labelled declaration, no mechanism promotion |
| Possible module replacement | `test_s8_cell_connections.py::test_init_member_is_not_reused_after_possible_replacement` | No invalid connection proof |
| Wrong scratch hash | `test_s8_source_override.py::test_wrong_hash_cannot_fall_back_to_installed_source` | Typed construction failure |
| Override never imported | `test_s8_source_override.py::test_unused_override_cannot_claim_the_builder_examined_it` | Reject source-examined claim |
| Preloaded module | `test_s8_source_override.py::test_preloaded_module_cannot_bypass_substitution` | Rejected |
| Duplicate module override | `test_s8_source_override.py::test_duplicate_import_address_is_rejected` | Rejected |
| Product config tries enabling old path | `test_s8_unet_routing.py::test_production_u_shape_uses_cutover_even_with_misleading_config` | New route only |
| Direct legacy invocation outside comparison | `test_s8_unet_routing.py::test_old_author_cannot_be_called_without_the_comparison_flag` | Rejected |

`counterexample/` additionally records the reproduced conditional class-member evidence gap. Its assertions passed. No new production class-lookup rule was introduced, so there is no accepted implementation claiming to close that poison. The proposed inherited/descriptor/custom-lookup closure controls are still future work. Synthetic tests do not count as the required six SDXL HTML conditions.
