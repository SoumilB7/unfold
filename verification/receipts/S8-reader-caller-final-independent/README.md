# Final caller/deletion audit — c38b008

ACCEPT within the bounded caller/deletion and legacy-routing scope. Reviewed immutable `c38b008af8153302d85bbaf2a8e3593841d87e1a` at `/private/tmp/unfold-s8-compiler-final`. This updates the historical 4637ef4 caller evidence; it does not rerun or replace the old classification results.

All 14 original `unet_*.py` modules present at accepted S7 commit `83140f1` remain in this tree and have live production callers. None of these original modules is orphaned, so their caller-or-delete requirement is met by retention. Calls passed to the orchestration `run` function are executed by that function, not merely imported. Dependency failures can prevent particular readers from running; caller presence is not model coverage.

| Original module | Production caller |
| --- | --- |
| `unet_attention_source.py` | `unet_runtime.py:91` |
| `unet_cell_mechanism.py` | `unet_runtime.py:79`, `unet_runtime.py:70` |
| `unet_nested_mechanism.py` | `unet_runtime.py:81`, `unet_runtime.py:97` |
| `unet_root_preprocess.py` | `unet_runtime.py:87` |
| `unet_selected_child_execution.py` | `unet_runtime.py:83` |
| `unet_selected_constructor.py` | `unet_root_preprocess.py:145`, `unet_selected_spatial.py:221`, `unet_cell_connections.py:189`, `unet_attention_source.py:696` |
| `unet_selected_spatial.py` | `unet_runtime.py:86` |
| `unet_selected_stage_children.py` | `unet_runtime.py:77` |
| `unet_stage_cells.py` | `unet_runtime.py:68` |
| `unet_stage_construction.py` | `unet_runtime.py:57` |
| `unet_stage_constructor_operands.py` | `unet_runtime.py:75` |
| `unet_stage_execution.py` | `unet_runtime.py:60` |
| `unet_stage_operands.py` | `unet_runtime.py:73` |
| `unet_stage_selection.py` | `unet_runtime.py:71` |

`unet_selected_constructor` is shared selected-constructor/guard proof infrastructure, not an independently invoked ReaderResult reader. Ordinary path: parser.py:653–667 → unet_cutover.py:32–94 → unet_runtime.py:46–100 → existing typed fact store → existing UNet projector. Exact runtime member intersection remains at unet_stage_cells.py:550–555. No parallel IR was introduced by these caller routes.

## New live modules

| Module | Concrete production caller(s) |
| --- | --- |
| `unet_runtime.py` | `unet_cutover.py:42` |
| `unet_claims.py` | `unet_cutover.py:61`, `unet_cutover.py:62` |
| `unet_cell_connections.py` | `unet_cutover.py:57` |
| `unet_primary_ports.py` | `unet_cutover.py:60` |
| `unet_call_binding.py` | `runtime_inventory.py:48`, `unet_runtime.py:99`, `unet_primary_ports.py:153` |
| `unet_lookup_closure.py` | `unet_call_binding.py:149`, `unet_call_binding.py:160` |
| `unet_wrapper_binding.py` | `unet_primary_ports.py:151` |
| `unet_iteration_binding.py` | `unet_call_binding.py:229` |
| `local_port_routes.py` | `unet_claims.py:261`, `unet_claims.py:282`, `unet_claims.py:456`, `unet_claims.py:423`, `unet_primary_ports.py:97`, `unet_primary_ports.py:49` |
| `runtime_inventory.py` | `unet_cutover.py:38` |
| `runtime_source.py` | `unet_claims.py:28`, `unet_claims.py:105`, `unet_claims.py:203`, `unet_claims.py:305`, `unet_claims.py:379`, `unet_claims.py:488`, `unet_claims.py:212`, `unet_claims.py:37` (additional exact references in result.json) |
| `instance_population_claim.py` | `unet_cutover.py:49`, `unet_cutover.py:50` |
| `instance_shape_claim.py` | `unet_cutover.py:52` |
| `attribute_bindings.py` | `instance_inventory.py:661`, `instance_inventory.py:615`, `instance_inventory.py:689` |
| `framework_primitives.py` | `instance_inventory.py:630`, `instance_inventory.py:688` |
| `source_override.py` | `runtime_inventory.py:22`, `runtime_inventory.py:40`, `runtime_inventory.py:46`, `runtime_inventory.py:56`, `unet_cutover.py:39`, `instance_inventory.py:445` |

Additional neutral producer accounting: existing program_index.py now emits TryObservation at line1789; live wrapper consumer calls try_observations_in at unet_wrapper_binding.py:44. Worker attribute observations are called at instance_inventory.py:661; exact primitive capture/selection at :688/:630; source override context at :445. Construction demand originates at runtime_inventory.py:48. These are construction/source-address observations and proof consumers, not a second architecture IR.

`execution_recipe.py` is live verification infrastructure: generate_s7_shadow.py:305 and verify_s8_unet_family.py:17 import its recipe functions. It is intentionally not a compulsory ordinary production parse probe. The temporary `unet_differential.py` bridge and old unet.py authority remain for explicit comparison. No wholly unused new module bridge was established in this bounded audit. Two live exact_callable helpers remain in call_binding and lookup_closure; duplication alone does not establish a dead bridge. This is not an exhaustive private-helper liveness or growth-budget audit.

## Legacy isolation and actual controlled probes

parser.py:655–656 invokes old _parse_unet_model only under legacy_comparison_enabled(). The old helper independently rejects ordinary calls at :49–51 before importing old unet.py at :52. ContextVar defaults false and resets in finally. The only non-test enabling caller in model_unfolder/physics/scripts is demonstrate_s8_unet.py:298, conditional on the explicit legacy demonstration condition. Product config does not enable the flag.

Five independent standalone controls passed on this exact tree: old direct call rejected; nested comparison flag restored after failure; misleading config fields still select new path; inventory failure returns a typed limited result; source-closure failure returns a typed limited result. The last two replace the legacy function with an assertion trap, so no silent old fallback can pass. They stub construction and topology; they do not run a model or establish model coverage. Unexpected uncaught exceptions likewise have no old-path recovery branch in the reviewed parser.

Legacy module functions remain directly importable by deliberate internal Python callers; this is ordinary routing isolation, not an access-control boundary. No output changes were blessed.

## Reproduction and pins

`python3 verification/receipts/S8-reader-caller-final-independent/replay.py`

`result.json` records exact entry definitions, real production reference lines, all 14 original modules, 16 new live modules, five probe outcomes, and source SHA256 pins. Immutable source hashes remained unchanged. No pytest, model execution, family classification, broad gate, source-grammar audit, production edits, commit or blessing.
