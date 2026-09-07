# Final caller and growth refresh —7168440

Read-only static source refresh against accepted83140f1. All14 original UNet modules have live production callers; none is deleted. The16 previously reviewed added support modules remain live, and construction_summary is the17th. This establishes public-entry caller paths, not complete model coverage or exhaustive private-helper necessity. No model, pytest, routing probe, production edit or blessing ran.

| Original module | Exact final caller references |
| --- | --- |
| `unet_attention_source.py` | `model_unfolder/evidence/unet_runtime.py:91` |
| `unet_cell_mechanism.py` | `model_unfolder/evidence/unet_runtime.py:79`; `model_unfolder/evidence/unet_runtime.py:70` |
| `unet_nested_mechanism.py` | `model_unfolder/evidence/unet_runtime.py:81`; `model_unfolder/evidence/unet_runtime.py:97` |
| `unet_root_preprocess.py` | `model_unfolder/evidence/unet_runtime.py:87` |
| `unet_selected_child_execution.py` | `model_unfolder/evidence/unet_runtime.py:83` |
| `unet_selected_constructor.py` | `model_unfolder/evidence/unet_root_preprocess.py:145`; `model_unfolder/evidence/unet_selected_spatial.py:221`; `model_unfolder/evidence/unet_cell_connections.py:189`; `model_unfolder/evidence/unet_attention_source.py:696` |
| `unet_selected_spatial.py` | `model_unfolder/evidence/unet_runtime.py:86` |
| `unet_selected_stage_children.py` | `model_unfolder/evidence/unet_runtime.py:77` |
| `unet_stage_cells.py` | `model_unfolder/evidence/unet_runtime.py:68` |
| `unet_stage_construction.py` | `model_unfolder/evidence/unet_runtime.py:57` |
| `unet_stage_constructor_operands.py` | `model_unfolder/evidence/unet_runtime.py:75` |
| `unet_stage_execution.py` | `model_unfolder/evidence/unet_runtime.py:60` |
| `unet_stage_operands.py` | `model_unfolder/evidence/unet_runtime.py:73` |
| `unet_stage_selection.py` | `model_unfolder/evidence/unet_runtime.py:71` |

The orchestration run helper actually invokes its reader argument. Dependency failures can lawfully skip later calls. selected_constructor remains shared constructor/guard proof infrastructure rather than a separate ReaderResult entry. The production chain remains parser→unet_cutover→unet_runtime→qualified existing fact store→existing ModelIR projection.

| Added live module | Exact final caller references |
| --- | --- |
| `unet_runtime.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:42` |
| `unet_claims.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:62`; `model_unfolder/adapters/diffusor/unet_cutover.py:63` |
| `unet_cell_connections.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:58` |
| `unet_primary_ports.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:61` |
| `unet_call_binding.py` | `model_unfolder/evidence/runtime_inventory.py:48`; `model_unfolder/evidence/unet_runtime.py:99`; `model_unfolder/evidence/unet_primary_ports.py:153` |
| `unet_lookup_closure.py` | `model_unfolder/evidence/unet_call_binding.py:149`; `model_unfolder/evidence/unet_call_binding.py:160` |
| `unet_wrapper_binding.py` | `model_unfolder/evidence/unet_primary_ports.py:151` |
| `unet_iteration_binding.py` | `model_unfolder/evidence/unet_call_binding.py:229` |
| `local_port_routes.py` | `model_unfolder/evidence/unet_claims.py:261`; `model_unfolder/evidence/unet_claims.py:282`; `model_unfolder/evidence/unet_claims.py:456`; `model_unfolder/evidence/unet_claims.py:423`; `model_unfolder/evidence/unet_primary_ports.py:97` (remaining references in result.json) |
| `runtime_inventory.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:38` |
| `runtime_source.py` | `model_unfolder/evidence/unet_claims.py:28`; `model_unfolder/evidence/unet_claims.py:105`; `model_unfolder/evidence/unet_claims.py:203`; `model_unfolder/evidence/unet_claims.py:305`; `model_unfolder/evidence/unet_claims.py:379` (remaining references in result.json) |
| `instance_population_claim.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:50`; `model_unfolder/adapters/diffusor/unet_cutover.py:51` |
| `instance_shape_claim.py` | `model_unfolder/adapters/diffusor/unet_cutover.py:53` |
| `attribute_bindings.py` | `physics/instance_inventory.py:661`; `physics/instance_inventory.py:615`; `physics/instance_inventory.py:689` |
| `framework_primitives.py` | `physics/instance_inventory.py:630`; `physics/instance_inventory.py:688` |
| `source_override.py` | `model_unfolder/evidence/runtime_inventory.py:22`; `model_unfolder/evidence/runtime_inventory.py:40`; `model_unfolder/evidence/runtime_inventory.py:46`; `model_unfolder/evidence/runtime_inventory.py:56`; `model_unfolder/adapters/diffusor/unet_cutover.py:39` (remaining references in result.json) |
| `construction_summary.py` | `model_unfolder/adapters/diffusor/unet_projection.py:141`; `model_unfolder/evidence/reconciliation.py:956`; `model_unfolder/adapters/diffusor/unet_cutover.py:88` |

construction_summary.py is called by unet_projection.py:141; its reverse guard is called by unet_cutover.py:88 and reconciliation.py:956. Its optional source-free DTO and terminal consumers belong to the same migration unit. The added execution_recipe.py is counted separately as shared verification infrastructure, including moved prior generator code. Thus18 newly added evidence/physics files correspond to17 support modules plus this recipe extraction; these are different accounting denominators.

Legacy isolation is unchanged in meaning: parser.py:662–666 chooses the old path only under the false-by-default comparison ContextVar. The old helper rejects ordinary entry before importing old unet.py at parser.py:50–53. The flag resets in finally. The only non-test enabling call is demonstrate_s8_unet.py:319, under the explicit legacy condition. Inventory/source closure failure returns a typed limited projection; it has no old fallback. This refresh inspects source only; the prior independent controlled routing probes retain their ownc38 pin. The bridge remains1added/0retired, one old ordinary-author entry removed,0old authority files deleted. Deliberate internal Python imports remain possible.

Exact committed growth agrees with S8-final-growth-7168440/growth.json: production50files,+6827/−102; scripts5files,+1204/−622; tests/support26files,+3026/−78. These gross counts do not imply6827 lines of independent architectural authority, nor a102-line authority deletion.

The fixture extraction is separate: test_support/unet_stage_fixture.py adds77lines while test_unet_stage_execution.py is+1/−70. ROOT,FACTORY,_write,_bundle,_read are AST-identical to their original source nodes; this moved test fixture is not product debt elimination.

The entire structural_debt.py register is byte-identical to83140f1. The three restored consumer fingerprints remain6857ba8217a73268 (expanded sampling loop),da27bb4cf97df230 (diffusion stats),07d25d5749cfe49b (loop view). Their original raw obligations remain;0old debt rows are eliminated. The new typed migration removes the added terminal reads without repinning or waiving existing debt. Exact rows and source hashes are retained in result.json.
