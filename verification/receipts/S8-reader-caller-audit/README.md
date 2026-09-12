# S8 reader callers and occurrence classification — frozen 4637ef4

Read-only audit. No production/test edits, pytest, new model execution or blessing for this audit. The separate family run supplies the exact persisted table and actual HTML.

The plan’s 14-reader denominator is the 14 `model_unfolder/evidence/unet_*.py` modules present at accepted S7 receipt commit `83140f1`. All 14 have production callers at this checkpoint; `unet_selected_constructor` is shared constructor/guard proof infrastructure rather than an independently invoked ReaderResult reader.

| Original module / entry | Production caller | Downstream fact or live dependency |
| --- | --- | --- |
| `model_unfolder/evidence/unet_attention_source.py:728` — read_unet_runtime_attention_sources | `model_unfolder/evidence/unet_runtime.py:91` | context_connections via unet_cutover.py:53 |
| `model_unfolder/evidence/unet_cell_mechanism.py:884` — read_unet_cell_mechanisms / read_unet_stage_join_connections | `model_unfolder/evidence/unet_runtime.py:79 / :70` | cell_connections, cell_arithmetic, stage_join_connections via unet_cutover.py:54–60 |
| `model_unfolder/evidence/unet_nested_mechanism.py:510` — read_unet_nested_mechanisms / read_unet_runtime_nested_ffns | `model_unfolder/evidence/unet_runtime.py:81 / :97` | ffn_mechanisms via unet_cutover.py:59 |
| `model_unfolder/evidence/unet_root_preprocess.py:475` — read_unet_root_preprocessing | `model_unfolder/evidence/unet_runtime.py:87` | attention source proof dependency at unet_runtime.py:91 |
| `model_unfolder/evidence/unet_selected_child_execution.py:486` — read_unet_selected_child_execution | `model_unfolder/evidence/unet_runtime.py:83` | spatial/context/cell proof dependency at unet_runtime.py:85–92 and unet_cutover.py:55–57 |
| `model_unfolder/evidence/unet_selected_constructor.py:692` — constructor_environments / selected_constructor_environment(s) | `model_unfolder/evidence/unet_root_preprocess.py:145; unet_selected_spatial.py:221; unet_attention_source.py:696` | shared selected constructor/guard proof dependency; helper module, not a ReaderResult entry |
| `model_unfolder/evidence/unet_selected_spatial.py:740` — read_unet_selected_spatial_operations | `model_unfolder/evidence/unet_runtime.py:86` | spatial_mechanisms via unet_cutover.py:50 |
| `model_unfolder/evidence/unet_selected_stage_children.py:522` — read_unet_selected_stage_children | `model_unfolder/evidence/unet_runtime.py:77` | selected child execution via unet_runtime.py:83 |
| `model_unfolder/evidence/unet_stage_cells.py:541` — read_unet_stage_cells | `model_unfolder/evidence/unet_runtime.py:68` | mechanism/selection dependencies; exact runtime stage filter at unet_stage_cells.py:550–555 |
| `model_unfolder/evidence/unet_stage_construction.py:660` — read_unet_stage_construction | `model_unfolder/evidence/unet_runtime.py:57` | stage execution and selected factory dependencies |
| `model_unfolder/evidence/unet_stage_constructor_operands.py:270` — read_unet_selected_stage_constructor_operands | `model_unfolder/evidence/unet_runtime.py:75` | selected children at unet_runtime.py:77 |
| `model_unfolder/evidence/unet_stage_execution.py:261` — read_unet_stage_execution | `model_unfolder/evidence/unet_runtime.py:60` | constructed_stage_relations via unet_cutover.py:58 and cell reader |
| `model_unfolder/evidence/unet_stage_operands.py:428` — read_unet_selected_stage_operands | `model_unfolder/evidence/unet_runtime.py:73` | constructor operands at unet_runtime.py:75 |
| `model_unfolder/evidence/unet_stage_selection.py:500` — read_unet_stage_selection | `model_unfolder/evidence/unet_runtime.py:71` | factory and root preprocessing at unet_runtime.py:73/:87 |

Ordinary production chain: `adapters/diffusor/parser.py:644–653` → `unet_cutover.py:31–44` → `evidence/unet_runtime.py:46–98` → qualified facts recorded in the existing store at `unet_cutover.py:64–67` → existing ModelIR projector at `unet_cutover.py:81–83`. The D/E/F cell candidate set is intersected with exact resolved runtime members at `unet_stage_cells.py:550–555`. Retained unsuccessful reader attempts are persisted in the context at `unet_cutover.py:62–63`; dependency failures can lawfully prevent later readers from running. Static caller presence alone is not full semantic coverage.

No original module deletion is owed merely to satisfy the caller-or-delete rule: all 14 are retained with live paths. Three additional modules (`unet_runtime`, `unet_claims`, `unet_cell_connections`) implement orchestration/qualified facts/local connection reading. This audit does not endorse their growth budget or establish that every private helper is necessary.

## Old adapter isolation

`adapters/diffusor/parser.py:645–647` enters the old path only when the ContextVar is enabled. `_parse_unet_model` repeats that guard at `parser.py:48–50` before importing old `unet.py` at line 51. `unet_differential.py:6` defaults false; lines 14–20 scope/reset the flag. The only non-test enabling caller found is `scripts/demonstrate_s8_unet.py:297`, explicitly condition legacy. Product config cannot enable this flag. Inventory/source closure failure at `parser.py:656–664` returns a typed limited result, without old fallback.

The old module and helpers remain directly importable for tests/comparison; this is ordinary product-routing isolation, not an access-control boundary against Python callers deliberately invoking internal functions. Old authority code remains the explicit temporary differential deletion unit. Comments at `parser.py:3–7`, `parser.py:640–643`, and `renderers/html/block_views/unet.py:9` still describe the old handoff and should be corrected as documentation debt.

## Classification finding

**239 are exact ModuleList containers, but 237 of them are visibly drawn.** Every container row is `eager_constructed`, exact `torch.nn.modules.container.ModuleList`, and carries reason `container`; all 239 execution rows remain class-1 unresolved. Actual HTML contains SVG nodes and cards for 237. Only top-level `down_blocks` and `up_blocks` have neither their own node nor card. `reconciliation.py:1115–1120` unconditionally overwrites previously built rendered/grouped claims with `non_architectural`. This is a container-policy override, not an unstamped-fact demotion. Nevertheless it makes the table under-report actual placement and can discard a container’s fact-level accounting. Owner decision/fix required before describing this table as a literal rendering inventory.

**All 1,621 projection-unresolved rows are guarded absent children.** Each has `construction=not_constructed`, a nonempty guard/source site, `execution=proven_inactive`, no runtime class, and a unique path absent from the 1,930 constructed occurrences. They are created from inventory `guarded_none_children` at `reconciliation.py:1337–1376`. Their projection is still `structure_unaccounted`; they are not 1,621 missing drawings of real constructed modules. This validates row shape and the existing producer route, not an independent rerun of every constructor guard.

All 1,691 rendered rows remain rendered, and the persisted family table has zero fact findings. Unstamped proof handling builds rendered/grouped rows with separate fact findings at `reconciliation.py:1054–1111`; no claim-proof-driven occurrence demotion appears in this checkpoint. The unconditional container override above is a separate issue.

Exact per-container HTML matches, guard counts, source hashes and assertions are in `classification-audit.json`; full 14-module ledger and source hashes are in `caller-ledger.json`.
