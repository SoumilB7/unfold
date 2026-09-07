# Independent correction review — RETURN

Reviewer: `/root/independent_review`, independent of implementation and correction authors. Checkpoint: `f58170931e3cd05b1b699df5428920f7fc4c0147`, clean detached checkout `/private/tmp/unfold-s8-corrections-review`.

Scope: the initial R1–R8 authority corrections. This is not a full S8/C-8 verdict; demonstration, recipes, final connected pages, complete family differential and output-delta approval remain separately owned. No pytest lane, production/test edit, commit, blessing or model execution was performed by this reviewer.

## Results

| Item | Bounded verdict | Evidence |
|---|---|---|
| R1 parent escapes | RETURN, partly fixed | Direct `alias=self` now rejected. `holder=[self]; helper(holder)` still leaves member binding positive. `authority-results.json`. |
| R2 exact primitive identity | Corrected for reviewed controls | Spoofed serialized class names supply no mechanism; actual worker capture accepts canonical SiLU and rejects the custom same-name class. `authority-results.json`. |
| R3 wrong concat | Corrected for reviewed control | Replacing the connected concat with the unrelated same-callable concat now raises `ValueError: join result does not reach the cited child input`. `authority-results.json`. |
| R4 syntactic dependency | Corrected at reviewed projection boundary | The claim no longer presents `side_parameter` as the proven origin of a helper result. It supplies operand call ports and explicitly unresolved helper dependency. The ignored-input helper probe retains an opaque result boundary. Source inspection plus `authority-results.json`; not a full conditioned SDXL HTML verdict. |
| R5 FFN callable premises | RETURN, partly fixed | FFN and selected-transform owner overrides suppress the claim; input/output affine descendant overrides and coherent changed descendant types do not. Fresh SDXL source investigation plus DTO poisons in `ffn-results.json`. |
| R6 spatial occurrence join | Corrected for reviewed control | Same-class stride-2 and stride-1 siblings no longer share one reduction. Only `down.0.spatial.0` receives the positive fact. `spatial-results.json`. |
| R7 reaching definitions | RETURN, partly fixed | Initial carried-local and inside-loop/with controls are limited correctly. Loop/with target bindings are still lost after leaving the region. `authority-results.json`. |
| R8 starred unpack | Corrected for reviewed control | The last target after a starred target is unresolved, not fixed slot 2. `authority-results.json`. |

## Remaining earliest false producers

**R1:** `model_unfolder/evidence/unet_cell_connections.py:63–69` propagates only simple-name aliases. `holder=[self]` is not recorded as carrying the parent, so the helper can mutate the parent while the reader continues to connect the original constructed child. The full minimal source is in `authority-results.json`. Propagate the bounded container escape or leave the affected call binding unresolved; no model-specific exception or general alias engine is required.

**R5:** `model_unfolder/evidence/unet_claims.py:136–147` checks the FFN owner and selected input-transform owner, but its positive computation also depends on the actual input projection and output projection. A recorded `forward` replacement at `.ff.net.0.proj` or `.ff.net.2` leaves the FFN fact positive. Replacing either descendant's actual class with ReLU, with its matching canonical framework witness and regenerated reconciliation table, also leaves the affine FFN claim positive. Qualify every invoked computation premise that the FFN fact asserts; class addresses and absence of an owner override do not establish the descendant's affine behavior.

The ordinary persisted inventory predates exact framework-type capture: its framework primitive witness count is zero, explicitly recorded. The changed-type control attaches a real canonical ReLU witness obtained by `witness_framework_type(nn.ReLU(), capture_framework_types())`, updates class/module/MRO coherently, and regenerates reconciliation. It does not carry a contradictory Linear witness. Existing parameters can remain registered but unused on a ReLU module; this is a typed-boundary poison, not a claim that the actual checkpoint constructed that replacement. No model was constructed or executed. The result also records matching installed/static and historical runtime root source hashes.

**R7:** `model_unfolder/evidence/local_port_routes.py:67–84` checks unsupported regions and loop targets only while the queried call is inside the region. Python target bindings survive afterward. Both `with manager() as seed: pass; consume(seed)` and `for state in items: pass; consume(state)` incorrectly route the incoming formal. Account for possible prior rebinding with existing observations or preserve an unresolved result.

## Commands and verification

Run serially from the immutable checkout; the scripts live in the primary checkout's receipt directory so the reviewed tree is never changed:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/replay_authority.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/authority-results.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/replay_spatial.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/spatial-results.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/replay_ffn.py > /Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-authority-corrections/independent/ffn-results.json
```

Assertions in these review scripts distinguish repaired controls from remaining defects. A zero exit confirms those observations; it does not mean the implementation passes all eight items. `replay_authority.py` reuses initial review helpers as definitions and strips their old failing-condition assertions only to collect current values; its new explicit checks are visible in the persisted script. It does not import test modules or alter tests.

Tracked-file fingerprints for the authority replay are identical: `a01ddb8fa1e062ad069bcb0ce08d24ca7cfb1e28f900ee9907e3256d259dd8a6`. The spatial probe uses an explicit synthetic inventory DTO and a real typed source-reader chain; its limit remains recorded. The FFN probe reconstructs static evidence from the in-repo historical SDXL inventory, not a private cached pickle. These are focused correction checks, not the required broad gate or preservation verdict.

The opaque argument/result boundary is lawful, and the fixes reviewed here do not intentionally turn unknowns into absence. Requiring supported exact descendant and reaching-definition proofs is bounded by the claims the product already elects to make; it is not a requirement for exhaustive Python interpretation. Return R1/R5/R7 to their earliest producers, then re-review the corrected immutable checkpoint. No output-delta approval or re-bless is recorded.
