# Existing third connection demonstration — independent review

ACCEPT as a bounded call-port connection demonstration on the unchanged f664 ordinary artifacts. Replace the spatial selection with: **the result of `get_time_embed(sample=sample, timestep=timestep)` feeds argument 0 of the exact constructed `time_embedding` invocation; root formal `timestep_cond` feeds argument 1.** Its returned result is a separate opaque call boundary. This does not assert internal computation, argument-to-result semantic dependence, full conditioning lineage, or observed execution.

Archived UNet source SHA256 `052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26`, lines 1083–1084, directly establishes the local assignment/use. Exact instance target selection is retained under all seven recorded conditions (wrapper PEFT exclusions, custom-lookup earlier-return exclusions and optional parent-exposing helper/root branch exclusions); these are not deployment guard values. The source hash and both call references occur in the existing qualified proof.

- Fact: `root.denoiser.primary_state_ports`, kind `connection`, proof `guarded_primary_state_argument_result_ports`.
- Selected value: `.regions[3].route.iteration_result.when_true.arguments[1].route.when_false.when_false.when_false`.
- Actual denoiser overview contains `unet_primary_region_3` (repeat boundary/down blocks) and `instance_time_embedding`. Source-port conditions and optional overwrite alternatives remain in the enclosing drill hierarchy.
- Routed drill/card: `unet_primary_region_3__iteration__when_true__arg_1__when_false__when_false__when_false`. Its actual SVG contains the `__arg_0` call-result node, `__arg_1` timestep_cond input, and `__call` time_embedding target. Two incoming arrows terminate at that call boundary and one outgoing arrow denotes its result.
- The `__call` node targets canonical card `instance_time_embedding`. Both route and canonical card cite the qualified primary-state fact. The canonical card also cites the shape fact and displays **2,050,560 parameters**.
- Shape fact: `root.denoiser.constructed_parameter_shapes.value.by_module.time_embedding`. Saved construction has linear_1 weight `[1280,320]`, bias `[1280]`, and linear_2 weight `[1280,1280]`, bias `[1280]`. Independent sum: `1280*320+1280+1280*1280+1280 = 2,050,560`. Child cards contain the actual shapes.

The overview's time_embedding bookend and canonical child drill remain containment; they do not acquire flow merely from this review. The actual routed connection is in the cited source-port drill, with an explicit target link to the occurrence/quantity card. This differs materially from the rejected spatial selection, which had no routed drill for its selected claim.

Reproduce from unfold-pkg with `python3 verification/receipts/S8-third-connection-independent/replay.py`. `result.json` preserves exact source/value/card paths, all conditions, shapes and input hashes; `time-embedding-call.svg` is extracted from the existing actual page. Artifacts were unchanged. No production edits, model run, pytest, blessing or new source-grammar audit.
