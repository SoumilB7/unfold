# Independent card / FFN projection-receipt review

RETURN on one concrete output/event mismatch in the new FFN receipt. Four source files and the focused test file are pinned/snapshotted. This is separate from the accepted typed15 contract. No model or pytest lane, production edits or blessing; one standalone toy renderer probe was run and persisted.

Reproduce from unfold-pkg:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 verification/receipts/S8-card-ffn-receipt-independent/reproduce.py

## Earliest false producer

`model_unfolder/renderers/html/block_views/unet.py::build_runtime_ffn_view`, new receipt block around lines240–249, accepts any returned string containing SVG opening/closing tags plus a nonempty ffn event emitted since entering the builder. These are separate observations. The returned SVG is never checked for the event's nodes.

The exact counterexample calls the real build_ffn_view (so its seven-node event is real), discards that SVG, and returns `<svg></svg>`. The runtime wrapper emits `runtime_ffn_fact` for root.denoiser.ffn_mechanisms anyway. The actual returned product contains only an empty SVG and empty constructed-children containment. result.json preserves both actual bytes and the unjustified receipt. This is the requested omitted-graph negative; it does not change any mechanism source or introduce a new grammar case.

The existing empty-graph tests replace the builder with a stub that emits no graph event. They pass while this concrete case remains false-positive. Require the claimed graph nodes to belong to the actual returned FFN SVG and correct block before stamping. An event produced while building subsequently discarded output cannot discharge a visible projection obligation. The same bound must refuse a partial graph whose claimed nodes are absent; do not settle for nonempty generic SVG tags.

## Supported conclusions from source inspection

The activation-label change is the existing label formatter applied only to an already-proven activation operation's display label/title. Fact values and op fn remain unchanged.

The producer authors exact display lines for spatial and omitted-default facts from the same values used for visible chip text. Spatial unresolved direction remains explicitly unresolved. Defaults remain labelled declarations rather than deployment facts.

The card helper requires the exact card ID, an explicitly cited key, a nonempty list/tuple of nonblank strings, and every exact escaped uf-fact span in the actual facts_html output. It emits only those keys, not the block's other source_fact_keys. Missing/changed/empty/description-only lines or an omitted chip renderer do not satisfy this predicate. Passing canonical block metadata through the existing card builders does not itself change card HTML. The mechanism receipt likewise selects only the FFN key, but currently overclaims that key in the discarded-output case above.

Final actual model pages, event comparisons and gate execution remain executor/owner tasks. This receipt grants no output approval.
