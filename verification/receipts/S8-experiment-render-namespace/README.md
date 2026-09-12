# S8 experiment render namespace — review required

The experiment used a random Diagram mount while requiring unchanged-source HTML to be byte-identical. The generator now sets `uf-s8-demonstration` before the real Diagram render. Normal production Diagram behavior remains unchanged.

Production is frozen c38b008. Existing raw ordinary/sparse/misleading/rewrite artifacts remain in `/private/tmp/unfold-s8-final-c38b008`; separate actual render-phase outputs are in `/private/tmp/unfold-s8-render-phase-c38-v2`. No page bytes are normalized or replaced in the raw campaign. Every new output comes from `Diagram.to_html()`.

The old public `ir.json` drops EvidenceWarning metadata. Replay uses the existing `apply_ship_findings` presentation producer on exact persisted `extras.ship_findings` records, maps its exact string-compatible results into the original positions, and requires complete `ModelIR.to_dict()` equality. Parameter totals must equal the original parse receipt. Actual rerendered HTML must equal the original page apart from the exact original mount token; this replacement is a diagnostic comparator only. Each phase case archives the original output trio and hashes every original artifact. New cases capture the exact Diagram render input, parameter record and typed warning metadata directly.

The two independent Diagram namespace control passed; normal Diagram instances retained distinct mounts. Production code is unchanged. Phase pins retain generator/reporter/replay/entry/runner hashes and exact source/input/IR/fact/proof links. All outputs remain proposals; no blessings.

All four actual saved-case rerenders passed exact page equivalence and before/after full production-source checks. An independent supplemental audit compares separately serialized actual render-input against fresh original IR JSON; all four equal. This closes the historical aliased comparison, and the current replay source now directly rereads the original JSON. Historical executed replay script bytes remain archived; do not overwrite them with the later one-line fix. Ordinary and equivalent rewrite actual fixed-mount HTML are raw-byte identical.
