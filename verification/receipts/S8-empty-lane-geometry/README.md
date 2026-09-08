# Empty direct-lane geometry correction

Owner delegated this bounded renderer correction to the independent reviewer;
the root owner separately reviewed the production patch. This receipt reports
implementation and verification, not self-approval of output baselines.

The saved actual Bloom canonical Graph and Region already contain both
`kv_cache → scaled_scores` and `kv_cache → attn_apply_v`. The initial renderer
draws both edges but places the value rail at x=78 through the scores and Softmax
boxes. XML also confirms zero source-top-to-tap stems: both empty lanes name the
same shared source, while the former shared-stem check only recognized implicit
sources.

The generic layout correction moves a colliding empty rail beyond the envelope
of actual boxes intersecting its vertical span. A lower destination remains an
obstacle when the same lane also reaches a higher destination. Boxes outside
the span do not determine clearance. Explicit references to the shared source
now receive the existing common stem once. Fit regions include both rail ends
and the destination attachment. No source fact, graph edge, Region, node or model
condition changed.

`reproduce.py` renders the retained canonical inputs with the archived preceding
engine and corrected engine. `replay.json` records their hashes and XML checks:

- Bloom value rail x=78 → x=176; no intermediate rectangle intersects it.
- Cache source stem 0 → 1. Arrow-path count and canvas bounds remain identical.
- Bloom FFN and fused-gated FFN complete SVG bytes remain identical.
- `canonical-graphs.json` is the exact original saved capture, including Regions.

Native before/after PNGs were generated from these SVGs using the existing
`preview.svg_to_png` renderer at scale 1 and visually inspected. The after image
has a connected cache tap and an outward value rail clear of the score and
Softmax boxes. Intersecting independent wires are not claimed to be junctions.
These are deterministic saved-graph renders, not new model executions or a full
document refresh.

Five new geometry controls plus the existing actual fused-bypass SVG control
passed: **6 passed in 0.03s** (`focused.log`). They cover both sides, the widest
intermediate box, irrelevant outside boxes, source connection and edge retention,
multi-target bypasses, unchanged adjacent lanes, and an unresolved nonempty lane
which must not become an invented direct edge. Pyflakes passed all three changed
Python files (engine, controls, reproduction script).

No model was run, no baseline was written, and no commit was made by this agent.
The two intended visible deltas are the outward rail and restored source stem.
Final document capture and broader gate belong to the executor's integration.
