# Seventeen non-UNet witnesses: restored drawing edges

**ACCEPT the bounded restored-edge deltas in all 22 changed views across 17 models.**
This is not a whole-view geometry clearance or a baseline blessing. One existing
Qwen gate obstruction is explicitly retained below; the owner accepts it as
unchanged by this patch and outside UNet S8.

Independent checks on the actual helper artifacts:

- Each witness's old-helper render recovers the complete immutable expected
  ordered distinct view list **and** exact expected HTML-metadata hash.
- Captured current Graph/Region values match the same-IR old-helper Graph/Region.
  Every restored empty-lane dependency exists in that canonical Region; no
  operation, node ID or label changed.
- All **43 listed dependencies** have the expected actual incoming SVG arrows,
  including MusicGen's two same-source score inputs. Their multiplicity is
  retained. Nineteen attention views restore cache→scores and cache→apply-values;
  two expert views restore split-value→multiply; MusicGen restores three routes
  from the explicitly unresolved projection boundary.
- Every restored source tap has exactly one connecting source-top stem.
  The new vertical rails intersect no intervening actual node rectangle,
  including wider score/sink/softcap boxes and off-spine activation boxes.
- All 22 current native PNGs were visually inspected. The restored rails remain
  within their fitted canvases, reach the intended visible attachment and clear
  the relevant boxes. All old/current SVG and native PNG pairs are retained.

`geometry-checks.json` contains each exact model/view, source-target dependency,
rail coordinate, output hashes and individual verdict. `input-ledger.json`
retains the helper's full actual Graph/Region observations and original paths.
`review.py` reproduces the bounded hash/canonical-edge/rectangle/stem checks and
native conversion without a model parse. Later multiplicity checks and visual
verdicts are recorded separately in the final JSON. Source parser proofs are not
re-authored by this review; the claim is preservation of the existing canonical
dependencies through the changed layout.

## Existing defect: Qwen3.5 text attention output gate

Owner-readable location: `12-qwen3-5-27b-text/current.png`, the **Sigmoid gate →
output multiply** vertical rail. It passes behind the right side of the scaled
scores box. The restored cache→apply-values rail is farther right and clear.

Exact XML evidence in `12-qwen3-5-27b-text/preexisting-gate-route.json`:

```
Old and current: M 118 760 L 118 152 Q 118 142 108 142 L 20 142
Scores rectangle: x=[-140,140], y=[408,490]
```

Both the obstruction path and the rectangle are byte-identical in the recovered
old locked view and current view. Its producer is the pre-existing nonempty-lane
route in `graph_engine._draw_parallel`, not the new empty-lane clearance path.
The old and current PNG pair makes the unchanged defect visible. No new
production work was started for it. The sheet/decision page must preserve this
named limitation instead of claiming every part of every view is geometry-clean.

No model/pytest lane, baseline update or production edit occurred in this
review. Native images were generated only from the retained actual SVGs, using
the existing `preview.svg_to_png` converter at scale 1. Root separately reviews
ledger and public example HTML deltas; user baseline approval remains pending.
