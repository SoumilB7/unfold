# S8 source-operator port wiring

Bounded reuse of the existing independent incoming-lane renderer for `source_operation` and `inplace_operation`. The projector retains each operand as a clickable child with its exact position, adds an unknown boundary labelled `Source +` / `Source +=` (or the supplied operator spelling), and routes both operands into that boundary then to `Operation result`. Source spelling is not promoted to established operand dispatch, internal dependence or mutation behavior; those remain explicit investigation findings on the card.

Only the operator branch of `_port_route_block` and `build_runtime_port_route` changed. No new graph primitive, canonical fact/IR, interpreter or model-name hook. Two synthetic native images were individually inspected. Both standalone probes verify three clickable cards, two separate operand rails, one result arrow, no synthetic common source/split and no edge/card crossings. Existing compact seven-call-port assertion and AST syntax passed. No pytest or model run, no output acceptance.

Exact source hashes, probe code/results, projected blocks and SVG/PNG artifacts are retained here. A fresh immutable ordinary page must confirm actual conditioning integration.
