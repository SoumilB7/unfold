# Product contract

## Input

The public entry point is `model_unfolder.unfold`. It accepts a configuration
dictionary, a Hugging Face configuration object, or a model/repository ID.
The loader also contains diffusion-pipeline and alternate-config-format rungs.
Model source is inspected statically; model code is not executed.

Verified in:

- `unfold-pkg/model_unfolder/__init__.py::unfold`
- `unfold-pkg/model_unfolder/parser.py::config_to_ir`
- `unfold-pkg/model_unfolder/parser.py::_coerce`
- `unfold-pkg/model_unfolder/evidence/sources.py::resolve_source_files`

## Output

`unfold` returns a `Diagram` unless `return_json=True`. The diagram supports:

- interactive HTML and notebook rendering;
- the parser IR as a dictionary;
- expanded machine-readable JSON;
- parameter estimates with incomplete-state support;
- exhaustive PNG view generation;
- warnings and wiring diagnostics;
- typed render events from the latest full render.

Verified in `unfold-pkg/model_unfolder/diagram.py`.

## Trust contract

Every visible structural claim should be traceable to an exact fact or geometry
target. When required evidence is unavailable, the output must remain unknown or
produce an actionable finding. A successful render does not imply that source
conformance was available; reports must disclose a missing code oracle.

## Output consistency

HTML, expanded JSON, cards, block drills, parameter estimates, and machine
audits are different projections of the same architecture. A fix in only one
projection is incomplete.
