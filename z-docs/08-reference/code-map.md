# Code map

Paths are relative to `unfold-pkg/`.

## Public surface

| Responsibility | Location |
|---|---|
| package exports and `unfold` | `model_unfolder/__init__.py` |
| diagram API and output formats | `model_unfolder/diagram.py` |
| public errors | `model_unfolder/errors.py` |
| root loading and adapter dispatch | `model_unfolder/parser.py` |
| structurally detected foreign input formats | `model_unfolder/input_formats.py`, scope-qualified entries in `everchanging/transformer/aliases.yaml` |

## Core model and projection

| Responsibility | Location |
|---|---|
| IR specs | `model_unfolder/ir.py` |
| operation regions | `model_unfolder/opgraph.py` |
| expanded schema | `model_unfolder/expanded/` |
| parameter estimates | `model_unfolder/params.py` |
| HTML rendering | `model_unfolder/renderers/html/` |
| HTML compatibility entry | `model_unfolder/html_renderer.py` |
| image/view generation | `model_unfolder/preview.py` |

## Evidence

| Responsibility | Location |
|---|---|
| call-local parse state and fact ledger | `model_unfolder/evidence/context.py` |
| exact config events and resolution | `model_unfolder/evidence/config_access.py` |
| typed observations/facts | `model_unfolder/evidence/facts.py` |
| typed projection receipts and exact receipt joins | `model_unfolder/evidence/receipts.py` |
| closed fact registry/debt | `model_unfolder/evidence/registry.py` |
| source acquisition | `model_unfolder/evidence/sources.py` |
| source evidence data models | `model_unfolder/evidence/models.py` |
| raw forward operation scan | `model_unfolder/evidence/forward_ops.py` |
| transitive callable scan | `model_unfolder/evidence/transitive.py` |
| construction/dataflow readers | `model_unfolder/evidence/patterns.py` |
| FFN/position/tower/projector/fusion readers | `model_unfolder/evidence/{ffn,position,vision,audio,projector,fusion}.py` |
| conformance | `model_unfolder/evidence/conformance.py` |
| identity/taint guards | `model_unfolder/evidence/identity_guard.py` |
| typed identity roles | `model_unfolder/evidence/identity_roles.py` |
| structural-author census | `model_unfolder/evidence/structural_writes.py` |

## Adapters

| Responsibility | Location |
|---|---|
| transformer parsing | `model_unfolder/adapters/transformer/parser.py` |
| transformer assembly/blocks | `model_unfolder/adapters/transformer/assembly.py`, `blocks/` |
| transformer modalities | `model_unfolder/adapters/transformer/special_parts/modalities/` |
| diffusion loading/parsing | `model_unfolder/adapters/diffusor/{loader,parser}.py` |
| diffusion blocks/UNet/compound | `model_unfolder/adapters/diffusor/{blocks,unet,compound}.py` |

## Config/data resources

| Responsibility | Location |
|---|---|
| YAML loader | `model_unfolder/everchanging/__init__.py` |
| transformer resources | `model_unfolder/everchanging/transformer/` |
| diffusion resources | `model_unfolder/everchanging/diffusor/` |
| conformance vocabulary | `model_unfolder/everchanging/conformance/` |

## Verification

| Responsibility | Location |
|---|---|
| Sable and blessing/regression API | `model_unfolder/sable.py` |
| preservation helpers | `test_support/preservation.py`, `test_support/tree_state.py` |
| shared metamorphic harness | `test_support/metamorphic.py` |
| identity/static/structural tests | `tests/test_identity_guard.py`, `test_h4_taint.py`, `test_static_guards.py`, `test_structural_writes.py` |
| config/fact/projection tests | `tests/test_config_access.py`, `test_config_intents.py`, `test_fact_registry.py`, `test_projection_audit.py`, `test_projection_obligations.py` |
| preservation/isolation tests | `tests/test_preservation.py`, `test_isolation.py` |
| domain evidence tests | `tests/test_{ffn,vision,audio,projector,fusion}_evidence.py` |
