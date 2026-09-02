# Documentation verification ledger

Verified against the code and active working tree on 2026-07-16.

The ratings mean:

- **implemented** — present in the committed code and covered by tests;
- **under verification** — present in the active working tree but not yet
  accepted as current behavior;
- **partial** — a real substrate exists, but another authority path remains;
- **target** — approved direction, not a statement of current behavior;
- **incorrect** — a claim active documentation must not make.

| Documentation claim | Code/test evidence | Rating |
|---|---|---|
| `unfold` accepts config-like input or a model identifier and returns a `Diagram` or JSON | `model_unfolder/__init__.py::unfold`, `model_unfolder/parser.py::config_to_ir` | implemented |
| source inspection is static and does not instantiate the model | `model_unfolder/evidence/sources.py` and AST evidence readers | implemented |
| parse evidence and config access are call-local | `ParseContext`, `ConfigAccessLedger`, context-local render capture | implemented |
| a config event records occurrence, owner, exact path, alias, presence state, intent and mechanism | `model_unfolder/evidence/config_access.py` | implemented |
| binding a value is distinct from consuming it | `ConfigResolution.bind` and `.consume` plus config-access tests | implemented |
| config consumption points at an exact fact or geometry target | `ProjectionTarget`, `ProjectionObligation`, claim audit | implemented |
| facts preserve status, provenance, completeness, premises and typed failures | `model_unfolder/evidence/facts.py::EvidenceFact` | implemented |
| typed fact writes are constrained by the registry | `FactLedger.record_typed`, `model_unfolder/evidence/registry.py` | implemented |
| all structural claims are registry-native | structural-write census and temporary debt registers still coexist | partial |
| core unknown geometry remains unknown instead of becoming zero | optional IR geometry and transformer/diffusion regression tests | implemented on the named core paths; every new consumer must preserve it |
| nested modality ownership follows the full recursive component namespace | diffusion slot contexts, transformer parser namespace propagation, modality builder ownership | under verification |
| projector output width is source-authoritative and generic width fallback is rejected | projector evidence reader, exact owner/path binding and conflict tests | implemented |
| foreign `params.json` input is normalized without model-specific identity | `model_unfolder/input_formats.py`, scope-qualified alias entries and `tests/test_input_formats.py` | under verification; focused loader, conflict and identity-free parse tests pass |
| installed Transformers source directories are not selected by a project-owned model table | `model_unfolder/evidence/sources.py`, upstream `model_type_to_module_name`, and `tests/test_code_evidence.py::test_source_directory_uses_installed_registry_not_a_project_identity_table` | under verification; focused registry and suffix-refusal test passes |
| a rendered projector width emits a typed receipt from the surface that draws it | `evidence/receipts.py`, declared operation views and receipt tests | under verification; current pilot covers vision/video projector output width |
| consumed projector width joins exact occurrence to target to render receipt | receipt join in `sable.py` and `tests/test_receipts.py` | under verification |
| a receipt for an unregistered or unclaimed fact is rejected | reverse-fabrication receipt check | under verification |
| all renderer and parameter claims consume the semantic graph only | dependency checks exist, but transitional labels, extras and independent parameter conventions remain | partial |
| one owner-bound raw program index feeds all evidence readers | specialized and overlapping readers remain | target |
| every supported config mechanism has an exact consumer and projection receipt | the occurrence-exact census still contains unclassified or unreceipted reads | partial |
| preservation and anti-vacuity checks run on an unchanged tree | preservation helpers, tree-state helpers and preservation tests | implemented; required again before accepting the active pilot |
| all model/config identity is eradicated | address and display identity remain lawful; identity must not select architecture | incorrect |
| config is eradicated entirely | checkpoint values and explicit user declarations remain valid config inputs | incorrect |
| every Hugging Face repository is supported | support is bounded to declared mechanisms and verified upstream surfaces | incorrect |
| package version is internally consistent | `pyproject.toml` is `0.2.17`; `model_unfolder.__version__` is `0.2.15` | incorrect; separate packaging fix required |

## Current acceptance boundary

The committed foundation includes owner-qualified config events, exact claims,
source-authoritative projector width, preservation witnesses and mechanical
anti-vacuity controls.

The active working tree adds two coupled changes:

1. typed projection receipts for the projector-width pilot; and
2. recursive component namespaces that correct a producer-side ownership error
   exposed by that pilot.

It also replaces the model-named foreign-config normalizer with a structurally
detected, scope-qualified alias dialect that creates no model/class identity.

Those changes are directionally consistent with the architecture, but they are
not accepted current state until targeted tests, receipt poisons, the grouped
verification checks, the full suite, the preservation corpus and intentional
visual review pass on the same unchanged tree.

## Verification rule

Historical test counts and commit receipts do not prove the current tree. Every
completion claim must identify the exact command, collection count, result,
duration, and unchanged tree fingerprint. If the tree changes during a run, the
result is invalid and must be repeated.
