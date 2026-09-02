# Remaining config and YAML authority

## Permanent config responsibility

These values remain checkpoint/config evidence:

- dimensions and counts selected by the checkpoint;
- selected values for branches whose meaning is proven by code;
- declared component slots and source addresses;
- exact public semantic enums where a typed contract or code binding establishes
  their meaning;
- format/syntax aliases and non-architectural metadata.

The goal is not zero configuration files. The goal is zero unsupported semantic
authority.

## Resource classification in the current tree

| Resource | Current role | Target fate |
|---|---|---|
| `transformer/aliases.yaml`, `diffusor/aliases.yaml` | global and scope-qualified syntax normalization, including structurally detected foreign `params.json` dialects | keep; scoped aliases cannot leak globally or prove mechanisms |
| `transformer/ignored_fields.yaml` | non-architectural ownership classification | keep only exact scoped ignores with reasons; no structural silence blanket |
| transformer/diffusion `typing.yaml` stage/ID taxonomies | schema and allowed presentation vocabulary mixed with markers | split lawful schema/display vocabulary from mechanism detection; code-bound evidence owns structure |
| conformance YAML | canonical operation/role vocabulary for verification | keep narrowly fingerprinted vocabulary; it cannot become parser authority |
| `transformer/layer_topology.yaml` | empty historical topology fallback; loader has no production consumer | delete loader/resource after a guard confirms no lawful consumer |
| `transformer/layer_types.yaml` | config token to mask category | retain only declared syntax normalization; internal attention mechanism requires source binding |
| `transformer/layer_schedules.yaml` | schedule field forms and token-to-mixer mapping | split syntax forms from semantic mixer interpretation; source/construction must prove the mechanism |
| `transformer/composite_slots.yaml` | component-slot/address vocabulary plus undrawn labels | keep address vocabulary; move role/mechanism meaning to owner binding |
| `transformer/decoderness.yaml` | declared architecture suffix to decoder role | quarantine as declared-role/address evidence; replace mask/causality semantics with source/typed role proof where possible |
| `diffusor/config_facts.yaml` | reads config and directly emits chips | dismantle into registered typed facts or scoped non-architectural ignores |
| `diffusor/conditioning.yaml` | enum-to-modality/projector story | bind enum to the component source and emit typed conditioning facts; YAML may retain display wording only |
| `diffusor/text_encoders.yaml` | class-to-friendly-label display map | keep only as fingerprinted display vocabulary; never structural |
| diffusion `dit_class_markers`, scheduler markers, norm mappings, temporal fields, stack lanes, companion fields | address, detection, display, and structure mixed in one file | split by authority; source-bound mechanisms replace structural marker inference |

## Hardcoded config authority outside YAML

The same audit must cover Python:

- default values in specs and parser calls;
- `or 0`, `or False`, and fallback strings at structural sinks;
- class-name substring detection;
- generic width fallback across components;
- renderer and parameter assumptions;
- broad exception-to-`None` behavior;
- compatibility normalizers that strengthen unknown evidence.

Moving such logic into YAML does not solve it. Authority is determined by how a
value reaches a structural sink, not by whether it is written in Python or data.

## End-state config residue

After the architecture conversion, config/data remains for:

1. checkpoint-selected numeric values;
2. values for code-proven gates;
3. syntax aliases and input-format translation;
4. exact component/source address vocabulary;
5. schema taxonomies and display-only wording;
6. scoped non-architectural ignore classifications;
7. verification vocabulary that cannot author parser output.

Input dialects are selected by file layout and required structural keys. They
may normalize field spellings and preserve repository provenance, but may not
manufacture a model type, class, architecture, or mechanism. A dialect-local
spelling such as `dim` remains scope-qualified when it is not universally
equivalent to the canonical field.

Everything else either becomes code-derived typed evidence or remains unknown.

## Classification is allowed to end work

The current occurrence census is not a demand for one reader per field. A field
can leave the worklist by becoming an exact:

- mechanism-bound checkpoint value;
- address/declaration;
- display-only value;
- non-architectural ignore with owner and reason;
- unresolved mechanism input;
- unsupported/out-of-scope input.

Only mechanism-bearing fields require new source interpretation. This boundary
prevents config cleanup from becoming endless checkpoint-by-checkpoint work.
