# Config and code authority

## The division line

| Question | Authoritative evidence |
|---|---|
| What value did this checkpoint select? | exact checkpoint config occurrence |
| What component owns that value? | construction/source binding plus exact config scope |
| What mechanism does the field control? | modeling source or a typed declared contract |
| What is the structural result? | typed interpretation/derivation |
| How is it shown? | projection plan consuming the fact |

Config is therefore not being eliminated. Unsupported config authority is.

## Legitimate config uses

- numeric checkpoint geometry after owner binding;
- declared enum values after code or a typed contract proves their meaning;
- exact component slots and source addresses;
- syntax aliases that normalize equivalent spellings;
- non-architectural display metadata;
- consciously scoped ignores for fields proven not to affect the architecture.

## Illegitimate config uses

- presence of a field as proof of a mechanism;
- a family/model type selecting a structure template;
- a generic `hidden_size` standing in for a component-specific width;
- a default conventional attention, FFN, norm, position, or fusion scheme;
- a YAML mapping that converts identity directly into architecture;
- a field read added solely to clear an audit.

## Exact occurrences

The config ledger in `model_unfolder/evidence/config_access.py` represents:

- component path;
- dotted config path;
- actual spelling;
- canonical field;
- missing, explicit-null, or value state;
- intent such as inspected, bound, consumed, ambiguous, ignored, or
  absent-default;
- fact/geometry target and reason.

Equal aliases may be redundant evidence. Unequal aliases are ambiguity. Missing
and explicit null are not interchangeable.

## Current implementation status

Exact occurrence, owner, alias, value-state, intent, mechanism, and target
records are implemented. Recursive component namespaces are being tightened so
nested towers cannot author top-level facts. Some compatibility views remain,
and actual consumer-emitted receipts currently cover only a pilot mechanism.
These are temporary implementation boundaries, not the final authority model.

The final test for a config field is:

```text
Does exact owner-bound source prove what this occurrence controls?
```

If yes, config supplies the checkpoint's selected value. If no, the value may
remain a declaration, display item, non-architectural field, unresolved input,
or unsupported input; it may not become a structural mechanism.
