# Projection architecture

## One fact, multiple projections

An architectural fact may appear in:

- the dominant architecture view;
- an attention or FFN drill;
- a card or chip;
- expanded JSON;
- parameter annotations;
- a conformance or projection receipt.

These are consumers, not independent authors. A renderer may choose layout and
wording, but it may not re-derive mechanism semantics from raw config, source,
class identity, or a convenient default.

## Current structural authors

- `model_unfolder/ir.py` defines the central specs.
- `model_unfolder/opgraph.py` represents operation regions.
- transformer blocks live under
  `model_unfolder/adapters/transformer/blocks/`.
- diffusion structure is assembled under
  `model_unfolder/adapters/diffusor/`.
- expanded views live under `model_unfolder/expanded/`.
- HTML projections live under `model_unfolder/renderers/html/`.
- parameter projection lives in `model_unfolder/params.py`.

## Projection receipts

Committed render events and `renderers/html/fact_projection.py` provide a
key-level projection witness. The current working tree is replacing that coarse
channel with typed receipts emitted by the actual consumer. A receipt names the
fact owner, mechanism, surface, node path, projection kind, and value/status
hash. `evidence/config_access.py` supplies exact occurrence-to-target
obligations, and the mechanical audit joins:

```text
config occurrence -> fact target -> actual projection receipt
```

Receipt coverage is owner-and-mechanism scoped. It is not a global Boolean.
Only scopes whose real consumer emits receipts can be blocking while the
cutover is incremental.

Applicability is still decided upstream. Receipt enforcement must never be
disabled merely because a view omitted the component. A false consumption for
an inactive or incorrectly namespaced owner is repaired at the producer.

## Editorial filtering

Minor source operations may be omitted from a high-level diagram only at the
projection/editorial boundary. Raw observations and semantic facts must remain
available for deeper views and conformance. Filtering during acquisition makes
it impossible to distinguish intentionally omitted detail from evidence that
was never found.

## Renderer firewall

Renderers consume normalized structure. They must not import parser/config
authority or choose structure by class/model family. Static guards enforce part
of this dependency direction; every new projection must preserve it.

## Temporary projection overlap

The key-level `facts_projected` channel and hand-maintained drawn-leaf sets are
temporary while typed receipts reach parity. They must be removed after the
real consumers are receipted; keeping both permanently would create two sources
of projection truth.
