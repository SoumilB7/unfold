"""First-class embedded sub-models — ONE recursive projection for every
supporting tower.

Any transformer stack that is not the root model (a pipeline text encoder, a
wrapper's delegated text stack, tomorrow's draft head or reward tower — and a
sub-tower inside any of those) is represented as a **facts-only, recursive,
JSON-safe spec** derived from its full :class:`~.ir.ModelIR` by exactly one
function (:func:`submodel_spec`), and rendered by exactly one projector
(:func:`submodel_cell_blocks` / :func:`submodel_block`).  There is no
per-context scalar relay: a fact added to ``AttentionSpec``/``FFNSpec`` and its
canonical serializer is available to every embedded context with zero plumbing,
at any nesting depth.

The three projection parameters (and the ONLY three):

* **namespace** — the id prefix; composed per group (``_g<k>``) and per nested
  sub-model (``_s<j>``), so arbitrary depth can never collide or mask a click.
* **component** — the qualified ownership path (``text_encoder``,
  ``text_encoder.text_config``); stamped on every emitted block so render
  events, evidence and conformance bind to the right oracle by construction.
* **altitude** — an editorial policy value (``tower`` today, ``hero`` reserved),
  applied as DATA (:data:`ALTITUDE_TRANSFORMS`), never as a separate code path.

The enforcement is the sub-model **parity net**
(``tests/test_submodel_parity.py``): projecting an embedded config must be
structurally identical to parsing the same config standalone, modulo namespace
and the declared altitude transforms.  Any divergence means a relay lost a
fact — the class of bug this module exists to make impossible.
"""
from __future__ import annotations

from typing import Any

from .block_schema import Block
from .ir import ModelIR, detect_layer_period, distinct_layer_groups

#: Altitude policy — declared fact transforms, applied at spec time.  ``tower``
#: is the supporting-tower policy: a prompt/feature encoder runs once, never
#: autoregressively, so its attention drills draw no KV-cache ports.  ``hero``
#: (the root model's own altitude) applies no transform and is reserved for the
#: future fold-in of the main tower onto this projector.
ALTITUDE_TRANSFORMS: dict[str, dict[str, dict[str, Any]]] = {
    "tower": {"attention": {"cached": False}},
    "hero": {},
}


# ---------------------------------------------------------------------------
# Spec derivation — ModelIR -> facts-only recursive spec (ONE function)
# ---------------------------------------------------------------------------

def submodel_spec(
    ir: ModelIR,
    *,
    component: str = "",
    altitude: str = "tower",
    scores_scaled: bool | None = None,
    norm_label: str | None = None,
    activation: str | None = None,
    gated: bool | None = None,
    structure_status: str | None = None,
    projection_mode: str | None = None,
    position_evidence: dict | None = None,
    ffn_evidence: dict | None = None,
    sub_models: list[dict] | None = None,
) -> dict:
    """Derive the embedded-model spec from a full sub-parse ``ModelIR``.

    Facts only — drill children, cards and regions are derived at PROJECTION
    time from these facts through the same canonical builders the root model
    uses, so the spec can never hold a stale copy of a drawable.

    The honesty overrides (``activation``/``gated``/``norm_label``/…) exist
    because the universal parser fills modern-LM defaults when a config is
    silent — right for a standalone decoder, invented facts for an encoder.
    The caller passes the evidence-resolved values; ``None`` keeps the
    tri-state unknown.  ``sub_models`` nests further embedded specs (a tower
    inside a tower) — the projector recurses through them with composed
    namespaces and dotted component paths.
    """
    from .adapters.transformer.blocks.attention import attention_detail
    from .adapters.transformer.blocks.feed_forward import ffn_detail

    groups = distinct_layer_groups(ir.layers)
    sigs = [layer.signature() for layer in ir.layers]
    attn_overrides = dict(ALTITUDE_TRANSFORMS.get(altitude, {}).get("attention", {}))

    def _attention_fact(group_layer) -> dict:
        fact = attention_detail(group_layer.attention)
        fact.update(attn_overrides)
        fact["hidden"] = ir.hidden_size
        if scores_scaled is not None:
            fact["scores_scaled"] = scores_scaled
        return fact

    def _ffn_fact(group_layer) -> dict:
        ffn = group_layer.ffn
        if ffn.kind == "moe":
            # The full canonical MoE serialization — routing, experts, shared,
            # clip — exactly what a decoder MoE block carries.
            fact = ffn_detail(ffn)
            fact["hidden"] = ir.hidden_size
            return fact
        fact = {
            "kind": "dense",
            "hidden": ir.hidden_size,
            "intermediate_size": ffn.intermediate_size,
            "activation": activation,
            "gated": gated,
            "structure_status": structure_status or "oracle_missing",
        }
        if structure_status == "proven" and projection_mode:
            fact["projection_mode"] = projection_mode
        return fact

    def _norm_for(group_layer) -> str | None:
        if not norm_label:
            return None
        return {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
            group_layer.norm_kind) or norm_label

    tags = submodel_group_tags(groups) if len(groups) > 1 else [""] * len(groups)
    spec_groups = [
        {
            "count": len(group["indices"]),
            "layers": list(group["indices"]),
            "runs": [list(run) for run in group["runs"]],
            "tag": tags[k],
            "attention": _attention_fact(group["layer"]),
            "ffn": _ffn_fact(group["layer"]),
            "norm": _norm_for(group["layer"]),
        }
        for k, group in enumerate(groups)
    ]

    sig_to_group = {group["sig"]: k for k, group in enumerate(groups)}
    runs: list[list[int]] = []
    for sig in sigs:
        k = sig_to_group[sig]
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])

    return {
        "component": component,
        "altitude": altitude,
        "layers": len(ir.layers),
        "hidden": ir.hidden_size,
        "vocab": ir.vocab_size,
        "max_pos": ir.max_position_embeddings,
        "norm": norm_label,
        "groups": spec_groups,
        "schedule": {
            "period": detect_layer_period(sigs),
            "runs": [tuple(run) for run in runs],
            "total": len(sigs),
        },
        # Envelope dicts are COPIED: the caller may also carry them as flat
        # fields, and ownership qualification mutates in place — sharing one
        # object would double-prefix the component path.
        "evidence": {
            "position": dict(position_evidence) if isinstance(position_evidence, dict) else None,
            "ffn": dict(ffn_evidence) if isinstance(ffn_evidence, dict) else None,
        },
        "sub_models": list(sub_models or []),
    }


def submodel_group_tags(groups: list) -> list[str]:
    """One short human tag per layer-type group, built ONLY from the facts that
    actually differ across the groups (mask flavour, mixer kind, FFN form,
    norm) — the tag names the distinction, never restates the whole spec."""
    from .adapters.transformer.blocks.attention import attention_detail
    from .labels import kind_short

    def _facts(group) -> dict:
        a, f = group["layer"].attention, group["layer"].ffn
        return {
            "kind": a.kind,
            "mask": ("sliding window" if a.mask == "sliding" else "global"),
            "ffn": ("MoE" if f.kind == "moe" else "dense FFN"),
            "norm": group["layer"].norm_kind,
        }

    table = [_facts(group) for group in groups]
    varying = [axis for axis in ("kind", "mask", "ffn", "norm")
               if len({row[axis] for row in table}) > 1]
    tags = []
    for group, row in zip(groups, table):
        parts = []
        for axis in varying:
            if axis == "kind":
                parts.append(kind_short(attention_detail(group["layer"].attention)))
            elif axis == "norm":
                parts.append({"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
                    row[axis], str(row[axis])))
            else:
                parts.append(row[axis])
        tags.append(" · ".join(dict.fromkeys(parts)) or f"type {chr(65 + len(tags))}")
    return tags


def qualify_component(spec: dict, component: str) -> None:
    """Stamp/qualify the slot path onto a spec IN PLACE, recursively.

    Inner components QUALIFY, never overwrite: a composite encoder (a VL
    wrapper) resolved its facts one component deeper (``text_config``); the
    exact owner is the nested dotted path (``text_encoder.text_config``),
    which the source bundle qualifies for conformance.
    """
    inner = str(spec.get("component") or "")
    spec["component"] = component if not inner else f"{component}.{inner}"
    for envelope in (spec.get("evidence") or {}).values():
        if isinstance(envelope, dict):
            env_component = str(envelope.get("component") or "root")
            envelope["component"] = (component if env_component == "root"
                                     else f"{component}.{env_component}")
    for nested in spec.get("sub_models") or []:
        if isinstance(nested, dict):
            qualify_component(nested, component)


# ---------------------------------------------------------------------------
# Projection — spec -> blocks/cards (ONE recursive projector)
# ---------------------------------------------------------------------------

def _owning_component(spec: dict, evidence: dict | None) -> str | None:
    """The MOST QUALIFIED ownership path wins.

    A composite tower (a VL wrapper as prompt encoder) resolves its facts one
    component deeper than its pipeline slot — the evidence envelope carries the
    dotted path (``text_encoder.text_config``), which is the exact oracle
    conformance must bind to; the slot path is only the fallback.
    """
    if isinstance(evidence, dict) and evidence.get("component"):
        return str(evidence["component"])
    return spec.get("component") or None


def submodel_attention_block(spec: dict, group: dict, prefix: str, *,
                             title: str, description: str,
                             facts: list[str]) -> Block:
    """One embedded attention block: the canonical region, namespaced.

    The drill SVG and its leaf cards derive from ONE region (ids can never
    drift apart), namespaced so two towers/groups at the same card depth
    cannot satisfy each other's clicks.
    """
    from .labels import cards_from_region
    from .opgraph import attention_region, prefix_region

    fact = group["attention"]
    namespace = f"{prefix}_attn_"
    region = attention_region(fact, fact.get("hidden"))
    namespaced = prefix_region(region, namespace)
    evidence = (spec.get("evidence") or {}).get("position")
    return {
        "id": f"{prefix}_op_selfattn",
        "role": "attention",
        "kind": "attention",
        "title": title,
        "description": description,
        "facts": facts,
        "view": "attention",
        "source_component": _owning_component(spec, evidence),
        "detail": {
            "attention": {**fact, "node_prefix": namespace},
            "evidence": evidence if isinstance(evidence, dict) else {},
        },
        "children": cards_from_region(namespaced),
    }


def submodel_ffn_block(spec: dict, group: dict, prefix: str) -> Block:
    """One embedded FFN block — dense / gated / honest-unknown / full MoE.

    An MoE group opens the SAME canonical router/top-k/expert drill a decoder
    MoE opens: the child subtree is derived here from the serialized facts
    through the one decoder builder — nothing re-authored, nothing cached.
    """
    from .labels import cards_from_region, ffn_summary
    from .opgraph import ffn_region, rename_ops

    fact = dict(group["ffn"])
    evidence = (spec.get("evidence") or {}).get("ffn")
    evidence = evidence if isinstance(evidence, dict) else {}
    component = _owning_component(spec, evidence)

    if fact.get("kind") == "moe":
        desc, facts = ffn_summary(fact)
        return {
            "id": f"{prefix}_op_ffn",
            "role": "ffn",
            "kind": "moe",
            "title": "Mixture of Experts",
            "description": desc,
            "facts": facts,
            "view": "ffn",              # dispatcher routes kind=moe -> moe view
            "source_component": component,
            "detail": {"ffn": fact, "evidence": evidence},
            "children": _moe_children_from_fact(fact, component),
        }

    region = ffn_region(fact, fact.get("hidden"))
    namespace = f"{prefix}_ffn_"
    namespaced = rename_ops(
        region,
        {op.id: f"{namespace}{op.id}" for op in region.ops if op.id != "hidden"},
    )
    desc, facts = ffn_summary(fact)
    if not region.resolved and region.ops:
        desc = str((region.ops[0].meta or {}).get("desc") or desc)
    facts = [item for item in facts if not item.endswith("?")]
    return {
        "id": f"{prefix}_op_ffn",
        "role": "ffn",
        "kind": "ffn",
        "title": "Feed-forward",
        "description": desc,
        "facts": facts,
        "view": "ffn",
        "source_component": component,
        "detail": {
            "ffn": fact,
            "op_namespace": namespace,
            "evidence": evidence,
        },
        "children": cards_from_region(namespaced),
    }


def _moe_children_from_fact(fact: dict, component: str | None) -> list[Block]:
    """The canonical MoE child subtree, rebuilt from serialized facts.

    ``ffn_child_blocks`` consumes the typed spec; ``ffn_detail`` is its exact
    serializer, so the round-trip is lossless.  Every child fact dict gains the
    tower's own width and ownership stamp — a sub-model must not inherit the
    host's hidden size, and its drill events must bind to its own oracle.
    """
    from .adapters.transformer.blocks.feed_forward import ffn_child_blocks
    from .ir import FFNSpec

    field_names = {f.name for f in FFNSpec.__dataclass_fields__.values()}
    ffn_spec = FFNSpec(**{k: v for k, v in fact.items() if k in field_names})
    children = ffn_child_blocks(ffn_spec, fact.get("hidden"))

    def _stamp(blocks_list):
        for child in blocks_list or []:
            if component:
                child["source_component"] = component
            inner = child.get("detail")
            if isinstance(inner, dict) and isinstance(inner.get("ffn"), dict):
                inner["ffn"]["hidden"] = fact.get("hidden")
            _stamp(child.get("children"))

    _stamp(children)
    return children


def submodel_cell_blocks(
    spec: dict,
    prefix: str,
    *,
    attn_description: str,
    norm_fallback: str,
    norm_card,
    residual_card,
) -> list[Block]:
    """Per-layer-type cell cards (attention + FFN + norm + residual) plus
    nested sub-model blocks — the recursive heart of the projector.

    A homogeneous stack keeps the bare ``{prefix}_op_*`` ids; each additional
    layer type gets ``{prefix}_g<k>_op_*``; each nested sub-model gets
    ``{prefix}_s<j>`` and recurses through :func:`submodel_block` with the
    composed namespace and dotted component — depth is unbounded by
    construction, never handled per node.
    """
    from .labels import attention_summary, kind_long

    groups = spec.get("groups") or []
    total = spec.get("layers")
    cards: list[Block] = []

    def _cell(group: dict, cell_prefix: str, *, chips: list[str]) -> list[Block]:
        attn = group["attention"]
        title = kind_long(attn).replace(" attention", " self-attention")
        facts = (attention_summary(attn)[1] if attn.get("num_heads") else [])
        return [
            submodel_attention_block(
                spec, group, cell_prefix, title=title,
                description=attn_description, facts=facts + chips,
            ),
            submodel_ffn_block(spec, group, cell_prefix),
            norm_card(cell_prefix, group.get("norm") or norm_fallback),
            residual_card(cell_prefix),
        ]

    if len(groups) > 1:
        for k, group in enumerate(groups):
            chips = [c for c in (
                group.get("tag"),
                f"{group.get('count')} of {total} layers"
                if group.get("count") and total else None,
            ) if c]
            cards.extend(_cell(group, f"{prefix}_g{k}", chips=chips))
    elif groups:
        cards.extend(_cell(groups[0], prefix, chips=[]))

    for j, nested in enumerate(spec.get("sub_models") or []):
        if isinstance(nested, dict):
            cards.append(submodel_block(
                nested, f"{prefix}_s{j}",
                attn_description=attn_description,
                norm_fallback=norm_fallback,
                norm_card=norm_card, residual_card=residual_card,
            ))
    return cards


def submodel_block(
    spec: dict,
    prefix: str,
    *,
    name: str | None = None,
    attn_description: str,
    norm_fallback: str,
    norm_card,
    residual_card,
) -> Block:
    """A whole nested sub-model as one clickable tower block (recursion step).

    Reuses the registered ``text_encoder`` tower view — the one grouped-tower
    renderer — with this spec in ``detail.sub_model``; its children are this
    same projector applied one level down.
    """
    title = str(name or spec.get("component") or "sub-model").split(".")[-1]
    return {
        "id": prefix,
        "role": "embedding",
        "kind": "embedding",
        "label": title,
        "title": f"{title} sub-model",
        "description": (
            f"An embedded transformer stack ({spec.get('layers') or '?'} layers, "
            f"{spec.get('hidden') or '?'}-d) with its own layer types and drills — "
            "projected through the same machinery as the main model."
        ),
        "view": "text_encoder",
        "source_component": spec.get("component") or None,
        "detail": {
            "name": title,
            "node_prefix": prefix,
            "layers": spec.get("layers"),
            "hidden": spec.get("hidden"),
            "norm": spec.get("norm"),
            "sub_model": spec,
        },
        # The tower view draws an embedding pre-stage node; every drawn node
        # gets a card — a generic one here (host adapters override prose for
        # their top-level slots; a nested tower keeps the neutral wording).
        "children": [{
            "id": f"{prefix}_op_embed",
            "title": "Token embedding",
            "description": ("Maps each input id to a vector at this tower's "
                            "own width before its layer stack."),
            "facts": [f for f in (
                f"{spec.get('vocab'):,} vocab" if spec.get("vocab") else "",
                f"{spec.get('hidden'):,}-d" if spec.get("hidden") else "",
            ) if f],
        }] + submodel_cell_blocks(
            spec, prefix,
            attn_description=attn_description,
            norm_fallback=norm_fallback,
            norm_card=norm_card, residual_card=residual_card,
        ),
    }


__all__ = [
    "ALTITUDE_TRANSFORMS",
    "qualify_component",
    "submodel_attention_block",
    "submodel_block",
    "submodel_cell_blocks",
    "submodel_ffn_block",
    "submodel_group_tags",
    "submodel_spec",
]
