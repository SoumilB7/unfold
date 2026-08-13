"""Owner-qualified, value-exact structural projection matrix.

The registry says a fact *kind* is lawful.  This matrix proves a concrete model
instance actually carries the fact whose value its specs/renderers/JSON/params
project.  A registered leaf without an instance fact cannot green this gate,
and a renderer/spec value that differs from the fact is a blocking mismatch.

Only fully migrated value families belong here.  Adding a row is therefore a
cutover declaration: every canonical-spec projection needs an owner-qualified
typed fact, unknown projects no conventional value, and heterogeneous values
require per-occurrence facts rather than a global compromise.  ``surfaces`` is
the mandatory consumer-test inventory; this runtime function validates the one
canonical spec those consumers must derive from, rather than pretending it has
rendered and decoded every downstream artifact during Sable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class QualificationRule:
    owner: str
    fact_key: str
    structural_fields: tuple[str, ...]
    surfaces: frozenset[str]
    ir_scope: str = "transformer_decoder"
    layer_kinds: frozenset[str] = frozenset()
    fact_value_shape: str = "mapping"

    def __post_init__(self) -> None:
        if not self.owner or not self.fact_key or not self.structural_fields:
            raise ValueError("a qualification rule names owner, fact and fields")
        if not self.surfaces:
            raise ValueError("a qualification rule names every authoritative sink")
        if self.ir_scope != "transformer_decoder":
            raise ValueError("a qualification rule has a closed IR scope")
        if self.fact_value_shape not in {
                "mapping", "scalar", "tuple", "sequence"}:
            raise ValueError("a qualification rule has a closed fact shape")
        if self.fact_value_shape == "scalar" \
                and len(self.structural_fields) != 1:
            raise ValueError("a scalar fact qualifies exactly one field")

    @property
    def fact_id(self) -> str:
        return f"{self.owner}.{self.fact_key}"


QUALIFICATION_MATRIX = (
    QualificationRule(
        "decoder.attention", "head_geometry",
        ("kind", "num_heads", "num_kv_heads", "head_dim",
         "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
         "qk_rope_head_dim", "v_head_dim"),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        layer_kinds=frozenset({"mha", "gqa", "mqa", "mla"})),
    QualificationRule(
        "decoder.attention", "head_geometry_schedule",
        ("kind", "num_heads", "num_kv_heads", "head_dim"),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        layer_kinds=frozenset({"mha", "gqa", "mqa"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.attention", "gated_delta_geometry",
        ("num_kv_heads", "num_heads", "head_dim", "v_head_dim",
         "conv_kernel_size"),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        layer_kinds=frozenset({"gated_delta"}),
        fact_value_shape="tuple"),
    QualificationRule(
        "decoder.attention", "mask_schedule", ("mask", "window_size"),
        frozenset({"spec", "card", "json"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.attention", "position_schedule",
        ("position_kind", "position_application", "rope_dim"),
        frozenset({"spec", "opgraph", "card", "json"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.input", "position_addition",
        ("position_kind", "position_application"),
        frozenset({"spec", "card", "json"}),
        fact_value_shape="mapping"),
    QualificationRule(
        "decoder.attention", "mixer_schedule", ("kind",),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.attention", "qk_norm_schedule", ("qk_norm",),
        frozenset({"spec", "opgraph", "card", "json"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.attention", "kv_sharing_schedule", ("kv_source_layer",),
        frozenset({"spec", "card", "json"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.attention", "cross_attention_schedule", ("cross_attention",),
        frozenset({"spec", "opgraph", "card", "json"}),
        fact_value_shape="sequence"),
    QualificationRule(
        "decoder.ffn", "intermediate_size", ("intermediate_size",),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        fact_value_shape="scalar"),
    QualificationRule(
        "decoder.ffn", "ffn_schedule", ("kind",),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        fact_value_shape="sequence"),
    # Fused expert geometry satisfies this row; split/flattened storage remains
    # honestly withheld until its axes can be distinguished exactly.
    QualificationRule(
        "decoder.ffn.expert", "expert_intermediate_size",
        ("expert_intermediate_size",),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        fact_value_shape="scalar"),
    QualificationRule(
        "decoder.ffn", "routing_policy",
        ("num_experts", "num_experts_per_tok"),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        layer_kinds=frozenset({"moe"}),
        fact_value_shape="mapping"),
    QualificationRule(
        "decoder.ffn.expert", "shared_expert_count",
        ("num_shared_experts",),
        frozenset({"spec", "opgraph", "card", "json", "params"}),
        layer_kinds=frozenset({"moe"}),
        fact_value_shape="scalar"),
    QualificationRule(
        "decoder", "codebook_streams",
        ("num", "embeddings_summed", "heads_stacked"),
        frozenset({"spec", "card", "json", "params"}),
        fact_value_shape="mapping"),
    QualificationRule(
        "decoder", "per_layer_embedding_pathway",
        ("hidden", "vocab"),
        frozenset({"spec", "card", "json", "params"}),
        fact_value_shape="mapping"),
)


def qualification_findings(ir: dict) -> list[str]:
    """Return instance-level projection/fact mismatches for migrated geometry."""
    if not isinstance(ir, dict):
        raise TypeError("qualification audit consumes the serialized IR")
    facts = ((ir.get("extras") or {}).get("fact_provenance") or {})
    findings = []
    for rule in QUALIFICATION_MATRIX:
        if not _applies(ir, rule, facts):
            continue
        values = _projected_values(ir, rule)
        fact = facts.get(rule.fact_id)
        if not values:
            # A complete negative schedule can truthfully carry one None per
            # occurrence while projecting no positive structure.  Normalize
            # the fact through the same per-family rule before calling that a
            # withheld positive claim.  This does not excuse a non-empty
            # schedule: any positive member still produces a mismatch here.
            normalized = (
                _fact_value(rule, (fact or {}).get("value"))
                if fact is not None else None)
            if fact is not None and normalized not in (None, (), []):
                findings.append(
                    f"{rule.fact_id} is ledgered as {(fact or {}).get('value')!r} "
                    "but every authoritative structural surface withholds it")
            continue
        if rule.fact_value_shape == "sequence":
            projected = values
            if fact is None:
                findings.append(
                    f"{rule.fact_id} projects a {len(values)}-layer schedule "
                    "without an owner-qualified instance fact "
                    f"(required consumers: {sorted(rule.surfaces)})")
                continue
            expected = _fact_value(rule, (fact or {}).get("value"))
            if _canonical(expected) != _canonical(projected):
                findings.append(
                    f"{rule.fact_id} fact schedule {expected!r} does not match "
                    f"the projected structural schedule {projected!r}")
            continue
        distinct = {_canonical(value): value for value in values}
        if len(distinct) != 1:
            findings.append(
                f"{rule.fact_id} has {len(distinct)} projected values across "
                "layer occurrences but only a global fact owner; migrate to "
                "per-occurrence facts before projecting the schedule")
            continue
        projected = next(iter(distinct.values()))
        if fact is None:
            findings.append(
                f"{rule.fact_id} projects {projected!r} on "
                "the canonical spec without an owner-qualified instance fact "
                f"(required consumers: {sorted(rule.surfaces)})")
            continue
        expected = _fact_value(rule, (fact or {}).get("value"))
        if _canonical(expected) != _canonical(projected):
            findings.append(
                f"{rule.fact_id} fact value {expected!r} does not match the "
                f"projected structural value {projected!r}")
    return findings


def _applies(ir, rule, facts):
    """Apply only at the owner altitude this unit has actually migrated.

    U6/U7 qualify the transformer adapter's decoder.  Diffusion transformer
    and UNet layers carry the typed ``extras.diffusion`` structural root and
    belong to U10/U11.  This is an adapter/owner boundary, never a family or
    model identity test.
    """
    extras = ir.get("extras") or {}
    if rule.ir_scope != "transformer_decoder" \
            or bool(extras.get("diffusion")):
        return False
    scheduled = "decoder.attention.head_geometry_schedule" in facts
    if rule.fact_key == "head_geometry_schedule":
        return scheduled
    if rule.fact_key == "head_geometry" and scheduled:
        return False
    if rule.fact_key == "cross_attention_schedule":
        if rule.fact_id in facts:
            return True
        return any(
            bool((layer or {}).get("cross_attention"))
            or bool(((layer or {}).get("attention") or {}).get(
                "cross_attention"))
            for layer in (ir.get("layers") or ()))
    return True


def _projected_values(ir, rule):
    if rule.owner == "decoder.input":
        model_blocks = ((((ir.get("extras") or {}).get("render") or {})
                         .get("model_blocks")) or ())
        block = next((item for item in model_blocks
                      if isinstance(item, dict)
                      and item.get("id") == "position_add"), None)
        value = (block or {}).get("detail")
        if not isinstance(value, dict):
            return []
        return ({field: value.get(field)
                 for field in rule.structural_fields},)
    if rule.owner == "decoder":
        extras = ir.get("extras") or {}
        model_blocks = ((extras.get("render") or {}).get("model_blocks") or ())
        if rule.fact_key == "codebook_streams":
            block = next((item for item in model_blocks
                          if item.get("id") == "tok_text"), None)
            value = (block or {}).get("detail")
        elif rule.fact_key == "per_layer_embedding_pathway":
            pathway = next((item for item in
                            (extras.get("external_pathways") or ())
                            if item.get("id") == "per_layer_input"), None)
            value = (pathway or {}).get("detail")
        else:
            value = None
        if not isinstance(value, dict):
            return []
        return [{field: value.get(field)
                 for field in rule.structural_fields}]
    layers = tuple(ir.get("layers") or ())
    out = []
    for layer in layers:
        layer = layer or {}
        if rule.owner == "decoder.attention":
            spec = layer.get("attention") or {}
            if rule.layer_kinds and spec.get("kind") not in rule.layer_kinds:
                continue
            if rule.fact_key == "mask_schedule" \
                    and spec.get("mask") in {None, "unknown"}:
                continue
            if rule.fact_key == "position_schedule" \
                    and spec.get("position_kind") in {None, "unknown"} \
                    and spec.get("position_application") in {None, "unknown"} \
                    and spec.get("rope_dim") is None:
                # Honest unknown is the absence of a conventional positional
                # claim.  It owes no positive instance fact; treating the
                # literal display sentinel "unknown" as a mechanism would make
                # source-missing models fail the very withholding law this net
                # is meant to enforce.
                continue
            if rule.fact_key == "mixer_schedule":
                kind = spec.get("kind")
                if kind is None:
                    continue
                out.append({"kind": (
                    "ordinary_attention"
                    if kind in {"mha", "gqa", "mqa", "mla"}
                    else kind)})
                continue
            if rule.fact_key == "qk_norm_schedule":
                value = spec.get("qk_norm")
                if value is not None:
                    out.append({"qk_norm": value})
                continue
            if rule.fact_key == "kv_sharing_schedule":
                value = spec.get("kv_source_layer")
                if value is not None:
                    out.append({"kv_source_layer": value})
                continue
            if rule.fact_key == "cross_attention_schedule":
                if isinstance(layer.get("cross_attention"), dict):
                    value = "additive_cross"
                elif spec.get("cross_attention") is True:
                    value = "replacement_cross"
                else:
                    value = "self"
                out.append({"cross_attention": value})
                continue
            value = {field: spec.get(field) for field in rule.structural_fields}
            if any(item is not None for item in value.values()):
                out.append(value)
        else:
            spec = layer.get("ffn") or {}
            if rule.layer_kinds and spec.get("kind") not in rule.layer_kinds:
                continue
            if rule.fact_key == "ffn_schedule":
                kind = spec.get("kind")
                if kind is not None:
                    out.append({"kind": kind})
                continue
            if rule.fact_value_shape == "mapping":
                value = {
                    field: spec.get(field) for field in rule.structural_fields}
                if any(item is not None for item in value.values()):
                    out.append(value)
                continue
            field = rule.structural_fields[0]
            value = spec.get(field)
            if value is not None:
                out.append(value)
    return tuple(out)


def _fact_value(rule, value):
    if rule.fact_value_shape == "mapping":
        if not isinstance(value, dict):
            return {field: None for field in rule.structural_fields}
        return {field: value.get(field) for field in rule.structural_fields}
    if rule.fact_value_shape == "scalar":
        return value
    if rule.fact_value_shape == "tuple":
        if not isinstance(value, (tuple, list)) \
                or len(value) != len(rule.structural_fields):
            return {field: None for field in rule.structural_fields}
        return dict(zip(rule.structural_fields, value))
    if rule.fact_value_shape == "sequence":
        if not isinstance(value, (tuple, list)):
            return ()
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append({field: item.get(field)
                            for field in rule.structural_fields})
                continue
            if rule.fact_key in {"mixer_schedule", "ffn_schedule"} \
                    and isinstance(item, str):
                out.append({"kind": item})
                continue
            if rule.fact_key == "qk_norm_schedule" \
                    and (isinstance(item, bool) or item is None):
                if item is not None:
                    out.append({"qk_norm": item})
                continue
            if rule.fact_key == "kv_sharing_schedule" \
                    and (isinstance(item, int) or item is None):
                if item is not None:
                    out.append({"kv_source_layer": item})
                continue
            if rule.fact_key == "cross_attention_schedule" \
                    and isinstance(item, str):
                out.append({"cross_attention": item})
                continue
            if not isinstance(item, (tuple, list)) \
                    or len(item) != len(rule.structural_fields):
                return ()
            out.append(dict(zip(rule.structural_fields, item)))
        return tuple(out)
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "QUALIFICATION_MATRIX", "QualificationRule", "qualification_findings",
]
