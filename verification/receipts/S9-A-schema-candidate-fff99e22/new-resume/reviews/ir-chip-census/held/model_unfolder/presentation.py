"""Typed annotations, separate from fact authority and structural occurrences.

Factories validate actual evidence. Decoded records are display transport only;
consumers bind them to the current ledger, never requalify a serialized proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Mapping

from .uncertainty import UnknownReason, UnknownReasonDisplay

CHIP_KINDS = frozenset({"unresolved", "class_default", "symbolic", "pending_design"})
_REFERENCE_FIELDS = {"owner", "fact_key", "status", "completeness", "value_status_hash", "claim_proof"}
_PROVEN = frozenset({"code_proven", "code_and_config", "config_declared", "class_default"})


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires nonempty text")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _parameter_value(value, path):
    try:
        for step in path:
            if type(step) is int and not isinstance(value, (list, tuple)):
                raise TypeError("integer step requires sequence")
            if type(step) is str and not isinstance(value, Mapping):
                raise TypeError("string step requires mapping")
            value = value[step]
    except (TypeError, KeyError, IndexError) as exc:
        raise ValueError("symbolic parameter path is not present in the fact") from exc
    return value


@dataclass(frozen=True)
class ChipFactReference:
    """A portable snapshot, not a proof object."""

    _record: str

    def __post_init__(self):
        data = self.to_dict()
        if not isinstance(data, dict) or set(data) != _REFERENCE_FIELDS:
            raise ValueError("chip fact reference schema is closed")
        for key in ("owner", "fact_key", "status", "completeness"):
            _text(data[key], key)
        if not data["fact_key"].startswith(data["owner"] + "."):
            raise ValueError("chip fact reference must name its exact owner")
        digest = data["value_status_hash"]
        if not isinstance(digest, str) or len(digest) != 16 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("chip reference requires the existing value/status hash")
        proof = data["claim_proof"]
        if proof is not None and (not isinstance(proof, dict) or proof.get("fact_id") != data["fact_key"]):
            raise ValueError("chip proof description must belong to its exact fact")

    @classmethod
    def from_fact(cls, fact):
        from .evidence.facts import EvidenceFact
        from .evidence.claim_evidence import validate_fact_claim
        from .evidence.receipts import value_status_hash
        if not isinstance(fact, EvidenceFact):
            raise TypeError("chip references originate from actual typed facts")
        proof = None
        if fact.claim_evidence is not None:
            validate_fact_claim(fact, fact.claim_evidence)
            proof = asdict(fact.claim_evidence.summary())
        return cls(_json({"owner": fact.owner, "fact_key": fact.ledger_key(),
                          "status": fact.status, "completeness": fact.completeness,
                          "value_status_hash": value_status_hash(fact.value, fact.status),
                          "claim_proof": proof}))

    @classmethod
    def from_dict(cls, record):
        return cls(_json(record))

    def to_dict(self):
        return json.loads(self._record)


@dataclass(frozen=True)
class SymbolicParameter:
    name: str
    fact_key: str
    value_path: tuple[str | int, ...]
    value: str | int | float

    def __post_init__(self):
        _text(self.name, "symbolic parameter name")
        _text(self.fact_key, "symbolic parameter fact")
        if not isinstance(self.value_path, tuple) or any(
                type(x) not in {str, int} or (type(x) is int and x < 0) for x in self.value_path):
            raise ValueError("symbolic value path uses exact keys/nonnegative indexes")
        if type(self.value) not in {str, int, float} or (
                type(self.value) is float and not math.isfinite(self.value)):
            raise ValueError("symbolic parameter requires a finite scalar")

    def to_dict(self):
        return {"name": self.name, "fact_key": self.fact_key,
                "value_path": list(self.value_path), "value": self.value}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict) or set(record) != {"name", "fact_key", "value_path", "value"} or not isinstance(record["value_path"], list):
            raise ValueError("symbolic parameter schema is closed")
        return cls(record["name"], record["fact_key"], tuple(record["value_path"]), record["value"])


@dataclass(frozen=True)
class PresentationChip:
    chip_kind: str
    text: str
    owner: str
    references: tuple[ChipFactReference, ...]
    unknown_reason: UnknownReasonDisplay | None = None
    decision_ref: str | None = None
    parameters: tuple[SymbolicParameter, ...] = ()

    def __post_init__(self):
        if self.chip_kind not in CHIP_KINDS:
            raise ValueError("chip kind is closed and separate from evidence status")
        _text(self.text, "chip text")
        _text(self.owner, "chip owner")
        if not isinstance(self.references, tuple) or not self.references or any(
                not isinstance(ref, ChipFactReference) for ref in self.references):
            raise TypeError("chip requires typed exact fact references")
        refs = [ref.to_dict() for ref in self.references]
        keys = [ref["fact_key"] for ref in refs]
        if len(set(keys)) != len(keys) or any(ref["owner"] != self.owner for ref in refs):
            raise ValueError("chip fact references must be unique and belong to its owner")
        if not isinstance(self.parameters, tuple) or any(not isinstance(p, SymbolicParameter) for p in self.parameters):
            raise TypeError("symbolic parameters must be a typed tuple")
        if self.chip_kind == "unresolved":
            if not isinstance(self.unknown_reason, UnknownReasonDisplay):
                raise ValueError("unresolved chip requires typed reason transport")
            if any(ref["status"] not in {"unknown", "ambiguous", "oracle_missing"} for ref in refs):
                raise ValueError("unresolved annotation cannot demote a proven fact")
            investigation = self.unknown_reason.to_dict()["investigation"]
            if investigation is not None and investigation["claim_key"] not in keys:
                raise ValueError("chip investigation belongs to another fact")
        elif self.unknown_reason is not None:
            raise ValueError("only unresolved chips carry an unknown reason")
        if self.chip_kind == "class_default" and any(ref["status"] != "class_default" or ref["claim_proof"] is None for ref in refs):
            raise ValueError("class_default chip requires qualified class-default references")
        if self.chip_kind in {"pending_design", "symbolic"} and any(
                ref["status"] not in _PROVEN or ref["completeness"] != "complete" or
                ref["claim_proof"] is None for ref in refs):
            raise ValueError("pending/symbolic chips require complete qualified evidence")
        if self.chip_kind == "pending_design":
            _text(self.decision_ref, "pending design decision reference")
        elif self.decision_ref is not None:
            raise ValueError("only pending-design chips carry a decision reference")
        if self.chip_kind == "symbolic":
            if not self.parameters or len({p.name for p in self.parameters}) != len(self.parameters) or any(p.fact_key not in keys for p in self.parameters):
                raise ValueError("symbolic chips require exact distinct proven parameters")
            by_key = {ref["fact_key"]: ref for ref in refs}
            if any(by_key[param.fact_key]["claim_proof"].get("claim_kind") != "value" for param in self.parameters):
                raise ValueError("symbolic parameters require value-qualified evidence")
        elif self.parameters:
            raise ValueError("only symbolic chips carry symbolic parameters")

    @classmethod
    def from_facts(cls, chip_kind, text, owner, facts, *, unknown_reason=None,
                   decision_ref=None, parameters=()):
        facts = tuple(facts)
        references = tuple(ChipFactReference.from_fact(fact) for fact in facts)
        if unknown_reason is not None and not isinstance(unknown_reason, UnknownReason):
            raise TypeError("producer needs an authoritative UnknownReason, not decoded display data")
        if unknown_reason is not None and any(
                fact.unknown_reason != unknown_reason for fact in facts):
            raise ValueError("chip unknown reason must be the actual cited facts' reason")
        by_key = {fact.ledger_key(): fact for fact in facts}
        for param in parameters:
            if not isinstance(param, SymbolicParameter) or param.fact_key not in by_key:
                raise ValueError("symbolic parameter has no actual cited fact")
            value = _parameter_value(by_key[param.fact_key].value, param.value_path)
            if type(value) is not type(param.value) or value != param.value:
                raise ValueError("symbolic parameter differs from its actual fact")
        return cls(chip_kind, text, owner, references,
                   UnknownReasonDisplay.from_dict(unknown_reason.to_dict()) if unknown_reason else None,
                   decision_ref, tuple(parameters))

    def to_dict(self):
        return {"chip_kind": self.chip_kind, "text": self.text, "owner": self.owner,
                "references": [ref.to_dict() for ref in self.references],
                "unknown_reason": self.unknown_reason.to_dict() if self.unknown_reason else None,
                "decision_ref": self.decision_ref,
                "parameters": [param.to_dict() for param in self.parameters]}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict) or set(record) != {
                "chip_kind", "text", "owner", "references", "unknown_reason", "decision_ref", "parameters"}:
            raise ValueError("presentation chip schema is closed")
        if not isinstance(record["references"], list) or not isinstance(record["parameters"], list):
            raise TypeError("chip references and parameters require lists")
        return cls(record["chip_kind"], record["text"], record["owner"],
                   tuple(ChipFactReference.from_dict(ref) for ref in record["references"]),
                   UnknownReasonDisplay.from_dict(record["unknown_reason"]) if record["unknown_reason"] is not None else None,
                   record["decision_ref"], tuple(SymbolicParameter.from_dict(p) for p in record["parameters"]))

    def bind(self, fact_rows):
        """Compare transport with current producer metadata; no evidence promotion."""
        for ref in self.references:
            data = ref.to_dict()
            row = fact_rows.get(data["fact_key"])
            if not isinstance(row, Mapping) or row.get("status") != data["status"] or row.get("presentation_reference") != data:
                raise ValueError("chip reference does not match the current producer ledger")
            if self.unknown_reason is not None and row.get("unknown_reason") != self.unknown_reason.to_dict():
                raise ValueError("chip unknown reason does not match the producer ledger")
        for param in self.parameters:
            actual = _parameter_value(fact_rows[param.fact_key].get("value"), param.value_path)
            if type(actual) is not type(param.value) or actual != param.value:
                raise ValueError("symbolic display parameter differs from the current ledger")


def chips_from_block(block):
    records = block.get("presentation_chips", [])
    if not isinstance(records, list):
        raise TypeError("presentation_chips must be a list")
    chips = tuple(PresentationChip.from_dict(record) for record in records)
    if chips:
        path = block.get("presentation_path")
        if not _valid_presentation_path(path):
            raise ValueError("chip-bearing card requires its exact producer JSON path")
        aliases = block.get("presentation_aliases", [])
        if not isinstance(aliases, list) or any(not _valid_presentation_path(alias) for alias in aliases):
            raise ValueError("presentation aliases require exact JSON paths")
        paths = [tuple(path), *(tuple(alias) for alias in aliases)]
        if len(set(paths)) != len(paths):
            raise ValueError("presentation canonical/alias paths must be unique")
    cited = block.get("source_fact_keys") or ()
    for chip in chips:
        if any(ref.to_dict()["fact_key"] not in cited for ref in chip.references):
            raise ValueError("chip references must be cited by this exact card")
    return chips


def _valid_presentation_path(path):
    return isinstance(path, list) and bool(path) and all(
        type(part) in {str, int} and
        (type(part) is not int or part >= 0) and
        (type(part) is not str or bool(part)) for part in path)


def architectural_envelopes(document):
    """Yield only declared architectural block/region slots, never raw payloads."""
    def blocks(values, path, active=frozenset()):
        if not isinstance(values, (list, tuple)):
            return
        for number, block in enumerate(values):
            if not isinstance(block, dict):
                continue
            if id(block) in active:
                raise ValueError('presentation projection cannot traverse cyclic blocks')
            location = (*path, number)
            yield location, block
            yield from blocks(block.get('children'), (*location, 'children'), active | {id(block)})
    layers = document.get('layers') or ()
    if isinstance(layers, (list, tuple)):
        for number, layer in enumerate(layers):
            if isinstance(layer, dict):
                yield from blocks(layer.get('blocks'), ('layers', number, 'blocks'))
    extras = document.get('extras') or {}
    render = (extras.get('render') or {}) if isinstance(extras, dict) else {}
    if isinstance(render, dict):
        for key in ('model_blocks', 'loop_blocks'):
            yield from blocks(render.get(key), ('extras', 'render', key))
        opaque = render.get('opaque_layer_block')
        if isinstance(opaque, dict):
            yield ('extras', 'render', 'opaque_layer_block'), opaque
            yield from blocks(opaque.get('children'),
                              ('extras', 'render', 'opaque_layer_block', 'children'))
        region = render.get('loop_region')
        if isinstance(region, dict):
            yield ('extras', 'render', 'loop_region'), region


def unresolved_spec_values(document):
    """Exact schema slots whose literal/state already explicitly means unknown."""
    def literals(mapping, prefix, names):
        if isinstance(mapping, dict):
            for name in names:
                if mapping.get(name) == 'unknown':
                    yield (*prefix, name), 'literal_unknown'
    def nulls(mapping, prefix, names):
        for name in names:
            if name in mapping and mapping[name] is None:
                yield (*prefix, name), 'null_unknown'
    yield from literals(document, (), ('embedding_norm_kind', 'final_norm_kind'))
    for name in ('hidden_size', 'tie_word_embeddings'):
        if name in document and document[name] is None:
            yield (name,), 'null_unknown'
    def layer_values(layer, prefix):
        if not isinstance(layer, dict):
            return
        yield from literals(layer, prefix, ('norm_kind', 'norm_placement', 'residual_topology'))
        for name in ('attention', 'cross_attention'):
            yield from literals(layer.get(name), (*prefix, name), ('kind', 'mask', 'position_kind', 'position_application'))
        yield from literals(layer.get('ffn'), (*prefix, 'ffn'), ('kind', 'activation'))
        if layer.get('residual_topology') in {'parallel', 'fused_parallel'}:
            yield from nulls(layer, prefix, ('parallel_norm_count',))
        for name in ('attention', 'cross_attention'):
            attention = layer.get(name)
            if not isinstance(attention, dict):
                continue
            address = (*prefix, name)
            yield from nulls(attention, address,
                ('kind', 'mask', 'qk_norm', 'cached', 'output_projection', 'bias', 'scores_scaled'))
            if attention.get('kind') in {None, 'mha', 'gqa', 'mqa'}:
                yield from nulls(attention, address, ('projection_mode',))
            if attention.get('cross_attention') is True:
                yield from nulls(attention, address, ('cross_kv_source_kind',))
        ffn = layer.get('ffn')
        if isinstance(ffn, dict):
            address = (*prefix, 'ffn')
            yield from nulls(ffn, address, ('kind', 'activation', 'gated'))
            if ffn.get('kind') in {None, 'dense', 'moe'}:
                yield from nulls(ffn, address, ('projection_mode',))
            if ffn.get('kind') == 'moe':
                yield from nulls(ffn, address, ('expert_projection_mode',))

    def submodel_values(spec, prefix, active=frozenset()):
        if not isinstance(spec, dict):
            return
        if id(spec) in active:
            raise ValueError('presentation projection cannot traverse cyclic submodels')
        active = active | {id(spec)}
        groups = spec.get('groups') or ()
        if isinstance(groups, (list, tuple)):
            for number, group in enumerate(groups):
                yield from layer_values(group, (*prefix, 'groups', number))
        nested = spec.get('sub_models') or ()
        if isinstance(nested, (list, tuple)):
            for number, child in enumerate(nested):
                yield from submodel_values(child, (*prefix, 'sub_models', number), active)

    layers = document.get('layers') or ()
    if isinstance(layers, (list, tuple)):
        for number, layer in enumerate(layers):
            yield from layer_values(layer, ('layers', number))
    # These are the typed attention/FFN snapshots consumed by block drills,
    # and the embedded model groups consumed by the tower renderer.  They
    # retain their own exact addresses, even when copied from a root layer.
    # Raw config, source_evidence, prose and arbitrary nested mappings are
    # deliberately outside this finite schema traversal.
    for prefix, block in architectural_envelopes(document):
        detail = block.get('detail')
        if not isinstance(detail, dict):
            continue
        # A detail envelope carries typed attention/FFN projections, not a
        # generic LayerSpec.  Only submodel groups carry layer topology.
        projections = {name: detail[name] for name in ('attention', 'cross_attention', 'ffn')
                       if name in detail}
        yield from layer_values(projections, (*prefix, 'detail'))
        yield from submodel_values(detail.get('sub_model'), (*prefix, 'detail', 'sub_model'))
    extras = document.get('extras') or {}
    modalities = extras.get('modalities') if isinstance(extras, dict) else None
    if isinstance(modalities, dict):
        inputs = modalities.get('inputs') or {}
        if isinstance(inputs, dict):
            for name in ('vision', 'video', 'audio', 'conditioning'):
                component = inputs.get(name)
                pipeline = component.get('pipeline') if isinstance(component, dict) else None
                if isinstance(pipeline, (list, tuple)):
                    for number, stage in enumerate(pipeline):
                        yield from literals(stage,
                            ('extras', 'modalities', 'inputs', name, 'pipeline', number),
                            ('operation',))
        yield from literals(modalities.get('fusion'), ('extras', 'modalities', 'fusion'),
                            ('operation', 'target'))




@dataclass(frozen=True)
class UnknownValueAnnotation:
    """A display-only unknown value address; never a fact or an occurrence."""
    path: tuple[str | int, ...]
    value_state: str
    reason: UnknownReasonDisplay

    def __post_init__(self):
        if not isinstance(self.path, tuple) or not self.path or any(
                type(part) not in (str, int) or (type(part) is int and part < 0)
                for part in self.path):
            raise ValueError('unknown annotation requires an exact schema path')
        if self.value_state not in {'unresolved_envelope', 'literal_unknown', 'null_unknown'}:
            raise ValueError('unknown annotation state is closed')
        if not isinstance(self.reason, UnknownReasonDisplay):
            raise TypeError('unknown annotation requires display reason metadata')

    def to_dict(self):
        return {'path': list(self.path), 'value_state': self.value_state,
                'unknown_reason': self.reason.to_dict()}


def unknown_annotations_from_ir(document):
    """Bind existing producer descriptions to exact current schema values."""
    annotations = []
    for path, envelope in architectural_envelopes(document):
        if (envelope.get('resolved') is False or envelope.get('kind') == 'unknown') and (
                envelope.get('unknown_reason') is not None):
            annotations.append(UnknownValueAnnotation(path, 'unresolved_envelope',
                UnknownReasonDisplay.from_dict(envelope['unknown_reason'])))
    expected = dict(unresolved_spec_values(document))
    rows = (document.get('extras') or {}).get('presentation_unresolved_values', [])
    if not isinstance(rows, list):
        raise ValueError('presentation_unresolved_values must be a list')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'path', 'value_state', 'unknown_reason'}:
            raise ValueError('unknown spec annotation schema is closed')
        if not isinstance(row['path'], list):
            raise ValueError('unknown spec path must be a list')
        annotation = UnknownValueAnnotation(tuple(row['path']), row['value_state'],
            UnknownReasonDisplay.from_dict(row['unknown_reason']))
        if annotation.path in seen or expected.get(annotation.path) != annotation.value_state:
            raise ValueError('unknown annotation does not bind one exact unresolved schema value')
        seen.add(annotation.path)
        annotations.append(annotation)
    return tuple(annotations)
