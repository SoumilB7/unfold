"""Closed, producer-owned projections of retained mechanism reader proofs.

The invocation seal establishes who produced the exact result, not semantic
strength. The concrete witness establishes that strength and derives every
published value. Neither a reader name nor arbitrary successful DTO/spans can
stand in for that witness. Local identity seals never enter persisted output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import wraps
from pathlib import Path
import sys
from threading import RLock
import weakref

from .config_access import (
    checkpoint_fingerprint, current_prepared_document,
    prepared_document_token, verify_prepared_document_token,
)
from .program_index import ProgramIndex, SourceSpan, portable_source_index_fingerprint
from .receipts import value_status_hash


class ReaderClaimUnavailable(ValueError):
    """The real reader has not yet proved one finite projection's premises.

    This is separate from a malformed/foreign proof. Existing unqualified facts
    can retain their visible migration debt; consumers must not catch arbitrary
    proof-validation errors as availability failures.
    """


@dataclass(frozen=True)
class ReaderFactProjection:
    """A finite projection computed by a concrete semantic witness."""

    owner: str
    key: str
    claim_kind: str
    value: object
    status: str
    config_paths: tuple[tuple[str, ...], ...] = ()
    completeness: str = "complete"
    required_spans: tuple[SourceSpan, ...] | None = None


@dataclass(frozen=True)
class ReaderOperand:
    """An exact source-requested scalar and its supplying document channel."""
    source_path: tuple[str, ...]
    value: object
    source_kind: str
    checkpoint_path: tuple[str, ...] | None


@dataclass(frozen=True)
class _SelectorRead:
    path: tuple[str, ...]
    value: object
    source_kind: str


def _checked_selector_reads(reads, document):
    for read in reads:
        operand = reader_operand(document, read.path)
        if value_status_hash(read.value, "operand") != value_status_hash(operand.value, "operand") \
                or read.source_kind and read.source_kind != operand.source_kind:
            raise ValueError("reader selector value/origin differs from its prepared document")


def validate_reader_operands(document, operands):
    """Compare producer-retained deciding operands with their exact channels."""
    seen = set()
    for path, source_kind, value in operands:
        marker = (tuple(path), source_kind, value_status_hash(value, "operand"))
        if marker in seen:
            continue
        seen.add(marker)
        actual = reader_operand(document, path)
        if actual.source_kind != source_kind or value_status_hash(
                actual.value, "operand") != marker[2]:
            raise ValueError("reader deciding operand differs from its prepared document")


def reader_operand(document, source_path, *, allow_aliases=True):
    """Resolve only an implementation-proven address through shared syntax.

    This performs no mechanism classification and emits no new config events.
    Conflicting aliases remain unavailable, and an alias-spelled source access
    never expands backwards into another property's vocabulary.
    """
    from ..everchanging import load_aliases
    from .config_access import _resolve_for_claim_proof
    path = tuple(source_path)
    if not path:
        raise ValueError("reader operand needs an exact source path")
    def container(mapping):
        current = mapping
        for segment in path[:-1]:
            if not isinstance(current, dict) or segment not in current:
                return {}
            current = current[segment]
        return current if isinstance(current, dict) else {}
    dotted = ".".join(path)
    defaults = ({path[-1]: document.class_overlay[dotted]}
                if dotted in document.class_overlay else {})
    aliases = load_aliases().get(path[-1], ()) if allow_aliases else ()
    # Use the ONE resolver, including its explicit-null and semantic-equality
    # policy. This inspection does not create another consumption or rewrite
    # the parser's ledger: its original events already own that lifecycle.
    resolution = _resolve_for_claim_proof(
        container(document.document), path[-1], aliases,
        path=path[:-1], class_defaults=defaults)
    if resolution.ambiguous:
        raise ValueError("reader operand has conflicting checkpoint spellings")
    if resolution.source_kind == "class_default" or resolution.provenance == "class_default":
        return ReaderOperand(path, resolution.value, "class_default", None)
    if resolution.present and resolution.selected_path is not None:
        return ReaderOperand(path, resolution.value, "config_declared",
                             tuple(resolution.selected_path.split(".")))
    raise ValueError("reader operand has no exact checkpoint or class-default evidence")



def repeated_count_operand(index, blocks, block_occurrence, document):
    """Exact repeated-child range operand, independently of a caller's count.

    The bounded protocol is unshadowed range(one source-bound config value).
    Arithmetic/ranges with offsets remain unavailable until explicitly proved.
    """
    from .attention import exact_config_path_for_expression
    proofs = tuple(proof for proof in blocks.repeated_child.proofs
                   if proof.child_occurrence == block_occurrence)
    if len(proofs) != 1:
        raise ValueError("claim has no unique exact repeated-child construction")
    container = proofs[0].template.container
    expression = container.count_expression
    callable_symbol = container.record.enclosing_callable
    if expression is None or expression.kind != "call" \
            or len(expression.children) != 2 or expression.keyword_children \
            or expression.children[0].kind != "name" \
            or expression.children[0].name != "range" \
            or any(item.name == "range" for item in index.module_bindings_in(callable_symbol.source)) \
            or any(item.name == "range" and item.context in {"parameter", "store", "del"}
                   for item in index.identifiers_in(callable_symbol)):
        raise ValueError("claim repetition has no supported exact builtin range")
    node = blocks.component_root.graph.node_for(container.owner_occurrence)
    path = exact_config_path_for_expression(
        index, node, expression.children[1], config_prefix=blocks.config_path)
    if path is None:
        raise ValueError("claim repetition count has no exact source operand")
    operand = reader_operand(document, path)
    if type(operand.value) is not int or operand.value <= 0:
        raise ValueError("claim repetition needs a positive integer operand")
    return operand

# Exact call-local authentication, analogous to prepared-document issuance.
# Weak values avoid keeping whole source graphs alive after a parse ends.
_CALLS: dict[int, tuple] = {}
_LOCK = RLock()
_PRODUCER_CALLABLES: dict[str, tuple] = {}
_PRODUCERS = frozenset({
    "model_unfolder.evidence.cell_topology.decoder_cell_topology_for_path",
    "model_unfolder.evidence.ffn_mechanism.decoder_ffn_mechanism_for_path",
    "model_unfolder.evidence.ffn_width.decoder_ffn_intermediate_width_for_path",
    "model_unfolder.evidence.expert_storage.decoder_routed_expert_storage_for_path",
    "model_unfolder.evidence.expert_width.decoder_expert_intermediate_width_for_path",
    "model_unfolder.evidence.diffusion_root.read_diffusion_root_topology",
    "model_unfolder.evidence.diffusion_stack.read_diffusion_stack_inventory",
    "model_unfolder.evidence.diffusion_block.read_diffusion_block_facts",
    "model_unfolder.evidence.diffusion_conditioning.read_diffusion_conditioning_graph",
    "model_unfolder.evidence.diffusion_stream.read_diffusion_stream_graph",
    "model_unfolder.evidence.diffusion_bookends.read_diffusion_bookends",
    "model_unfolder.evidence.attention_sinks.decoder_attention_sinks_for_path",
    "model_unfolder.evidence.attention_storage.decoder_attention_projection_storage_for_path",
    "model_unfolder.evidence.decoder_norm.decoder_norm_kind_for_path",
    "model_unfolder.evidence.embedding_bookend.embedding_stage_norm_evidence",
    "model_unfolder.evidence.attention.decoder_attention_qkv_clip_for_path",
    "model_unfolder.evidence.position_linear_bias.decoder_alibi_score_bias_for_path",
    "model_unfolder.evidence.cross_attention_schedule.decoder_cross_attention_all_layers_for_path",
    "model_unfolder.evidence.mixer_schedule.decoder_mixer_schedule_for_path",
    "model_unfolder.evidence.ffn_schedule.decoder_ffn_schedule_for_path",
    "model_unfolder.evidence.final_bookend.final_stage_norm_evidence",
    "model_unfolder.evidence.attention_mask.decoder_attention_mask_execution_for_path",
    "model_unfolder.evidence.position_schedule.decoder_position_application_schedule_for_path",
    "model_unfolder.evidence.position_initialization.position_frequency_initialization",
    "model_unfolder.evidence.attention_output.decoder_attention_output_projection_for_path",
    "model_unfolder.evidence.attention.decoder_attention_cache_for_path",
    "model_unfolder.evidence.attention.decoder_attention_mechanism_for_path",
    "model_unfolder.evidence.attention.decoder_attention_logit_softcap_for_path",
    "model_unfolder.evidence.attention_geometry.decoder_attention_head_geometry_for_path",
    "model_unfolder.evidence.projection_bias.decoder_attention_bias_for_path",
    "model_unfolder.evidence.qk_norm_schedule.decoder_qk_norm_schedule_for_path",
    "model_unfolder.evidence.kv_sharing_schedule.decoder_kv_sharing_schedule_for_path",
    "model_unfolder.evidence.codebook_streams.decoder_codebook_streams_for_path",
    "model_unfolder.evidence.per_layer_side_input.decoder_per_layer_side_input_for_path",
})


def _validate_producer_callable(symbol):
    declaration = _PRODUCER_CALLABLES.get(symbol)
    if declaration is None or declaration[0].__code__ is not declaration[1] \
            or getattr(sys.modules.get(declaration[0].__module__),
                       declaration[0].__name__, None) is not declaration[2]:
        raise ValueError("reader producer callable changed after its invocation")


def _prepared_scope_fingerprint(document):
    return checkpoint_fingerprint({"document": document.document,
                                   "class_overlay": document.class_overlay,
                                   "provenance": document.provenance})


def retained_claim_reader(function):
    """Retain actual calls at the explicitly enumerated producer boundaries."""
    symbol = function.__module__ + "." + function.__qualname__
    if symbol not in _PRODUCERS:
        raise ValueError("reader claim issuance requires a migrated producer")
    module = sys.modules.get(function.__module__)
    code = function.__code__
    if module is None or function.__globals__ is not vars(module) \
            or code.co_qualname != function.__qualname__ \
            or Path(code.co_filename).resolve() != Path(module.__file__).resolve():
        raise ValueError("reader claim issuance requires the exact original producer callable")
    with _LOCK:
        previous = _PRODUCER_CALLABLES.get(symbol)
        if previous is not None:
            raise ValueError("reader claim producer already has its exact callable registration")

    @wraps(function)
    def invoke(*args, **kwargs):
        if function.__code__ is not code or getattr(module, function.__name__, None) is not invoke \
                or _PRODUCER_CALLABLES.get(symbol) != (function, code, invoke):
            raise ValueError("reader claim producer callable changed after registration")
        index = args[0] if args else kwargs.get("index")
        prepared = current_prepared_document.get()
        fingerprint = (checkpoint_fingerprint(prepared.checkpoint)
                       if prepared is not None else "")
        token = (prepared_document_token(prepared, fingerprint)
                 if prepared is not None else "")
        scope_fingerprint = (_prepared_scope_fingerprint(prepared)
                             if prepared is not None else "")
        selector_reads = []
        def captured(selector):
            def select(path):
                selected = selector(path)
                present, value, kind = selected is not None, selected, ""
                if isinstance(selected, tuple) and len(selected) in {2, 3} \
                        and isinstance(selected[0], bool):
                    present, value = selected[:2]
                    kind = selected[2] if len(selected) == 3 else ""
                else:
                    from .framework_config import FrameworkConfigDefaultValue
                    if isinstance(selected, FrameworkConfigDefaultValue):
                        value, kind = selected.value, "class_default"
                if present:
                    from .document import _snapshot
                    selector_reads.append(_SelectorRead(tuple(path), _snapshot(value), kind))
                return selected
            return select
        kwargs = dict(kwargs)
        for name in ("config_selector", "guard_config_selector"):
            if callable(kwargs.get(name)):
                kwargs[name] = captured(kwargs[name])
        result = replace(function(*args, **kwargs), claim_reader_symbol=symbol)
        witness = result.claim_witness
        if not isinstance(index, ProgramIndex):
            raise ValueError("reader witness belongs to another source index")
        if witness is not None:
            if witness.index is not index or witness.reader_symbol != symbol:
                raise ValueError("reader witness belongs to another producer or source index")
            witness.validate_result(result)
        identity = id(result)

        def expired(reference):
            with _LOCK:
                existing = _CALLS.get(identity)
                if existing is not None and existing[0] is reference:
                    _CALLS.pop(identity, None)

        reference = weakref.ref(result, expired)
        with _LOCK:
            _CALLS[identity] = (reference, index, witness, prepared,
                                fingerprint, token, symbol, scope_fingerprint,
                                tuple(selector_reads))
        return result

    with _LOCK:
        _PRODUCER_CALLABLES[symbol] = (function, code, invoke)
    return invoke


def _indexed_spans(index):
    """One linear neutral span-membership census per immutable source index."""
    key = ("reader_claims", "indexed_spans")
    cached = index._call_memo.get(key)
    if cached is not None:
        return cached
    pending, visited, spans = [index], set(), set()
    while pending:
        value = pending.pop()
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            continue
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(value, SourceSpan):
            spans.add(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    result = frozenset(spans)
    index._call_memo[key] = result
    return result


def _closed_witness(witness):
    from .diffusion_bookends import DiffusionBookendClaimWitness
    from .diffusion_root import DiffusionRootClaimWitness
    from .diffusion_stack import DiffusionStackClaimWitness
    from .diffusion_block import DiffusionBlockClaimWitness
    from .diffusion_stream import DiffusionStreamClaimWitness
    from .diffusion_conditioning import DiffusionConditioningClaimWitness
    from .position_linear_bias import AlibiClaimWitness
    from .cross_attention_schedule import AdditiveCrossClaimWitness
    from .ffn_mechanism import FFNMechanismClaimWitness
    from .ffn_width import FFNWidthClaimWitness
    from .expert_storage import ExpertStorageClaimWitness
    from .expert_width import ExpertWidthClaimWitness
    from .cell_topology import CellTopologyClaimWitness
    from .mixer_schedule import MixerScheduleClaimWitness
    from .ffn_schedule import FFNScheduleClaimWitness
    from .final_bookend import FinalNormClaimWitness
    from .decoder_norm import DecoderNormClaimWitness
    from .embedding_bookend import EmbeddingNormClaimWitness
    from .attention_mask import MaskExecutionClaimWitness
    from .position_schedule import PositionScheduleClaimWitness
    from .position_initialization import PositionInitializationClaimWitness
    from .attention_output import AttentionOutputClaimWitness
    from .attention import AttentionCacheClaimWitness, AttentionMechanismClaimWitness, AttentionSoftcapClaimWitness, AttentionClipClaimWitness
    from .attention_sinks import AttentionSinkClaimWitness
    from .attention_storage import AttentionStorageClaimWitness
    from .attention_geometry import AttentionGeometryClaimWitness
    from .projection_bias import AttentionBiasClaimWitness
    from .qk_norm_schedule import QKNormScheduleClaimWitness
    from .kv_sharing_schedule import KVSharingClaimWitness
    from .codebook_streams import CodebookClaimWitness
    from .per_layer_side_input import PerLayerInputClaimWitness
    if type(witness) not in (
            DiffusionBookendClaimWitness, DiffusionRootClaimWitness, DiffusionStackClaimWitness,
            DiffusionBlockClaimWitness, DiffusionStreamClaimWitness, DiffusionConditioningClaimWitness,
            AlibiClaimWitness, AdditiveCrossClaimWitness,
            FFNMechanismClaimWitness, FFNWidthClaimWitness, ExpertStorageClaimWitness,
            ExpertWidthClaimWitness, CellTopologyClaimWitness, MixerScheduleClaimWitness,
            FFNScheduleClaimWitness, FinalNormClaimWitness, DecoderNormClaimWitness,
            EmbeddingNormClaimWitness, MaskExecutionClaimWitness,
            PositionScheduleClaimWitness, PositionInitializationClaimWitness,
            AttentionOutputClaimWitness, AttentionCacheClaimWitness,
            AttentionMechanismClaimWitness, AttentionSoftcapClaimWitness, AttentionClipClaimWitness,
            AttentionSinkClaimWitness, AttentionStorageClaimWitness,
            AttentionGeometryClaimWitness, AttentionBiasClaimWitness,
            QKNormScheduleClaimWitness, KVSharingClaimWitness,
            CodebookClaimWitness, PerLayerInputClaimWitness):
        raise TypeError("claim projection requires a closed producer witness")


@dataclass(frozen=True)
class ReaderProjectionClaimProof:
    fact_id: str
    reader_result: object = field(repr=False, compare=False)
    index: ProgramIndex = field(repr=False, compare=False)
    prepared_document: object = field(repr=False, compare=False)
    _projection: ReaderFactProjection = field(init=False, repr=False, compare=False)
    _value_status_hash: str = field(init=False, repr=False, compare=False)
    _required_spans: tuple[SourceSpan, ...] = field(init=False, repr=False, compare=False)

    proof_kind = "retained_reader_projection"

    def _binding(self):
        result = self.reader_result
        _closed_witness(result.claim_witness)
        with _LOCK:
            binding = _CALLS.get(id(result))
        if binding is None or binding[0]() is not result:
            raise ValueError("claim needs the actual result emitted by its reader")
        _, index, witness, document, fingerprint, token, _symbol, scope_fingerprint, _reads = binding
        _validate_producer_callable(_symbol)
        if index is not self.index or witness is not result.claim_witness:
            raise ValueError("claim carries another source index or witness")
        if document is not self.prepared_document or document is None \
                or not verify_prepared_document_token(document, fingerprint, token) \
                or _prepared_scope_fingerprint(document) != scope_fingerprint:
            raise ValueError("claim belongs to another or changed prepared document")
        return binding

    def projection(self):
        self._binding()
        return self._projection

    def __post_init__(self):
        binding = self._binding()
        _checked_selector_reads(binding[8], self.prepared_document)
        owner, _, key = self.fact_id.rpartition(".")
        projection = self.reader_result.claim_witness.project(
            owner, key, self.prepared_document)
        if not isinstance(projection, ReaderFactProjection) \
                or (projection.owner, projection.key) != (owner, key):
            raise ValueError("reader did not prove this finite fact projection")
        required_spans = projection.required_spans
        if required_spans is None:
            required_spans = tuple(dict.fromkeys(
                span for origin in self.reader_result.provenance for span in origin.spans))
        elif not isinstance(required_spans, tuple) or not required_spans \
                or any(not isinstance(span, SourceSpan) for span in required_spans) \
                or len(set(required_spans)) != len(required_spans):
            raise ValueError("finite question evidence requires non-empty unique typed spans")
        if projection.required_spans is not None and not set(required_spans) <= _indexed_spans(self.index):
            raise ValueError("finite question evidence contains a span absent from its source index")
        object.__setattr__(self, "_required_spans", required_spans)
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(self, "_value_status_hash", value_status_hash(
            projection.value, projection.status))

    @property
    def claim_kind(self):
        return self.projection().claim_kind

    @property
    def reader_symbols(self):
        return (self.reader_result.claim_witness.reader_symbol,)

    @property
    def document_token(self):
        return self._binding()[5]

    def validate_fact(self, fact):
        projection = self.projection()
        if fact.claim_document_token != self.document_token:
            raise ValueError("reader fact carries another document seal")
        if fact.status != projection.status \
                or fact.completeness != projection.completeness \
                or value_status_hash(fact.value, fact.status) != self._value_status_hash:
            raise ValueError("reader fact differs from its whole typed projection")
        if set(fact.config_paths) != {
                ".".join(path) for path in projection.config_paths}:
            raise ValueError("reader fact omits or changes decisive config paths")
        required = {
            (span.source.component_key or "root", span.source.canonical_path,
             span.line)
            for span in self._required_spans
        }
        cited = {(span.component, span.file, span.line) for span in fact.source_spans}
        if not required or not required <= cited:
            raise ValueError("reader fact omits its retained source evidence")

    def summary(self):
        from .claim_evidence import ClaimProofSummary
        projection = self.projection()
        binding = self._binding()
        refs = {f"projection:{projection.owner}.{projection.key}:"
                f"{self._value_status_hash}",
                *(f"source:{span.source.component_key or 'root'}:"
                  f"{span.source.content_fingerprint}:{span.line}:{span.col}:"
                  f"{span.end_line}:{span.end_col}"
                  for span in self._required_spans)}
        return ClaimProofSummary(
            self.fact_id, projection.claim_kind, self.proof_kind,
            self.reader_symbols, tuple(sorted(refs)),
            document_fingerprints=(binding[4],),
            index_fingerprints=(portable_source_index_fingerprint(self.index),))


def qualify_reader_fact(fact, reader_result, index, prepared_document):
    """Attach only a migrated reader's exact finite projection, or abstain."""
    if reader_result is None or reader_result.claim_witness is None:
        return fact
    proof = ReaderProjectionClaimProof(
        fact.ledger_key(), reader_result, index, prepared_document)
    return replace(fact, claim_kind=proof.claim_kind,
                   claim_readers=proof.reader_symbols, claim_evidence=proof,
                   claim_document_token=proof.document_token)


def reader_requires_claim_scope(result):
    """A marker only requests scope validation; it never grants authority."""
    return result.claim_witness is not None or bool(result.claim_reader_symbol)


def reader_claim_scope_matches(result, index, prepared_document):
    """Whether a cached migrated result was actually read in this document.

    A pre-parse unbound call remains a valid reader result, but the parse must
    run it once at its document boundary before using it as qualified evidence.
    """
    if not reader_requires_claim_scope(result):
        return True
    with _LOCK:
        binding = _CALLS.get(id(result))
    return bool(binding is not None and binding[0]() is result
                and binding[1] is index and binding[2] is result.claim_witness
                and binding[3] is prepared_document and prepared_document is not None
                and verify_prepared_document_token(
                    prepared_document, binding[4], binding[5])
                and _prepared_scope_fingerprint(prepared_document) == binding[7])


def retained_reader_attempt(result):
    """Return authenticated attempt identity, including an actual failed read.

    Consumers retain the original result and call this validator; serializing
    this tuple does not make a reconstructed ReaderResult authentic.
    """
    with _LOCK:
        binding = _CALLS.get(id(result))
    if binding is None or binding[0]() is not result:
        raise ValueError("investigation requires an actual retained reader invocation")
    _reference, index, witness, document, fingerprint, token, symbol, scope, reads = binding
    _validate_producer_callable(symbol)
    if document is not None and (
            not verify_prepared_document_token(document, fingerprint, token)
            or _prepared_scope_fingerprint(document) != scope):
        raise ValueError("investigation document changed after its invocation")
    if document is not None:
        _checked_selector_reads(reads, document)
    return (symbol, index, document, result.owner)


def validate_reader_investigation(result, *, owner, key, claim_kind):
    """Bind a failed attempt to its finite producer-declared fact vocabulary.

    Unmigrated failure vocabularies remain investigation-missing; an authentic
    read of one question cannot manufacture exhaustion of another question.
    """
    from .final_bookend import FinalNormClaimWitness
    from .decoder_norm import DecoderNormClaimWitness
    from .embedding_bookend import EmbeddingNormClaimWitness
    symbol, index, document, occurrence = retained_reader_attempt(result)
    if result.status not in {"failed", "ambiguous", "incomplete", "absent"}:
        raise ValueError("resolved reader evidence is not an exhaustion result")
    if document is None:
        raise ValueError("reader investigation has no exact prepared-document binding")
    if occurrence is None:
        raise ValueError("reader investigation has no exact source owner")
    if symbol != FinalNormClaimWitness.reader_symbol \
            or (owner, key, claim_kind) not in FinalNormClaimWitness.attempted_projections:
        raise ValueError("reader did not attempt this finite architectural claim")
    return (symbol, index, document, occurrence)
