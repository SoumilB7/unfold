"""Finite scalar default declaration for the existing model.hidden_size consumer.

The source-bound config class supplies the VALUE. This makes no claim about
embedding execution, forward connections, parameter ownership, or tensor effects.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import resolve_component_root
from .framework_config import (
    framework_config_alias, framework_config_class, framework_config_class_default,
)
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
from .reader_claims import ReaderFactProjection, reader_operand, retained_claim_reader
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class ModelHiddenSizeDefault:
    value: int
    owner: object
    alias: object
    config_class: object
    declaration: object
    spans: tuple[SourceSpan, ...]


def _declaration(index, bundle, config_path, document):
    # This first finite scalar migration covers the root consumer only. A
    # child needs its own exact selected config-class binding; no root fallback.
    from .document import PreparedDocument
    if config_path or not isinstance(document, PreparedDocument) or document.failure is not None:
        return None
    # The reader-facing document may differ from the checkpoint snapshot.
    # Both must show absence under the same canonical/alias arbitration.
    from ..everchanging import load_aliases
    from .config_access import _resolve_for_claim_proof
    checkpoint = _resolve_for_claim_proof(
        document.checkpoint, 'hidden_size', load_aliases().get('hidden_size', ()))
    if checkpoint.state != 'absent':
        return None
    root = resolve_component_root(index, bundle, 'root')
    if root.status != 'resolved':
        return None
    owner = root.graph.root.occurrence
    alias = framework_config_alias(index, root, owner)
    if alias.status != 'resolved' or alias.value.config_binding.resolved_prefix != ():
        return None
    config_class = framework_config_class(index, alias.value)
    if config_class.status != 'resolved':
        return None
    declaration = framework_config_class_default(index, config_class.value, ('hidden_size',))
    if declaration.status != 'resolved' or type(declaration.value.value) is not int \
            or declaration.value.value <= 0:
        return None
    # Shared resolver retains checkpoint/explicit-null/conflicting-alias
    # precedence. An overlay entry alone is not enough to qualify this value.
    try:
        operand = reader_operand(document, ('hidden_size',))
    except (ValueError, KeyError):
        return None
    if operand.source_kind != 'class_default' or operand.checkpoint_path is not None \
            or type(operand.value) is not int or operand.value != declaration.value.value:
        return None
    if document.class_overlay.get('hidden_size') != declaration.value.value:
        return None
    spans = tuple(dict.fromkeys(span for result in (alias, config_class, declaration)
                                for origin in result.provenance for span in origin.spans))
    if not spans or any(not isinstance(span, SourceSpan) for span in spans):
        return None
    return ModelHiddenSizeDefault(operand.value, owner, alias.value, config_class.value,
                                  declaration.value, spans)


@dataclass(frozen=True)
class ModelHiddenSizeDefaultClaimWitness:
    index: ProgramIndex
    bundle: SourceBundle
    config_path: tuple[str, ...]
    document: object
    default: ModelHiddenSizeDefault
    reader_symbol = 'model_unfolder.evidence.class_default_value.model_hidden_size_class_default'

    def _checked(self):
        actual = _declaration(self.index, self.bundle, self.config_path, self.document)
        if actual is None or actual != self.default:
            raise ValueError('hidden-size default differs from its exact source class/document declaration')
        return actual

    def validate_result(self, result):
        default = self._checked()
        if result.status != 'resolved' or result.value is not self.default or result.owner != default.owner:
            raise ValueError('default result differs from its actual retained scalar declaration')
        if not set(default.spans) <= {span for origin in result.provenance for span in origin.spans}:
            raise ValueError('default result omitted its exact class/default source provenance')

    def project(self, owner, key, document):
        if (owner, key) != ('model', 'hidden_size') or document is not self.document:
            raise ValueError('this scalar default proves only its actual model.hidden_size value')
        default = self._checked()
        return ReaderFactProjection(owner, key, 'value', default.value, 'class_default', (),
                                    completeness='complete', required_spans=default.spans)


@retained_claim_reader
def model_hidden_size_class_default(index, bundle, config_path):
    from .config_access import current_prepared_document
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle) \
            or type(config_path) is not tuple or any(type(item) is not str or not item for item in config_path):
        raise TypeError('scalar default requires exact source index, source bundle and selected path')
    document = current_prepared_document.get()
    default = _declaration(index, bundle, config_path, document)
    if default is None:
        return ReaderResult.failed(None, (ReaderFailure('unsupported_syntax',
            'no exact own-class hidden-size literal default for this selected document/absence'),))
    return ReaderResult.resolved(default.owner, default,
        claim_witness=ModelHiddenSizeDefaultClaimWitness(index, bundle, config_path, document, default),
        provenance=(ReaderProvenance('source', spans=default.spans,
            detail='exact annotated config class own-literal scalar default; no execution/connection claim'),))
