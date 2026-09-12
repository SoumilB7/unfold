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



def _constructor_keeps_default(index, alias):
    """Refuse relevant direct constructor writes; no helper-effect theorem.

    Root graph bindings name the loader actual, but do not include the local
    write invalidations collected at child construction sites. Inspect that
    exact constructor's existing binding census, retaining possible local
    aliases conservatively across guards/reassignments.
    """
    from .component_owner import _arg_config_flow, _self_attribute_chain, _paths_overlap
    from .program_index import SymbolId
    constructor = SymbolId(alias.owner_symbol.source,
                           alias.owner_symbol.qualified_name + '.__init__')
    environment = {alias.constructor_parameter: ((),)}

    def paths(expression):
        if expression is None:
            return ()
        if expression.kind == 'attribute' and expression.name == '__dict__' and expression.children:
            return paths(expression.children[0])
        direct = _arg_config_flow(expression, environment)
        if direct is not None:
            return direct.prefixes
        stored = _self_attribute_chain(expression)
        if stored and stored[0] == alias.stored_field:
            return (tuple(stored[1:]),)
        if expression.kind == 'subscript' and expression.children:
            bases = paths(expression.children[0])
            if not bases:
                return ()
            key = expression.children[1] if len(expression.children) == 2 else None
            if key is not None and key.kind == 'constant' and type(key.const_value) is str:
                return tuple((*base, key.const_value) for base in bases)
            return bases  # Unknown key can overwrite any field of this object.
        return ()

    def namespace(expression):
        return (expression is not None and expression.kind == 'attribute'
                and expression.name == '__dict__' and expression.children
                and bool(paths(expression.children[0])))

    def contains_namespace(expression):
        if expression is None:
            return False
        if namespace(expression):
            return True
        return expression.kind in {'tuple', 'list', 'dict', 'set', 'ifexp', 'boolop'} and any(
            contains_namespace(child) for child in (*expression.children,
                *(value for _key, value in expression.keyword_children)))

    bindings = tuple(index.bindings_in(constructor))

    def leaves(target):
        if target.kind in {'tuple', 'list', 'starred'}:
            return tuple(leaf for child in target.children for leaf in leaves(child))
        return (target,)

    def transported_config(expression):
        if expression is None:
            return False
        if () in paths(expression):
            return True
        return expression.kind in {'tuple', 'list', 'dict', 'set'} and any(
            transported_config(child) for child in expression.children)

    # Collect possible aliases before testing writes. This is deliberately
    # conservative across branches and chained assignment target ordering.
    # No aliases are invented for arbitrary returned/helper values.
    for _ in range(len(bindings) + 1):
        changed = False
        for binding in bindings:
            if contains_namespace(binding.value):
                return False  # Namespace alias/transport is not an admitted config relay.
            if not transported_config(binding.value):
                continue
            if binding.value.kind in {'tuple', 'list', 'dict', 'set'}:
                return False  # Container transport has no exact config alias binding.
            if any(target.kind != 'name' for target in binding.targets):
                return False  # Untracked field/container/destructuring transport.
            for target in binding.targets:
                if target.name and target.name not in environment:
                    environment[target.name] = ((),)
                    changed = True
        if not changed:
            break
    for binding in bindings:
        for target in (leaf for target in binding.targets for leaf in leaves(target)):
            if target.kind == 'name':
                if target.name == alias.constructor_parameter:
                    return False
                continue
            if any(_paths_overlap(path, ('hidden_size',)) for path in paths(target)):
                return False

    # The neutral name-access census includes delete targets and nested loop
    # bodies, which the binding/statement tables do not fully represent.
    init = index.callable_by_symbol(constructor)
    if init is None or init.span is None:
        return False
    for module in index.modules:
        if module.source.source_id != constructor.source:
            continue
        for access in module.name_accesses:
            if not ((init.span.line, init.span.col) <= (access.span.line, access.span.col)
                    < (init.span.end_line, init.span.end_col)):
                continue
            # calls_in observes assigned values, not calls embedded in store
            # targets. The full name census retains their exact parent Call:
            # vars(config)[key] = value must cross the same namespace boundary.
            parent = access.parent_expression
            if parent is not None and parent.kind == 'call' and parent.children \
                    and parent.children[0].name == 'vars' and any(
                        paths(argument) for argument in parent.children[1:]):
                return False
            if access.context not in {'store', 'del'}:
                continue
            if access.expression.kind == 'name':
                if access.name == alias.constructor_parameter:
                    return False
                continue
            if any(_paths_overlap(path, ('hidden_size',)) for path in paths(access.expression)):
                return False

    # Ordinary explicit attribute mutation syntax is not a helper-effect
    # interpreter. Refuse direct setattr/delattr protocols on this config,
    # including unknown property names that may name the selected operand.
    for call in index.calls_in(constructor):
        callee = call.callee
        # Namespace dictionaries are mutable protocol objects. Refuse their
        # whole direct-call boundary, including vars(config), mapping mutators
        # and unknown namespace methods/escapes; do not infer helper effects.
        if callee.name == 'vars' and any(paths(argument) for argument in call.args):
            return False
        if callee.kind == 'attribute' and callee.children and namespace(callee.children[0]):
            return False
        if any(contains_namespace(argument) for argument in (
                *call.args, *(value for _key, value in call.kwargs))):
            return False
        if callee.name not in {'setattr', 'delattr', '__setattr__', '__delattr__'}:
            continue
        receiver = paths(callee.children[0]) if callee.kind == 'attribute' and callee.children else ()
        args = call.args
        if not receiver and args:
            receiver, args = paths(args[0]), args[1:]
        if not receiver:
            continue
        key = args[0] if args else None
        suffix = (key.const_value,) if key is not None and key.kind == 'constant' \
            and type(key.const_value) is str else ()
        if any(_paths_overlap((*base, *suffix), ('hidden_size',)) for base in receiver):
            return False
    return True


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
    if not _constructor_keeps_default(index, alias.value):
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
