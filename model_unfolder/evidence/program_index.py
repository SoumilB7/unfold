"""U3 — the ONE raw program index (master plan §20.6; docs/U3_RUNBOOK.md).

This module is the TYPE LAYER only.  The walker that populates these records
and the immutable assembly that freezes them per SourceBundle (both U3-A) are
built ON this schema — never before it.

OBSERVATION-ONLY (binding boundary 1): every record here stores an AST fact —
a symbol, a span, a normalized expression, a construction candidate — and
NEVER decides what a class IS ("this is the FFN", "gated", best candidate).
Interpretation belongs to owner-bound readers wrapping their domain evidence
in ``ReaderResult[T]`` (U3-C); owner resolution belongs to the ComponentOwner
resolver (U3-B).

Qualified identity, never a bare name (2026-07-19 schema contract).  A class or
function name is NOT authoritative identity: two modules can define the same
class name, and the same class can be constructed at two sites for two roles.
Identity is therefore always qualified:

    SourceId            path + CONTENT fingerprint + component + external prov.
    SymbolId            a SourceId + a qualified dotted name within it
    SourceSpan          a SourceId + (line, col, end_line, end_col)
    ConstructionSiteId  owner SymbolId + enclosing-callable SymbolId + span +
                        the assignment/slot ORDINAL (two ctor calls on one line)

Identity is a DETERMINISTIC CONTENT FINGERPRINT (boundary 3): two files with
equal bytes share identity; a content change with a preserved mtime MUST
invalidate.  The index unit of ownership is the CONSTRUCTION-SITE OCCURRENCE
(boundary 2): the same class instantiated twice for different roles is two
occurrences; a ComponentOwner (U3-B) resolves the chain ``parent owner ->
construction site -> field/slot -> child class``.

Query-bearing expressions are NEVER stored as strings.  Every assignment,
return, kwarg, guard, control, call and dataflow edge carries a normalized
:class:`ExprNode`; a reader queries that STRUCTURE and may never read semantics
out of ``ast.unparse()`` or a substring search over the diagnostic segment.

Construction is honest.  A :class:`ConstructionSite` records ZERO, ONE, or
SEVERAL proof-bearing child :class:`ChildCandidate` edges — zero for dynamic
construction, several for rivals — and the walker NEVER collapses them to one
``child_class``.  A :class:`ConflictRecord` is NOT emitted merely because the
walker sees several candidates; the raw index records the candidates and the
U3-B resolver emits the typed rival-owner / config-prefix conflict when it
cannot prove uniqueness.

Config reads are reported as PATHS only (boundary 4): the syntactic config-root
binding plus exact typed path segments, with dynamic/unresolved segments made
explicit; the VALUE resolves through the owner-qualified U1 ledger and is
consumed into a fact by the reader.  The index never reads checkpoints.

External library files (a shared diffusers attention module) enter as
provenance-bearing EXTERNAL :class:`SourceFileNode`s (boundary 8), never blended
into the model bundle.  Read/decode/parse failures are :class:`ParseFailure`
records; a HEALTHY file that merely contains unsupported dynamic syntax stays
indexed and carries an :class:`UnsupportedSyntaxRecord` (never a skip).  The ONE
immutable index lives call-local on the ParseContext/SourceBundle (boundary 7);
any cross-parse cache must be content-fingerprint keyed and immutable.
"""
from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

def content_fingerprint(text: str) -> str:
    """Deterministic identity of one source file's CONTENT (never mtime)."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class SourceId:
    """The stable qualified identity of one source file.

    ``content_fingerprint`` is over CONTENT (boundary 3).  ``component_key`` is
    the bundle component ('root', 'text_config', 'text_encoder.text_config', ...)
    for internal files and MUST be None for external files.  ``external`` +
    ``external_provenance`` describe a file outside the model bundle (boundary 8);
    provenance is HOW it entered (e.g. 'diffusers.models.attention').  The
    internal/external invariants are enforced where this identity becomes a
    :class:`SourceFileNode`.
    """

    canonical_path: str
    content_fingerprint: str
    component_key: str | None = None
    external: bool = False
    external_provenance: str = ""


@dataclass(frozen=True)
class SymbolId:
    """A source identity + a qualified dotted name within it.

    ``qualified_name`` is the dotted path INSIDE the module: 'LlamaModel',
    'LlamaModel.__init__', 'LlamaAttention.forward', or a free function name.
    The (SourceId, qualified_name) pair is authoritative; the bare name is not.
    """

    source: SourceId
    qualified_name: str


@dataclass(frozen=True)
class SourceSpan:
    """Exact source location: a source identity + full line/column extent."""

    source: SourceId
    line: int
    col: int = 0
    end_line: int = 0
    end_col: int = 0


@dataclass(frozen=True)
class ConstructionSiteId:
    """Identity of one construction OCCURRENCE (boundary 2).

    Two constructions on the SAME source line are disambiguated by ``ordinal``
    (assignment / container-slot position), so a site is uniquely identified even
    when line+column collide.
    """

    owner: SymbolId               # the class whose __init__/helper constructs
    enclosing_callable: SymbolId  # the exact method/function it happens in
    span: SourceSpan
    ordinal: int = 0


# --------------------------------------------------------------------------- #
# Source nodes, parse outcomes, modules
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceFileNode:
    """One indexed file as a provenance-bearing node.

    Invariants (Soumil 2026-07-19) enforced in ``__post_init__``: an internal
    node requires a component key and carries no external provenance; an
    external node requires non-empty provenance and carries no component key;
    contradictory internal/external metadata raises.  The content-vs-fingerprint
    match is an ASSEMBLY-time invariant (:meth:`verify_content`), checked when
    the assembler supplies the actual source bytes.
    """

    source_id: SourceId

    def __post_init__(self) -> None:
        sid = self.source_id
        if sid.external:
            if not sid.external_provenance:
                raise ValueError(
                    f"external source node requires non-empty provenance: "
                    f"{sid.canonical_path!r}")
            if sid.component_key:
                raise ValueError(
                    f"external source node must not carry a bundle component "
                    f"key: {sid.canonical_path!r} (component={sid.component_key!r})")
        else:
            if not sid.component_key:
                raise ValueError(
                    f"internal source node requires a component key: "
                    f"{sid.canonical_path!r}")
            if sid.external_provenance:
                raise ValueError(
                    f"internal source node must not carry external provenance: "
                    f"{sid.canonical_path!r}")

    def verify_content(self, text: str) -> None:
        """Assembly-time invariant (boundary 3): the recorded fingerprint MUST
        match the actual source content, so a stale index is impossible."""
        actual = content_fingerprint(text)
        if actual != self.source_id.content_fingerprint:
            raise ValueError(
                f"fingerprint mismatch for {self.source_id.canonical_path!r}: "
                f"recorded {self.source_id.content_fingerprint[:12]}… != "
                f"content {actual[:12]}…")


@dataclass(frozen=True)
class ParseFailure:
    """A file the index could not READ/DECODE/PARSE — never a silent skip.

    This is ONLY for read/decode/AST-parse failures.  A healthy file that merely
    contains unsupported dynamic syntax is still parsed and indexed; that carries
    an :class:`UnsupportedSyntaxRecord` instead.
    """

    source: SourceId
    kind: str                    # read_error | decode_error | syntax_error
    detail: str


@dataclass(frozen=True)
class UnsupportedSyntaxRecord:
    """A healthy, indexed file contains a construct the walker cannot normalize
    (dynamic dispatch, star-construction, a dynamic default, ...).  The file
    stays fully usable; this record marks the one spot that is unresolved."""

    owner: SymbolId | None
    enclosing_callable: SymbolId | None
    syntax_kind: str
    detail: str
    span: SourceSpan


@dataclass(frozen=True)
class ModuleRecord:
    """One module: its module/package identity and its source node."""

    module: str                  # dotted module name as resolved
    package: str                 # dotted package (containing package), '' if top
    source: SourceFileNode
    span: SourceSpan


# --------------------------------------------------------------------------- #
# Normalized expressions (boundary: no query-bearing strings)
# --------------------------------------------------------------------------- #

#: The structural expression kinds a reader may query.  ``unsupported`` marks an
#: expression the walker could not normalize (a reader treats it as opaque, never
#: guesses from the diagnostic segment).
EXPR_KINDS = frozenset({
    "name", "attribute", "constant", "call", "subscript", "slice",
    "tuple", "list", "dict", "set", "boolop", "binop", "unaryop",
    "compare", "ifexp", "comprehension", "starred", "lambda", "fstring",
    "unsupported",
})


@dataclass(frozen=True)
class ExprNode:
    """A frozen, normalized expression.  Readers query this STRUCTURE; they may
    never read semantics out of ``ast.unparse()`` or a substring search over
    ``source_segment`` (which is DIAGNOSTIC ONLY).

    Field use by ``kind``:
      * name        -> ``name`` = identifier
      * attribute   -> ``name`` = attr, ``children`` = (value,)
      * constant    -> ``const_value`` = the literal (int/str/bool/None/…);
                       ``kind == 'constant'`` with ``const_value is None`` IS a
                       literal ``None`` (distinct from "field absent")
      * call        -> ``children[0]`` = callee, ``children[1:]`` = positional
                       args, ``keyword_children`` = (name, ExprNode) kwargs
      * subscript   -> ``children`` = (value, index)
      * slice       -> ``children`` = (lower?, upper?, step?)
      * tuple/list/set -> ``children`` = elements
      * dict        -> ``keyword_children`` = ('' or key-repr, value) pairs and
                       ``children`` mirrors keys as ExprNodes
      * boolop/binop/unaryop/compare -> ``operator`` = token, ``children`` =
                       operands
      * ifexp       -> ``children`` = (body, test, orelse)
      * comprehension -> ``children`` = (element, *generators)
      * unsupported -> opaque; ``source_segment`` retained for diagnostics only
    """

    kind: str
    name: str = ""
    const_value: object = None
    operator: str = ""
    children: tuple = ()          # tuple[ExprNode]
    keyword_children: tuple = ()  # tuple[(str, ExprNode)]
    span: SourceSpan | None = None
    source_segment: str = ""      # DIAGNOSTIC ONLY — never a query surface


# --------------------------------------------------------------------------- #
# Callables and their parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ParamRecord:
    """One parameter.  The three default cases are distinguished (Soumil):
      * NO default            -> ``has_default is False`` (``default`` ignored)
      * literal ``None``      -> ``has_default is True`` and ``default`` is an
                                 ExprNode ``kind == 'constant'`` / value None
      * dynamic/unsupported   -> ``has_default is True`` and ``default.kind ==
                                 'unsupported'``
    """

    name: str
    has_default: bool = False
    default: ExprNode | None = None
    kind: str = "positional"     # positional | keyword_only | vararg | kwarg


@dataclass(frozen=True)
class GuardStep:
    """One static branch/loop step in an enclosing control/guard chain.  The
    controlling expression is STRUCTURAL (ExprNode), never a string; ``test`` is
    None for a bare ``else``."""

    kind: str                    # if | elif | else | for | while | comprehension
    test: ExprNode | None
    span: SourceSpan


@dataclass(frozen=True)
class CallableRecord:
    """One method or free function: qualified symbol, enclosing class (None for a
    free function), typed params, returned expressions, and the OBSERVED
    ``self.<name>`` call targets (raw names — the U3-B resolver binds these to
    concrete :class:`SymbolId`s via the MRO; they are not authoritative here)."""

    symbol: SymbolId
    owner: SymbolId | None
    kind: str                    # method | function
    params: tuple = ()           # tuple[ParamRecord]
    returns: tuple = ()          # tuple[ExprNode]
    self_method_calls: tuple = ()  # tuple[str] — observed helper-fold targets
    span: SourceSpan | None = None


# --------------------------------------------------------------------------- #
# Exact callable-scoped identifiers
# --------------------------------------------------------------------------- #

IDENTIFIER_CONTEXTS = frozenset({
    "parameter", "load", "store", "del",
    "annotation", "default", "decorator",
})


@dataclass(frozen=True)
class IdentifierObservation:
    """One exact ``ast.Name``/``ast.arg`` occurrence in one callable.

    This is neutral syntax, not a role or mechanism claim.  ``context`` records
    where the spelling occurred so a specialized reader can distinguish a
    runtime load/store from a parameter, annotation, default or decorator.
    Nested named callables/classes and lambdas are separate lexical scopes and
    are deliberately excluded from their parent's census; a reader requiring
    transitive behavior must resolve/follow that callable explicitly.

    Python stores a few bindings (for example ``except ... as name`` and match
    pattern captures) as plain strings rather than ``ast.Name`` nodes.  They are
    not part of this record family's contract: inventing a token span for them
    would violate the exact-span law.  Add a separately proven token-aware
    observation if a future reader genuinely needs those bindings.
    """

    owner: SymbolId | None
    enclosing_callable: SymbolId
    name: str
    context: str
    span: SourceSpan

    def __post_init__(self) -> None:
        if self.owner is not None and not isinstance(self.owner, SymbolId):
            raise TypeError("identifier owner must be a SymbolId or None")
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("identifier enclosing_callable must be a SymbolId")
        if not self.name:
            raise ValueError("an identifier observation requires a spelling")
        if self.context not in IDENTIFIER_CONTEXTS:
            raise ValueError(f"unknown identifier context {self.context!r}")
        if not isinstance(self.span, SourceSpan):
            raise TypeError("an identifier observation requires an exact SourceSpan")
        if self.span.source != self.enclosing_callable.source:
            raise ValueError("identifier span must belong to its callable source")
        if self.owner is not None and \
                self.owner.source != self.enclosing_callable.source:
            raise ValueError("identifier owner and callable must share a source")


# --------------------------------------------------------------------------- #
# Classes, imports, dispatch registries
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassBodyAssign:
    """A class-BODY assignment (config_class = ..., is_causal = True, ...)."""

    attr: str
    value: ExprNode
    span: SourceSpan


@dataclass(frozen=True)
class ClassRecord:
    """One class: qualified symbol, base-class REFERENCE expressions (ExprNode,
    never bare strings), and class-body assignments."""

    symbol: SymbolId
    bases: tuple = ()            # tuple[ExprNode]
    body_assigns: tuple = ()     # tuple[ClassBodyAssign]
    span: SourceSpan | None = None


@dataclass(frozen=True)
class ImportRecord:
    """One import binding: the local alias and the dotted import target as
    written.  (An import target IS a name binding, not a query-bearing
    expression; the alias->target map is what the resolver later uses to turn a
    construction reference into an external :class:`SourceId`.)"""

    source: SourceId
    alias: str                   # local name bound in the module
    target: str                  # dotted module or module.symbol imported
    span: SourceSpan
    guard: tuple = ()            # tuple[GuardStep] for module-scope conditional imports

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceId):
            raise TypeError("an import is qualified by its SourceId")
        if not self.alias or not self.target:
            raise ValueError("an import carries exact non-empty binding names")
        if not isinstance(self.span, SourceSpan) or self.span.source != self.source:
            raise ValueError("an import carries an exact same-source span")
        if any(not isinstance(step, GuardStep) for step in self.guard):
            raise TypeError("an import guard is a tuple of GuardStep values")


@dataclass(frozen=True)
class ModuleBindingObservation:
    """One name bound in a module's lexical scope.

    This is neutral Python name-resolution evidence.  It exists so a later
    protocol reader can prove that an unqualified builtin reference was not
    shadowed, rather than treating a familiar spelling as semantic evidence.
    The collector is deliberately conservative: conditional/deleted bindings
    are still reported because either can prevent a sound builtin proof.
    """

    source: SourceId
    name: str
    kind: str
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceId):
            raise TypeError("a module binding is qualified by its SourceId")
        if not self.name:
            raise ValueError("a module binding carries a non-empty name")
        if self.kind not in {
                "import", "class", "function", "store", "delete",
                "exception", "pattern"}:
            raise ValueError(f"unknown module-binding kind {self.kind!r}")
        if not isinstance(self.span, SourceSpan) or self.span.source != self.source:
            raise ValueError("a module binding carries an exact same-source span")


@dataclass(frozen=True)
class DispatchRegistryRecord:
    """A module-level literal dict registry (ATTENTION_CLASSES = {...}).  Keys
    and values are structural ExprNodes (a value is typically a class-reference
    Name); resolution of a ``REGISTRY[key](...)`` construction to candidates is
    the walker's job, uniqueness is the resolver's."""

    symbol: SymbolId             # module-level registry name
    entries: tuple = ()          # tuple[(ExprNode key, ExprNode value)]
    span: SourceSpan | None = None


# --------------------------------------------------------------------------- #
# Construction occurrences (boundary 2) — honest candidate edges
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChildCandidate:
    """One statically-resolved child-class candidate for a construction site.

    ``reference`` is the class reference AS WRITTEN (structural).  ``symbol`` is
    the resolved child :class:`SymbolId` when — and only when — resolution is
    statically PROVABLE (a direct name bound to a class in the closure, a factory
    classmethod whose base is known, a registry value); otherwise it is None.
    ``provenance`` records HOW the candidate was derived — a factory candidate is
    a proof-bearing edge, never a guessed single name.
    """

    reference: ExprNode
    symbol: SymbolId | None
    provenance: str              # direct_name | attribute | alias_import:<alias>
    #                              | factory:<method> | registry_subscript:<name>
    #                              | base_of:<factory> | ...


@dataclass(frozen=True)
class ConstructionSite:
    """One place a class is constructed: the OCCURRENCE the owner chain walks.

    Records the exact site identity, owner + enclosing callable as qualified
    symbols, the target kind and field/slot, the raw constructor expression, the
    positional and keyword arguments (all ExprNode), the enclosing guard/control
    chain, and ZERO / ONE / SEVERAL proof-bearing :class:`ChildCandidate` edges.
    Zero candidates == dynamic construction; several == rivals.  The walker NEVER
    chooses one child class — uniqueness is the U3-B resolver's job.
    """

    site_id: ConstructionSiteId
    owner: SymbolId
    enclosing_callable: SymbolId
    target_kind: str             # field | slot | element | return | bare
    target: str                  # self.<field> name / slot marker / ''
    constructor: ExprNode        # raw constructor expression (structural)
    args: tuple = ()             # tuple[ExprNode]
    kwargs: tuple = ()           # tuple[(str, ExprNode)]
    guard: tuple = ()            # tuple[GuardStep]
    candidates: tuple = ()       # tuple[ChildCandidate] — 0=dynamic, >1=rivals
    via: str = "init"            # init | helper:<name> | factory:<name> |
    #                              modulelist | sequential | moduledict |
    #                              add_module
    span: SourceSpan | None = None


@dataclass(frozen=True)
class FieldAssignRecord:
    """One ``self.<field> = <expr>`` in __init__/helpers: the structural value
    expression plus the enclosing guard chain.  When the value constructs, the
    matching :class:`ConstructionSite` carries the candidate edges."""

    owner: SymbolId
    enclosing_callable: SymbolId
    field: str
    value: ExprNode              # structural, never a string
    guard: tuple = ()            # tuple[GuardStep]
    span: SourceSpan | None = None


@dataclass(frozen=True)
class ContainerElementsRecord:
    """ModuleList/Sequential/ModuleDict elements + the count controlling
    expression (range()/len(), as an ExprNode)."""

    owner: SymbolId
    enclosing_callable: SymbolId
    field: str
    kind: str                    # modulelist | sequential | moduledict
    elements: tuple = ()         # tuple[ConstructionSite]
    count: ExprNode | None = None
    span: SourceSpan | None = None


# --------------------------------------------------------------------------- #
# Behavioural observations (observation separate from resolution)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CallObservation:
    """One call within a callable, in LEXICAL order (not runtime order — a reader
    claims execution order only after evaluating a proven branch/path).  The
    enclosing control/guard path is preserved so that "call A textually precedes
    call B" is never mistaken for "A runs before B"."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    lexical_order: int           # textual position within the callable
    callee: ExprNode             # structural func expression
    receiver: ExprNode | None    # self.<field> receiver when called through one
    args: tuple = ()             # tuple[ExprNode]
    kwargs: tuple = ()           # tuple[(str, ExprNode)]
    guard: tuple = ()            # tuple[GuardStep] — enclosing control/guard path
    span: SourceSpan | None = None


@dataclass(frozen=True)
class AttributeAccessRecord:
    """One attribute read/write observation (.T, .weight chains, subscripts) as a
    structural target expression."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    target: ExprNode             # attribute/subscript chain (structural)
    mode: str                    # read | write
    span: SourceSpan | None = None


@dataclass(frozen=True)
class ConfigSegment:
    """One segment of a config path.  ``dynamic`` marks an unresolved/dynamic
    segment (``config[some_var]``); ``name`` is '' when dynamic."""

    name: str
    dynamic: bool = False


@dataclass(frozen=True)
class ConfigPathObservation:
    """Source reads config.<path> — the PATH only (boundary 4).  Carries the
    syntactic config-root binding, the exact typed path segments with dynamic
    segments made explicit, the owner callable and the span.  The VALUE resolves
    through the owner-qualified U1 ledger, never here; the index never reads a
    checkpoint."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    root_binding: ExprNode       # the config-root expression (Name 'config', …)
    segments: tuple = ()         # tuple[ConfigSegment]
    form: str = "attr"           # attr | subscript | act2fn | get_activation
    span: SourceSpan | None = None


@dataclass(frozen=True)
class ControlRecord:
    """One static branch/loop/comprehension + its controlling expression
    (structural)."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    kind: str                    # if | elif | else | for | while | comprehension
    controlling: ExprNode | None
    span: SourceSpan | None = None


@dataclass(frozen=True)
class DataflowObservation:
    """A raw definition-use / structural-relationship edge.  ``producer`` and
    ``consumer`` are structural expressions; ``op`` is the SYNTACTIC operator or
    call token as written.  This edge is NEVER labelled with an architectural
    role (attention, FFN, gating, projection) — that is a reader's job."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    producer: ExprNode
    consumer: ExprNode | None    # None == returned / terminal sink
    op: str = ""                 # syntactic operator/call token only
    span: SourceSpan | None = None


# --------------------------------------------------------------------------- #
# Execution-flow observations (U3-A2 kernel) — NEUTRAL SYNTAX ONLY.
# These records add exact statement/binding/loop/return/transfer identities so a
# LATER resolver can prove def-use and happens-before OUTSIDE the index.  The index
# itself NEVER emits SSA versions, architectural roles, resolved module owners,
# happens-before edges, "layer stack" labels, or semantic callee classifications.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StatementId:
    """Exact identity of one statement: enclosing callable + exact span + a
    same-span ordinal (two statements sharing a start position are disambiguated,
    mirroring :class:`ConstructionSiteId`)."""

    enclosing_callable: SymbolId
    span: SourceSpan
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a statement id is qualified by its enclosing callable")
        if self.span is None:
            raise ValueError("a statement id carries an exact span")
        if self.span.source != self.enclosing_callable.source:
            raise ValueError("a statement lives in its callable's source")
        if self.ordinal < 0:
            raise ValueError("a statement ordinal is non-negative")


@dataclass(frozen=True)
class CallSiteId:
    """Exact identity of one call site, DERIVED from a :class:`CallObservation`
    (enclosing callable + the call's exact span + its existing lexical order).  It
    does not re-walk calls; use :meth:`of`."""

    enclosing_callable: SymbolId
    span: SourceSpan
    lexical_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a call-site id is qualified by its enclosing callable")
        if self.span is None:
            raise ValueError("a call-site id carries the call's exact span")
        if self.span.source != self.enclosing_callable.source:
            raise ValueError("a call site lives in its callable's source")
        if self.lexical_order < 0:
            raise ValueError("a lexical order is non-negative")

    @classmethod
    def of(cls, call: "CallObservation") -> "CallSiteId":
        if not isinstance(call, CallObservation):
            raise TypeError("CallSiteId.of requires a CallObservation")
        if call.span is None:
            raise ValueError("the call observation carries no span")
        return cls(call.enclosing_callable, call.span, call.lexical_order)


@dataclass(frozen=True)
class BindingObservation:
    """One assignment as RAW SYNTAX (no SSA / no versioning): the exact target
    pattern(s) (a chained assignment keeps every target; destructuring is preserved
    structurally), the value expression, the assignment kind and the guard path."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    statement: StatementId
    targets: tuple               # tuple[ExprNode] — exact target patterns
    value: ExprNode | None       # None only for a bare AnnAssign (`x: int`)
    assignment_kind: str         # assign | annassign | augassign | walrus
    guard: tuple = ()            # tuple[GuardStep]
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a binding is qualified by its enclosing callable")
        if not isinstance(self.statement, StatementId):
            raise TypeError("a binding carries its StatementId")
        if self.statement.enclosing_callable != self.enclosing_callable:
            raise ValueError("a binding's statement shares its enclosing callable")
        if not self.targets:
            raise ValueError("a binding carries >=1 exact target pattern")
        if self.assignment_kind not in {"assign", "annassign", "augassign", "walrus"}:
            raise ValueError(f"unknown assignment kind {self.assignment_kind!r}")


@dataclass(frozen=True)
class LoopObservation:
    """One loop as raw syntax: the target pattern + iterable (a ``for``/async-for),
    or the test as ``iterable`` (a ``while``), plus guard path, body/else spans and
    the async flag.  Never a claim that the loop body runs — order is proven
    later."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    statement: StatementId
    kind: str                    # for | while
    target: ExprNode | None      # loop target pattern (for); None for while
    iterable: ExprNode | None    # iterable (for) / test (while)
    async_flag: bool = False
    guard: tuple = ()
    body_span: SourceSpan | None = None
    else_span: SourceSpan | None = None
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a loop is qualified by its enclosing callable")
        if not isinstance(self.statement, StatementId):
            raise TypeError("a loop carries its StatementId")
        if self.statement.enclosing_callable != self.enclosing_callable:
            raise ValueError("a loop's statement shares its enclosing callable")
        if self.kind not in {"for", "while"}:
            raise ValueError(f"unknown loop kind {self.kind!r}")
        if self.kind == "for" and self.target is None:
            raise ValueError("a for-loop carries an exact target pattern")


@dataclass(frozen=True)
class ReturnObservation:
    """One ``return`` statement: the value expression (or None) and the guard
    path."""

    owner: SymbolId | None
    enclosing_callable: SymbolId
    statement: StatementId
    value: ExprNode | None
    guard: tuple = ()
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a return is qualified by its enclosing callable")
        if not isinstance(self.statement, StatementId):
            raise TypeError("a return carries its StatementId")
        if self.statement.enclosing_callable != self.enclosing_callable:
            raise ValueError("a return's statement shares its enclosing callable")


@dataclass(frozen=True)
class ControlTransferObservation:
    """One control-transfer point that can break straight-line ordering:
    ``return`` | ``break`` | ``continue`` | ``raise`` | ``yield`` | ``yield_from``.
    (yield/yield_from are expressions; the rest are statements.)"""

    kind: str
    enclosing_callable: SymbolId
    guard: tuple = ()
    value: ExprNode | None = None
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("a control transfer is qualified by its enclosing callable")
        if self.kind not in {"return", "break", "continue", "raise", "yield", "yield_from"}:
            raise ValueError(f"unknown control-transfer kind {self.kind!r}")


@dataclass(frozen=True)
class UnsupportedExecutionRegion:
    """An execution construct NOT classified by the records above (match/case,
    with/async-with, try, await, dynamic dispatch, comprehension-built calls, ...).
    This is a published, explicitly non-exhaustive coverage gap.  Its absence is
    never a whole-callable completeness proof."""

    enclosing_callable: SymbolId
    construct_kind: str
    guard: tuple = ()
    span: SourceSpan | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.enclosing_callable, SymbolId):
            raise TypeError("an unsupported region is qualified by its enclosing callable")
        if not self.construct_kind:
            raise ValueError("an unsupported region names its construct kind")


# --------------------------------------------------------------------------- #
# Conflicts — emitted by the U3-B resolver, NOT the walker
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConflictRecord:
    """Two observations that cannot both hold under a UNIQUE resolution (rival
    owner chains, rival config prefixes).  This is NOT emitted merely because the
    walker saw several :class:`ChildCandidate`s — the raw index records the
    candidates; the U3-B ComponentOwner resolver emits this typed conflict when
    it cannot prove uniqueness (boundary 5)."""

    kind: str                    # rival_owner_chain | rival_config_prefix
    rivals: tuple = ()           # the competing candidate/observation tuples
    spans: tuple = ()            # tuple[SourceSpan]


__all__ = [
    "content_fingerprint",
    # identity
    "SourceId", "SymbolId", "SourceSpan", "ConstructionSiteId",
    # source nodes / parse outcomes / modules
    "SourceFileNode", "ParseFailure", "UnsupportedSyntaxRecord", "ModuleRecord",
    # expressions
    "EXPR_KINDS", "ExprNode",
    # callables
    "ParamRecord", "GuardStep", "CallableRecord",
    "IDENTIFIER_CONTEXTS", "IdentifierObservation",
    # classes / imports / registries
    "ClassBodyAssign", "ClassRecord", "ImportRecord",
    "ModuleBindingObservation", "DispatchRegistryRecord",
    # construction occurrences
    "ChildCandidate", "ConstructionSite", "FieldAssignRecord",
    "ContainerElementsRecord",
    # behavioural observations
    "CallObservation", "AttributeAccessRecord", "ConfigSegment",
    "ConfigPathObservation", "ControlRecord", "DataflowObservation",
    # execution-flow observations (U3-A2 kernel)
    "StatementId", "CallSiteId", "BindingObservation", "LoopObservation",
    "ReturnObservation", "ControlTransferObservation", "UnsupportedExecutionRegion",
    # resolver-only
    "ConflictRecord",
    # assembly
    "ProgramIndex", "build_program_index",
]


# =========================================================================== #
# U3-A walker — populates the record families above from one parsed source.
# OBSERVATION ONLY: no interpretation, no name-based architectural selection,
# no candidate is ever collapsed to one winner.
# =========================================================================== #

_BINOP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.MatMult: "@", ast.BitOr: "|",
    ast.BitAnd: "&", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_UNARY = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not", ast.Invert: "~"}
_BOOL = {ast.And: "and", ast.Or: "or"}
_CMP = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
    ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in",
    ast.NotIn: "not in",
}

# Direct AST expression nodes normalized by _SourceWalker._expr.  The callable
# coverage walk tests membership in O(1) per node; calling _expr at every node
# would recursively normalize the same subtrees and turn indexing into O(n^2).
_NORMALIZED_EXPR_NODES = (
    ast.Name, ast.Attribute, ast.Constant, ast.Call, ast.Subscript, ast.Slice,
    ast.Tuple, ast.List, ast.Set, ast.Dict, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.Compare, ast.IfExp, ast.ListComp, ast.SetComp, ast.GeneratorExp,
    ast.DictComp, ast.Starred, ast.Lambda, ast.JoinedStr,
)
_IDENTIFIER_COVERED_EXPR_NODES = _NORMALIZED_EXPR_NODES + (ast.FormattedValue,)
_MATCH = getattr(ast, "Match", None)   # structural pattern matching (py>=3.10)
# statements that never affect submodule execution ordering; everything else that
# reaches the _stmt fallback is published as an unsupported coverage gap.
_BENIGN_STATEMENTS = (ast.Pass, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


def _op_token(op) -> str:
    t = type(op)
    return (_BINOP.get(t) or _UNARY.get(t) or _BOOL.get(t) or _CMP.get(t)
            or t.__name__)


class _CallableScan:
    """Mutable per-callable accumulator (lexical counter, returns, self-calls).
    Reentrant: a nested def gets its own instance, so the outer callable's
    lexical order is never disturbed."""

    __slots__ = ("enclosing", "owner", "order", "returns", "self_calls")

    def __init__(self, enclosing: SymbolId, owner: SymbolId | None):
        self.enclosing = enclosing
        self.owner = owner
        self.order = 0
        self.returns: list = []
        self.self_calls: list = []

    def next_order(self) -> int:
        self.order += 1
        return self.order


class _CallableIdentifierCollector(ast.NodeVisitor):
    """Complete exact-span ``ast.Name``/``ast.arg`` census for ONE callable.

    The collector owns lexical-scope boundaries rather than relying on a broad
    ``ast.walk``: nested defs/classes/lambdas never contaminate the parent.  The
    normal walker indexes named nested callables separately.  Every emitted
    record is syntax only; no spelling is interpreted here.
    """

    def __init__(self, sid: SourceId, owner: SymbolId | None,
                 enclosing: SymbolId, *, segment):
        self.sid = sid
        self.owner = owner
        self.enclosing = enclosing
        self._segment = segment
        self.records: list[IdentifierObservation] = []
        self.unsupported: list[UnsupportedSyntaxRecord] = []
        self._region: str | None = None

    def collect(self, node):
        args = node.args
        params = (tuple(getattr(args, "posonlyargs", ())) + tuple(args.args)
                  + tuple(args.kwonlyargs)
                  + ((args.vararg,) if args.vararg is not None else ())
                  + ((args.kwarg,) if args.kwarg is not None else ()))
        for arg in params:
            self._emit(arg.arg, "parameter", arg)
            if arg.annotation is not None:
                self._visit_region(arg.annotation, "annotation")
        for default in tuple(args.defaults) + tuple(
                item for item in args.kw_defaults if item is not None):
            self._visit_region(default, "default")
        for decorator in node.decorator_list:
            self._visit_region(decorator, "decorator")
        if node.returns is not None:
            self._visit_region(node.returns, "annotation")
        for stmt in node.body:
            self.visit(stmt)
        return (tuple(sorted(self.records, key=_identifier_sort_key)),
                tuple(sorted(self.unsupported, key=_unsupported_sort_key)))

    def _visit_region(self, node, region: str) -> None:
        previous = self._region
        self._region = region
        try:
            self.visit(node)
        finally:
            self._region = previous

    def _emit(self, name: str, context: str, node) -> None:
        self.records.append(IdentifierObservation(
            self.owner, self.enclosing, name, context,
            SourceSpan(
                self.sid,
                getattr(node, "lineno", 0) or 0,
                getattr(node, "col_offset", 0) or 0,
                getattr(node, "end_lineno", 0) or 0,
                getattr(node, "end_col_offset", 0) or 0,
            ),
        ))

    def visit_Name(self, node: ast.Name) -> None:
        if self._region is not None:
            context = self._region
        elif isinstance(node.ctx, ast.Store):
            context = "store"
        elif isinstance(node.ctx, ast.Del):
            context = "del"
        else:
            context = "load"
        self._emit(node.id, context, node)

    def visit(self, node):
        # ExprNode normalization has a closed set.  Record every expression
        # outside it even when that expression appears only as a bare statement
        # or binding construct, so a reader can prove its negative did not skip
        # an opaque part of the exact callable.
        # FormattedValue is normalized as part of its supported JoinedStr
        # parent; it is not a standalone expression surface.
        if isinstance(node, ast.expr) and not isinstance(
                node, _IDENTIFIER_COVERED_EXPR_NODES):
            self._unsupported(type(node).__name__, node)
        return super().visit(node)

    def _unsupported(self, kind: str, node) -> None:
        span = SourceSpan(
            self.sid,
            getattr(node, "lineno", 0) or 0,
            getattr(node, "col_offset", 0) or 0,
            getattr(node, "end_lineno", 0) or 0,
            getattr(node, "end_col_offset", 0) or 0,
        )
        self.unsupported.append(UnsupportedSyntaxRecord(
            self.owner, self.enclosing, kind,
            self._segment(node) or kind, span))

    # These are distinct lexical scopes.  _SourceWalker indexes named nested
    # defs/classes separately; lambda bodies stay opaque to the parent's census.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._unsupported("nested_callable", node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._unsupported("nested_callable", node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._unsupported("nested_class", node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._unsupported("nested_lambda", node)


def _identifier_sort_key(item: IdentifierObservation):
    span = item.span
    return (span.line, span.col, span.end_line, span.end_col,
            item.context, item.name)


def _unsupported_sort_key(item: UnsupportedSyntaxRecord):
    span = item.span
    return (span.line, span.col, span.end_line, span.end_col,
            item.syntax_kind, item.detail)


class _ModuleBindingCollector(ast.NodeVisitor):
    """Collect exact names bound in one module scope.

    Function/class/lambda bodies and comprehension targets are separate scopes.
    Control-flow statement bodies at module level remain in the module scope and
    are visited normally.  The result is syntax-only and deliberately
    conservative for builtin-name proof.
    """

    def __init__(self, sid: SourceId):
        self.sid = sid
        self.records: list[ModuleBindingObservation] = []

    def collect(self, tree: ast.Module) -> tuple:
        for stmt in tree.body:
            self.visit(stmt)
        return tuple(sorted(self.records, key=lambda r: (
            r.span.line, r.span.col, r.span.end_line, r.span.end_col,
            r.kind, r.name)))

    def _span(self, node) -> SourceSpan:
        return SourceSpan(
            self.sid,
            getattr(node, "lineno", 0) or 0,
            getattr(node, "col_offset", 0) or 0,
            getattr(node, "end_lineno", 0) or 0,
            getattr(node, "end_col_offset", 0) or 0,
        )

    def _emit(self, name: str | None, kind: str, node) -> None:
        if name:
            self.records.append(
                ModuleBindingObservation(self.sid, name, kind, self._span(node)))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._emit(node.id, "store", node)
        elif isinstance(node.ctx, ast.Del):
            self._emit(node.id, "delete", node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._emit(alias.asname or alias.name.split(".")[0], "import", alias)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._emit(alias.asname or alias.name, "import", alias)

    def _visit_function_header(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in tuple(node.args.defaults) + tuple(
                d for d in node.args.kw_defaults if d is not None):
            self.visit(default)
        for arg in (tuple(getattr(node.args, "posonlyargs", ()))
                    + tuple(node.args.args) + tuple(node.args.kwonlyargs)):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._emit(node.name, "function", node)
        self._visit_function_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._emit(node.name, "function", node)
        self._visit_function_header(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._emit(node.name, "class", node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # A lambda's parameters/body are a separate lexical scope.  Its defaults
        # do not exist on ast.Lambda; nothing here binds the containing module.
        return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._emit(node.name, "exception", node)
        if node.type is not None:
            self.visit(node.type)
        for stmt in node.body:
            self.visit(stmt)

    def visit_MatchAs(self, node) -> None:
        self._emit(getattr(node, "name", None), "pattern", node)
        if getattr(node, "pattern", None) is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node) -> None:
        self._emit(getattr(node, "name", None), "pattern", node)

    def visit_MatchMapping(self, node) -> None:
        self._emit(getattr(node, "rest", None), "pattern", node)
        for pattern in node.patterns:
            self.visit(pattern)

    def _visit_comprehension(self, node, value_nodes) -> None:
        # Comprehension targets are local to the implicit comprehension scope.
        # The surrounding expressions are still visited so a NamedExpr target
        # (which Python binds in the containing scope) remains visible.
        for generator in node.generators:
            self.visit(generator.iter)
            for cond in generator.ifs:
                self.visit(cond)
        for value in value_nodes:
            self.visit(value)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, (node.key, node.value))


class _SourceWalker:
    """Walk ONE parsed source file into observation records.  Two passes: pass 1
    collects the definitions a construction reference may resolve against (local
    classes, module-level dispatch registries, import aliases); pass 2 walks
    bodies, emitting the behavioural records with proof-bearing candidate edges.
    """

    def __init__(self, sid: SourceId, text: str, tree: ast.Module,
                 *, config_vocab: dict, factory_names: frozenset):
        self.sid = sid
        self.text = text
        self.tree = tree
        self._config_roots = config_vocab["config_roots"]
        self._act_dispatch = config_vocab["activation_dispatch"]
        self._act_calls = config_vocab["activation_calls"]
        self._container_classes = config_vocab["container_classes"]
        self._factory_names = factory_names
        # outputs
        self.imports: list = []
        self.module_bindings: list = []
        self.classes: list = []
        self.callables: list = []
        self.identifiers: list = []
        self.field_assigns: list = []
        self.sites: list = []
        self.containers: list = []
        self.registries: list = []
        self.calls: list = []
        self.attrs: list = []
        self.configs: list = []
        self.controls: list = []
        self.dataflow: list = []
        self.unsupported: list = []
        # execution-flow observations (U3-A2 kernel)
        self.bindings: list = []
        self.loops: list = []
        self.return_obs: list = []
        self.transfers: list = []
        self.unsupported_exec: list = []
        self._stmt_ordinals: dict = {}
        # pass-1 tables
        self._local_classes: set = set()          # qualified names defined here
        self._registry_values: dict = {}          # registry name -> tuple[ExprNode]
        self._import_aliases: dict = {}           # alias -> dotted target
        # per-(callable, line) construction ordinal, so two constructions on one
        # source line (``self.a = A(); self.b = B()`` or ``Sequential(A(), B())``)
        # get distinct ConstructionSiteIds (boundary: honest site identity).
        self._line_ordinals: dict = {}

    # -- passes ------------------------------------------------------------- #

    def run(self) -> "_SourceWalker":
        self.module_bindings.extend(
            _ModuleBindingCollector(self.sid).collect(self.tree))
        self._collect_definitions(self.tree.body, scope="", guard=())
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                self._class(node, scope="")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._callable(node, owner=None, scope="")
        return self

    def _collect_definitions(self, body, scope: str, guard: tuple) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qual = f"{scope}.{node.name}" if scope else node.name
                self._local_classes.add(qual)
                self._collect_definitions(node.body, scope=qual, guard=guard)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._record_import(node, guard)
            elif isinstance(node, ast.If) and scope == "":
                branch = GuardStep("if", self._expr(node.test), self._span(node))
                self._collect_definitions(
                    node.body, scope=scope, guard=(*guard, branch))
                if node.orelse:
                    alternative = GuardStep("else", None, self._span(node))
                    self._collect_definitions(
                        node.orelse, scope=scope,
                        guard=(*guard, alternative))
            elif isinstance(node, ast.Assign) and scope == "":
                self._maybe_registry(node)

    # -- imports / registries ---------------------------------------------- #

    def _record_import(self, node, guard=()) -> None:
        span = self._span(node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                if not guard:
                    self._import_aliases[local] = alias.name
                self.imports.append(
                    ImportRecord(self.sid, local, alias.name, span, guard))
        else:  # ImportFrom
            module = node.module or ""
            level = "." * (node.level or 0)
            for alias in node.names:
                target = f"{level}{module}.{alias.name}" if module else \
                    f"{level}{alias.name}"
                local = alias.asname or alias.name
                if not guard:
                    self._import_aliases[local] = target
                self.imports.append(
                    ImportRecord(self.sid, local, target, span, guard))

    def _maybe_registry(self, node: ast.Assign) -> None:
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)):
            return
        name = node.targets[0].id
        entries = []
        values = []
        for k, v in zip(node.value.keys, node.value.values):
            key_expr = self._expr(k) if k is not None else ExprNode("unsupported")
            val_expr = self._expr(v)
            entries.append((key_expr, val_expr))
            values.append(val_expr)
        self._registry_values[name] = tuple(values)
        self.registries.append(DispatchRegistryRecord(
            SymbolId(self.sid, name), tuple(entries), self._span(node)))

    # -- classes / callables ----------------------------------------------- #

    def _class(self, node: ast.ClassDef, scope: str) -> None:
        qual = f"{scope}.{node.name}" if scope else node.name
        sym = SymbolId(self.sid, qual)
        bases = tuple(self._expr(b) for b in node.bases)
        body_assigns = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        body_assigns.append(ClassBodyAssign(
                            tgt.id, self._expr(stmt.value), self._span(stmt)))
            elif isinstance(stmt, ast.AnnAssign) and \
                    isinstance(stmt.target, ast.Name) and stmt.value is not None:
                body_assigns.append(ClassBodyAssign(
                    stmt.target.id, self._expr(stmt.value), self._span(stmt)))
        self.classes.append(ClassRecord(sym, bases, tuple(body_assigns),
                                        self._span(node)))
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._callable(stmt, owner=sym, scope=qual)
            elif isinstance(stmt, ast.ClassDef):
                self._class(stmt, scope=qual)

    def _callable(self, node, owner: SymbolId | None, scope: str) -> None:
        qual = f"{scope}.{node.name}" if scope else node.name
        sym = SymbolId(self.sid, qual)
        identifiers, unsupported = _CallableIdentifierCollector(
            self.sid, owner, sym, segment=self._seg).collect(node)
        self.identifiers.extend(identifiers)
        self.unsupported.extend(unsupported)
        scan = _CallableScan(sym, owner)
        self._visit_stmts(node.body, guard=(), scan=scan)
        self.callables.append(CallableRecord(
            symbol=sym, owner=owner,
            kind="method" if owner is not None else "function",
            params=self._params(node.args),
            returns=tuple(scan.returns),
            self_method_calls=tuple(dict.fromkeys(scan.self_calls)),
            span=self._span(node)))

    def _params(self, args: ast.arguments) -> tuple:
        out = []
        posonly = list(getattr(args, "posonlyargs", []))
        positional = posonly + list(args.args)
        defaults = list(args.defaults)
        # defaults align to the LAST len(defaults) positional params
        n_no_default = len(positional) - len(defaults)
        for i, a in enumerate(positional):
            di = i - n_no_default
            if di >= 0:
                out.append(self._param(a.arg, defaults[di], "positional"))
            else:
                out.append(ParamRecord(a.arg, has_default=False, kind="positional"))
        if args.vararg:
            out.append(ParamRecord(args.vararg.arg, has_default=False, kind="vararg"))
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            if d is None:
                out.append(ParamRecord(a.arg, has_default=False, kind="keyword_only"))
            else:
                out.append(self._param(a.arg, d, "keyword_only"))
        if args.kwarg:
            out.append(ParamRecord(args.kwarg.arg, has_default=False, kind="kwarg"))
        return tuple(out)

    def _param(self, name: str, default_ast, kind: str) -> ParamRecord:
        return ParamRecord(name, has_default=True,
                           default=self._expr(default_ast), kind=kind)

    # -- statement visiting (guard stack + lexical order) ------------------ #

    def _visit_stmts(self, stmts, guard: tuple, scan: _CallableScan) -> None:
        for stmt in stmts:
            self._stmt(stmt, guard, scan)

    def _statement_id(self, stmt, scan: _CallableScan) -> "StatementId":
        span = self._span(stmt)
        key = (scan.enclosing.qualified_name, span.line, span.col)
        ordinal = self._stmt_ordinals.get(key, 0)
        self._stmt_ordinals[key] = ordinal + 1
        return StatementId(scan.enclosing, span, ordinal)

    def _stmt(self, stmt, guard: tuple, scan: _CallableScan) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._callable(stmt, owner=None, scope=scan.enclosing.qualified_name)
            return
        if isinstance(stmt, ast.ClassDef):
            self._class(stmt, scope=scan.enclosing.qualified_name)
            return
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            self._assign(stmt, guard, scan)
            return
        if isinstance(stmt, ast.AugAssign):
            self._walk_expr(stmt.value, guard, scan)
            self.dataflow.append(DataflowObservation(
                scan.owner, scan.enclosing, self._expr(stmt.value),
                self._expr(stmt.target), f"aug:{_op_token(stmt.op)}",
                self._span(stmt)))
            self.bindings.append(BindingObservation(
                scan.owner, scan.enclosing, self._statement_id(stmt, scan),
                (self._expr(stmt.target),), self._expr(stmt.value), "augassign",
                guard, self._span(stmt)))
            return
        if isinstance(stmt, ast.Return):
            sid = self._statement_id(stmt, scan)
            value_expr = self._expr(stmt.value) if stmt.value is not None else None
            self.return_obs.append(ReturnObservation(
                scan.owner, scan.enclosing, sid, value_expr, guard, self._span(stmt)))
            self.transfers.append(ControlTransferObservation(
                "return", scan.enclosing, guard, value_expr, self._span(stmt)))
            if stmt.value is not None:
                scan.returns.append(self._expr(stmt.value))
                # a helper that RETURNS a provable construction is a site
                # (target_kind='return'), so the resolver can follow a
                # helper-built field to its child class; a plain function-call
                # return is NOT a construction and stays a call observation.
                if isinstance(stmt.value, ast.Call) and \
                        scan.owner is not None and \
                        self._provable_construction(stmt.value.func):
                    self.sites.append(self._site(
                        "", "return", stmt.value, guard, scan, via="return"))
                self._walk_expr(stmt.value, guard, scan)
                self.dataflow.append(DataflowObservation(
                    scan.owner, scan.enclosing, self._expr(stmt.value), None,
                    "return", self._span(stmt)))
            return
        if isinstance(stmt, ast.If):
            self.controls.append(ControlRecord(
                scan.owner, scan.enclosing, "if", self._expr(stmt.test),
                self._span(stmt)))
            self._walk_expr(stmt.test, guard, scan)
            body_guard = guard + (GuardStep("if", self._expr(stmt.test),
                                            self._span(stmt)),)
            self._visit_stmts(stmt.body, body_guard, scan)
            if stmt.orelse:
                self.controls.append(ControlRecord(
                    scan.owner, scan.enclosing, "else", None, self._span(stmt)))
                else_guard = guard + (GuardStep("else", None, self._span(stmt)),)
                self._visit_stmts(stmt.orelse, else_guard, scan)
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self.controls.append(ControlRecord(
                scan.owner, scan.enclosing, "for", self._expr(stmt.iter),
                self._span(stmt)))
            self.loops.append(LoopObservation(
                scan.owner, scan.enclosing, self._statement_id(stmt, scan), "for",
                self._expr(stmt.target), self._expr(stmt.iter),
                isinstance(stmt, ast.AsyncFor), guard,
                self._body_span(stmt.body), self._body_span(stmt.orelse),
                self._span(stmt)))
            self._walk_expr(stmt.iter, guard, scan)
            loop_guard = guard + (GuardStep("for", self._expr(stmt.iter),
                                            self._span(stmt)),)
            self._visit_stmts(stmt.body, loop_guard, scan)
            self._visit_stmts(stmt.orelse, guard, scan)
            return
        if isinstance(stmt, ast.While):
            self.controls.append(ControlRecord(
                scan.owner, scan.enclosing, "while", self._expr(stmt.test),
                self._span(stmt)))
            self.loops.append(LoopObservation(
                scan.owner, scan.enclosing, self._statement_id(stmt, scan), "while",
                None, self._expr(stmt.test), False, guard,
                self._body_span(stmt.body), self._body_span(stmt.orelse),
                self._span(stmt)))
            self._walk_expr(stmt.test, guard, scan)
            loop_guard = guard + (GuardStep("while", self._expr(stmt.test),
                                            self._span(stmt)),)
            self._visit_stmts(stmt.body, loop_guard, scan)
            self._visit_stmts(stmt.orelse, guard, scan)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            kind = "async_with" if isinstance(stmt, ast.AsyncWith) else "with"
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, kind, guard, self._span(stmt),
                "context-manager execution semantics are not modelled"))
            for item in stmt.items:
                self._walk_expr(item.context_expr, guard, scan)
            self._visit_stmts(stmt.body, guard, scan)
            return
        if isinstance(stmt, ast.Try):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "try", guard, self._span(stmt),
                "exception control flow is not modelled"))
            self._visit_stmts(stmt.body, guard, scan)
            for handler in stmt.handlers:
                self._visit_stmts(handler.body, guard, scan)
            self._visit_stmts(stmt.orelse, guard, scan)
            self._visit_stmts(stmt.finalbody, guard, scan)
            return
        if isinstance(stmt, ast.Raise):
            self.transfers.append(ControlTransferObservation(
                "raise", scan.enclosing, guard,
                self._expr(stmt.exc) if stmt.exc is not None else None,
                self._span(stmt)))
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    self._walk_expr(child, guard, scan)
            return
        if isinstance(stmt, ast.Break):
            self.transfers.append(ControlTransferObservation(
                "break", scan.enclosing, guard, None, self._span(stmt)))
            return
        if isinstance(stmt, ast.Continue):
            self.transfers.append(ControlTransferObservation(
                "continue", scan.enclosing, guard, None, self._span(stmt)))
            return
        if _MATCH is not None and isinstance(stmt, _MATCH):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "match", guard, self._span(stmt),
                "structural pattern matching is not modelled"))
            self._walk_expr(stmt.subject, guard, scan)
            for case in stmt.cases:
                self._visit_stmts(case.body, guard, scan)
            return
        if isinstance(stmt, ast.Expr):
            self._walk_expr(stmt.value, guard, scan)
            return
        # Any other statement is published as an UNKNOWN/unmodelled coverage gap,
        # then its contained expressions are still swept for calls/reads.  This
        # list is intentionally non-exhaustive and never proves completeness.
        if not isinstance(stmt, _BENIGN_STATEMENTS):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "unknown_statement", guard, self._span(stmt),
                type(stmt).__name__))
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                self._walk_expr(child, guard, scan)

    @staticmethod
    def _has_call(node) -> bool:
        return any(isinstance(n, ast.Call) for n in ast.walk(node))

    def _body_span(self, body) -> "SourceSpan | None":
        if not body:
            return None
        first, last = body[0], body[-1]
        start = self._span(first)
        end = self._span(last)
        return SourceSpan(start.source, start.line, start.col,
                          end.end_line or end.line, end.end_col or end.col)

    def _assign(self, stmt, guard: tuple, scan: _CallableScan) -> None:
        value = stmt.value
        targets = (stmt.targets if isinstance(stmt, ast.Assign)
                   else [stmt.target])
        if value is None:  # bare AnnAssign (`x: int`) — no observation
            return
        # dataflow: <value> -> each target
        for tgt in targets:
            self.dataflow.append(DataflowObservation(
                scan.owner, scan.enclosing, self._expr(value),
                self._expr(tgt), "assign", self._span(stmt)))
            if isinstance(tgt, ast.Attribute):
                self.attrs.append(AttributeAccessRecord(
                    scan.owner, scan.enclosing, self._expr(tgt), "write",
                    self._span(tgt)))
        # neutral binding observation: exact target pattern(s) + value, no SSA.
        self.bindings.append(BindingObservation(
            scan.owner, scan.enclosing, self._statement_id(stmt, scan),
            tuple(self._expr(tgt) for tgt in targets), self._expr(value),
            "assign" if isinstance(stmt, ast.Assign) else "annassign",
            guard, self._span(stmt)))
        # self.<field> = ... : field assign + construction / container
        field_name = self._self_field(targets)
        if field_name is not None and scan.owner is not None:
            self.field_assigns.append(FieldAssignRecord(
                scan.owner, scan.enclosing, field_name, self._expr(value),
                guard, self._span(stmt)))
            self._maybe_construction(field_name, value, guard, scan)
        # always sweep the value for calls / config reads / attribute reads
        self._walk_expr(value, guard, scan)

    def _self_field(self, targets) -> str | None:
        for tgt in targets:
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                return tgt.attr
        return None

    # -- construction sites + containers ----------------------------------- #

    def _maybe_construction(self, field_name: str, value, guard: tuple,
                            scan: _CallableScan) -> None:
        # container? self.x = nn.ModuleList([...]) / Sequential(...) / ModuleDict
        if isinstance(value, ast.Call):
            cname = _short_name(value.func)
            if cname in self._container_classes:
                self._container(field_name, value, guard, scan)
                return
            self.sites.append(self._site(
                field_name, "field", value, guard, scan, via="init"))

    def _container(self, field_name: str, call: ast.Call, guard: tuple,
                   scan: _CallableScan) -> None:
        # KIND derived by lowercasing the class name (ModuleList -> modulelist),
        # so the vocabulary stays a flat list, never a class -> fact identity map.
        kind = _short_name(call.func).lower()
        elements: list = []
        count: ExprNode | None = None
        for arg in call.args:
            elems, cnt = self._container_elements(arg)
            for e in elems:
                elements.append(self._site(field_name, "element", e, guard,
                                           scan, via=kind))
            if cnt is not None and count is None:
                count = cnt
        self.containers.append(ContainerElementsRecord(
            scan.owner, scan.enclosing, field_name, kind, tuple(elements),
            count, self._span(call)))

    def _container_elements(self, arg):
        """Return (list of construction Call asts, count ExprNode|None)."""
        if isinstance(arg, (ast.List, ast.Tuple)):
            return [e for e in arg.elts if isinstance(e, ast.Call)], None
        if isinstance(arg, (ast.ListComp, ast.GeneratorExp)):
            calls = [arg.elt] if isinstance(arg.elt, ast.Call) else []
            count = None
            for gen in arg.generators:
                if isinstance(gen.iter, ast.Call) and \
                        _short_name(gen.iter.func) in ("range", "len"):
                    count = self._expr(gen.iter)
            return calls, count
        if isinstance(arg, ast.Dict):
            return [v for v in arg.values if isinstance(v, ast.Call)], None
        return [], None

    def _site(self, target: str, target_kind: str, call: ast.Call,
              guard: tuple, scan: _CallableScan, *, via: str) -> ConstructionSite:
        key = (scan.enclosing.qualified_name, call.lineno)
        ordinal = self._line_ordinals.get(key, 0)
        self._line_ordinals[key] = ordinal + 1
        span = self._span(call)
        site_id = ConstructionSiteId(scan.owner, scan.enclosing, span, ordinal)
        candidates, resolved_via = self._candidates(call.func)
        if not candidates:
            kind = ("unresolved_registry" if isinstance(call.func, ast.Subscript)
                    else "dynamic_construction")
            self.unsupported.append(UnsupportedSyntaxRecord(
                scan.owner, scan.enclosing, kind, self._seg(call.func), span))
        return ConstructionSite(
            site_id=site_id, owner=scan.owner, enclosing_callable=scan.enclosing,
            target_kind=target_kind, target=target, constructor=self._expr(call),
            args=tuple(self._expr(a) for a in call.args),
            kwargs=tuple((k.arg or "**", self._expr(k.value))
                         for k in call.keywords),
            guard=guard, candidates=candidates,
            via=resolved_via or via, span=span)

    def _candidates(self, func) -> tuple:
        """Zero, one or several proof-bearing child candidates.  The walker
        never picks a winner; ambiguity is preserved for the U3-B resolver."""
        # direct name: Foo(...)
        if isinstance(func, ast.Name):
            sym = (SymbolId(self.sid, func.id)
                   if func.id in self._local_classes else None)
            prov = "direct_name"
            if sym is None and func.id in self._import_aliases:
                prov = f"alias_import:{func.id}"
            return (ChildCandidate(self._expr(func), sym, prov),), None
        # attribute: nn.Linear(...) or X._from_config(...) factory
        if isinstance(func, ast.Attribute):
            if func.attr in self._factory_names:
                base = func.value
                base_name = _short_name(base)
                sym = (SymbolId(self.sid, base_name)
                       if base_name in self._local_classes else None)
                cand = ChildCandidate(self._expr(base), sym,
                                      f"factory:{func.attr}")
                return (cand,), f"factory:{func.attr}"
            return (ChildCandidate(self._expr(func), None, "attribute"),), None
        # registry dispatch: REGISTRY[key](...)
        if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
            name = func.value.id
            values = self._registry_values.get(name)
            if values:
                cands = tuple(
                    ChildCandidate(
                        v, (SymbolId(self.sid, v.name)
                            if v.kind == "name" and v.name in self._local_classes
                            else None),
                        f"registry_subscript:{name}")
                    for v in values)
                return cands, f"registry_subscript:{name}"
            # unknown/dynamic registry -> no proven candidate
            return (), f"registry_subscript:{name}"
        # anything else (a call returning a class, dynamic dispatch) -> dynamic
        return (), None

    def _provable_construction(self, func) -> bool:
        """Is this call PROVABLY a construction (not a plain function call)?  A
        factory classmethod, or any candidate that resolves to a local class, is
        proof; a bare attribute call (``nn.Linear``, ``F.softmax``) is not — the
        walker will not mint a return/bare site it cannot prove constructs."""
        cands, via = self._candidates(func)
        if via and via.startswith("factory:"):
            return True
        return any(c.symbol is not None for c in cands)

    # -- calls / attribute reads / config reads ---------------------------- #

    def _walk_expr(self, node, guard: tuple, scan: _CallableScan) -> None:
        if node is None:
            return
        # execution-flow expression observations (yield / yield from / await /
        # walrus) — neutral syntax, recorded then swept normally.
        if isinstance(node, ast.Yield):
            self.transfers.append(ControlTransferObservation(
                "yield", scan.enclosing, guard,
                self._expr(node.value) if node.value is not None else None,
                self._span(node)))
        elif isinstance(node, ast.YieldFrom):
            self.transfers.append(ControlTransferObservation(
                "yield_from", scan.enclosing, guard, self._expr(node.value),
                self._span(node)))
        elif isinstance(node, ast.Await):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "await", guard, self._span(node),
                "await suspension is not modelled"))
        elif isinstance(node, ast.NamedExpr):
            self.bindings.append(BindingObservation(
                scan.owner, scan.enclosing, self._statement_id(node, scan),
                (self._expr(node.target),), self._expr(node.value), "walrus",
                guard, self._span(node)))
        # Publish known executable value forms that carry or defer a call as
        # unsupported regions.  This list is deliberately non-exhaustive; absence
        # of one of these records is never a completeness certificate.
        if isinstance(node, ast.IfExp) and self._has_call(node):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "ifexp", guard, self._span(node),
                "conditional expression carries a call"))
        elif isinstance(node, ast.BoolOp) and self._has_call(node):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "boolop", guard, self._span(node),
                "short-circuit boolean carries a call"))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)) and self._has_call(node):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "comprehension", guard, self._span(node),
                "comprehension iteration carries a call"))
        elif isinstance(node, ast.Lambda):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "lambda", guard, self._span(node),
                "lambda defers execution"))
        elif isinstance(node, ast.Compare) and len(node.ops) > 1 and self._has_call(node):
            self.unsupported_exec.append(UnsupportedExecutionRegion(
                scan.enclosing, "chained_comparison", guard, self._span(node),
                "chained comparison carries a call"))
        # activation-dispatch subscript: ACT2FN[config.hidden_act]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in self._act_dispatch:
            segs, base = self._config_segments(node.slice)
            if segs and self._is_config_root(base):
                self.configs.append(ConfigPathObservation(
                    scan.owner, scan.enclosing, self._expr(base), tuple(segs),
                    "act2fn", self._span(node)))
                return
        # config path chain: config.a.b / config["a"] / self.config.a
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            segs, base = self._config_segments(node)
            if segs and self._is_config_root(base):
                form = "subscript" if isinstance(node, ast.Subscript) else "attr"
                self.configs.append(ConfigPathObservation(
                    scan.owner, scan.enclosing, self._expr(base), tuple(segs),
                    form, self._span(node)))
                for dyn in self._dynamic_index_parts(node):
                    self._walk_expr(dyn, guard, scan)
                return
        if isinstance(node, ast.Call):
            self._record_call(node, guard, scan)
            for child in ast.iter_child_nodes(node):
                self._walk_expr(child, guard, scan)
            return
        if isinstance(node, ast.Attribute):  # a non-config attribute read
            self.attrs.append(AttributeAccessRecord(
                scan.owner, scan.enclosing, self._expr(node), "read",
                self._span(node)))
            self._walk_expr(node.value, guard, scan)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._walk_expr(child, guard, scan)

    def _record_call(self, node: ast.Call, guard: tuple,
                     scan: _CallableScan) -> None:
        callee = self._expr(node.func)
        receiver = None
        if isinstance(node.func, ast.Attribute):
            receiver = self._expr(node.func.value)
            if isinstance(node.func.value, ast.Name) and \
                    node.func.value.id == "self":
                scan.self_calls.append(node.func.attr)
        # get_activation(config.hidden_act) -> a config read, form get_activation
        if isinstance(node.func, ast.Name) and node.func.id in self._act_calls \
                and node.args:
            segs, base = self._config_segments(node.args[0])
            if segs and self._is_config_root(base):
                self.configs.append(ConfigPathObservation(
                    scan.owner, scan.enclosing, self._expr(base), tuple(segs),
                    "get_activation", self._span(node)))
        self.calls.append(CallObservation(
            owner=scan.owner, enclosing_callable=scan.enclosing,
            lexical_order=scan.next_order(), callee=callee, receiver=receiver,
            args=tuple(self._expr(a) for a in node.args),
            kwargs=tuple((k.arg or "**", self._expr(k.value))
                         for k in node.keywords),
            guard=guard, span=self._span(node)))

    # -- config-chain helpers ---------------------------------------------- #

    def _config_segments(self, node):
        """Unwind an attribute/subscript chain into (segments, base-most node)."""
        segs: list = []
        cur = node
        while True:
            if isinstance(cur, ast.Attribute):
                segs.append(ConfigSegment(cur.attr, False))
                cur = cur.value
            elif isinstance(cur, ast.Subscript):
                key = cur.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    segs.append(ConfigSegment(key.value, False))
                else:
                    segs.append(ConfigSegment("", True))
                cur = cur.value
            else:
                break
        segs.reverse()
        return segs, cur

    def _is_config_root(self, node) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._config_roots
        # self.config / self.text_config
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            return node.attr in self._config_roots
        return False

    def _dynamic_index_parts(self, node):
        cur = node
        while isinstance(cur, (ast.Attribute, ast.Subscript)):
            if isinstance(cur, ast.Subscript):
                key = cur.slice
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    yield key
            cur = cur.value

    # -- expression normalization + spans ---------------------------------- #

    def _span(self, node) -> SourceSpan:
        return SourceSpan(
            self.sid, getattr(node, "lineno", 0), getattr(node, "col_offset", 0),
            getattr(node, "end_lineno", 0) or 0,
            getattr(node, "end_col_offset", 0) or 0)

    def _seg(self, node) -> str:
        # get_source_segment returns None for a node without position info; the
        # only raises are line/col misalignment against the source text.
        try:
            return ast.get_source_segment(self.text, node) or ""
        except (ValueError, IndexError):
            return ""

    def _expr(self, node) -> ExprNode | None:
        if node is None:
            return None
        sp = self._span(node)
        seg = self._seg(node)
        if isinstance(node, ast.Name):
            return ExprNode("name", name=node.id, span=sp, source_segment=seg)
        if isinstance(node, ast.Attribute):
            return ExprNode("attribute", name=node.attr,
                            children=(self._expr(node.value),), span=sp,
                            source_segment=seg)
        if isinstance(node, ast.Constant):
            return ExprNode("constant", const_value=node.value, span=sp,
                            source_segment=seg)
        if isinstance(node, ast.Call):
            kids = (self._expr(node.func),) + tuple(self._expr(a) for a in node.args)
            kw = tuple((k.arg or "**", self._expr(k.value)) for k in node.keywords)
            return ExprNode("call", children=kids, keyword_children=kw, span=sp,
                            source_segment=seg)
        if isinstance(node, ast.Subscript):
            return ExprNode("subscript",
                            children=(self._expr(node.value), self._expr(node.slice)),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.Slice):
            return ExprNode("slice",
                            children=tuple(self._expr(x) for x in
                                           (node.lower, node.upper, node.step)),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.Tuple):
            return ExprNode("tuple", children=tuple(self._expr(e) for e in node.elts),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.List):
            return ExprNode("list", children=tuple(self._expr(e) for e in node.elts),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.Set):
            return ExprNode("set", children=tuple(self._expr(e) for e in node.elts),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.Dict):
            keys = tuple(self._expr(k) if k is not None else ExprNode("starred")
                         for k in node.keys)
            vals = tuple((str(i), self._expr(v)) for i, v in enumerate(node.values))
            return ExprNode("dict", children=keys, keyword_children=vals, span=sp,
                            source_segment=seg)
        if isinstance(node, ast.BoolOp):
            return ExprNode("boolop", operator=_op_token(node.op),
                            children=tuple(self._expr(v) for v in node.values),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.BinOp):
            return ExprNode("binop", operator=_op_token(node.op),
                            children=(self._expr(node.left), self._expr(node.right)),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.UnaryOp):
            return ExprNode("unaryop", operator=_op_token(node.op),
                            children=(self._expr(node.operand),), span=sp,
                            source_segment=seg)
        if isinstance(node, ast.Compare):
            ops = "/".join(_op_token(o) for o in node.ops)
            kids = (self._expr(node.left),) + \
                tuple(self._expr(c) for c in node.comparators)
            return ExprNode("compare", operator=ops, children=kids, span=sp,
                            source_segment=seg)
        if isinstance(node, ast.IfExp):
            return ExprNode("ifexp",
                            children=(self._expr(node.body), self._expr(node.test),
                                      self._expr(node.orelse)),
                            span=sp, source_segment=seg)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            kids = [self._expr(node.elt)]
            for gen in node.generators:
                kids.append(self._expr(gen.iter))
                kids.extend(self._expr(c) for c in gen.ifs)
            return ExprNode("comprehension", children=tuple(kids), span=sp,
                            source_segment=seg)
        if isinstance(node, ast.DictComp):
            kids = [self._expr(node.key), self._expr(node.value)]
            for gen in node.generators:
                kids.append(self._expr(gen.iter))
                kids.extend(self._expr(c) for c in gen.ifs)
            return ExprNode("comprehension", children=tuple(kids), span=sp,
                            source_segment=seg)
        if isinstance(node, ast.Starred):
            return ExprNode("starred", children=(self._expr(node.value),),
                            span=sp, source_segment=seg)
        if isinstance(node, ast.Lambda):
            return ExprNode("lambda", children=(self._expr(node.body),), span=sp,
                            source_segment=seg)
        if isinstance(node, ast.JoinedStr):
            kids = tuple(self._expr(v.value) for v in node.values
                         if isinstance(v, ast.FormattedValue))
            return ExprNode("fstring", children=kids, span=sp, source_segment=seg)
        return ExprNode("unsupported", span=sp, source_segment=seg)


def _short_name(node) -> str:
    """The trailing name of a call target (``nn.ModuleList`` -> 'ModuleList',
    ``Foo`` -> 'Foo'); '' when the target is not a plain name/attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


# =========================================================================== #
# The ONE immutable index + assembly
# =========================================================================== #

@dataclass(frozen=True)
class ProgramIndex:
    """The single immutable observation index for one SourceBundle (boundary 7).

    Every record is address-qualified; the query surface below selects only by
    :class:`SourceId` / :class:`SymbolId` (ADDRESS), never by a class-name
    substring or an architectural role — interpretation is a reader's job.
    ``fingerprint`` is the aggregate bundle identity: a canonical, order-
    independent digest over each source's identity, component ownership, external
    provenance and content fingerprint (never raw file contents)."""

    bundle_source: str
    source_nodes: tuple = ()
    modules: tuple = ()
    imports: tuple = ()
    module_bindings: tuple = ()
    classes: tuple = ()
    callables: tuple = ()
    identifiers: tuple = ()
    field_assigns: tuple = ()
    construction_sites: tuple = ()
    containers: tuple = ()
    dispatch_registries: tuple = ()
    calls: tuple = ()
    attribute_accesses: tuple = ()
    config_paths: tuple = ()
    controls: tuple = ()
    dataflow: tuple = ()
    unsupported_syntax: tuple = ()
    bindings: tuple = ()
    loops: tuple = ()
    return_observations: tuple = ()
    control_transfers: tuple = ()
    unsupported_execution: tuple = ()
    parse_failures: tuple = ()
    fingerprint: str = ""

    # -- address-only query surface (no name/substring/role selection) ----- #

    def source_node(self, path: str, component: str | None) -> SourceFileNode | None:
        for node in self.source_nodes:
            if node.source_id.canonical_path == path and \
                    node.source_id.component_key == component:
                return node
        return None

    def sources_for_component(self, component: str | None) -> tuple:
        return tuple(n for n in self.source_nodes
                     if n.source_id.component_key == component)

    def classes_in(self, source: SourceId) -> tuple:
        return tuple(c for c in self.classes if c.symbol.source == source)

    def module_bindings_in(self, source: SourceId) -> tuple:
        if not isinstance(source, SourceId):
            raise TypeError("module_bindings_in requires a SourceId")
        return tuple(b for b in self.module_bindings if b.source == source)

    def class_by_symbol(self, symbol: SymbolId) -> ClassRecord | None:
        for c in self.classes:
            if c.symbol == symbol:
                return c
        return None

    def callables_of(self, owner: SymbolId) -> tuple:
        return tuple(c for c in self.callables if c.owner == owner)

    def callable_by_symbol(self, symbol: SymbolId) -> CallableRecord | None:
        for c in self.callables:
            if c.symbol == symbol:
                return c
        return None

    def identifiers_in(self, callable_symbol: SymbolId) -> tuple:
        """Exact identifiers for one callable address, in source order."""
        if not isinstance(callable_symbol, SymbolId):
            raise TypeError("identifiers_in requires a SymbolId")
        return tuple(item for item in self.identifiers
                     if item.enclosing_callable == callable_symbol)

    def field_assigns_of(self, owner: SymbolId) -> tuple:
        return tuple(f for f in self.field_assigns if f.owner == owner)

    def construction_sites_of(self, owner: SymbolId) -> tuple:
        return tuple(s for s in self.construction_sites if s.owner == owner)

    def construction_sites_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(s for s in self.construction_sites
                     if s.enclosing_callable == callable_symbol)

    def calls_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(sorted(
            (c for c in self.calls if c.enclosing_callable == callable_symbol),
            key=lambda c: c.lexical_order))

    def config_paths_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(c for c in self.config_paths
                     if c.enclosing_callable == callable_symbol)

    # -- execution-flow query surface (address-only) ----------------------- #

    def call_sites_in(self, callable_symbol: SymbolId) -> tuple:
        """CallSiteIds for one callable, derived from CallObservations in lexical
        order — no re-walk of calls."""
        if not isinstance(callable_symbol, SymbolId):
            raise TypeError("call_sites_in requires a SymbolId")
        return tuple(CallSiteId.of(c) for c in self.calls_in(callable_symbol))

    def bindings_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(b for b in self.bindings
                     if b.enclosing_callable == callable_symbol)

    def loops_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(loop for loop in self.loops
                     if loop.enclosing_callable == callable_symbol)

    def return_observations_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(r for r in self.return_observations
                     if r.enclosing_callable == callable_symbol)

    def control_transfers_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(t for t in self.control_transfers
                     if t.enclosing_callable == callable_symbol)

    def unsupported_execution_in(self, callable_symbol: SymbolId) -> tuple:
        return tuple(u for u in self.unsupported_execution
                     if u.enclosing_callable == callable_symbol)


def _aggregate_fingerprint(source_ids) -> str:
    """Canonical, order-independent digest over source IDENTITY (path +
    component ownership + external provenance + content fingerprint) — never raw
    contents.  Sorting makes iteration order irrelevant (boundary/acceptance)."""
    rows = sorted(
        "\x1f".join((
            sid.canonical_path, sid.component_key or "",
            "1" if sid.external else "0", sid.external_provenance,
            sid.content_fingerprint,
        ))
        for sid in source_ids
    )
    h = hashlib.sha256()
    for row in rows:
        h.update(row.encode("utf-8", "surrogatepass"))
        h.update(b"\x1e")
    return h.hexdigest()


def _canonical_path(path: str) -> str:
    # realpath is pure string work here; it can only raise on an embedded NUL
    # (ValueError) or a platform path-resolution error (OSError).
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def build_program_index(bundle, *, external_nodes=()) -> ProgramIndex:
    """Assemble the ONE immutable index for a :class:`SourceBundle`.

    Ownership comes from ``bundle.component_files`` (each component key -> its
    files); a shared file owned by several components is indexed once PER owning
    component, so text/vision/audio observations are never blended (boundary 2).
    Read/decode/parse failures become :class:`ParseFailure` records (never
    skips); a healthy file is fully indexed even if it contains unsupported
    dynamic syntax.  ``external_nodes`` are provenance-bearing EXTERNAL sources
    resolved by import closure (boundary 8) — the U3-A core indexes the bundle's
    own component files; the resolver (U3-B) supplies external nodes.
    """
    from ..everchanging import (
        load_program_index_vocab, load_constructor_classmethods)
    config_vocab = load_program_index_vocab()
    factory_names = load_constructor_classmethods()

    component_files = dict(getattr(bundle, "component_files", {}) or {})
    if not component_files and getattr(bundle, "files", None):
        component_files = {"root": tuple(bundle.files)}

    source_nodes: list = []
    parse_failures: list = []
    agg_ids: list = []
    out = {k: [] for k in (
        "modules", "imports", "module_bindings", "classes", "callables", "identifiers",
        "field_assigns", "sites",
        "containers", "registries", "calls", "attrs", "configs", "controls",
        "dataflow", "unsupported",
        "bindings", "loops", "returns", "transfers", "unsupported_exec")}

    for component in sorted(component_files):
        for raw_path in component_files[component]:
            path = _canonical_path(str(raw_path))
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                sid = SourceId(path, "", component_key=component)
                parse_failures.append(ParseFailure(sid, "read_error", str(exc)))
                agg_ids.append(sid)
                continue
            fp = content_fingerprint(
                data.decode("utf-8", "surrogatepass"))
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                sid = SourceId(path, fp, component_key=component)
                parse_failures.append(ParseFailure(sid, "decode_error", str(exc)))
                agg_ids.append(sid)
                continue
            sid = SourceId(path, fp, component_key=component)
            try:
                tree = ast.parse(text, filename=path)
            except SyntaxError as exc:
                parse_failures.append(ParseFailure(sid, "syntax_error", str(exc)))
                agg_ids.append(sid)
                continue
            node = SourceFileNode(sid)
            node.verify_content(text)
            source_nodes.append(node)
            agg_ids.append(sid)
            module_name = os.path.splitext(os.path.basename(path))[0]
            out["modules"].append(ModuleRecord(
                module_name, "", node, SourceSpan(sid, 1)))
            walker = _SourceWalker(sid, text, tree, config_vocab=config_vocab,
                                   factory_names=factory_names).run()
            out["imports"].extend(walker.imports)
            out["module_bindings"].extend(walker.module_bindings)
            out["classes"].extend(walker.classes)
            out["callables"].extend(walker.callables)
            out["identifiers"].extend(walker.identifiers)
            out["field_assigns"].extend(walker.field_assigns)
            out["sites"].extend(walker.sites)
            out["containers"].extend(walker.containers)
            out["registries"].extend(walker.registries)
            out["calls"].extend(walker.calls)
            out["attrs"].extend(walker.attrs)
            out["configs"].extend(walker.configs)
            out["controls"].extend(walker.controls)
            out["dataflow"].extend(walker.dataflow)
            out["unsupported"].extend(walker.unsupported)
            out["bindings"].extend(walker.bindings)
            out["loops"].extend(walker.loops)
            out["returns"].extend(walker.return_obs)
            out["transfers"].extend(walker.transfers)
            out["unsupported_exec"].extend(walker.unsupported_exec)

    for ext in external_nodes:
        if not isinstance(ext, SourceFileNode):
            raise TypeError("external_nodes must be SourceFileNode instances")
        if not ext.source_id.external:
            raise ValueError("external_nodes must carry external=True identity")
        source_nodes.append(ext)
        agg_ids.append(ext.source_id)

    return ProgramIndex(
        bundle_source=getattr(bundle, "source", ""),
        source_nodes=tuple(source_nodes),
        modules=tuple(out["modules"]),
        imports=tuple(out["imports"]),
        module_bindings=tuple(out["module_bindings"]),
        classes=tuple(out["classes"]),
        callables=tuple(out["callables"]),
        identifiers=tuple(out["identifiers"]),
        field_assigns=tuple(out["field_assigns"]),
        construction_sites=tuple(out["sites"]),
        containers=tuple(out["containers"]),
        dispatch_registries=tuple(out["registries"]),
        calls=tuple(out["calls"]),
        attribute_accesses=tuple(out["attrs"]),
        config_paths=tuple(out["configs"]),
        controls=tuple(out["controls"]),
        dataflow=tuple(out["dataflow"]),
        unsupported_syntax=tuple(out["unsupported"]),
        bindings=tuple(out["bindings"]),
        loops=tuple(out["loops"]),
        return_observations=tuple(out["returns"]),
        control_transfers=tuple(out["transfers"]),
        unsupported_execution=tuple(out["unsupported_exec"]),
        parse_failures=tuple(parse_failures),
        fingerprint=_aggregate_fingerprint(agg_ids),
    )
