"""The ONE document-preparation primitive.

A parser never reads "the config". It reads a DOCUMENT that three different
things contributed to:

* the **checkpoint** — what the file itself declared;
* the installed **config class** — defaults the checkpoint never serialized
  (Gemma-2's sliding/global alternation), located by ``model_type``;
* the **loader** — stamps and fetched component context it injected
  (``_repo_id``, a pipeline's ``_text_encoder_configs``).

Mixing those into one dict and handing it to a reader destroys the only
distinction that matters: a fact detected from evidence versus a fact detected
from identity. ``model_type`` may LOCATE the class — identity as *address* is
lawful — but what the class supplies may never masquerade as what the checkpoint
declared.

Preparation happened in three places before this module (a root by-id path, an
encoder-slot path, and a class-defaults tier), which is why provenance was
partial and why the SAME model parsed differently embedded than standalone: only
some paths hydrated. One primitive, used at every document boundary, is what
makes both properties reachable at all.

Key origin and VALUE origin are different claims, and this keeps them apart: a
key the checkpoint declared whose value the class then changed is class-derived,
because the value the reader sees is not the value the file gave.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config_access import (
    CHECKPOINT_DECLARED,
    CLASS_DEFAULT,
    LOADER_METADATA,
)


@dataclass(frozen=True)
class PreparationFailure:
    """Why a document could not be prepared — typed, never a prose string.

    A failure is evidence too: "the class rejected this config" and "there is no
    class at this address" are different facts, and a downstream reader that has
    to regex a message cannot act on either."""

    kind: str        # no_mapping | class_rejected
    stage: str       # snapshot | hydrate
    message: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "message": self.message}


def _snapshot(value: Any) -> Any:
    """A deep, immutable-enough record of what the checkpoint said.

    ``dict(raw)`` is a SHALLOW copy: the nested dicts stay shared, so a config
    class normalizing a sub-config in place would rewrite the very record kept
    to prove what the file declared — and the provenance map would then compare
    the class's work against itself and find it unchanged.  A snapshot that the
    thing it audits can mutate is not a snapshot."""
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    # Preserve the CONTAINER TYPE.  ``_values_agree`` is type-strict, so
    # retyping a list to a tuple here made an untouched checkpoint list compare
    # unequal to itself — and the file's own declaration was then blamed on the
    # config class.  A snapshot that changes the record is not a snapshot.
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return value


@dataclass(frozen=True)
class DocumentBinding:
    """A prepared document bound to the OWNER whose parse reads it (§5.1).

    The binding is the single object a migrated ``document_scope`` accepts, so
    ``obj`` and ``provenance`` can no longer be supplied independently and drift
    apart.  ``prepared.document`` is the ONLY object this binding may describe;
    pairing a preparation with a different object is the nested-scope leak
    re-entering through the front door, so it is refused at construction."""

    owner: str
    document_path: tuple
    prepared: "PreparedDocument"

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedDocument):
            raise TypeError("DocumentBinding.prepared must be a PreparedDocument")

    @property
    def document(self):
        return self.prepared.document

    @property
    def provenance(self) -> dict:
        return self.prepared.provenance

    def describes(self, obj: Any) -> bool:
        return self.prepared.describes(obj)


@dataclass(frozen=True)
class PreparedDocument:
    """A document, its checkpoint snapshot, and where every field came from."""

    document: dict                       # what the parser reads
    checkpoint: dict                     # the checkpoint's OWN words, snapshotted
    #: {dotted path: value} the installed config class WOULD supply — kept as a
    #: SEPARATE CHANNEL, never merged into the document the readers see.
    #:
    #: Merging is what made a class default indistinguishable from a
    #: declaration, and an untyped ``_g()`` read cannot rank what it cannot
    #: tell apart: Falcon declares ``multi_query=True`` (which MEANS one KV
    #: head), the class emits its own ``num_kv_heads=71``, and a reader seeing
    #: one blended dict prefers the explicit-looking number and answers MHA —
    #: the class overriding the checkpoint, and the parse getting the
    #: architecture WRONG.  Candidates must stay separable so an arbiter can
    #: rank them by evidence strength.
    class_overlay: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)   # {dotted path: kind}
    failure: "PreparationFailure | None" = None

    @property
    def hydrated(self) -> bool:
        return any(k == CLASS_DEFAULT for k in self.provenance.values())

    def describes(self, obj: Any) -> bool:
        """Is this preparation ABOUT that object?

        A prepared document paired with a document it never described would
        attribute one file's provenance to another's fields — the nested-scope
        leak, re-introduced through the front door."""
        return obj is self.document


def _member(obj: Any, key: str, default=None):
    return (obj.get(key, default) if isinstance(obj, dict)
            else getattr(obj, key, default))


def _values_agree(a: Any, b: Any) -> bool:
    """Did the class leave this value alone?

    Deliberately conservative: anything we cannot compare confidently counts as
    CHANGED, so an unclear case is attributed to the class rather than credited
    to the checkpoint. Over-crediting the checkpoint is the failure that
    matters — it is what lets a class-derived value pose as a declaration."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return (set(a) == set(b)
                and all(_values_agree(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (len(a) == len(b)
                and all(_values_agree(x, y) for x, y in zip(a, b)))
    try:
        return bool(a == b)
    except Exception:
        return False


def _map_provenance(checkpoint: Any, document: Any, loader_keys: frozenset,
                    prefix: tuple = ()) -> dict:
    """``{dotted path: kind}`` for every addressable field of ``document``.

    Only what can be PROVEN is claimed:

    * declared by the loader (the loader names its own injections — it is the
      only thing that knows)            -> ``loader_metadata``
    * absent from the checkpoint        -> ``class_default``
    * present but the class changed its value -> ``class_default`` (the value
      the reader sees is the class's, whatever the key's origin)
    * present and unchanged             -> ``checkpoint_declared``

    ``class_normalized_alias`` is never inferred. A class-added key that
    RESEMBLES a vanished raw one (``rope_scaling`` → ``rope_parameters``) is not
    evidence of a rename: asserting the link would invent the very source
    relationship this map exists to record. Class-derived is the honest answer
    until an explicit trace names both paths.
    """
    out: dict[str, str] = {}
    if not isinstance(document, dict):
        return out
    for key, value in document.items():
        # A dotted path addresses string keys; a label map keyed by ints has no
        # dotted address, so it is not mapped rather than given a false one.
        if not isinstance(key, str):
            continue
        path = ".".join((*prefix, key))
        if not prefix and key in loader_keys:
            out[path] = LOADER_METADATA
            if key in LOADER_FETCHED_VERBATIM and isinstance(value, dict):
                # Its own checkpoint file, stored untouched: the subtree is that
                # component speaking for itself, so it is its own checkpoint.
                out.update(_map_provenance(value, value, frozenset(),
                                           (*prefix, key)))
            continue
        in_checkpoint = isinstance(checkpoint, dict) and key in checkpoint
        declared = _member(checkpoint, key) if in_checkpoint else None
        if not in_checkpoint:
            out[path] = CLASS_DEFAULT
        elif _values_agree(declared, value):
            out[path] = CHECKPOINT_DECLARED
        else:
            out[path] = CLASS_DEFAULT    # key origin != value origin
        if isinstance(value, dict):
            out.update(_map_provenance(
                declared if in_checkpoint else None, value, loader_keys,
                (*prefix, key)))
    return out


def _overlay(checkpoint: Any, hydrated: Any, prefix: tuple = ()) -> dict:
    """{dotted path: value} for everything the CLASS supplied — the class's
    candidates, kept apart from the checkpoint's."""
    out: dict[str, Any] = {}
    if not isinstance(hydrated, dict):
        return out
    for key, value in hydrated.items():
        if not isinstance(key, str):
            continue
        path = ".".join((*prefix, key))
        in_checkpoint = isinstance(checkpoint, dict) and key in checkpoint
        declared = _member(checkpoint, key) if in_checkpoint else None
        if not in_checkpoint or not _values_agree(declared, value):
            out[path] = value
        if isinstance(value, dict):
            out.update(_overlay(declared if in_checkpoint else None, value,
                                (*prefix, key)))
    return out


def prepare_document(raw: Any, *, loader_keys: frozenset = frozenset(),
                     merge: bool = True,
                     already_prepared: "PreparedDocument | None" = None,
                     ) -> PreparedDocument:
    """Prepare ONE document for parsing, and record where every field came from.

    ``raw`` is the document as received. ``loader_keys`` are the top-level keys
    the LOADER injected — passed in because only the loader knows what it added,
    and inferring it from an underscore prefix would be a guess (``_name_or_path``
    and ``_class_name`` are genuinely the checkpoint's words).

    ``already_prepared`` carries a preparation done earlier (the loader hydrates
    a component at fetch time, so the value reaching the parser is no longer the
    file's words). Without it the second preparation would compare a hydrated
    dict against itself and credit the CLASS's work to the checkpoint — the
    exact confusion of "present before hydration" with "declared by the file".

    Preparation never raises: an unknown/unregistered ``model_type``, or a class
    that rejects the config, yields the raw document with a typed ``failure`` and
    NO provenance claims. Failing to hydrate is a known unknown; guessing is not.
    """
    if already_prepared is not None:
        # IDEMPOTENT, but only about the document it actually describes:
        # returning any handed-in preparation would let a caller pair one file's
        # provenance with another file's fields.
        if not already_prepared.describes(raw):
            raise ValueError(
                "already_prepared describes a different document than the one "
                "being prepared — a preparation may not be transplanted onto "
                "an object it never examined")
        return already_prepared
    if not isinstance(raw, dict):
        return PreparedDocument(
            document=raw, checkpoint=raw, provenance={},
            failure=PreparationFailure("no_mapping", "snapshot",
                                       f"{type(raw).__name__} is not a mapping"))
    checkpoint = _snapshot(raw)
    model_type = raw.get("model_type")
    if not model_type:
        # No address for a class, so nothing was hydrated: the document IS the
        # checkpoint (minus whatever the loader declared it injected).
        # No address for a config class, so nothing was hydrated -- an OUTCOME,
        # not a failure: the document IS the checkpoint, and every field in it
        # is provably the file's own word.  Marking this broken would condemn
        # every plain config.json that omits model_type.
        return PreparedDocument(
            document=raw, checkpoint=checkpoint,
            provenance=_map_provenance(checkpoint, raw, loader_keys))
    try:
        from transformers import AutoConfig
        document = AutoConfig.for_model(
            str(model_type),
            **{k: v for k, v in raw.items()
               if not k.startswith("_") and k != "model_type"}).to_dict()
    except Exception as exc:
        return PreparedDocument(
            document=raw, checkpoint=checkpoint,
            provenance=_map_provenance(checkpoint, raw, loader_keys),
            failure=PreparationFailure(
                "class_rejected", "hydrate", f"{type(exc).__name__}: {exc}"[:200]))
    for key, value in raw.items():       # loader stamps / private context survive
        if key.startswith("_"):
            document[key] = value
    overlay = _overlay(checkpoint, document)
    # ``merge=False`` is SHADOW MODE: prepare and record everything, but let the
    # readers keep seeing only the checkpoint.  Universal hydration may not
    # change one architecture until an arbiter can rank a class default BELOW a
    # declaration — otherwise the parse computes a wrong answer and provenance
    # merely labels the wrong answer honestly.
    return PreparedDocument(
        document=document if merge else raw, checkpoint=checkpoint,
        class_overlay=overlay,
        provenance=_map_provenance(checkpoint,
                                   document if merge else raw, loader_keys))


#: Top-level keys the LOADER synthesizes — it is the only thing that knows, so
#: they are DECLARED here rather than guessed from an underscore prefix
#: (``_name_or_path`` and ``_class_name`` are genuinely the checkpoint's words,
#: which is why "starts with _" is not the test).
LOADER_STAMPS = frozenset({
    "_repo_id",                # the model tag the user typed (parser/loader)
    "_pipeline_class_name",    # read off the pipeline index by the loader
    "_text_encoder_configs",   # component configs the loader FETCHED
    "_vae_config",
    "_scheduler_config",
})

#: Loader-fetched subtrees the loader downloads VERBATIM — another component's
#: own config.json, stored untouched.  The loader authored the KEY; it did not
#: author the CONTENTS, so the subtree is that component's checkpoint speaking
#: for itself and its fields are checkpoint_declared.  Marking the key
#: loader_metadata and then skipping its children left every VAE and scheduler
#: read with no origin at all — the loader's envelope was mistaken for the
#: letter inside it.
#:
#: ``_text_encoder_configs`` is deliberately NOT here: the loader hydrates each
#: encoder through its config class at fetch time and discards the pre-hydration
#: snapshot, so those children are a checkpoint/class MIXTURE that no later
#: reader can separate.  They stay unestablished — an honest unknown, and the
#: exact reason the loader must retain a PreparedDocument per component.
LOADER_FETCHED_VERBATIM = frozenset({"_vae_config", "_scheduler_config"})


def checkpoint_provenance(raw: Any, *,
                          loader_keys: frozenset = LOADER_STAMPS) -> dict:
    """Provenance for a document that was NOT hydrated: the file's own words.

    Every field is the checkpoint's except the loader's declared stamps. Used at
    a document boundary that performs no class hydration, so nothing here is a
    class default — and saying so explicitly is what stops "" from silently
    counting as the checkpoint's word downstream."""
    if not isinstance(raw, dict):
        return {}
    return _map_provenance(raw, raw, loader_keys)


__all__ = ["PreparedDocument", "DocumentBinding", "PreparationFailure",
           "prepare_document",
           "checkpoint_provenance", "LOADER_STAMPS",
           "LOADER_FETCHED_VERBATIM"]
