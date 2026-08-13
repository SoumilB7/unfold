"""U2.2a — a recorded config path must be TRUE of the document that was read.

The occurrence key is defined as "what was actually supplied, where".  Nothing
enforced the "where": a container scope glued its prefix onto every enclosed
read, including reads of a DIFFERENT object, so the ledger asserted exact dotted
paths that resolved nowhere and silently defeated every exact-path join.

Two laws hold the producer honest here:

* **A container applies only to reads OF the object it names.**  A builder
  legitimately reads both its sub-config and its host; the host read keeps its
  own true path.
* **A container names the spelling the document supplies.**  Choosing it with an
  alias-resolving accessor names a canonical key the document may not have.

and one records what a path is relative to:

* **A recursively-parsed slot is a DOCUMENT, not a container.**  Paths stay
  document-relative (the stable key a claim binding matches, identically
  standalone or embedded); the document's address is recorded beside them, so
  ``document_path + config_path`` is resolvable and therefore checkable.

The corpus tests are the real invariants — a false path anywhere in 25 witnesses
fails them.  Model names appear here (a test) and nowhere in the production fix,
which is expressed as ownership/spelling predicates.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import model_unfolder as mu
from test_support import bind_document
from model_unfolder.encoder_panel import hydrate_encoder_config_facts
from model_unfolder.evidence.document import LOADER_STAMPS, prepare_document
from model_unfolder.evidence import config_access as ca

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"


def _witnesses():
    return sorted(_CORPUS.glob("*.json"))


def _has(doc, parts) -> bool:
    cur = doc
    for key in parts:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False
    return True


def _parsed_document(cfg: dict) -> dict:
    """The document the parse actually READS.

    Production hydrates each encoder slot through its config class before
    parsing it (``hydrate_encoder_config_facts``), so the read document is the
    hydrated one.  Applying the SAME production function keeps this test honest
    about what it proves: paths are true of the document that was read.  It does
    NOT prove the source declared them — that is the separate provenance law
    below."""
    doc = json.loads(json.dumps(cfg))
    slots = doc.get("_text_encoder_configs")
    if isinstance(slots, dict):
        for key, sub in list(slots.items()):
            if isinstance(sub, dict):
                slots[key] = hydrate_encoder_config_facts(sub)
    return doc


def _ledger_for(cfg: dict) -> ca.ConfigAccessLedger:
    ledger = ca.ConfigAccessLedger()
    with ca.capture_events(ledger):
        mu.unfold(cfg).to_ir()
    return ledger


def _exact_present_events(ledger):
    for e in ledger.events:
        if e.present and e.path_exact and e.config_path:
            yield e


# --------------------------------------------------------------------------
# The corpus invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("witness", _witnesses(), ids=lambda p: p.stem)
def test_every_exact_path_resolves_in_the_document_that_was_read(witness):
    """An occurrence claiming an EXACT path must name a real location.

    Resolution is ``document_path + config_path`` against the top-level witness:
    ``config_path`` alone is document-relative, so a slot's occurrence is only
    checkable once its document address is recorded."""
    cfg = json.loads(witness.read_text())["config"]
    read_doc = _parsed_document(cfg)
    unresolvable = sorted({
        (e.component, ".".join(e.document_path), e.config_path)
        for e in _exact_present_events(_ledger_for(cfg))
        if not _has(read_doc, (*e.document_path, *e.config_path.split(".")))})
    assert not unresolvable, (
        f"{witness.stem}: occurrences claim an exact path that resolves nowhere "
        f"in the document the parser read — the join key is fabricated: "
        f"{unresolvable}")


@pytest.mark.parametrize("witness", _witnesses(), ids=lambda p: p.stem)
def test_every_checkpoint_census_row_exists_in_the_raw_checkpoint(witness):
    """The strongest form, and the point of the whole unit.

    The census claims to list what the CHECKPOINT declared. So every row must
    address a real location in the raw checkpoint file itself — not in a
    hydrated view of it, and not merely in some document. Three separate
    producer defects each broke this: fabricated container prefixes, class
    defaults presented as the checkpoint's word, and reads whose location was
    never established."""
    cfg = json.loads(witness.read_text())["config"]
    ir = mu.unfold(cfg).to_ir()
    access = (ir.get("extras") or {}).get("config_access") or {}
    roots = access.get("document_roots") or {}
    missing = sorted(
        (row["component"], row["path"])
        for row in access.get("accessed_unconsumed_exact") or []
        if not _has(cfg, (*roots.get(row["component"], []),
                          *row["path"].split("."))))
    assert not missing, (
        f"{witness.stem}: the census presents rows the raw checkpoint does not "
        f"contain: {missing}")


# --------------------------------------------------------------------------
# Hydration provenance — the class's words are not the checkpoint's
#
# An encoder slot is parsed through its installed config class, located by
# model_type.  That is identity-as-ADDRESS, which is lawful.  What is not lawful
# is letting what the class SUPPLIES masquerade as what the checkpoint DECLARED,
# because a fact detected from identity would then be indistinguishable from one
# detected from evidence.
#
# An earlier version of this file asserted "no class key is ever CONSUMED" and
# passed at zero.  It proved nothing: it measured the legacy inspected-vs-consumed
# gap that U2 exists to repair.  A class-supplied ``layer_types`` is recorded as
# merely `inspected` and still decides the entire per-layer mask schedule — the
# structural influence was real, the metric was blind to it.  Provenance is
# tracked per key instead.
# --------------------------------------------------------------------------

def _gemma2_raw():
    """A Gemma-2 checkpoint that declares NO layer schedule of its own."""
    return {"model_type": "gemma2", "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "vocab_size": 100, "intermediate_size": 512,
            "sliding_window": 128, "head_dim": 64}


def test_gemma2_class_supplied_schedule_is_class_default_never_checkpoint():
    """THE motivating case, both halves.

    Gemma-2's sliding/global alternation is never serialized — the config class
    supplies it (as ``layer_types`` on transformers 5.x, ``sliding_window_pattern``
    earlier). It is genuinely structural: it decides the mask of every layer. So
    it must be recorded as the CLASS's word, never the checkpoint's."""
    raw = _gemma2_raw()
    _prepared = prepare_document(raw, loader_keys=LOADER_STAMPS)
    doc, provenance = _prepared.document, _prepared.provenance

    schedule_keys = [k for k in ("layer_types", "sliding_window_pattern")
                     if k in doc and doc[k] is not None]
    assert schedule_keys, (
        "the installed Gemma-2 class must supply a layer schedule — if this "
        f"fails the class changed shape; doc keys: {sorted(doc)}")
    for key in schedule_keys:
        assert key not in raw, "the checkpoint must not declare it (fixture)"
        assert provenance[key] == ca.CLASS_DEFAULT, (
            f"{key} was supplied by the config class but is recorded as "
            f"{provenance[key]!r}")
        assert provenance[key] != ca.CHECKPOINT_DECLARED

    # and a field the checkpoint DID declare keeps its own provenance
    assert provenance["sliding_window"] == ca.CHECKPOINT_DECLARED


def test_gemma2_missing_executable_schedule_stays_unknown_after_u8():
    """A class overlay is still not checkpoint evidence after U8.

    An earlier version of this test asserted the EMBEDDED encoder builds the
    heterogeneous stack — but that only happened because the embedded prep used
    merge=True, letting the class overlay author architecture.  That was the
    R1-vet violation.  In U2 shadow mode (merge=False everywhere) the class
    overlay authors NOTHING, so both the embedded and the standalone Gemma-2 are
    uniformly sliding.  The schedule's real influence is a U8 counterfactual —
    recorded, deferred — never a U2 success condition."""
    from model_unfolder.encoder_panel import normalize_encoder_config

    # embedded: shadow mode -> NO class-authored alternation
    spec = normalize_encoder_config(_gemma2_raw())
    groups = (spec.get("sub_model") or {}).get("groups") or []
    tags = [g.get("tag") for g in groups]
    assert not any("sliding window" in str(t) for t in tags), (
        f"the class overlay authored an embedded schedule (tags={tags}) — §3.3 "
        "forbids a merge-driven structural delta before U8")

    # Standalone is equally honest: the raw checkpoint omits the executable
    # selector operand, so source cannot choose a schedule and stays unknown.
    flat = mu.unfold(_gemma2_raw()).to_ir()
    masks = {(layer.get("attention") or {}).get("mask")
             for layer in (flat.get("layers") or [])}
    assert masks == {"unknown"}, (
        f"missing executable schedule evidence must stay unknown, got {masks}")


def test_a_class_supplied_field_is_not_a_checkpoint_occurrence():
    """The census asks "classify this declaration" — an incoherent task for a
    declaration the checkpoint never made. Excluded from the checkpoint census,
    and VISIBLE in its own class rather than vanished."""
    doc = {"declared": 1, "from_class": 2}
    provenance = {"declared": ca.CHECKPOINT_DECLARED,
                  "from_class": ca.CLASS_DEFAULT}
    with ca.capture_events() as ledger, \
            ca.bound_document(bind_document(doc, provenance)):
        ca.emit("declared", intent="inspected", present=True,
                source_obj_id=id(doc))
        ca.emit("from_class", intent="inspected", present=True,
                source_obj_id=id(doc))
    paths = {k.config_path for k in ledger.unconsumed_occurrences()}
    assert paths == {"declared"}, paths
    assert ("root", "from_class", ca.CLASS_DEFAULT) in \
        ledger.non_checkpoint_occurrences()


@pytest.mark.parametrize("witness", _witnesses(), ids=lambda p: p.stem)
def test_no_checkpoint_occurrence_is_really_a_class_supplied_field(witness):
    """Corpus-wide: nothing the config class supplied may appear in the
    checkpoint census under any witness."""
    cfg = json.loads(witness.read_text())["config"]
    if not isinstance(cfg.get("_text_encoder_configs"), dict):
        pytest.skip("no recursively-parsed encoder slot in this witness")
    ledger = _ledger_for(cfg)
    class_paths = {(c, p) for c, p, _ in ledger.non_checkpoint_occurrences()}
    census = {(k.component_path, k.config_path)
              for k in ledger.unconsumed_occurrences()}
    assert not (census & class_paths), sorted(census & class_paths)


# --------------------------------------------------------------------------
# Path-proof poisons — exactness is PROVEN, never asserted
#
# Exactness is two claims and both must be discharged: the path addresses a real
# location in the document, AND the object the document places there is the
# object the reader actually read.  Proving only the first lets an unrelated
# object borrow a real path; proving neither lets a producer invent one.
# --------------------------------------------------------------------------

def test_a_nonexistent_explicit_path_is_an_error():
    """A producer may not author an address the document disproves."""
    doc = {"real": 1}
    with ca.capture_events(), ca.bound_document(bind_document(doc)):
        with pytest.raises(ValueError, match="not proven by the document"):
            ca.emit("x", intent="inspected", present=True,
                    config_path="does.not.exist", source_obj_id=id(doc))


def test_a_real_path_cannot_be_borrowed_by_an_unrelated_object():
    """The sharpest form: the path EXISTS, so mere resolution certifies it.

    Only identity separates the document's own ``real`` from a foreign object's
    ``real`` — the value would be someone else's and the occurrence key ("what
    was supplied, WHERE") a fiction that resolves."""
    doc = {"real": 1}
    unrelated = {"real": 999}
    with ca.capture_events(), ca.bound_document(bind_document(doc)):
        with pytest.raises(ValueError, match="not proven by the document"):
            ca.emit("real", intent="inspected", present=True,
                    config_path="real", source_obj_id=id(unrelated))
    # control: the document's OWN object, same path, is lawful and exact
    with ca.capture_events() as ledger, ca.bound_document(bind_document(doc)):
        ca.emit("real", intent="inspected", present=True,
                config_path="real", source_obj_id=id(doc))
    assert ledger.events[0].path_exact is True


def test_a_named_container_may_not_speak_for_an_unidentified_read():
    """A read that does not say which object it came from cannot be shown to
    belong in the container — absence of a contradiction is not evidence."""
    doc = {"sub": {"leaf": 2}}
    with ca.capture_events() as ledger, ca.bound_document(bind_document(doc)), \
            ca.config_container(("sub",), obj=doc["sub"]):
        ca.emit("leaf", intent="inspected", present=True, source_obj_id=None)
    assert ledger.events[0].config_path == "leaf"      # NOT sub.leaf
    assert ledger.events[0].path_exact is False


def test_consuming_an_occurrence_keeps_the_proof_its_inspection_had():
    """Occurrence identity must survive the whole lifecycle.

    A consumption that re-emits from a value alone loses the location its
    inspection proved — and the consumption is what a claim binding joins on,
    so the fact ends up bound to an unproven address."""
    doc = {"sub": {"leaf": 2}}
    with ca.capture_events() as ledger, ca.bound_document(bind_document(doc)):
        resolution = ca.resolve(doc["sub"], "leaf", (), path=("sub",))
        resolution.consume(fact_owner="root", fact_key="x")
    by_intent = {e.intent: e for e in ledger.events}
    assert by_intent["inspected"].config_path == "sub.leaf"
    assert by_intent["consumed"].config_path == "sub.leaf"
    assert by_intent["consumed"].path_exact is True


def test_priority_resolution_consumes_the_field_it_actually_found():
    """A priority chain runs over DISTINCT fields, not rival spellings: the
    declared order is the meaning, disagreement is not ambiguity, and only the
    winner is consumed — carrying the exact path it was found at."""
    doc = {"vision_config": {"embed_dim": 1280, "hidden_size": 3584}}
    vision = doc["vision_config"]
    with ca.capture_events() as ledger, ca.bound_document(bind_document(doc)), \
            ca.config_container(("vision_config",), obj=vision):
        resolution = ca.resolve_priority(vision, ("embed_dim", "hidden_size"))
        value = resolution.consume(fact_owner="root.vision", fact_key="hidden_size",
                                   mechanism="encoder_width")
    assert value == 1280                                   # priority, not the sibling
    consumed = [e for e in ledger.events if e.intent == "consumed"]
    assert len(consumed) == 1
    assert consumed[0].config_path == "vision_config.embed_dim"
    assert consumed[0].path_exact is True
    # the LOSING sibling is a different field, not this fact's to clear
    assert not any(e.config_path == "vision_config.hidden_size"
                   for e in ledger.events)


def test_qwen2vl_vision_width_is_consumed_at_its_exact_path():
    """The real witness: the tower width must be consumed at
    ``vision_config.embed_dim`` — the path a claim binding joins on."""
    cfg = json.loads((_CORPUS / "qwen2-vl-7b-instruct.json").read_text())["config"]
    consumed = [e for e in _ledger_for(cfg).events
                if e.intent == "consumed" and e.mechanism == "encoder_width"
                and e.component == "root.vision"]
    assert consumed, "the witness must consume a vision tower width"
    assert all(e.config_path == "vision_config.embed_dim" for e in consumed), \
        [e.config_path for e in consumed]
    assert all(e.path_exact for e in consumed)


# --------------------------------------------------------------------------
# Poisons — each fails if its producer law is removed
# --------------------------------------------------------------------------

def test_container_does_not_prefix_a_read_of_another_object():
    """The law that makes an exact path trustworthy: a builder reading its HOST
    inside a sub-config's container must not inherit that prefix."""
    sub = {"hidden_size": 8}
    host = {"sub_config": sub, "token_id": 3}
    with ca.capture_events() as ledger:
        with ca.config_container(("sub_config",), obj=sub):
            ca.emit("hidden_size", intent="inspected", present=True,
                    source_obj_id=id(sub))
            ca.emit("token_id", intent="inspected", present=True,
                    source_obj_id=id(host))
    by_leaf = {e.canonical: e for e in ledger.events}
    assert by_leaf["hidden_size"].config_path == "sub_config.hidden_size"
    assert by_leaf["hidden_size"].path_exact is True
    # the host read keeps its own true path and admits it is not exact
    assert by_leaf["token_id"].config_path == "token_id"
    assert by_leaf["token_id"].path_exact is False


def test_container_names_the_spelling_the_document_supplies():
    """A container chosen by an alias-resolving read would name a canonical key
    the document never had."""
    doc = {"rope_scaling": {"rope_type": "default"}}
    assert ca.present_spelling(doc, ("rope_parameters", "rope_scaling")) == "rope_scaling"
    # and when the document supplies neither, no container may be declared
    assert ca.present_spelling({"unrelated": 1},
                               ("rope_parameters", "rope_scaling")) is None


def test_container_scoped_resolves_the_object_it_names():
    """The decorator declares "these reads are of cfg.<path>" — so a read of cfg
    itself is outside it by construction, with no hand-placed escape."""
    inner = {"num_train_timesteps": 1000}
    cfg = {"_scheduler_config": inner, "scheduler": "Euler"}

    @ca.container_scoped(("_scheduler_config",))
    def _read(config):
        ca.emit("num_train_timesteps", intent="inspected", present=True,
                source_obj_id=id(inner))
        ca.emit("scheduler", intent="inspected", present=True,
                source_obj_id=id(config))

    with ca.capture_events() as ledger:
        _read(cfg)
    paths = {e.canonical: e.config_path for e in ledger.events}
    assert paths["num_train_timesteps"] == "_scheduler_config.num_train_timesteps"
    assert paths["scheduler"] == "scheduler"      # NOT _scheduler_config.scheduler


def test_container_scoped_names_nothing_when_its_object_is_absent():
    """A declared container whose object this document lacks may not prefix
    anything — no read can be a read of an object that is not there."""
    cfg = {"scheduler": "Euler"}

    @ca.container_scoped(("_scheduler_config",))
    def _read(config):
        ca.emit("scheduler", intent="inspected", present=True,
                source_obj_id=id(config))

    with ca.capture_events() as ledger:
        _read(cfg)
    assert ledger.events[0].config_path == "scheduler"


def test_bound_document_keeps_paths_relative_and_records_the_address():
    """A slot is a document: its reads keep the host-independent join key a
    claim binding matches, and the address travels beside the path.  U2-R7:
    entered through a BINDING — the path-only overload is deleted."""
    doc = {"vision_config": {"hidden_size": 3584}}
    binding = bind_document(
        doc, path=("_text_encoder_configs", "text_encoder"))
    with ca.capture_events() as ledger:
        with ca.bound_document(binding):
            # the source object is the one the document PLACES at the address
            # (the binding makes the path-proof law bite — the deleted loose
            # overload could not check this)
            ca.emit("hidden_size", intent="inspected", present=True,
                    config_path="vision_config.hidden_size",
                    source_obj_id=id(doc["vision_config"]))
    event = ledger.events[0]
    assert event.config_path == "vision_config.hidden_size"     # unchanged key
    assert event.document_path == ("_text_encoder_configs", "text_encoder")


def test_bound_document_clears_an_enclosing_container():
    """A container names an object in the document being LEFT; it can never
    describe a read inside the new one."""
    outer = {"a": 1}
    binding = bind_document(
        {"hidden_size": 8}, path=("_text_encoder_configs", "text_encoder"))
    with ca.capture_events() as ledger:
        with ca.config_container(("_vae_config",), obj=outer):
            with ca.bound_document(binding):
                ca.emit("hidden_size", intent="inspected", present=True,
                        source_obj_id=id(outer))
    assert ledger.events[0].config_path == "hidden_size"


def test_the_loose_document_scope_overload_is_deleted():
    """U2-R7: entering a document REQUIRES a DocumentBinding — the loose
    ``document_scope(path, obj=, provenance=)`` entry no longer exists as
    public API, so an object and an unrelated provenance map can never be
    paired again."""
    assert not hasattr(ca, "document_scope")
    assert "document_scope" not in ca.__all__
