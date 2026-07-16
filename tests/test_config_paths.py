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
from model_unfolder.encoder_panel import hydrate_encoder_config_facts
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
def test_no_structural_fact_is_authored_from_a_key_the_source_never_supplied(witness):
    """Hydration fills class defaults keyed by ``model_type`` — identity.

    That is sanctioned as code evidence (the config class states what the model
    constructs), but a fact CONSUMED from a class-supplied key would be a
    structural claim the document never made, recorded indistinguishably from a
    declared one — detection from identity wearing a config's clothes.

    This currently holds at zero corpus-wide and is locked here so it can never
    start: a class default may be inspected, never consumed into a fact."""
    cfg = json.loads(witness.read_text())["config"]
    slots = cfg.get("_text_encoder_configs")
    if not isinstance(slots, dict):
        pytest.skip("no recursively-parsed encoder slot in this witness")
    raw = {k: v for k, v in slots.items() if isinstance(v, dict)}
    hydrated = {k: hydrate_encoder_config_facts(v) for k, v in raw.items()}

    offenders = []
    for e in _ledger_for(cfg).events:
        if e.intent != "consumed" or not e.present or not e.document_path:
            continue
        slot = e.document_path[-1]
        if slot not in raw:
            continue
        parts = e.config_path.split(".")
        if not _has(raw[slot], parts) and _has(hydrated[slot], parts):
            offenders.append((e.component, e.config_path, e.fact_key))
    assert not offenders, (
        f"{witness.stem}: structural facts consumed from keys supplied by the "
        f"config CLASS, not the document: {sorted(offenders)}")


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


def test_document_scope_keeps_paths_relative_and_records_the_address():
    """A slot is a document: its reads keep the host-independent join key a
    claim binding matches, and the address travels beside the path."""
    with ca.capture_events() as ledger:
        with ca.document_scope(("_text_encoder_configs", "text_encoder")):
            ca.emit("hidden_size", intent="inspected", present=True,
                    config_path="vision_config.hidden_size")
    event = ledger.events[0]
    assert event.config_path == "vision_config.hidden_size"     # unchanged key
    assert event.document_path == ("_text_encoder_configs", "text_encoder")


def test_document_scope_clears_an_enclosing_container():
    """A container names an object in the document being LEFT; it can never
    describe a read inside the new one."""
    outer = {"a": 1}
    with ca.capture_events() as ledger:
        with ca.config_container(("_vae_config",), obj=outer):
            with ca.document_scope(("_text_encoder_configs", "text_encoder")):
                ca.emit("hidden_size", intent="inspected", present=True,
                        source_obj_id=id(outer))
    assert ledger.events[0].config_path == "hidden_size"
