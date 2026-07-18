"""U0 (§20.3 / §9-R0.3) — the §5 adversarial probes, pinned BEFORE production changes.

Each ``xfail(strict=True)`` encodes the CORRECT contract from
``docs/EVIDENCE_CODE_AUTHORITY_MASTER_PLAN.md`` §5, currently fails for the
exact defect the independent audit named, and cites the unit that must flip it
(strict: the fixing unit MUST remove the marker in the same commit, or the
suite fails on the unexpected pass).  Controls pin the taint shapes the guard
ALREADY catches — U0 pre-verification showed the net is stronger than §5.5's
prose (helper / bool / mapping / ternary / decorated-class shapes all fire),
so those become regression pins and the xfails mark only the true holes.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import textwrap

import pytest

import model_unfolder as mu
from model_unfolder.evidence.config_access import capture_events, owner_scope
from model_unfolder.evidence.identity_guard import scan_identity_source
from model_unfolder.evidence.patterns import decoder_attention_sinks_from_files
from model_unfolder.evidence.structural_writes import StructuralWrite
from test_support import LLAMA

# A GPT-2-shaped checkpoint: ONLY the historical alias spellings are present.
GPT2_ALIASED = {
    "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
    "n_embd": 64, "n_layer": 2, "n_head": 4, "n_inner": 256, "vocab_size": 50257,
}
_CANONICALS = {"hidden_size", "num_hidden_layers", "num_attention_heads",
               "intermediate_size"}
_ALIASES = {"n_embd", "n_layer", "n_head", "n_inner"}


def _capture(cfg):
    with capture_events() as ledger:
        with owner_scope("root"):
            ir = mu.unfold(cfg).to_ir()
    return ledger, ir


def _unread(cfg) -> list[str]:
    ir = mu.unfold(cfg).to_ir()
    return ((ir.get("extras") or {}).get("config_audit") or {}).get("unread", [])


# --------------------------------------------------------------------------- #
# §5.1 — H3's live path does not use the exact alias resolver (unit U1 / T-01)
# --------------------------------------------------------------------------- #

def test_p1_every_consumed_event_cites_a_spelling_present_in_the_config():
    """FIXED by U1 (Contract A resolver): permanent guard — a consumed event
    cites the exact supplying spelling, never an absent canonical name."""
    ledger, _ = _capture(GPT2_ALIASED)
    fictional = sorted({
        (e.alias or e.canonical) for e in ledger.events
        if e.intent == "consumed" and (e.alias or e.canonical) not in GPT2_ALIASED
    })
    assert not fictional, f"consumed events citing absent spellings: {fictional}"


def test_p2_the_supplying_alias_spelling_is_the_consumed_read():
    """FIXED by U1: permanent guard — the supplying spelling IS the recorded
    consumed read; the real aliases never sit in accessed-but-unconsumed."""
    ledger, _ = _capture(GPT2_ALIASED)
    consumed_spellings = {e.alias for e in ledger.events if e.intent == "consumed"}
    missing = _ALIASES - consumed_spellings
    assert missing == set(), (
        f"supplying spellings not cited by any consumed event: {sorted(missing)}; "
        f"cited instead: {sorted(s for s in consumed_spellings if s)}")


def test_p3_absent_fields_never_enter_accessed():
    """FIXED by U1 (§5.1 Decision): permanent guard — accessed is PRESENT-ONLY
    file spellings; absence lives only in absent_default premises."""
    _, ir = _capture(GPT2_ALIASED)
    accessed = set(((ir.get("extras") or {}).get("config_audit") or {}).get("accessed", []))
    leaked = _CANONICALS & accessed
    assert leaked == set(), f"absent canonicals inside accessed: {sorted(leaked)}"


def test_p4_conflicting_aliases_become_typed_ambiguity():
    """FIXED by U1: permanent guard — unequal simultaneous aliases record a
    typed ambiguity and no value is silently chosen."""
    ledger, _ = _capture({**GPT2_ALIASED, "hidden_size": 96})  # 96 vs n_embd=64
    ambiguous = [e for e in ledger.events
                 if e.intent == "ambiguous" and e.canonical == "hidden_size"]
    assert ambiguous, "hidden_size=96 vs n_embd=64 selected silently — no ambiguity event"


# --------------------------------------------------------------------------- #
# §5.2 — blocking unread coverage is bare-key based (unit U1)
# --------------------------------------------------------------------------- #

def test_p5_a_sibling_scope_cannot_clear_nested_unread_debt():
    """FIXED by U1 (§20.4.8): permanent guard — unread coverage is an
    exact-path/owner join; a nested path under an unmapped container has no
    owner, so no sibling's read can clear it."""
    unread = _unread({**LLAMA, "aux_tower_config": {"hidden_size": 123}})
    assert "aux_tower_config.hidden_size" in unread, (
        "the nested component's unread hidden_size was cleared by the root's read")


# --------------------------------------------------------------------------- #
# §5.4 — pending-projection debt is excused by leaf key (unit U1 step 5 / R2)
# --------------------------------------------------------------------------- #

def test_p6_pending_debt_excusal_is_owner_tight():
    """FIXED by U1 (§5.4 Decision): permanent guard — the excusal joins on the
    registered owner + canonical (root.vae's act_fn never excuses a
    transformer-root field), and excused paths stay visible as
    ``pending_projection`` diagnostics."""
    unread = _unread({**LLAMA, "temporal_compression_ratio": 4})
    assert "temporal_compression_ratio" in unread, (
        "a transformer-root temporal_compression_ratio was excused by the "
        "root.vae-registered debt entry (key-only join)")


# --------------------------------------------------------------------------- #
# §5.5 — taint shapes: pin what the guard ALREADY catches (controls)
# --------------------------------------------------------------------------- #

_CAUGHT_SHAPES = [
    ("identity_name_branch", "identity_branch", '''
        def label(model_type):
            if model_type == "flux2":
                return {"kind": "mmdit"}
            return {"kind": "dit"}
    '''),
    ("helper_indirection", "class_identity_branch", '''
        def _role(class_name):
            if class_name == "Flux2Transformer2DModel":
                return "mmdit"
            return "dit"
        def build(model):
            spec = {}
            spec["kind"] = _role(type(model).__name__)
            return spec
    '''),
    ("bool_then_sink", "class_identity_branch", '''
        def build(class_name):
            is_flux = class_name == "Flux2Transformer2DModel"
            return {"kind": "mmdit" if is_flux else "dit"}
    '''),
    ("mapping_lookup", "class_keyed_literal", '''
        _KIND_BY_CLASS = {"Flux2Transformer2DModel": "mmdit",
                          "PixArtTransformer2DModel": "dit"}
        def build(class_name):
            spec = {}
            spec["kind"] = _KIND_BY_CLASS.get(class_name, "dit")
            return spec
    '''),
    ("ternary_assign", "class_identity_branch", '''
        def build(class_name):
            spec = {}
            spec["kind"] = "mmdit" if class_name == "Flux2Transformer2DModel" else "dit"
            return spec
    '''),
    ("decorated_class_identity_branch", "class_identity_branch", '''
        from x import identity_display
        @identity_display
        def label(class_name):
            if class_name == "Flux2Transformer2DModel":
                return {"kind": "mmdit"}
            return {"kind": "dit"}
    '''),
]


@pytest.mark.parametrize("shape,expected_kind,src",
                         _CAUGHT_SHAPES, ids=[s[0] for s in _CAUGHT_SHAPES])
def test_p7_controls_identity_taint_shapes_stay_caught(shape, expected_kind, src):
    """U0 pre-verification proved these shapes FIRE today — pin them so no later
    unit can regress the guard while fixing the xfail holes below."""
    findings = scan_identity_source(textwrap.dedent(src), path=f"{shape}.py")
    assert any(f.kind == expected_kind for f in findings), (
        f"{shape} no longer caught (expected {expected_kind}; "
        f"got {[(f.kind, f.line) for f in findings]})")


# --------------------------------------------------------------------------- #
# §5.6 — the decorator holes that DO exist (unit R3)
# --------------------------------------------------------------------------- #

def test_p8_decorated_identity_name_branch_is_not_blessed():
    # U2 (Soumil's proviso): CLOSED — the @identity marker no longer exempts a
    # branch that writes a STRUCTURAL SINK, so this decorated shape is caught.
    findings = scan_identity_source(textwrap.dedent('''
        from x import identity_display
        @identity_display
        def label(model_type):
            if model_type == "flux2":
                return {"kind": "mmdit"}
            return {"kind": "dit"}
    '''), path="decorated_identity_name.py")
    assert findings, "identity-name branch writing a structural sink was blessed by the decorator"


@pytest.mark.xfail(strict=True, reason="§5.6/R3: a decorator alone can never bless a "
                   "structural sink — an @identity_display function returning a "
                   "structural-keyed dict unconditionally is invisible today")
def test_p9_decorator_cannot_bless_an_unconditional_structural_return():
    findings = scan_identity_source(textwrap.dedent('''
        from x import identity_display
        @identity_display
        def label(config):
            return {"kind": "flux"}
    '''), path="decorated_structural_return.py")
    assert findings, "structural return under @identity_display produced no finding"


# --------------------------------------------------------------------------- #
# §5.7 — structural-write identity is too weak (unit U2 / C-09)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="§5.7/U2(C-09): the census key must be "
                   "(module, symbol, sink, target) so a NEW author of a familiar "
                   "target grows the census; today key is (sink, target)")
def test_p10_structural_write_key_is_site_qualified():
    first = StructuralWrite(sink="spec", target="AttentionSpec",
                            module="adapters/transformer/parser.py", symbol="parse")
    second = StructuralWrite(sink="spec", target="AttentionSpec",
                             module="adapters/diffusor/parser.py", symbol="build")
    assert first.key != second.key, (
        "two different authoring sites collapse to one census key — a new module "
        "can write a familiar structural target invisibly")


# --------------------------------------------------------------------------- #
# §5.8 — debt rows must carry their owner (unit U2)
# --------------------------------------------------------------------------- #

def test_p11_drawn_unledgered_debt_carries_an_owner_field():
    """CLOSED by U2-R6: the ONE StructuralDebt register replaced
    DrawnUnledgeredFact (which claimed an owner in its docstring but had no
    field) — every debt row now carries a typed owner plus writer, consumer,
    unit and a checkable deletion condition."""
    from model_unfolder.evidence.structural_debt import StructuralDebt
    field_names = {f.name for f in dataclasses.fields(StructuralDebt)}
    assert {"owner", "writer_module", "writer_symbol", "last_consumer",
            "migration_unit", "deletion_condition"} <= field_names


# --------------------------------------------------------------------------- #
# §5.9 — sinks reader must bind to the decoder owner (unit U6)
# --------------------------------------------------------------------------- #

_SIBLING_ONLY_SINKS = '''
import torch
import torch.nn as nn


class AuxiliaryPoolerAttention(nn.Module):
    """NOT the decoder's attention — but it carries the sink signal."""

    def __init__(self, config):
        super().__init__()
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        scores = self.q_proj(x)
        combined = torch.cat([scores, self.sinks.expand(1)], dim=-1)
        return torch.softmax(combined, dim=-1)


class MainDecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return torch.softmax(self.q_proj(x), dim=-1)
'''


@pytest.mark.xfail(strict=True, reason="§5.9/U6: the sinks reader scans ALL classes "
                   "in the files, so a sibling attention component votes for the "
                   "decoder; it must bind to the exact decoder-attention owner")
def test_p12_sinks_reader_binds_to_the_decoder_owner():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "modeling_probe.py"
        path.write_text(_SIBLING_ONLY_SINKS)
        assert not decoder_attention_sinks_from_files([path]), (
            "a sibling class's sinks parameter was attributed to the decoder")


# --------------------------------------------------------------------------- #
# §5.3 — the promised nets are not live (unit U1 steps 9-10)
# --------------------------------------------------------------------------- #

def test_p13_consumed_but_unprojected_net_is_published():
    """FIXED by U1 (§20.4.9), cut over by U2: net-2 is published as the
    ``config_consumed_unreceipted`` Sable check — blocking inside receipted
    (owner, mechanism) scopes and advisory elsewhere."""
    from model_unfolder.sable import sable

    report = sable(LLAMA, render_images=False)
    names = {check.name for check in report.checks}
    assert any(("consumed" in name and ("unreceipted" in name or "unprojected" in name))
               for name in names), (
        f"no published consumed-but-unreceipted net among checks: {sorted(names)}")


def test_p13_content_exact_unreceipted_targets():
    """REC-6 (§12.8): net-2 asserts CONTENT — known exact consumed targets on
    both adapters, and the primitive clears ONLY a matching receipt."""
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.evidence.config_access import (
        ConfigAccessEvent, ConfigAccessLedger)

    ir = mu.unfold(LLAMA).to_ir()
    net2 = ((ir.get("extras") or {}).get("config_access") or {}).get(
        "consumed_unprojected") or []
    assert "root:num_hidden_layers" in net2, net2[:6]

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    flux = json.loads((corpus / "flux-2-dev.json").read_text())["config"]
    ir2 = mu.unfold(flux).to_ir()
    net2d = ((ir2.get("extras") or {}).get("config_access") or {}).get(
        "consumed_unprojected") or []
    assert "root.denoiser:num_layers" in net2d, net2d[:6]

    # primitive: a synthetic exact receipt clears ONLY its own target
    ev = ConfigAccessEvent(component="root", config_path="hidden_size",
                           canonical="hidden_size", alias="hidden_size",
                           present=True, intent="consumed",
                           fact_owner="model", fact_key="hidden_size")
    ev2 = ConfigAccessEvent(component="root", config_path="vocab_size",
                            canonical="vocab_size", alias="vocab_size",
                            present=True, intent="consumed",
                            fact_owner="model", fact_key="vocab_size")
    led = ConfigAccessLedger([ev, ev2])
    left = led.consumed_but_unprojected(projected={("model", "hidden_size")})
    assert ("root", "vocab_size") in left and ("root", "hidden_size") not in left


def test_audit_incomplete_names_unmigrated_owners():
    """REC-6 (§12.6): an owner with zero consumed events is NAMED, never
    empty-clean.  U2-R7 gave every flux owner a real consumption (scheduler
    included), so the live witness is clean — the MECHANISM is pinned by
    stripping the scheduler's consumed fields, leaving root.scheduler with
    only ignored/inspected events."""
    import copy
    import json
    import pathlib

    import model_unfolder as mu
    from model_unfolder.sable import sable

    corpus = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
    flux = json.loads((corpus / "flux-2-dev.json").read_text())["config"]
    rep = sable(flux, render_images=False)
    check = next(c for c in rep.checks if c.name == "config_audit_incomplete")
    assert check.blocking is True           # U2-R8: the staged period ended
    assert check.findings == [], check.findings   # every owner consumes now

    starved = copy.deepcopy(flux)
    for consumed_key in ("num_train_timesteps", "prediction_type"):
        (starved.get("_scheduler_config") or {}).pop(consumed_key, None)
    rep2 = sable(starved, render_images=False)
    check2 = next(c for c in rep2.checks if c.name == "config_audit_incomplete")
    assert "root.scheduler" in check2.findings, check2.findings


# --------------------------------------------------------------------------- #
# COR-3 (§8) — unknown-safety at EVERY depth (permanent guards)
# --------------------------------------------------------------------------- #

def _depth_surfaces(cfg):
    import re
    from model_unfolder.params import estimate_params
    diagram = mu.unfold(cfg)
    html = diagram.to_html(standalone=True)
    return {
        "ir": diagram.to_ir(),
        "html_zero": bool(re.search(r"\b(dim|hidden) 0\b", html)),
        "expanded": json.dumps(diagram.to_json()),
        "params": estimate_params(diagram.ir),
    }


def test_cor3_conflicting_width_stays_unknown_at_every_depth():
    """§8.C.1: hidden_size=4096 + n_embd=64 — unknown through IR, HTML,
    expanded, and params; never dim 0 / hidden 0 / zero-math totals."""
    s = _depth_surfaces({**LLAMA, "n_embd": 64})
    assert s["ir"].get("hidden_size") in (None,)          # unknown, not 0
    assert not s["html_zero"]
    assert '"hidden_size": 0' not in s["expanded"]
    assert s["params"]["total"] is None and "incomplete" in s["params"]
    rows = (s["ir"].get("extras") or {}).get("config_ambiguity") or []
    assert len(rows) == 1                                  # idempotent, §8.B


def test_cor3_conflicting_heads_never_fabricate_geometry():
    """§8.C.2: FLUX head rivals — no zero-width claims at any depth."""
    from test_support import FLUX

    s = _depth_surfaces({**FLUX, "n_heads": 16})
    assert not s["html_zero"]
    assert '"hidden_size": 0' not in s["expanded"]
    assert s["params"]["total"] is None


def test_cor3_activation_rivals_author_nothing_and_block():
    """§8.C.3: hidden_act=gelu vs act_fn=silu — ONE ambiguity, no retry, the
    render is byte-identical to control (only code evidence draws), Sable
    blocks."""
    import sys
    sys.path.insert(0, str(pathlib.Path(mu.__file__).parent.parent))
    from model_unfolder.sable import sable
    from test_support import FLUX
    from test_support.preservation import html_meta

    conflicted = {**FLUX, "hidden_act": "gelu", "act_fn": "silu"}
    control = html_meta(mu.unfold(FLUX).to_html(standalone=True))
    actual = html_meta(mu.unfold(conflicted).to_html(standalone=True))
    assert actual["structural_sha256"] == control["structural_sha256"]
    rep = sable(conflicted, render_images=False)
    assert not rep.mechanical_passed
    amb = next(c for c in rep.checks if c.name == "config_ambiguity")
    assert any("hidden_act" in f for f in amb.findings)


def test_cor3_equal_aliases_preserve_current_output_exactly():
    """§8.C.4: equal redundant aliases change NOTHING."""
    from model_unfolder.params import estimate_params

    control = estimate_params(mu.unfold(LLAMA).ir)
    redundant = estimate_params(mu.unfold({**LLAMA, "n_embd": 4096}).ir)
    assert control["total"] == redundant["total"] and control["total"]


# --------------------------------------------------------------------------- #
# COR-4 (§9) — exact modality scopes and source-authoritative projector width
# --------------------------------------------------------------------------- #

_CORPUS = pathlib.Path(__file__).parent / "sable_test_corpus"


def _qwen2vl_corpus_cfg():
    return json.loads(
        (_CORPUS / "qwen2-vl-7b-instruct.json").read_text())["config"]


def _resolves(doc, config_path: str) -> bool:
    """Does this claimed path address a real location in the document?

    The predicate that makes an exact path mean something: a fabricated prefix
    looks precise and resolves nowhere."""
    cur = doc
    for key in config_path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False
    return True


# A modality host with NO resolvable modeling source: the projector width can
# only be honest-unknown, and sibling towers share the same leaf spelling.
_NO_SOURCE_MM = {
    "architectures": ["NoSuchThingForConditionalGeneration"],
    "model_type": "no_such_thing",
    "hidden_size": 4096, "num_hidden_layers": 2, "num_attention_heads": 8,
    "vocab_size": 100,
    "vision_config": {"hidden_size": 768, "num_hidden_layers": 2,
                      "num_attention_heads": 4, "image_size": 224,
                      "patch_size": 14},
    "image_token_id": 5,
}


def test_cor4_ce1_qwen2vl_events_are_dotted_and_width_is_source_bound():
    """§9.C.1: on the source-present corpus witness every authoritative
    vision/video event carries its exact dotted path (the legacy
    ``root.vision:leaf`` label is retired), and the drawn projector width
    flows from the construction-site binding, not the language width."""
    from model_unfolder.evidence.projector import projector_evidence

    cfg = _qwen2vl_corpus_cfg()
    ledger, ir = _capture(cfg)
    tower_events = [e for e in ledger.events
                    if e.component in {"root.vision", "root.video"} and e.present]
    assert tower_events, "witness must exercise the modality owners"
    bare = [e for e in tower_events if not getattr(e, "path_exact", False)]
    assert bare == [], f"bare funnel leaves under modality owners: {bare[:3]}"
    # U2.2a: the legacy ``owner:leaf`` label is retired — but "the path contains
    # a dot" is NOT what proves it.  A host field genuinely lives at the top of
    # its document, so its exact path has no dot; demanding one is demanding a
    # fabricated prefix (this assertion previously passed on
    # ``vision_config.image_token_id``, a path present in no document, and the
    # single ``vision_config`` carve-out below was the first bare leaf that hit
    # it).  Resolution is the real, and strictly stronger, predicate: a dotted
    # path can be fiction, a resolving path cannot.
    assert all(not e.config_path.startswith(f"{e.component}:")
               for e in tower_events), "legacy owner:leaf label"
    unresolvable = sorted({e.config_path for e in tower_events
                           if not _resolves(cfg, e.config_path)})
    assert not unresolvable, (
        f"modality-owner events claim paths that exist nowhere in the witness: "
        f"{unresolvable}")

    evidence = projector_evidence(cfg)
    assert evidence.out_width_source == "config_bound"
    assert tuple(evidence.out_width_path) == ("vision_config", "hidden_size")
    projector = ir["extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert projector["out_features"] == cfg["vision_config"]["hidden_size"] == 3584


def test_cor4_ce2_same_leaf_under_text_and_vision_stays_distinct():
    """§9.C.2: ``hidden_size`` under the text scope and under the vision scope
    are distinct occurrences with distinct exact paths — never one row."""
    ledger, _ = _capture(_qwen2vl_corpus_cfg())
    paths = {(e.component, e.config_path) for e in ledger.events
             if e.canonical == "hidden_size" and e.present}
    vision_rows = {p for p in paths if p[0] == "root.vision"}
    text_rows = {p for p in paths if p[0] not in {"root.vision", "root.video"}}
    assert any(path == "vision_config.hidden_size" for _, path in vision_rows)
    assert all(not path.startswith("vision_config")
               for _, path in text_rows if "." in path or path == "hidden_size")
    assert vision_rows.isdisjoint(text_rows)


def test_cor4_ce3_equivalent_wrapper_layouts_produce_the_same_facts():
    """§9.C.3: the same component under either declared wrapper spelling
    yields the same built path (scope normalization, not spelling forks)."""
    canonical = mu.unfold(_NO_SOURCE_MM).to_ir()
    alt = {k: v for k, v in _NO_SOURCE_MM.items() if k != "vision_config"}
    alt["vision_model_config"] = dict(_NO_SOURCE_MM["vision_config"])
    renamed = mu.unfold(alt).to_ir()
    a = canonical["extras"]["modalities"]["inputs"]["vision"]
    b = renamed["extras"]["modalities"]["inputs"]["vision"]
    assert a == b


def test_cor4_ce4_conflicting_wrappers_are_ambiguity_not_first_match():
    """§9.C.4: rival wrapper spellings with unequal values author NOTHING —
    a structured ambiguity event replaces the silent first-match pick."""
    conflicted = dict(_NO_SOURCE_MM)
    conflicted["vision_model_config"] = dict(
        _NO_SOURCE_MM["vision_config"], hidden_size=999)
    with capture_events() as ledger:
        with owner_scope("root"):
            ir = mu.unfold(conflicted).to_ir()
    inputs = ((ir["extras"].get("modalities") or {}).get("inputs")) or {}
    assert "vision" not in inputs
    rows = [e for e in ledger.events
            if e.intent == "ambiguous" and e.component == "root.vision"]
    assert rows and "vision_model_config" in rows[0].reason

    # EQUAL rivals are redundant evidence: the path builds, facts unchanged.
    redundant = dict(_NO_SOURCE_MM)
    redundant["vision_model_config"] = dict(_NO_SOURCE_MM["vision_config"])
    built = mu.unfold(redundant).to_ir()["extras"]["modalities"]["inputs"]
    assert built["vision"]["encoder"]["hidden_size"] == 768


def test_cor4_ce5_source_owned_width_beats_language_width():
    """§9.C.5: when the construction site says the merger's output is the
    VISION config's field, that value wins even when it differs from the
    language width — the generic width can no longer author it."""
    from copy import deepcopy
    from test_support import QWEN2VL_STYLE

    cfg = deepcopy(QWEN2VL_STYLE)
    cfg["vision_config"]["hidden_size"] = 1234          # != any text width
    projector = mu.unfold(cfg).to_ir()[
        "extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert projector["out_features"] == 1234


def test_cor4_ce6_missing_source_leaves_width_and_mechanism_unknown():
    """§9.C.6: no modeling source -> no out_features, no callable ops — and
    no family/config/language-width fallback revives them."""
    from model_unfolder.evidence.projector import projector_evidence

    evidence = projector_evidence(_NO_SOURCE_MM)
    assert evidence.status == "oracle_missing"
    assert evidence.out_width_source == "unavailable"
    projector = mu.unfold(_NO_SOURCE_MM).to_ir()[
        "extras"]["modalities"]["inputs"]["vision"]["projector"]
    assert "out_features" not in projector
    assert "ops" not in projector


def test_cor4_ce7_audio_and_vision_sharing_a_leaf_never_cross_clear():
    """§9.C.7: audio and vision towers both declaring ``hidden_size`` keep
    separate owner-scoped rows — neither clears the other's debt."""
    cfg = dict(_NO_SOURCE_MM)
    cfg["audio_config"] = {"hidden_size": 512, "num_hidden_layers": 2,
                           "num_attention_heads": 4}
    cfg["audio_token_index"] = 6
    with capture_events() as ledger:
        with owner_scope("root"):
            mu.unfold(cfg).to_ir()
    rows = {(e.component, e.config_path) for e in ledger.events
            if e.canonical == "hidden_size" and e.present
            and e.component in {"root.vision", "root.audio"}}
    assert ("root.vision", "vision_config.hidden_size") in rows
    assert ("root.audio", "audio_config.hidden_size") in rows
    consumed_by = {}
    for e in ledger.events:
        if e.intent == "consumed" and e.canonical == "hidden_size" \
                and e.component in {"root.vision", "root.audio"}:
            consumed_by.setdefault(e.component, set()).add(e.config_path)
    for owner, paths in consumed_by.items():
        prefix = "vision_config." if owner == "root.vision" else "audio_config."
        assert all(p.startswith(prefix) for p in paths), (owner, paths)
