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
from model_unfolder.evidence.registry import DrawnUnledgeredFact
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

@pytest.mark.xfail(strict=True, reason="§5.1/U1(T-01): the live _resolve funnel "
                   "records the CANONICAL as consumed even though that spelling "
                   "is absent from the checkpoint — a fictional consumed field")
def test_p1_absent_canonicals_are_never_recorded_consumed():
    ledger, _ = _capture(GPT2_ALIASED)
    fictional = _CANONICALS & set(ledger.consumed_names())
    assert fictional == set(), f"canonicals consumed without existing: {sorted(fictional)}"


@pytest.mark.xfail(strict=True, reason="§5.1/U1(T-01): the spelling that actually "
                   "supplied the value must be the consumed read; today the real "
                   "aliases land in accessed-but-unconsumed debt")
def test_p2_the_supplying_alias_spelling_is_the_consumed_read():
    ledger, _ = _capture(GPT2_ALIASED)
    missing = _ALIASES - set(ledger.consumed_names())
    assert missing == set(), f"supplying spellings not recorded consumed: {sorted(missing)}"


@pytest.mark.xfail(strict=True, reason="§5.1 Decision/U1: accessed is PRESENT-ONLY; "
                   "absence lives only in absent_default — today absent canonicals "
                   "are inserted into the compat accessed list")
def test_p3_absent_fields_never_enter_accessed():
    _, ir = _capture(GPT2_ALIASED)
    accessed = set(((ir.get("extras") or {}).get("config_audit") or {}).get("accessed", []))
    leaked = _CANONICALS & accessed
    assert leaked == set(), f"absent canonicals inside accessed: {sorted(leaked)}"


@pytest.mark.xfail(strict=True, reason="§5.1/U1: simultaneous unequal aliases must "
                   "become a typed ambiguity (no silent winner); today one value is "
                   "silently chosen and no ambiguous event exists")
def test_p4_conflicting_aliases_become_typed_ambiguity():
    ledger, _ = _capture({**GPT2_ALIASED, "hidden_size": 96})  # 96 vs n_embd=64
    ambiguous = [e for e in ledger.events
                 if e.intent == "ambiguous" and e.canonical == "hidden_size"]
    assert ambiguous, "hidden_size=96 vs n_embd=64 selected silently — no ambiguity event"


# --------------------------------------------------------------------------- #
# §5.2 — blocking unread coverage is bare-key based (unit U1)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="§5.2/U1: unread coverage subtracts a flat "
                   "leaf-name set, so the root's hidden_size read clears a nested "
                   "sibling component's UNREAD hidden_size")
def test_p5_a_sibling_scope_cannot_clear_nested_unread_debt():
    unread = _unread({**LLAMA, "aux_tower_config": {"hidden_size": 123}})
    assert "aux_tower_config.hidden_size" in unread, (
        "the nested component's unread hidden_size was cleared by the root's read")


# --------------------------------------------------------------------------- #
# §5.4 — pending-projection debt is excused by leaf key (unit U1 step 5 / R2)
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="§5.4/U1: the pending-debt excusal ignores "
                   "the registered owner (root.vae), so a ROOT-owner field of the "
                   "same name is silently excused from unread accounting")
def test_p6_pending_debt_excusal_is_owner_tight():
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

@pytest.mark.xfail(strict=True, reason="§5.6/R3: an @identity_display decorator "
                   "exempts identity-NAME branches inside the function body — the "
                   "one decorated shape that IS blessed today")
def test_p8_decorated_identity_name_branch_is_not_blessed():
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

@pytest.mark.xfail(strict=True, reason="§5.8/U2: DrawnUnledgeredFact's docstring "
                   "claims an owner but the dataclass has no owner field")
def test_p11_drawn_unledgered_debt_carries_an_owner_field():
    field_names = {f.name for f in dataclasses.fields(DrawnUnledgeredFact)}
    assert "owner" in field_names, f"debt row fields lack owner: {sorted(field_names)}"


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

@pytest.mark.xfail(strict=True, reason="§5.3/U1: consumed-but-unprojected exists as "
                   "a unit-tested ledger method but is not published as a Sable "
                   "check; both owner-qualified nets must be live")
def test_p13_consumed_but_unprojected_net_is_published():
    from model_unfolder.sable import sable

    report = sable(LLAMA, render_images=False)
    names = {check.name for check in report.checks}
    assert any(("consumed" in name and "unprojected" in name) for name in names), (
        f"no published consumed-but-unprojected net among checks: {sorted(names)}")
