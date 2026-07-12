"""H6 (§16.6) — registry-driven projection obligations + reverse-fabrication audit.

The REGISTRY is the single source of truth for what may be drawn.  Two directions,
both closed here:

* **reverse fabrication** — every leaf name the renderer DRAWS
  (``fact_projection.ATTENTION_DRAWN`` etc.) must be either a REGISTERED fact or a
  structured drawn-but-unledgered debt (``DRAWN_UNLEDGERED_DEBT``, each carrying
  owner/reason/unit).  A drawn leaf that is neither is fabrication — a picture with
  no fact behind it.
* **projection obligation** — every REGISTERED fact whose registry ``projections``
  include a drawable surface must actually be drawn on that surface, so the
  registry's obligation is honored, not decorative.

(The renderer/parser dependency firewall — the renderer never reads raw
config/source — is the third H6 leg, pinned by
``test_renderer_parser_dependency_firewall`` in ``test_h4_taint.py``.)
"""
from __future__ import annotations

from model_unfolder.evidence.registry import DRAWN_UNLEDGERED_DEBT, REGISTRY
from model_unfolder.renderers.html.fact_projection import (
    ATTENTION_DRAWN,
    FFN_DRAWN,
    LAYER_DRAWN,
    MODEL_DRAWN,
)

# registry projection surface -> the renderer drawn-leaf set for that surface
_SURFACE_DRAWN = {
    "attention_detail": ATTENTION_DRAWN,
    "ffn_detail": FFN_DRAWN,
    "architecture_view": LAYER_DRAWN | MODEL_DRAWN,
}
_ALL_DRAWN = ATTENTION_DRAWN | FFN_DRAWN | LAYER_DRAWN | MODEL_DRAWN
_DEBT_NAMES = frozenset(d.name for d in DRAWN_UNLEDGERED_DEBT)


def test_every_drawn_leaf_is_registered_or_pinned_debt():
    """Reverse-fabrication audit: no leaf is drawn without a fact behind it."""
    fabricated = _ALL_DRAWN - set(REGISTRY) - _DEBT_NAMES
    assert not fabricated, (
        f"drawn leaves with no registered fact and no pinned debt (fabrication): "
        f"{sorted(fabricated)} — register the fact or add a DrawnUnledgeredFact")


def test_every_registry_drawable_fact_is_actually_drawn():
    """Projection obligation: a registered fact that DECLARES a drawable surface
    must be drawn on it — the registry's obligation is real, not decorative."""
    unmet = []
    for name, definition in REGISTRY.items():
        for surface in definition.projections:
            drawn = _SURFACE_DRAWN.get(surface)
            if drawn is not None and name not in drawn:
                unmet.append(f"{name}: registry promises {surface} but it is not drawn there")
    assert not unmet, "unmet projection obligations:\n" + "\n".join(unmet)


def test_drawn_unledgered_debt_is_fully_qualified():
    """§16.6 / the H2/H5 discipline: every debt entry carries owner-reason-unit-
    becomes, never a bare string; each is actually drawn (not stale)."""
    for entry in DRAWN_UNLEDGERED_DEBT:
        assert entry.name and entry.surface and entry.reason and entry.becomes
        assert entry.unit in {"H7", "H8"}, entry
        assert entry.name in _ALL_DRAWN, f"{entry.name}: pinned as drawn debt but not drawn (stale)"
    names = [d.name for d in DRAWN_UNLEDGERED_DEBT]
    assert len(names) == len(set(names)), "duplicate drawn-debt names"


def test_debt_and_registry_do_not_overlap():
    """A leaf is EITHER a registered fact OR pinned debt — never both (a debt
    entry that gained a writer must move OUT of the debt register into REGISTRY)."""
    both = set(REGISTRY) & _DEBT_NAMES
    assert not both, f"drawn leaves in BOTH registry and debt — remove from debt: {sorted(both)}"


def test_reverse_fabrication_poison():
    """Anti-vacuous: a fabricated drawn leaf (registered nowhere, pinned nowhere)
    is caught by the reverse audit."""
    poison = (_ALL_DRAWN | {"fabricated_cell_kind"}) - set(REGISTRY) - _DEBT_NAMES
    assert "fabricated_cell_kind" in poison
