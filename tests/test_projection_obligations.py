"""H6 (§16.6) — registry-driven projection obligations + reverse-fabrication audit.

The REGISTRY is the single source of truth for what may be drawn.  Two directions,
both closed here:

* **reverse fabrication** — every leaf name the renderer DRAWS
  (``fact_projection.ATTENTION_DRAWN`` etc.) must be either a REGISTERED fact or
  an exact drawn_leaf row in the ONE StructuralDebt register (U2-R6:
  ``drawn_unledgered_names()``, each row carrying owner/writer/consumer/unit/
  checkable deletion condition).  A drawn leaf that is neither is fabrication —
  a picture with no fact behind it.
* **projection obligation** — every REGISTERED fact whose registry ``projections``
  include a drawable surface must actually be drawn on that surface, so the
  registry's obligation is honored, not decorative.

(The renderer/parser dependency firewall — the renderer never reads raw
config/source — is the third H6 leg, pinned by
``test_renderer_parser_dependency_firewall`` in ``test_h4_taint.py``.)
"""
from __future__ import annotations

from model_unfolder.evidence.registry import REGISTRY
from model_unfolder.evidence.structural_debt import (
    STRUCTURAL_DEBT,
    drawn_unledgered_names,
)
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
_DEBT_NAMES = drawn_unledgered_names()


def test_every_drawn_leaf_is_registered_or_pinned_debt():
    """Reverse-fabrication audit: no leaf is drawn without a fact behind it."""
    fabricated = _ALL_DRAWN - set(REGISTRY) - _DEBT_NAMES
    assert not fabricated, (
        f"drawn leaves with no registered fact and no pinned debt (fabrication): "
        f"{sorted(fabricated)} — register the fact or add an exact drawn_leaf "
        f"StructuralDebt row")


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
    """§16.6 / U2-R6: every unledgered drawn leaf is an exact drawn_leaf row
    (owner/writer/consumer/U3–U14 unit/checkable deletion condition enforced by
    the StructuralDebt constructor); each is actually drawn (not stale)."""
    for name in _DEBT_NAMES:
        assert name in _ALL_DRAWN, \
            f"{name}: pinned as drawn debt but not drawn (stale)"
    rows = [r for r in STRUCTURAL_DEBT if r.sink_kind == "drawn_leaf"
            and r.structural_target in _DEBT_NAMES]
    assert {r.structural_target for r in rows} == set(_DEBT_NAMES)
    assert all(r.migration_unit in {"U6", "U8"} for r in rows), \
        "unledgered attention leaves belong to the U6/U8 attention units"


def test_debt_and_registry_do_not_overlap():
    """A leaf is EITHER a registered fact OR pinned debt — never both.  U2-R6:
    ``drawn_unledgered_names()`` derives the exclusive-or from the register
    (a drawn_leaf row whose fact is registered is a CONVENTION row, never a
    re-excusal), so the overlap is empty by construction — pinned here so the
    derivation cannot regress."""
    both = set(REGISTRY) & _DEBT_NAMES
    assert not both, f"drawn leaves in BOTH registry and debt — remove from debt: {sorted(both)}"


def test_reverse_fabrication_poison():
    """Anti-vacuous: a fabricated drawn leaf (registered nowhere, pinned nowhere)
    is caught by the reverse audit."""
    poison = (_ALL_DRAWN | {"fabricated_cell_kind"}) - set(REGISTRY) - _DEBT_NAMES
    assert "fabricated_cell_kind" in poison
