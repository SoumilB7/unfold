"""Schema + validation for render blocks — the dict tree the diagram draws.

Blocks have to be dicts (they cross the parser→renderer boundary and serialize
to JSON), but they were entirely stringly-typed: a typo'd key, an `view` that
isn't registered, or a view drawing a node-id with no backing card all failed
**silently** — the worst outcome for a tool whose value is being trusted.

This module gives blocks a contract:

* :class:`Block` — a ``TypedDict`` documenting every legal key, so editors and
  type-checkers flag typos and unknown keys at author time.
* :func:`validate_block_tree` — runtime checks that catch the silent failures a
  type-checker can't see at runtime: a missing/duplicate ``id``, an unknown key,
  a ``view`` that isn't in the renderer's registry, or malformed ``children``.
* :func:`validate_click_coupling` — over rendered HTML, every clickable node
  (``data-id``) must resolve to a card (``data-card-id``).  This is the
  view↔block-id drift guard: if a view draws a node-id no block declares, the
  card is missing and the click does nothing.

Run both over a config corpus in CI (see ``tests/test_block_schema.py``) and the
whole "silently renders wrong" class becomes a failing test, not a surprise.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, Optional, TypedDict

from .everchanging import load_diffusion_typing, load_transformer_typing


class Block(TypedDict, total=False):
    """One node in the render tree.  ``id`` is the only required key.

    ``id`` links the drawn SVG node (``data-id``) to its inspect card
    (``data-card-id``); everything else is presentation, drill-down, or layout.
    """

    id: str                       # REQUIRED — node ↔ card link
    role: str                     # semantic slot: attention/ffn/norm/residual/…
    kind: str                     # glyph hint: attention/residual_add/embedding/…
    diffusion_stage: str          # approved diffusion slot/stage, when applicable
    diffusion_part_kind: str      # approved compound diffusion region, when applicable
    label: "str | list[str]"      # on-block text (one line, or stacked lines)
    title: str                    # card heading
    description: str              # card body — explanation prose, no numbers
    facts: "list[str]"            # numeric/spec chips ("32 heads", "4,096 → 12,288")
    view: str                     # drill-down archetype; MUST be a registered view
    children: "list[Block]"       # sub-blocks (recursed by the inspect panel)
    detail: dict                  # extra structured payload (e.g. MTP module counts)
    components: list[dict]        # typed sub-facts inside a compound stage
    # --- block-worthiness paradigm (Gate C tiers; see docs/BLOCK_STANDARD.md) ---
    static: bool                  # Tier-2 CONNECTOR: render as a glyph on the join
                                  # (residual ⊕, gate ×, split, concat), NON-clickable,
                                  # no card.  The renderer uses `clickable = not static`
                                  # and the card builder skips static blocks.  A True
                                  # here is the single switch that demotes a candidate
                                  # from a box to wiring — the inverse of a Tier-1 block.
    # --- layout hints (renderer geometry; non-default topologies) ---
    branch_side: str              # "left" | "right" — this block is a parallel branch
                                  # drawn off the central column (not in the chain),
                                  # converging into the `feeds` merge (e.g. dense MLP ∥
                                  # MoE).  Distinct from `lane`, which is a side rail.
    lane: str                     # side lane placement (parallel/PLE/cross-attn)
    tap_from: str                 # block id this one taps its input from
    feeds: str                    # block id this one feeds into
    also_feeds: "list[str]"       # additional block ids this side source fans into
                                  # (e.g. AdaLN conditioning → each gate × it drives)
    side_align: str               # alignment of a side block against its tap
    residual_from: str            # for residual_add: the block the skip starts at
    offset_y: float
    w: float
    h: float
    font: float


#: Every key a render block may legally carry — derived from the actual tree,
#: not guessed.  Anything outside this set is treated as a typo by the validator.
KNOWN_BLOCK_KEYS: frozenset[str] = frozenset(Block.__annotations__)

#: Semantic roles in use.  Informational — not hard-validated, since modality
#: pathways can introduce new ones; kept here as the canonical list.
KNOWN_ROLES: frozenset[str] = frozenset({
    "input", "embedding", "norm", "attention", "ffn", "residual",
    "output", "mtp", "ple", "vision", "audio", "video",
})

#: APPROVED diffusion block stages — the static type for diffusion diagrams.
#: A diffusion block tags itself with ``diffusion_stage``; only stages in this
#: set render as solid, first-class blocks.  A block whose stage is missing or
#: NOT in this set renders pale/light (same label) to flag that its place in the
#: diagram is *not decided yet* — a guardrail so a new adapter fact can't quietly
#: become a real block.  The set is *data*, edited in
#: ``everchanging/diffusor/typing.yaml`` — bless a new slot there, not here.
#:
#: ``DIFFUSION_BLOCK_IDS`` are ids legitimately solid inside a DiT block without a
#: ``diffusion_stage`` — they come from the reused transformer ``decoder_layer``
#: assembly (norm / attention / residual-add / FFN), not the diffusion adapter.
_diffusion_typing = load_diffusion_typing()
DIFFUSION_STAGES: frozenset[str] = frozenset(_diffusion_typing["stages"])
DIFFUSION_BLOCK_IDS: frozenset[str] = frozenset(_diffusion_typing["block_ids"])
DIFFUSION_PART_KINDS: frozenset[str] = frozenset(_diffusion_typing["part_kinds"])

#: APPROVED transformer block stages — the known decoder-only-transformer block
#: taxonomy (data in ``everchanging/transformer/typing.yaml``).  Documented now;
#: the transformer renderer doesn't draw pale-when-unapproved yet (only diffusion
#: does), but this is the single place to bless transformer stages when it does.
TRANSFORMER_STAGES: frozenset[str] = frozenset(load_transformer_typing()["stages"])


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

def iter_block_tree(ir: Any) -> Iterator[tuple[str, dict]]:
    """Yield ``(scope, block)`` for every render block in an IR.

    Covers per-layer blocks and model-level blocks, recursing into ``children``.
    ``scope`` is a dotted path for readable error messages.  Accepts a ModelIR
    or anything exposing ``.layers`` / ``.extras``.
    """
    layers = getattr(ir, "layers", None) or []
    for li, layer in enumerate(layers):
        yield from _walk(getattr(layer, "blocks", None) or [], f"layer[{li}]")
    extras = getattr(ir, "extras", None) or {}
    model_blocks = (extras.get("render") or {}).get("model_blocks") or []
    yield from _walk(model_blocks, "model")


def _walk(blocks: Any, scope: str) -> Iterator[tuple[str, dict]]:
    if not isinstance(blocks, list):
        return
    for b in blocks:
        if not isinstance(b, dict):
            continue
        yield scope, b
        children = b.get("children")
        if isinstance(children, list):
            yield from _walk(children, f"{scope}/{b.get('id', '?')}")


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_block_tree(ir: Any, *, known_views: Optional[set[str]] = None) -> list[str]:
    """Return a list of structural problems (empty == valid).

    Catches: missing/duplicate ``id``, unknown keys (typo guard), a ``view`` not
    in the registry, and malformed ``children``.
    """
    if known_views is None:
        known_views = _registry_views()

    problems: list[str] = []
    # Duplicate-id check is scoped to siblings (the renderer's card lookup is
    # per-panel), so track ids per parent scope.
    seen_by_scope: dict[str, set[str]] = {}

    for scope, block in iter_block_tree(ir):
        bid = block.get("id")
        if not isinstance(bid, str) or not bid:
            problems.append(f"{scope}: block has no string 'id' ({_short(block)})")
            continue

        seen = seen_by_scope.setdefault(scope, set())
        if bid in seen:
            problems.append(f"{scope}: duplicate id {bid!r}")
        seen.add(bid)

        unknown = set(block) - KNOWN_BLOCK_KEYS
        if unknown:
            problems.append(f"{scope}/{bid}: unknown key(s) {sorted(unknown)} — typo?")

        view = block.get("view")
        if view is not None and view not in known_views:
            problems.append(
                f"{scope}/{bid}: view {view!r} is not registered "
                f"(known: {sorted(known_views)})"
            )

        children = block.get("children")
        if children is not None and not isinstance(children, list):
            problems.append(f"{scope}/{bid}: 'children' must be a list, got {type(children).__name__}")

    return problems


# ---------------------------------------------------------------------------
# Render-coupling validation (view ↔ block-id drift)
# ---------------------------------------------------------------------------

_DATA_ID = re.compile(r'data-id="([^"]+)"')
_CARD_ID = re.compile(r'data-card-id="([^"]+)"')

#: Card ids that legitimately exist without a clickable node (fallbacks).
_NODELESS_CARDS = frozenset({"default"})


def validate_click_coupling(html: str) -> list[str]:
    """Every clickable node must resolve to a card; return any that don't.

    A ``data-id`` with no matching ``data-card-id`` means clicking that node
    opens nothing — exactly the silent failure a view drawing an undeclared
    node-id produces.
    """
    node_ids = set(_DATA_ID.findall(html))
    card_ids = set(_CARD_ID.findall(html)) | _NODELESS_CARDS
    orphans = sorted(node_ids - card_ids)
    return [f"clickable node {nid!r} has no card (click would do nothing)" for nid in orphans]


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _registry_views() -> set[str]:
    # Lazy import keeps this module dependency-free of the renderer at import time.
    from .renderers.html.block_views.registry import VIEW_REGISTRY
    return {k for k in VIEW_REGISTRY if k is not None}


def _short(block: dict, n: int = 80) -> str:
    text = repr(block)
    return text if len(text) <= n else text[: n - 1] + "…"


__all__ = [
    "Block",
    "KNOWN_BLOCK_KEYS",
    "KNOWN_ROLES",
    "DIFFUSION_STAGES",
    "DIFFUSION_BLOCK_IDS",
    "DIFFUSION_PART_KINDS",
    "TRANSFORMER_STAGES",
    "iter_block_tree",
    "validate_block_tree",
    "validate_click_coupling",
]
