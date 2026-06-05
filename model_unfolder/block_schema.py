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


class Block(TypedDict, total=False):
    """One node in the render tree.  ``id`` is the only required key.

    ``id`` links the drawn SVG node (``data-id``) to its inspect card
    (``data-card-id``); everything else is presentation, drill-down, or layout.
    """

    id: str                       # REQUIRED — node ↔ card link
    role: str                     # semantic slot: attention/ffn/norm/residual/…
    kind: str                     # glyph hint: attention/residual_add/embedding/…
    label: "str | list[str]"      # on-block text (one line, or stacked lines)
    title: str                    # card heading
    description: str              # card body
    view: str                     # drill-down archetype; MUST be a registered view
    children: "list[Block]"       # sub-blocks (recursed by the inspect panel)
    detail: dict                  # extra structured payload (e.g. MTP module counts)
    # --- layout hints (renderer geometry; non-default topologies) ---
    lane: str                     # side lane placement (parallel/PLE/cross-attn)
    tap_from: str                 # block id this one taps its input from
    feeds: str                    # block id this one feeds into
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
    "iter_block_tree",
    "validate_block_tree",
    "validate_click_coupling",
]
