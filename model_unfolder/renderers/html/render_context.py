"""Call-local rendering state: theme, diagnostics and provenance events."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class RenderEvent:
    """One exact graph projection emitted during a render."""

    view: str
    block_path: tuple[str, ...]
    component: str
    variant: str
    source_owner: str
    source_file: str
    source_line: int | None
    drawn_ops: frozenset[str]
    node_ids: frozenset[str]
    # U2 P4 net #13 witness channel: the ledger keys ("owner.fact") this
    # projection visibly carries.  Populated by the drill / architecture
    # renderers (renderers/html/fact_projection.py); the projection-audit net
    # unions these across events and diffs them against the evidenced facts in
    # ``ir.extras['fact_provenance']`` so no proven fact is silently dropped.
    facts_projected: frozenset[str] = frozenset()

    def legacy_tuple(self) -> tuple[str, frozenset[str], frozenset[str]]:
        return self.view, self.drawn_ops, self.node_ids


@dataclass
class RenderContext:
    """All mutable state owned by one HTML/SVG render call."""

    theme: str = "teal"
    wiring_findings: list[str] = field(default_factory=list)
    events: list[RenderEvent] = field(default_factory=list)
    block_stack: list[dict] = field(default_factory=list)
    id_sequence: int = 0

    def next_id(self) -> int:
        value = self.id_sequence
        self.id_sequence += 1
        return value

    @contextmanager
    def block(self, value: dict) -> Iterator[None]:
        self.block_stack.append(value)
        try:
            yield
        finally:
            self.block_stack.pop()

    def record_graph(self, view: str, drawn_ops, node_ids,
                     facts_projected=frozenset()) -> None:
        block = self.block_stack[-1] if self.block_stack else {}
        detail = block.get("detail") if isinstance(block.get("detail"), dict) else {}
        evidence = detail.get("evidence") if isinstance(detail.get("evidence"), dict) else {}
        component = str(
            block.get("source_component") or block.get("component")
            or evidence.get("component") or "root"
        )
        source_owner = str(
            block.get("source_owner") or evidence.get("owner_class")
            or evidence.get("class_name") or ""
        )
        source_file = str(block.get("source_file") or evidence.get("source_file") or "")
        source_line = block.get("source_line") or evidence.get("line")
        variant = str(
            block.get("variant") or block.get("group_variant")
            or evidence.get("variant") or ""
        )
        path = tuple(str(item.get("id") or item.get("view") or "?") for item in self.block_stack)
        self.events.append(RenderEvent(
            view=view,
            block_path=path,
            component=component,
            variant=variant,
            source_owner=source_owner,
            source_file=source_file,
            source_line=source_line if isinstance(source_line, int) else None,
            drawn_ops=frozenset(drawn_ops),
            node_ids=frozenset(node_ids),
            facts_projected=frozenset(facts_projected or ()),
        ))

    def note_facts_projected(self, view: str, facts_projected, *, node_ids=()) -> None:
        """Record a facts-only projection witness (U2 P4 net #13).

        For surfaces that visibly carry ledger facts WITHOUT going through the
        graph engine — the model-level architecture view draws the norm cells,
        the head-tying, and the position/mask chips as raw SVG, not a
        :class:`~.graph.Graph`.  ``view`` must NOT be a sub-module drill role, so
        the nested-conformance net (which keys on drill roles) skips this event;
        it carries no ``drawn_ops`` and so never enters an op/closure diff."""
        if not facts_projected:
            return
        self.record_graph(view, (), node_ids, facts_projected=facts_projected)


_CURRENT: ContextVar[RenderContext | None] = ContextVar(
    "model_unfolder_render_context", default=None,
)


def current_render_context() -> RenderContext | None:
    return _CURRENT.get()


@contextmanager
def activate_render_context(context: RenderContext) -> Iterator[RenderContext]:
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def ensure_render_context(*, theme: str = "teal") -> RenderContext:
    context = current_render_context()
    if context is None:
        context = RenderContext(theme=theme)
        _CURRENT.set(context)
    return context


def release_render_context(context: RenderContext) -> None:
    """Drop a compatibility capture when it is still the active context."""
    if current_render_context() is context:
        _CURRENT.set(None)


__all__ = [
    "RenderContext", "RenderEvent", "activate_render_context",
    "current_render_context", "ensure_render_context", "release_render_context",
]
