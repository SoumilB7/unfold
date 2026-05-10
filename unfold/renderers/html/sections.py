"""Reusable HTML sections and header fragments."""
from __future__ import annotations

from .metadata import _arch_badges
from .utils import _attr, _fmt_int, _html


def _details_section(label: str, sub: str, svg: str) -> str:
    """Collapsible section using <details>; closed by default."""
    return (
        '<details class="uf-section uf-section-collapsible">'
        '<summary class="uf-section-head">'
        f'<span class="uf-section-label">{_html(label)}</span>'
        f'<span class="uf-section-sub">{_html(sub)}</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        "</summary>"
        f'<div class="uf-section-body">{svg}</div>'
        "</details>"
    )


def _header(ir: dict, info: dict) -> str:
    badges = []
    for badge in _arch_badges(ir, info):
        title = badge.get("title") or ""
        badges.append(
            f'<span class="uf-badge" title="{_attr(title)}">{_html(badge["text"])}</span>'
        )
    return f"""
<div class="uf-header">
  <div class="uf-name">{_html(ir.get("name", "model"))}</div>
  <div class="uf-arch">{_html(ir.get("architecture", ""))}</div>
  <div class="uf-badges">{''.join(badges)}</div>
</div>
"""


def _stats_banner(ir: dict) -> str:
    params = ir.get("params") or {}
    param_text = (
        f"{params.get('total_h')} ({params.get('active_h')} act.)"
        if params.get("is_sparse")
        else params.get("total_h", "?")
    )
    items = [
        ("Layers", str(len(ir.get("layers", [])))),
        ("Hidden", _fmt_int(ir.get("hidden_size"))),
        ("Vocab", _fmt_int(ir.get("vocab_size"))),
        ("Context", _fmt_int(ir.get("max_position_embeddings")) if ir.get("max_position_embeddings") else "-"),
        ("Params", param_text or "?"),
    ]
    cells = []
    for key, value in items:
        cells.append(
            '<div class="uf-stat">'
            f'<div class="uf-stat-key">{_html(key.upper())}</div>'
            f'<div class="uf-stat-val">{_html(value)}</div>'
            "</div>"
        )
    return f'<div class="uf-stats">{"".join(cells)}</div>'
