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


def _msg_bar(css_class: str, messages: list[str]) -> str:
    lines = "".join(f'<div class="uf-msg-line">{_html(m)}</div>' for m in messages)
    return f'<div class="uf-msg-bar {css_class}">{lines}</div>'


def _header(ir: dict, info: dict, mount_id: str) -> str:
    # No hover anywhere: badges carry no `title` tooltip.  The two message
    # badges (config gaps, advisory notes) instead CLICK to open a full-width
    # line at the top of the card — a pure-CSS checkbox toggle (label ↔ hidden
    # checkbox ↔ `:checked ~` bar), consistent with the "JS only opens/closes"
    # rule.  Arch badges are plain, non-interactive chips.
    badges = [
        f'<span class="uf-badge">{_html(badge["text"])}</span>'
        for badge in _arch_badges(ir, info)
    ]

    toggles: list[str] = []
    bars: list[str] = []
    # Only genuine config GAPS warrant the "partial config" alarm; by-design
    # advisories (e.g. a CFG twin we deliberately don't draw twice) are notes.
    warnings = ir.get("warnings") or []
    if warnings:
        wid = f"{mount_id}-msg-warn"
        toggles.append(f'<input type="checkbox" id="{_attr(wid)}" class="uf-msg-toggle" hidden>')
        bars.append(_msg_bar("uf-msg-bar-warn", warnings))
        badges.append(
            f'<label for="{_attr(wid)}" class="uf-badge uf-badge-warn">⚠ partial config</label>'
        )

    notes = ir.get("notes") or []
    if notes:
        nid = f"{mount_id}-msg-note"
        label = "ⓘ note" if len(notes) == 1 else f"ⓘ {len(notes)} notes"
        toggles.append(f'<input type="checkbox" id="{_attr(nid)}" class="uf-msg-toggle" hidden>')
        bars.append(_msg_bar("uf-msg-bar-note", notes))
        badges.append(
            f'<label for="{_attr(nid)}" class="uf-badge uf-badge-note">{_html(label)}</label>'
        )

    return f"""
{''.join(toggles)}
{''.join(bars)}
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
    extras = ir.get("extras") or {}
    if (extras.get("render") or {}).get("family") == "diffusion":
        items = _diffusion_stats(ir, extras, param_text)
    else:
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


def _diffusion_stats(ir: dict, extras: dict, param_text: str) -> list[tuple[str, str]]:
    """Diffusion replaces the (meaningless) Vocab / Context cells with the
    denoising schedule length and the latent channels it operates on."""
    meta = extras.get("diffusion") or {}
    timesteps = meta.get("scheduler_train_timesteps")
    latent = meta.get("in_channels")
    return [
        ("Layers", str(len(ir.get("layers", [])))),
        ("Hidden", _fmt_int(ir.get("hidden_size"))),
        ("Timesteps", _fmt_int(timesteps) if timesteps else "-"),
        ("Latent", f"{_fmt_int(latent)} ch" if latent else "-"),
        ("Params", param_text or "?"),
    ]
