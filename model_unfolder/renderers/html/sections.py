"""Reusable HTML sections and header fragments."""
from __future__ import annotations

from ...ir import EvidenceWarning
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


def _warning_bar(messages: list[str], groups: list[EvidenceWarning]) -> str:
    """Render producer-authored summaries; exact receipts stay disclosed."""
    lines = [f'<div class="uf-msg-line">{_html(message)}</div>'
             for message in messages]
    for group in groups:
        summary = group.summary
        details = group.details
        detail_lines = "".join(
            f'<div class="uf-msg-line">{_html(detail)}</div>'
            for detail in details)
        lines.append(
            '<details class="uf-evidence-disclosure" style="margin-top:7px">'
            f'<summary style="cursor:pointer;font-weight:650">{_html(summary)}</summary>'
            '<div class="uf-evidence-details" '
            f'style="margin:7px 0 0 16px;opacity:.9">{detail_lines}</div>'
            '</details>')
    return f'<div class="uf-msg-bar uf-msg-bar-warn">{"".join(lines)}</div>'


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
    warning_rows = ir.get("warnings") or []
    groups_by_check = {}
    for warning in warning_rows:
        if isinstance(warning, EvidenceWarning):
            groups_by_check.setdefault(warning.check, warning)
    groups = list(groups_by_check.values())
    warnings = [str(warning) for warning in warning_rows
                if isinstance(warning, str)
                and not isinstance(warning, EvidenceWarning)]
    if warnings or groups:
        wid = f"{mount_id}-msg-warn"
        toggles.append(f'<input type="checkbox" id="{_attr(wid)}" class="uf-msg-toggle" hidden>')
        bars.append(_warning_bar(warnings, groups))
        # Evidence producers mark their diagnostic class explicitly. The
        # renderer only maps that transport prefix to display text; it neither
        # inspects raw evidence extras nor infers a mechanism from prose.
        typed_unresolved = bool(groups) or any(
            warning.startswith("Unresolved evidence — ")
            for warning in warnings
        )
        legacy_unresolved_only = all(
            warning.startswith("Unresolved code-defined facts")
            for warning in warnings
        )
        warning_label = (
            "⚠ unresolved evidence"
            if typed_unresolved or legacy_unresolved_only
            else "⚠ partial config"
        )
        badges.append(
            f'<label for="{_attr(wid)}" class="uf-badge uf-badge-warn">'
            f'{warning_label}</label>'
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
    # U2: an estimate that had to pick a counting convention for an UNKNOWN
    # fact (tie / FFN gating) says so — starred value with the conventions in
    # the hover title, never a silently-branched number.
    _assumption_note = "; ".join(params.get("assumptions") or [])
    if _assumption_note:
        param_text = f"~{param_text}*"
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
        title = (f' title="{_attr("estimate uses conventions for unknowns: " + _assumption_note)}"'
                 if (_assumption_note and value == param_text) else "")
        cells.append(
            f'<div class="uf-stat"{title}>'
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
