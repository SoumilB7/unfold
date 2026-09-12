"""Escaping and formatting helpers for renderer modules."""
from __future__ import annotations

from html import escape
from typing import Any


def _fmt_int(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _num(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _attr(value: Any) -> str:
    return escape(str(value), quote=True)


def facts_html(facts) -> str:
    """The one chips row every inspect card uses for its numeric/spec facts."""
    if not facts:
        return ""
    chips = "".join(f'<span class="uf-fact">{_html(f)}</span>' for f in facts if f)
    return f'<div class="uf-card-facts">{chips}</div>' if chips else ""


def _presentation_chip_html(chip) -> str:
    """One typed annotation; labels never determine kind or evidence status."""
    from ...presentation import PresentationChip
    if not isinstance(chip, PresentationChip):
        raise TypeError("chip renderer requires typed display metadata")
    label = chip.chip_kind.replace("_", " ") if chip.chip_kind == "pending_design" else chip.chip_kind
    extra = ""
    if chip.unknown_reason is not None:
        reason = chip.unknown_reason.to_dict()
        label += " · " + reason["reason_class"]
        extra = f' data-reason-class="{_attr(reason["reason_class"])}"'
    if chip.decision_ref is not None:
        label += " · " + chip.decision_ref
        extra += f' data-decision-ref="{_attr(chip.decision_ref)}"'
    return (f'<span class="uf-fact uf-chip-{chip.chip_kind}" '
            f'data-chip-kind="{chip.chip_kind}" data-chip-owner="{_attr(chip.owner)}"{extra}>'
            f'{_html(label)} · {_html(chip.text)}</span>')


def presentation_chip_html(chip) -> str:
    return _presentation_chip_html(chip)


def is_exact_chip_emission(fragment: str, chip) -> bool:
    """Bind receipts to returned bytes, including kind, owner and full text."""
    return fragment == _presentation_chip_html(chip)


def _annotation_path_text(path, *, relative=False):
    """Readable, reversible exact keys and indexes, including unusual keys."""
    import json
    parts = []
    for index, part in enumerate(path):
        if type(part) is int:
            parts.append('[' + str(part) + ']')
        elif part.isidentifier():
            parts.append(('.' if index or relative else '') + part)
        else:
            parts.append('[' + json.dumps(part, ensure_ascii=False) + ']')
    return ''.join(parts)


def _unknown_annotation_group_html(annotations) -> str:
    from ...presentation import UnknownValueAnnotation
    if not annotations or any(not isinstance(row, UnknownValueAnnotation) for row in annotations):
        raise TypeError('unknown group requires typed exact producer annotations')
    first = annotations[0]
    if any((row.value_state, row.reason) != (first.value_state, first.reason) for row in annotations):
        raise ValueError('unknown groups share only exact state and reason metadata')
    prefix = first.path
    for annotation in annotations[1:]:
        count = 0
        while count < min(len(prefix), len(annotation.path)) and prefix[count] == annotation.path[count]:
            count += 1
        prefix = prefix[:count]
    reason = first.reason.to_dict()
    readable_class = reason['reason_class'].replace('_', ' ')
    readable_reason = reason['concrete_reason'].replace('_', ' ')
    scope = _annotation_path_text(prefix)
    paths = [_annotation_path_text(row.path[len(prefix):], relative=bool(prefix)) or '(this value)'
             for row in annotations]
    scope_html = (f'<div>Under <code class="uf-value-scope">{_html(scope)}</code>; '
                  'append each suffix below:</div>' if prefix else '<div>Exact paths:</div>')
    return ('<section class="uf-unknown-value-group">'
            '<span class="uf-fact uf-chip-unresolved uf-value-reason" '
            'data-annotation-kind="unresolved-value-group" '
            f'data-value-state="{_attr(first.value_state)}" '
            f'data-reason-class="{_attr(reason["reason_class"])}">'
            f'unresolved · {_html(readable_class)}: {_html(readable_reason)}</span>'
            f'<div>{len(annotations)} value annotations; not a constructed-occurrence count.</div>'
            + scope_html + '<pre class="uf-value-paths">' + _html('\n'.join(paths))
            + '</pre></section>')


def unknown_annotation_group_html(annotations) -> str:
    return _unknown_annotation_group_html(annotations)


def append_unknown_value_report(fragment: str, ir: dict) -> str:
    """Add a collapsed exact-value strip to the shared family card shell."""
    from ...presentation import unknown_annotations_from_ir
    from .render_context import current_render_context
    annotations = unknown_annotations_from_ir(ir)
    if not annotations:
        return fragment
    marker = '<div class="uf-card">'
    if fragment.count(marker) != 1:
        raise ValueError('unknown report requires the exact single family card shell')
    groups = {}
    for annotation in annotations:
        groups.setdefault((annotation.value_state, annotation.reason), []).append(annotation)
    emissions = [(tuple(group), unknown_annotation_group_html(tuple(group)))
                 for group in groups.values()]
    shown = [(group, markup) for group, markup in emissions if markup]
    report = ('<details class="uf-unknown-values"><summary>Unresolved values · '
              + str(len(annotations)) + ' · reasons and exact locations</summary>'
              '<div class="uf-unknown-value-list">'
              + ''.join(markup for _, markup in shown) + '</div></details>')
    output = fragment.replace(marker, marker + report, 1)
    context = current_render_context()
    if context is not None:
        emitted = {annotation for group, markup in shown
                   if markup == _unknown_annotation_group_html(group) for annotation in group}
        # Retain original annotation order and every exact path independently
        # of the shared display badge/prefix. No occurrence/fact receipt added.
        for annotation in annotations:
            if annotation in emitted:
                context.record_unknown_value(annotation)
    return output
