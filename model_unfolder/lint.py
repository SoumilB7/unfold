"""Label lint — cheap, mechanical guards on the TEXT a block draws.

The op-conformance net checks the diagram against the *code*; click-coupling and
wiring check it against *itself*.  Neither looks at whether a block's **label**
reads cleanly.  Two real regressions slipped past every other net and were caught
only by eye:

* a nested-parenthesis label — ``"Joint Attention (MM-DiT (dual-stream))"`` — from
  blindly wrapping a tag that already had parens;
* a raw backend activation spelling on a block (``"gelu-approximate"`` instead of
  the clean math name ``"GELU"``).

Both have an unambiguous textual signature, so they belong in a fast lint, not in
the (irreducibly manual) visual pass.  This walks every block label an IR draws —
layer blocks + their drill children, plus the model/loop bookends — and returns one
message per offence.  Deliberately conservative: it flags only patterns that are
*always* wrong, never stylistic repeats (two ``"Normalization"`` boxes in a layer
are correct, so duplicate-label detection is left to the visual rubric).
"""
from __future__ import annotations

import re

from .labels import activation_label

#: clean activation display names the renderer is allowed to draw (the values of
#: ``labels._ACTIVATION_LABELS`` plus the bare extras). An activation-role block
#: whose label is NOT one of these (or resolves to one) is a raw backend spelling.
_CLEAN_ACTIVATIONS = {
    "GELU", "ReLU", "SiLU", "GEGLU", "SwiGLU", "GLU", "Tanh", "Sigmoid",
    "Softmax", "ReLU²", "Mish", "Activation",
}

#: HF class-name task-head suffixes. A block label ending in one of these is a raw
#: modeling class name leaked into the diagram (e.g. a text encoder rendered as
#: "Mistral3ForConditionalGeneration"), which overflows its box — give it a clean
#: family name in the relevant everchanging map instead.
_RAW_CLASS_SUFFIXES = (
    "ForConditionalGeneration", "ForCausalLM", "ForTextEncoding",
    "ForImageGeneration", "ForSequenceClassification",
)

_CONNECTOR_KINDS = frozenset({"residual_add", "gate_mul", "dot_product", "concat"})

# Numeric facts belong on cards/chips, never block labels.  Keep this deliberately
# targeted: real PyTorch op names like ``Conv2d`` / ``Conv3d`` are allowed, while
# dimensions/counts/tokens painted on boxes are not.
_FACTY_NUMBER = re.compile(
    r"("
    r"×\s*\d"                                  # Encoder ×30
    # A *dimension* is hyphenated (768-d / 1,280-d) or a ≥2-digit run (1024d).
    # NOTE: a bare single-digit + D (2D / 3D / axial 3D) is a TOPOLOGY descriptor
    # (N-dimensional RoPE / patches), not a channel dimension — it must NOT flag.
    r"|\b\d[\d,]*\s*-\s*d\b"                   # 768-d / 1,280-d
    r"|\b\d[\d,]+\s*d\b"                       # 1024d  (≥2-digit dimension)
    r"|\b\d[\d,]*\s*(?:tokens?|heads?|experts?|layers?|blocks?|channels?)\b"
    r"|\b\d[\d,]*\s*(?:→|->)"                  # 1024 -> ...
    r"|(?:→|->)\s*\d[\d,]*"                    # ... -> 4096
    r")",
    re.IGNORECASE,
)
_ALLOWED_NUMERIC_OP_LABELS = frozenset({"Conv1d", "Conv2d", "Conv3d"})
_BACKEND_ALTERNATIVE = re.compile(r"\b(?:Linear|Conv[123]d)\s*/\s*(?:Linear|Conv[123]d)\b")


def lint_labels(ir: dict) -> list[str]:
    """Every block-label offence in ``ir`` (``Diagram.to_ir()``), as messages.

    Empty ⇒ clean.  Used by the Sable harness and pinned as a corpus test."""
    problems: list[str] = []
    for block in _walk_blocks(ir):
        problems.extend(_lint_block(block))
    return problems


def _lint_block(block: dict) -> list[str]:
    out: list[str] = []
    bid = block.get("id", "?")
    if _is_static_connector(block):
        out.append(
            f"block {bid!r}: Tier-2 connector kind {block.get('kind')!r} is static — "
            "connectors must be clickable glyphs with a one-line card; only ports/layout helpers are static.")
    for part in _label_parts(block.get("label")):
        # 1. Nested / doubled parentheses — a block label is a short name with at
        #    most one parenthetical; ``"(MM-DiT (dual-stream))"`` is a tag wrapped
        #    in parens it already had.
        if part.count("(") > 1 or part.count(")") > 1:
            out.append(
                f"block {bid!r}: label part {part!r} has nested/doubled parentheses "
                "— wrap the short discriminator, not the full tag.")
        # 2. A raw HF modeling class name leaked into a label (it overflows the box
        #    and reads as garbage) — give it a clean family name in everchanging.
        if part.endswith(_RAW_CLASS_SUFFIXES):
            out.append(
                f"block {bid!r}: label part {part!r} is a raw model class name — "
                "map it to a clean family name (e.g. an everchanging text_encoders row).")
        # 3. Backend alternatives are not operation labels.  The parser/source
        #    evidence must resolve the real op or render an honest unknown.
        if _BACKEND_ALTERNATIVE.search(part):
            out.append(
                f"block {bid!r}: label part {part!r} hedges between backend ops — "
                "draw the real code op, or mark the block honestly unknown.")
        # 4. Counts/dimensions/tokens are facts, not operation identity.
        if _leaks_numeric_fact(part):
            out.append(
                f"block {bid!r}: label part {part!r} includes dimensions/counts — "
                "move numeric facts to card chips, not the block label.")
    # 5. Activation-role blocks must draw the clean math name, never the config's
    #    backend spelling (gelu-approximate / gelu_pytorch_tanh / quick_gelu …).
    if _is_activation_block(block):
        label = " ".join(_label_parts(block.get("label"))).strip()
        if label and _is_raw_activation(label):
            clean = activation_label(label)
            out.append(
                f"block {bid!r}: activation label {label!r} is a raw backend "
                f"spelling — draw the clean math name (e.g. {clean!r}).")
    return out


def _is_static_connector(block: dict) -> bool:
    return block.get("kind") in _CONNECTOR_KINDS and block.get("static") is True


def _leaks_numeric_fact(part: str) -> bool:
    if part in _ALLOWED_NUMERIC_OP_LABELS:
        return False
    # Allow clean ordinal-free operator names with digits embedded only in the
    # PyTorch ConvNd family above; all architecture dimensions/counts are chips.
    return bool(_FACTY_NUMBER.search(part))


def _is_activation_block(block: dict) -> bool:
    return block.get("kind") == "activation" or block.get("role") == "activation" \
        or str(block.get("id", "")).endswith(("activation", "act"))


def _is_raw_activation(label: str) -> bool:
    """A label is a raw activation spelling if it carries backend cruft (an
    underscore, a hyphen, or a known implementation suffix) instead of a clean
    name — and a clean name exists for it."""
    low = label.lower()
    cruft = ("_" in label or "-" in label
             or any(t in low for t in ("approximate", "pytorch", "quick")))
    if not cruft:
        return label not in _CLEAN_ACTIVATIONS and activation_label(label) != label
    return True


def _label_parts(label) -> list[str]:
    if label is None:
        return []
    if isinstance(label, (list, tuple)):
        return [str(p) for p in label if p is not None]
    return [str(label)]


def _walk_blocks(ir: dict):
    """Yield every block dict an IR draws — layer blocks + their drill children
    (recursively), plus the model and sampling-loop bookends."""
    seen_lists = []
    for layer in (ir.get("layers") or []):
        seen_lists.append(layer.get("blocks") or [])
    render = (ir.get("extras") or {}).get("render") or {}
    seen_lists.append(render.get("model_blocks") or [])
    seen_lists.append(render.get("loop_blocks") or [])
    for blocks in seen_lists:
        yield from _walk_block_list(blocks)


def _walk_block_list(blocks):
    for block in (blocks or []):
        if not isinstance(block, dict):
            continue
        yield block
        children = block.get("children")
        if isinstance(children, list):
            yield from _walk_block_list(children)
