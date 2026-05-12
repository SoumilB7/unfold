"""State-space and recurrent attention-like detail views."""
from __future__ import annotations

from ...utils import _fmt_int
from .common import vertical_attention_stack


def build_ssm(ir: dict, info: dict, mount_id: str) -> str:
    attn = info["dominant"]["spec"].get("attention") or {}
    state = _fmt_int(attn.get("head_dim"))
    subtitle = f"state dim {state}" if state != "?" else "selective recurrence"
    return vertical_attention_stack(
        ir,
        info,
        mount_id,
        "ssm",
        "selective state-space block",
        [
            ("ssm_out_proj", "Output projection", None),
            ("ssm_gate", "Gate", None),
            ("ssm_scan", ["Selective Scan", subtitle], 16),
            ("ssm_conv", "Local Conv", None),
            ("ssm_in_proj", "Input projection", None),
        ],
    )


def build_recurrent(ir: dict, info: dict, mount_id: str) -> str:
    attn = info["dominant"]["spec"].get("attention") or {}
    width = _fmt_int(attn.get("head_dim"))
    return vertical_attention_stack(
        ir,
        info,
        mount_id,
        "recurrent",
        "linear recurrent unit",
        [
            ("lru_out_proj", "Output projection", None),
            ("lru_gate", "Gate", None),
            ("lru_state", ["Recurrent State", f"width {width}"], 16),
            ("lru_in_proj", "Input projection", None),
        ],
        h=520,
    )
