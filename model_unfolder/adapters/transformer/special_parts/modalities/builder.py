"""Top-level multimodal extras assembly.

Generic over the modality registry: every input modality is one entry in
``MODALITY_REGISTRY``.  This loop never names a specific modality, so adding
a new input type touches only the registry (and its path builder).
"""
from __future__ import annotations

from typing import Any

from .....evidence import config_access as _config_access
from ...common import TEXT_WRAPPER_KEYS, get_config_value as _g
from .fusion import fusion_path
from .registry import MODALITY_REGISTRY
from .schema import multimodal_payload


def _modality_host(cfg: Any, _depth: int = 0) -> Any:
    """The config level that OWNS the modality sub-configs.

    Usually the root; a composite wrapper (Qwen-Omni's ``thinker_config``)
    hides the whole multimodal host one declared level down — found through
    the SAME wrapper vocabulary the LM unwrap uses, never by name guessing.
    The host level also carries the modality token-id fields
    (``audio_token_index``, ``vision_token_id``), which is why the WHOLE level
    is returned, not just the sub-configs."""
    if any(spec.resolve_config(cfg) is not None for spec in MODALITY_REGISTRY):
        return cfg
    if _depth >= 3:
        return None
    for key in TEXT_WRAPPER_KEYS:
        sub = _g(cfg, key)
        if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
            sub = sub.to_dict()          # composite AutoConfig carries OBJECTS
        if isinstance(sub, dict):
            host = _modality_host(sub, _depth + 1)
            if host is not None:
                return host
    return None


def _wrapper_dict(host: Any, key: str) -> Any:
    """Normalized wrapper value for the rivalry comparison (objects compare
    by their dict form so an AutoConfig child and a plain dict can be equal)."""
    value = host.get(key) if isinstance(host, dict) else getattr(host, key, None)
    if not isinstance(value, dict) and hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def multimodal_extras(cfg: Any, text_cfg: Any, text_hidden_size: int,
                      namespace: str = "root") -> dict | None:
    """Return structured multimodal extras, if the config declares them.

    ``namespace`` is the ambient ownership path of this parse (root for a
    top-level parse, root.<slot> for a recursively-parsed encoder slot).
    Every modality owner is namespaced under it, so a sub-component's tower
    is owned by ``<namespace>.<modality>`` — never falsely attributed to the
    pipeline's top-level ``root.<modality>``."""
    host = _modality_host(cfg)
    if host is None:
        return None
    modalities: dict[str, Any] = {}
    for spec in MODALITY_REGISTRY:
        sub_cfg = spec.resolve_config(host)
        if sub_cfg is None:
            continue
        # H3 (§16.5): attribute this tower's config reads to its OWN owner, so a
        # vision ``hidden_size`` and the text ``hidden_size`` are distinct ledger
        # entries and one sibling never clears another's accessed-but-unconsumed
        # debt.  Generic over the registry — the loop still names no modality.
        _owner = f"{namespace}.{spec.name}"
        _present_keys = [
            k for k in (getattr(spec, "config_keys", ()) or ())
            if ((k in host) if isinstance(host, dict)
                else getattr(host, k, None) is not None)]
        # COR-4 (§9): rival spellings of ONE component slot obey the alias law —
        # equal wrappers are redundant evidence (the registry's declared order
        # is the named precedence), unequal wrappers are structured ambiguity
        # and author NOTHING.  Never a silent first-match.
        if (len(_present_keys) > 1
                and getattr(spec, "keys_are_rival_spellings", True)):
            _wrapper_values = [_wrapper_dict(host, k) for k in _present_keys]
            if any(v != _wrapper_values[0] for v in _wrapper_values[1:]):
                _config_access.emit(
                    f"{spec.name}_config", intent="ambiguous", present=True,
                    alias=_present_keys[0], component=_owner,
                    config_path=_present_keys[0],
                    reason=("rival component wrappers with unequal values: "
                            + ", ".join(_present_keys)))
                continue
        _matched_key = _present_keys[0] if _present_keys else None
        with _config_access.owner_scope(_owner), \
                _config_access.config_container(
                    (_matched_key,) if _matched_key else ()):
            path = spec.build(host, text_cfg, sub_cfg, text_hidden_size)
            if not path:
                continue                 # a builder may veto on closer evidence
            modalities[spec.name] = path
            if spec.companion is not None:
                extra = spec.companion(host, sub_cfg, text_hidden_size)
                if extra:
                    modalities.update(extra)

    if not modalities:
        return None

    return multimodal_payload(modalities, fusion_path(host, text_cfg, modalities, text_hidden_size))


__all__ = ["multimodal_extras"]
