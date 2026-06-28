"""User-editable config vocabulary — *data, not code*.

Files here are loaded at runtime so new config dialects can be supported by
editing YAML, with the parser/renderer code left untouched.  Anything that is
config-variable vocabulary — aliases, ignore lists, word types, approved block
typing — lives here as YAML, never hardcoded.

Organised by **domain** (one folder per adapter family), so it scales as new
families are added — drop a folder + YAML, no code change beyond a thin loader::

    everchanging/
      transformer/
        aliases.yaml          canonical field -> config key spellings (in order)
        ignored_fields.yaml   non-architectural keys/suffixes (unparsed diagnostic)
      diffusor/
        aliases.yaml          DiT/MMDiT field aliases
        typing.yaml           APPROVED block stages / ids / DiT detection markers
        text_encoders.yaml    text-encoder class name -> friendly label

Loading prefers PyYAML when installed; otherwise a tiny built-in reader handles
the flow-style (``key: [a, b, c]``) and block-style (``key:`` + ``- item``)
formats the shipped files use, so the package keeps working with no third-party
dependency.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(domain: str, name: str) -> dict:
    """Load ``everchanging/<domain>/<name>.yaml`` (e.g. ``load("transformer", "aliases")``)."""
    text = (_DIR / domain / f"{name}.yaml").read_text(encoding="utf-8")
    try:
        import yaml  # optional; not a hard dependency
    except ImportError:
        return _parse_flow_yaml(text)
    return yaml.safe_load(text) or {}


# --- transformer domain -----------------------------------------------------

def load_aliases() -> dict[str, list[str]]:
    """Transformer field-alias table (``transformer/aliases.yaml``)."""
    return load("transformer", "aliases")


def load_ignored_fields() -> dict[str, list[str]]:
    """Non-architectural config keys/suffixes (``transformer/ignored_fields.yaml``)."""
    data = load("transformer", "ignored_fields")
    return {
        "keys": data.get("keys") or [],
        "suffixes": data.get("suffixes") or [],
        "opaque_scopes": data.get("opaque_scopes") or [],
    }


def load_transformer_typing() -> dict[str, list[str]]:
    """APPROVED transformer block stages (``transformer/typing.yaml``) — the known
    decoder-only-transformer block taxonomy."""
    data = load("transformer", "typing")
    return {"stages": data.get("stages") or []}


def load_layer_type_labels() -> dict[str, list[str]]:
    """Per-layer attention-type label groups (``transformer/layer_types.yaml``):
    the ``layer_types`` config spellings mapped to the mask the renderer draws
    (``full`` / ``sliding`` / ``compressed_sparse``)."""
    data = load("transformer", "layer_types")
    return {k: data.get(k) or []
            for k in ("full", "sliding", "compressed_sparse", "heavily_compressed")}


def load_layer_topology() -> dict:
    """Per-family macro-topology (``transformer/layer_topology.yaml``): which
    model_types use post/sandwich norm placement or flag-less parallel residual.

    Returns ``{"norm_placement": {model_type: pre|post|double},
    "parallel_residual": [model_type, ...]}``."""
    data = load("transformer", "layer_topology")
    placement: dict[str, str] = {}
    for item in data.get("norm_placement") or []:
        if isinstance(item, str) and "=" in item:
            mt, _, place = item.partition("=")
            placement[mt.strip()] = place.strip()
    return {"norm_placement": placement,
            "parallel_residual": list(data.get("parallel_residual") or []),
            "no_rope": list(data.get("no_rope") or [])}


# --- diffusor domain --------------------------------------------------------

def load_diffusion_aliases() -> dict[str, list[str]]:
    """Diffusion (DiT/MMDiT) field-alias table (``diffusor/aliases.yaml``)."""
    return load("diffusor", "aliases")


def load_diffusion_typing() -> dict[str, list[str]]:
    """APPROVED diffusion block typing (``diffusor/typing.yaml``): ``stages``
    (blessed diffusion_stage values), ``block_ids`` (reused transformer-layer
    ids), ``part_kinds`` (compound detail-view regions),
    ``dit_class_markers`` (detection substrings), and ``scheduler_display``
    ("Class=Display" name overrides)."""
    data = load("diffusor", "typing")
    return {
        "stages": data.get("stages") or [],
        "block_ids": data.get("block_ids") or [],
        "part_kinds": data.get("part_kinds") or [],
        "dit_class_markers": data.get("dit_class_markers") or [],
        "scheduler_display": data.get("scheduler_display") or [],
        "scheduler_flow_matching_markers": data.get("scheduler_flow_matching_markers") or [],
        "norm_type_kind": data.get("norm_type_kind") or [],
    }


def load_diffusion_text_encoders() -> dict[str, str]:
    """Text-encoder class name -> friendly label (``diffusor/text_encoders.yaml``;
    the whole file is the flat map)."""
    return {k: v for k, v in load("diffusor", "text_encoders").items() if isinstance(v, str)}


def load_diffusion_class_defaults() -> dict[str, dict[str, str]]:
    """Architectural facts HARDCODED in the diffusers model class but NOT in the
    config (``diffusor/class_defaults.yaml``) — surfaced (marked code-derived)
    when the config is silent (e.g. Flux's axial RoPE / QK-norm).

    Stored flow-style as ``field: ["<_class_name>=<value>", ...]``; returned as
    ``{canonical_field: {class_name: value}}``."""
    out: dict[str, dict[str, str]] = {}
    for field, entries in load("diffusor", "class_defaults").items():
        mapping: dict[str, str] = {}
        for entry in entries or []:
            if isinstance(entry, str) and "=" in entry:
                cls, _, val = entry.partition("=")
                mapping[cls.strip()] = val.strip()
        out[field] = mapping
    return out


# --- conformance domain (op-conformance diff: diagram structure vs HF forward) ---

def load_conformance_op_tokens() -> dict[str, str]:
    """``conformance/op_tokens.yaml`` inverted to ``{call_token: canonical_op}``.

    The file is authored canonical->[tokens] for readability; the extractor wants
    token->canonical. ``ignore`` maps its tokens to the sentinel ``""``."""
    out: dict[str, str] = {}
    for canonical, tokens in load("conformance", "op_tokens").items():
        for tok in tokens or []:
            out[str(tok)] = "" if canonical == "ignore" else str(canonical)
    return out


def load_conformance_type_roles() -> tuple[list[str], dict[str, list[str]]]:
    """``conformance/type_roles.yaml`` -> (priority_order, {role: [substrings]}).

    The extractor matches a constructed-class name against substrings in
    ``priority`` order (first hit wins), so a specific role beats generic
    ``linear``."""
    data = load("conformance", "type_roles")
    priority = [str(r) for r in (data.get("priority") or [])]
    roles = {k: [str(s) for s in (v or [])]
             for k, v in data.items() if k != "priority"}
    if not priority:                       # fall back to declared order
        priority = list(roles.keys())
    return priority, roles


def load_conformance_fact_markers() -> dict[str, list[str]]:
    """``conformance/fact_markers.yaml`` -> ``{marker: [substrings]}`` for the
    fact-conformance net (the SAME-op-kind, different-SEMANTICS dimensions
    op-presence is blind to): ``rotary`` (block applies RoPE -> a NoPE claim is
    fabricated) and ``linear_attn`` (self-attn is linear, not softmax)."""
    data = load("conformance", "fact_markers")
    return {k: [str(s) for s in (v or [])] for k, v in data.items()}


def load_conformance_map() -> dict:
    """``conformance/conformance_map.yaml`` -> view<->code overrides + markers.

    Returns ``{"views": {"<family>/<view>": "Class.method"},
    "single_stream_class_markers": [...]}``."""
    data = load("conformance", "conformance_map")
    views: dict[str, str] = {}
    for entry in data.get("views") or []:
        if isinstance(entry, str) and "=" in entry:
            key, _, val = entry.partition("=")
            views[key.strip()] = val.strip()
    return {
        "views": views,
        "single_stream_class_markers": [str(m) for m in (data.get("single_stream_class_markers") or [])],
    }


def load_conformance_abstractions() -> dict:
    """``conformance/abstractions.yaml`` -> the deliberate-abstraction allow-list.

    Returns ``{"omit_global": set, "omit_scoped": {"<family>/<view>": set(ops)},
    "composite": {drawn_op: set(absorbed_code_ops)}, "draw_extra":
    {"<family>/<view>": set(ops)}, "since": {citation_key: set(tokens)}}``."""
    data = load("conformance", "abstractions")

    def _scoped(entries):                  # ["family/view=op[,op…]", ...] -> {key: {ops}}
        out: dict[str, set[str]] = {}
        for e in entries or []:
            if isinstance(e, str) and "=" in e:
                key, _, ops = e.partition("=")
                out.setdefault(key.strip(), set()).update(
                    o.strip() for o in ops.split(",") if o.strip())
        return out

    composite: dict[str, set[str]] = {}
    for e in data.get("composite") or []:  # ["ffn=linear,activation", ...]
        if isinstance(e, str) and "=" in e:
            drawn, _, absorbed = e.partition("=")
            composite[drawn.strip()] = {a.strip() for a in absorbed.split(",") if a.strip()}
    since: dict[str, set[str]] = {}
    for e in data.get("since") or []:      # ["family/view=tokA,tokB", ...]
        if isinstance(e, str) and "=" in e:
            key, _, toks = e.partition("=")
            since[key.strip()] = {t.strip() for t in toks.split(",") if t.strip()}
    return {
        "omit_global": {str(o) for o in (data.get("omit_global") or [])},
        "omit_scoped": _scoped(data.get("omit_scoped")),
        "composite": composite,
        "draw_extra": _scoped(data.get("draw_extra")),
        "since": since,
    }


def load_conformance_wiring_roles() -> tuple[dict[str, str], dict[str, list[str]]]:
    """``conformance/wiring_roles.yaml`` -> (stage_role, role_params).

    ``stage_role``: a side-input's ``diffusion_stage`` -> conditioning role.
    ``role_params``: role -> forward()-parameter-name substrings that satisfy it.
    Both authored as ``"key=val[,val…]"`` flow entries so the loader needs no PyYAML."""
    data = load("conformance", "wiring_roles")
    stage_role: dict[str, str] = {}
    for e in data.get("stage_role") or []:
        if isinstance(e, str) and "=" in e:
            k, _, v = e.partition("=")
            stage_role[k.strip()] = v.strip()
    role_params: dict[str, list[str]] = {}
    for e in data.get("role_params") or []:
        if isinstance(e, str) and "=" in e:
            role, _, subs = e.partition("=")
            role_params[role.strip()] = [s.strip().lower() for s in subs.split(",") if s.strip()]
    return stage_role, role_params


def load_conformance_transitive() -> dict:
    """``conformance/transitive.yaml`` -> the recursive drill-conformance vocab.

    Returns a dict with: ``attention_compute_ops`` (frozenset), ``attention_compute_tokens``
    (frozenset), ``drawn_ignore`` (frozenset of drawn node-kinds dropped from the
    op diff), ``drawn_op_map`` (drawn-kind -> code-op), ``semantic_kinds`` (drawn
    kinds checked by marker presence, not op-kind), ``semantic_markers``
    ({kind: [substrings]}), and ``library_helpers`` (token -> frozenset(ops))."""
    data = load("conformance", "transitive")

    def _list(key):
        return [str(x) for x in (data.get(key) or [])]

    drawn_op_map: dict[str, str] = {}
    for e in _list("drawn_op_map"):            # ["select=route", ...]
        if "=" in e:
            k, _, v = e.partition("=")
            drawn_op_map[k.strip()] = v.strip()
    helpers: dict[str, frozenset[str]] = {}
    for e in _list("library_helpers"):         # ["repeat_kv=reshape", "dropout="]
        if "=" in e:
            name, _, ops = e.partition("=")
            helpers[name.strip()] = frozenset(o.strip() for o in ops.split(",") if o.strip())

    def _kv_list(key):                         # ["role=a,b", ...] -> {role: [a, b]}
        out: dict[str, list[str]] = {}
        for e in _list(key):
            if "=" in e:
                k, _, v = e.partition("=")
                out[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
        return out

    def _kv_str(key):                          # ["role=type", ...] -> {role: type}
        return {k: (v[0] if v else "") for k, v in _kv_list(key).items()}

    return {
        "attention_compute_ops": frozenset(_list("attention_compute_ops")),
        "attention_compute_tokens": frozenset(_list("attention_compute_tokens")),
        "drawn_ignore": frozenset(_list("drawn_ignore")),
        "drawn_op_map": drawn_op_map,
        "semantic_kinds": frozenset(_list("semantic_kinds")),
        "semantic_markers": {
            "rope": [s.lower() for s in _list("semantic_rope_markers")],
            "cache": [s.lower() for s in _list("semantic_cache_markers")],
        },
        "library_helpers": helpers,
        "drill_role_markers": _kv_list("drill_role_markers"),
        "drill_role_to_type": _kv_str("drill_role_to_type"),
        "drill_category": _kv_str("drill_category"),
        "drill_salient_missing": {k: frozenset(v) for k, v in _kv_list("drill_salient_missing").items()},
        "drill_op_equivalents": {k: frozenset(v) for k, v in _kv_list("drill_op_equivalents").items()},
        "selection_presentation_kinds": frozenset(_list("selection_presentation_kinds")),
        "composite_container_map": _kv_str("composite_container_map"),
        "processor_markers": frozenset(_list("processor_markers")),
    }


def _parse_flow_yaml(text: str) -> dict[str, list[str]]:
    """Minimal reader for ``key: [a, b, c]`` / ``key:`` + ``- item`` blocks.

    Handles ``#`` comments and quoted scalars — enough for the alias file when
    PyYAML is absent.  Not a general YAML parser.
    """
    data: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0] if " #" in raw else raw
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current is not None:
            data[current].append(_unquote(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            data[key] = [_unquote(x) for x in val[1:-1].split(",") if x.strip()]
            current = None
        elif val:
            data[key] = _unquote(val)
            current = None
        else:
            data[key] = []
            current = key
    return data


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


__all__ = [
    "load",
    "load_aliases",
    "load_ignored_fields",
    "load_transformer_typing",
    "load_layer_type_labels",
    "load_layer_topology",
    "load_diffusion_aliases",
    "load_diffusion_typing",
    "load_diffusion_text_encoders",
]
